"""Model/support reuse, atomic caches, and explicit provenance."""
import copy
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from adaptive_world_model.data import collect_sequential, TransitionBatch
from adaptive_world_model.environments.cartpole import CartPolePhysics, observable_features
from adaptive_world_model.utils.randomness import stable_seed
from adaptive_world_model.utils.artifacts import sanitized
from control_evaluate import load_models
from .protocol import ROOT, HISTORICAL


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(sanitized(data), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def transition_batch(s, a, n, provenance):
    count = len(s)
    return TransitionBatch(torch.as_tensor(s).float().cpu(), torch.as_tensor(a).float().cpu(),
                           torch.as_tensor(n).float().cpu(), torch.zeros(count, dtype=torch.long),
                           torch.arange(count), provenance)


class Context:
    def __init__(self, run, cfg, device):
        self.run, self.cfg, self.device = Path(run), cfg, device
        self.models, self.pools, self.supports, self.trains = {}, {}, {}, {}
        self.digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    def physics(self, name):
        env = next(x for x in self.cfg['environments'] if x['name'] == name)
        return CartPolePhysics(**{k:v for k,v in env.items() if k not in ['name', 'category']})

    def templates(self, seed):
        if seed not in self.models:
            self.models[seed] = load_models(seed)
        return self.models[seed]

    def support(self, seed, condition, trial, mode, k):
        key = (seed, condition['name'], trial)
        if key not in self.pools:
            p = self.physics(condition['environment'])
            regime = 'perturbed_balance' if condition['task'] == 'balance' else 'broad'
            pool = collect_sequential(p, self.cfg['support_pool_steps'], stable_seed(seed, 'diagnostic_support', *key[1:]),
                                      regime, batch_size=self.cfg['support_pool_trajectories'], stream='diagnostic_passive_pool')
            self.pools[key] = pool
        cache = (*key, mode, k)
        if cache not in self.supports:
            pool = self.pools[key]
            base, _ = self.templates(seed)
            z = base.normalizer.encode_x(pool.features())
            if mode == 'consecutive':
                ix = torch.where(pool.trajectory_id == 0)[0][:k]
            elif mode == 'random_multi':
                g = torch.Generator().manual_seed(stable_seed(seed, 'support_random', condition['name'], trial))
                ix = torch.randperm(len(pool), generator=g)[:k]
            else:
                # Selection uses observables only, never next states, query errors, or control outcomes.
                z = z[:, [2, 3, 4]] if mode == 'state_diverse' else z
                distance = torch.full((len(z),), float('inf'))
                chosen, first = [], int(torch.linalg.vector_norm(z-z.mean(0), dim=1).argmin())
                for _ in range(k):
                    chosen.append(first)
                    distance = torch.minimum(distance, (z-z[first]).square().sum(1))
                    distance[chosen] = -1
                    first = int(distance.argmax())
                ix = torch.tensor(chosen)
            support = pool.select(ix)
            self.supports[cache] = support
            target = self.run/'support'/f'{seed}_{condition["name"]}_{trial}_{mode}_{k}.npz'
            save_npz(target, states=support.state.numpy(), actions=support.action.numpy(), next_states=support.next_state.numpy(),
                     pool_indices=ix.numpy(), trajectory_id=support.trajectory_id.numpy(), transition_id=support.transition_id.numpy())
        return self.supports[cache]

    def deploy(self, seed, settings, supports=None):
        base, templates = self.templates(seed)
        models, logs = [], []
        for i, s in enumerate(settings):
            m = base if s['method'] in ['frozen', 'oracle'] else copy.deepcopy(templates[s['method']])
            log = None
            if m is not base:
                m.reset_adaptation()
                if supports is not None and supports[i] is not None:
                    log = m.adapt(supports[i])
            models.append(m)
            logs.append(log)
        return base, models, logs

    def training_reference(self, seed):
        if seed not in self.trains:
            from adaptive_world_model.experiments.cartpole_v1 import nominal_data
            cfg = json.loads((ROOT/'configs/cartpole/v1.json').read_text())
            data = nominal_data(cfg, seed, cfg['base']['train_count'], 'cpu', 'base_train')
            g = torch.Generator().manual_seed(stable_seed(seed, 'diagnostic_training_reference'))
            ix = torch.randperm(len(data), generator=g)[:self.cfg['training_reference_count']]
            self.trains[seed] = data.select(ix)
        return self.trains[seed]

    def cached(self, path):
        if not path.exists():
            return None
        packet = json.loads(path.read_text())
        if packet['config_sha256'] != self.digest:
            raise RuntimeError(f'Incompatible cache: {path}')
        if not path.with_suffix('.npz').exists():
            return None
        return packet


def features_numpy(states, actions):
    s, u = torch.as_tensor(states).float(), torch.as_tensor(actions).float()
    return observable_features(s, u).numpy()
