"""Read-only access and deterministic replay of the frozen V1 experiment."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from . import cartpole_v1 as v1
from ..adaptation import AdaptationSettings, MetaResidualModel, RidgeResidualModel
from ..data import collect_iid, collect_sequential
from ..evaluation import one_step_metrics, rollout_metrics
from ..utils.randomness import stable_seed

METHOD_MAP = {'frozen': 'frozen', 'ridge': 'ridge', 'meta': 'meta_residual_1step'}
IDENTIFIERS = ['seed', 'trial', 'environment', 'category', 'regime', 'support_mode', 'K']


def baseline_config(directory):
    return json.loads((Path(directory)/'config.json').read_text(encoding='utf-8'))


def verify_frozen(directory, manifest):
    directory = Path(directory); saved = json.loads(Path(manifest).read_text(encoding='utf-8'))
    for name, expected in saved['files'].items():
        path = directory/name; value = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024*1024), b''): value.update(chunk)
        if value.hexdigest() != expected: raise AssertionError(f'Immutable V1 artifact changed: {name}')
    return len(saved['files'])


def load_baselines(directory, seed):
    config = baseline_config(directory)
    base, path = v1.require_base(Path(directory), seed, config, torch.device('cpu'))
    adapter = v1.require_adapter(Path(directory)/'checkpoints'/f'meta_seed{seed}.pt',
                                 v1.fingerprint(path), seed, config, torch.device('cpu'))
    return base, {'frozen': base, 'ridge': RidgeResidualModel(base, config['ridge']['regularization']),
                  'meta_residual_1step': MetaResidualModel(base, adapter, AdaptationSettings(**config['adaptation']))}


def final_data(config, seed, environment, regime, trial):
    settings = config['evaluation']; physics = v1.physics_for(config, environment)
    data_seed = stable_seed(seed, 'final_evaluation', environment['name'], regime, trial)
    sequential = collect_sequential(physics, max(settings['K']), data_seed, regime, 'cpu', stream='final/support')
    iid = collect_iid(physics, max(settings['K']), data_seed, regime, 'cpu', stream='final/iid_support')
    query = collect_sequential(physics, settings['query_steps'], data_seed, regime, 'cpu',
                               settings['query_trajectories'], stream='final/query')
    trajectory = collect_sequential(physics, max(settings['rollout_horizons']), data_seed, regime, 'cpu',
                                    settings['rollout_trajectories'], stream='final/rollout')
    for support in (sequential, iid):
        support.assert_disjoint(query); support.assert_disjoint(trajectory)
    query.assert_disjoint(trajectory)
    return {'sequential': sequential, 'iid': iid, 'query': query,
            'rollout': v1.rollout_inputs(trajectory, max(settings['rollout_horizons']), settings['rollout_trajectories'])}


class ReferenceMetrics:
    def __init__(self, directory):
        self.trials = pd.read_csv(Path(directory)/'per_trial_metrics.csv').set_index(IDENTIFIERS+['method'])
        self.rollouts = pd.read_csv(Path(directory)/'rollout_metrics.csv').set_index(IDENTIFIERS+['method','horizon'])

    def checks(self, identifiers, method, metric, horizons, support_hash, query_hash, tolerance=1e-6):
        legacy = next(old for old, new in METHOD_MAP.items() if new == method)
        key = tuple(identifiers[name] for name in IDENTIFIERS)+(legacy,)
        reference = self.trials.loc[key]; rows = []
        for name, candidate, expected in [('support_hash', support_hash, reference.support_hash),
                                           ('query_hash', query_hash, reference.query_hash)]:
            rows.append({**identifiers, 'method': method, 'check': name, 'passed': candidate == expected,
                         'candidate': candidate, 'reference': expected, 'absolute_difference': 0 if candidate==expected else 1,
                         'tolerance': 0, 'support_hash': support_hash, 'query_hash': query_hash})
        values = [('one_step_nmse', metric['nmse'], reference.nmse)]
        values.extend((f'rollout_h{h}', horizons[h]['nmse'], self.rollouts.loc[key+(h,), 'nmse']) for h in (5,20,50,100))
        for name, candidate, expected in values:
            difference = abs(candidate-float(expected)) if candidate is not None else float('inf')
            rows.append({**identifiers, 'method': method, 'check': name, 'passed': difference <= tolerance,
                         'candidate': candidate, 'reference': float(expected), 'absolute_difference': difference,
                         'tolerance': tolerance, 'support_hash': support_hash, 'query_hash': query_hash})
        return rows


def sampled_reproduction(directory):
    """Twenty exact legacy cases, all five seeds, both regimes and support modes."""
    torch.set_num_threads(1); config = baseline_config(directory); reference = ReferenceMetrics(directory)
    environment = next(item for item in config['test_environments'] if item['name']=='near_pole_mass')
    rows = []
    for seed in config['seeds']:
        base, models = load_baselines(directory, seed)
        for regime in config['evaluation']['regimes']:
            data = final_data(config, seed, environment, regime, 0)
            for mode in config['evaluation']['support_modes']:
                support = data[mode].prefix(20); support_hash = support.support_hash(); query_hash = data['query'].support_hash()
                identifiers = dict(seed=seed, trial=0, environment=environment['name'], category=environment['category'],
                                   regime=regime, support_mode=mode, K=20)
                for method, model in models.items():
                    model.reset_adaptation(); model.adapt(support)
                    metric = one_step_metrics(model, data['query'])
                    horizons = rollout_metrics(model, *data['rollout'], (5,20,50,100), config['evaluation']['rollout_state_scale'])
                    rows.extend(reference.checks(identifiers, method, metric, horizons, support_hash, query_hash))
                    base.assert_frozen(); model.reset_adaptation()
    if not all(row['passed'] for row in rows):
        raise AssertionError(f"Legacy V1 reproduction failed: {[row for row in rows if not row['passed']][:3]}")
    return rows
