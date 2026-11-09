"""Explicitly invoked future training/evaluation. Importing this file does no work."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from ..adaptation import AdaptationSettings, MetaResidualModel, RidgeResidualModel, meta_objective
from ..data import TransitionBatch, collect_iid, collect_sequential
from ..environments import CartPolePhysics, step
from ..evaluation import one_step_metrics, rollout_metrics, recovery_ratio, fraction_error_removed
from ..models import FeatureNormalizer, FrozenDynamicsModel, ResidualAdapter
from ..models.dynamics import PredictionSafetyError
from ..utils.artifacts import append_jsonl, new_run_directory, sanitized, write_json
from ..utils.randomness import seed_everything, stable_seed
from ..utils.runtime import Deadline, hardware_info, select_device


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / 'configs/cartpole/v1.json'


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def validate_config(config):
    """Fail before creating artifacts or allocating a training optimizer."""
    CartPolePhysics(**config['nominal'])
    AdaptationSettings(**config['adaptation'])
    if config['evaluation']['methods'] != ['frozen', 'ridge', 'meta']:
        raise ValueError('V1 primary methods must be frozen, ridge, meta')
    if type(config['ridge']['degree']) is not int or config['ridge']['degree'] != 1:
        raise ValueError('Main V1 Ridge must be linear; polynomial is a separate optional baseline')
    if not math.isfinite(config['ridge']['regularization']) or config['ridge']['regularization'] <= 0:
        raise ValueError('Ridge regularization must be finite and positive')
    seeds = config['seeds']
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(type(s) is not int for s in seeds):
        raise ValueError('V1 requires five independent integer training seeds')
    for section in ('base', 'meta', 'evaluation'):
        for key, value in config[section].items():
            if key in ('first_order', 'methods', 'regimes', 'support_modes', 'hidden_sizes',
                       'K', 'support_K', 'rollout_horizons', 'rollout_state_scale'):
                continue
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{section}.{key} must be finite and positive')
        for key in ('steps', 'batch_size', 'train_count', 'validation_count', 'sequential_horizon', 'validation_every',
                    'min_steps', 'task_pool', 'tasks_per_step', 'query_steps', 'query_trajectories',
                    'trials_per_seed', 'rollout_trajectories', 'cpu_threads'):
            if key in config[section] and type(config[section][key]) is not int:
                raise ValueError(f'{section}.{key} must be an integer')
        if config[section].get('device', config['device']) not in ('cuda', 'cpu'):
            raise ValueError('Stage device must be cpu or cuda')
    for section, key in (('evaluation', 'K'), ('evaluation', 'rollout_horizons'), ('meta', 'support_K')):
        values = config[section][key]
        if not values or any(type(x) is not int or x < 1 for x in values) or len(set(values)) != len(values):
            raise ValueError(f'{section}.{key} must contain distinct positive integers')
    if len(config['evaluation']['regimes']) != 2 or set(config['evaluation']['regimes']) != {'perturbed_balance', 'broad'}:
        raise ValueError('V1 requires both excited data regimes')
    if len(config['evaluation']['support_modes']) != 2 or set(config['evaluation']['support_modes']) != {'sequential', 'iid'}:
        raise ValueError('V1 requires sequential main support and IID reference')
    names = [item['name'] for item in config['test_environments'] + config['development']]
    if len(names) != len(set(names)):
        raise ValueError('Development and final environment names must be distinct')
    for item in config['test_environments'] + config['development']:
        physics_for(config, item)
    for name, bounds in config['meta_family'].items():
        if name not in config['nominal'] or len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
            raise ValueError(f'Invalid meta-family bounds: {name}')
        for value in bounds:
            CartPolePhysics(**{**config['nominal'], name: value})
    if config['base']['train_count'] % 4 or config['base']['validation_count'] % 4:
        raise ValueError('Nominal counts must be divisible by four for balanced data strata')
    if type(config['base'].get('sequential_horizon')) is not int or config['base']['sequential_horizon'] < 1:
        raise ValueError('base.sequential_horizon must be a positive integer')
    if type(config['meta']['first_order']) is not bool:
        raise ValueError('meta.first_order must be a boolean')
    for section in ('base', 'meta'):
        widths = config[section]['hidden_sizes']
        if not widths or any(type(w) is not int or w < 1 for w in widths):
            raise ValueError(f'{section}.hidden_sizes must be positive integers')
    scales = config['evaluation']['rollout_state_scale']
    if len(scales) != 4 or any(not math.isfinite(v) or v <= 0 for v in scales):
        raise ValueError('Rollout state scales must be four finite positive values')


def physics_for(config, item):
    unknown = set(item) - set(config['nominal']) - {'name', 'category'}
    if unknown:
        raise ValueError(f'Unknown physics/config keys: {sorted(unknown)}')
    return CartPolePhysics(**{**config['nominal'], **{k: v for k, v in item.items() if k in config['nominal']}})


def nominal_data(config, seed, count, device, namespace):
    """Equal IID/sequential shares in each regime; only exact nominal physics."""
    physics = CartPolePhysics(**config['nominal'])
    pieces = []
    for regime in config['evaluation']['regimes']:
        part = count // 4
        iid = collect_iid(physics, part, stable_seed(seed, namespace, regime, 'iid'),
                          regime, device, stream=namespace + '/iid')
        horizon = config['base']['sequential_horizon']
        trajectories = math.ceil(part / horizon)
        seq = collect_sequential(physics, horizon, stable_seed(seed, namespace, regime, 'seq'),
                                 regime, device, trajectories, stream=namespace + '/seq').prefix(part)
        pieces.extend([iid, seq])
    # Concatenation gets globally distinct identities; feature/target bytes are unchanged.
    return TransitionBatch(torch.cat([p.state for p in pieces]), torch.cat([p.action for p in pieces]),
                           torch.cat([p.next_state for p in pieces]), torch.arange(count, device=device),
                           torch.zeros(count, device=device, dtype=torch.int64),
                           provenance=f'nominal/{seed}/{namespace}', metadata={'physics': asdict(physics)})


def save_stage_status(run, stage, **details):
    path = run / 'status.json'
    status = json.loads(path.read_text()) if path.exists() else {'complete': False, 'stages': {}}
    status['stages'][stage] = sanitized(details)
    status['complete'] = bool(status['stages'].get('evaluate', {}).get('complete', False))
    write_json(path, status)
    return status


def require_base(run, seed, config, device):
    path = run / 'checkpoints' / f'base_seed{seed}.pt'
    if not path.exists():
        raise RuntimeError(f'Missing competent base checkpoint: {path}')
    base = FrozenDynamicsModel.load_checkpoint(path, device, freeze=True)
    metadata = base.checkpoint_metadata
    if not base.trained or metadata.get('competent') is not True:
        raise RuntimeError('Nominal competence gate has not passed; meta-training/evaluation is blocked')
    if base.seed != seed or metadata.get('training_seed') != seed or metadata.get('config_hash') != config_hash(config):
        raise RuntimeError('Base seed/config provenance mismatch')
    return base, path


def train_base(run, config, device):
    settings = config['base']; deadline = Deadline(settings['max_minutes']); completed = []
    save_stage_status(run, 'train-base', complete=False, state='running')
    try:
        for seed in config['seeds']:
            deadline.check()
            final = run / 'checkpoints' / f'base_seed{seed}.pt'
            if final.exists():
                require_base(run, seed, config, device); completed.append(seed); continue
            seed_everything(seed)
            training = nominal_data(config, seed, settings['train_count'], device, 'base_train')
            validation = nominal_data(config, seed, settings['validation_count'], device, 'base_validation')
            training.assert_disjoint(validation)
            norm = FeatureNormalizer.fit(training.features(), training.delta())
            base = FrozenDynamicsModel(norm, seed, tuple(settings['hidden_sizes'])).to(device)
            optimizer = torch.optim.Adam(base.parameters(), lr=settings['learning_rate'])
            sampler = torch.Generator(device=device).manual_seed(stable_seed(seed, 'base_batches'))
            xn = base.normalizer.encode_x(training.features()); yn = base.normalizer.encode_y(training.delta())
            resume = run / 'checkpoints' / f'base_resume_seed{seed}.pt'
            iteration = 0; best = float('inf'); best_state = None; best_metrics = None
            if resume.exists():
                saved = torch.load(resume, map_location=device, weights_only=True)
                if saved['config_hash'] != config_hash(config): raise RuntimeError('Base resume configuration mismatch')
                base.load_state_dict(saved['current']); optimizer.load_state_dict(saved['optimizer'])
                sampler.set_state(saved['sampler'].cpu()); iteration = saved['iteration']
                best, best_state, best_metrics = saved['best'], saved['best_state'], saved['best_metrics']
            else:
                atomic_save(resume, {'current': base.state_dict(), 'optimizer': optimizer.state_dict(),
                            'sampler': sampler.get_state(), 'iteration': 0, 'best': best, 'best_state': None,
                            'best_metrics': None, 'config_hash': config_hash(config)})
            started = time.perf_counter()
            try:
                for iteration in range(iteration + 1, settings['steps'] + 1):
                    deadline.check()
                    indices = torch.randint(len(training), (settings['batch_size'],), generator=sampler, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = (base.network(xn[indices]) - yn[indices]).square().mean()
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nominal training loss nonfinite')
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    if not all(bool(torch.isfinite(p).all()) for p in base.parameters()):
                        raise FloatingPointError('Nominal parameters nonfinite; resume from last saved state')
                    if iteration % settings['validation_every'] == 0 or iteration == settings['steps']:
                        metrics = one_step_metrics(base, validation)
                        append_jsonl(run / 'logs/base.jsonl', {'seed': seed, 'step': iteration, 'train_loss': float(loss.detach()),
                                      'gradient_norm': float(gradient), 'validation': metrics, 'elapsed_seconds': time.perf_counter()-started})
                        if metrics['finite'] and metrics['nmse'] < best:
                            best = metrics['nmse']; best_metrics = metrics
                            best_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
                        atomic_save(resume, {'current': base.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'sampler': sampler.get_state(), 'iteration': iteration, 'best': best,
                                    'best_state': best_state, 'best_metrics': best_metrics, 'config_hash': config_hash(config)})
                        if iteration >= settings['min_steps'] and nominal_competent(best_metrics, settings): break
            except (TimeoutError, KeyboardInterrupt):
                # Last committed validation boundary is the reproducible restart point.
                raise
            if best_state is not None: base.load_state_dict(best_state)
            competent = nominal_competent(best_metrics, settings)
            base.trained = competent
            base.save_checkpoint(final if competent else run/'checkpoints'/f'base_incompetent_seed{seed}.pt',
                                 {'training_seed': seed, 'config_hash': config_hash(config), 'competent': competent,
                                  'validation': best_metrics, 'steps_completed': iteration, 'training_count': len(training),
                                  'normalizer_fit': 'nominal training only'})
            if not competent: raise RuntimeError(f'Seed {seed} failed predeclared nominal competence gate: {best_metrics}')
            base.freeze(); base.assert_frozen(); completed.append(seed)
        save_stage_status(run, 'train-base', complete=True, completed_seeds=completed)
    except BaseException as exc:
        save_stage_status(run, 'train-base', complete=False, completed_seeds=completed, reason=str(exc), exception=type(exc).__name__)
        raise


def nominal_competent(metrics, settings):
    return bool(metrics and metrics['finite'] and metrics['nmse'] <= settings['competence_nmse']
                and max(metrics['component_nmse']) <= settings['component_competence_nmse'])


def task_data(config, physics, seed, device, namespace, regime, support_count=None):
    support = collect_sequential(physics, support_count or max(config['meta']['support_K']), seed,
                                 regime, device, stream=namespace + '/support')
    query = collect_sequential(physics, config['meta']['query_steps'], seed, regime, device,
                               config['meta']['query_trajectories'], stream=namespace + '/query')
    support.assert_disjoint(query)
    return support, query


def make_meta_pool(config, seed, device, deadline):
    generator = torch.Generator().manual_seed(stable_seed(seed, 'meta_physics'))
    pool = []
    for task in range(config['meta']['task_pool']):
        deadline.check()
        parameters = dict(config['nominal'])
        for name, (low, high) in config['meta_family'].items():
            parameters[name] = low + (high-low) * float(torch.rand((), generator=generator))
        physics = CartPolePhysics(**parameters)
        regime = config['evaluation']['regimes'][task % 2]
        support, query = task_data(config, physics, stable_seed(seed, 'meta_task', task), device, f'meta_train/{task}', regime)
        pool.append((support, query))
    return pool


def development_tasks(config, seed, device):
    return [task_data(config, physics_for(config, item), stable_seed(seed, 'development', item['name'], regime),
                      device, f"development/{item['name']}/{regime}", regime)
            for item in config['development'] for regime in config['evaluation']['regimes']]


def validate_meta(base, adapter, tasks, config):
    model = MetaResidualModel(base, adapter, AdaptationSettings(**config['adaptation']))
    scores = []; logs = []
    for support, query in tasks:
        for k in config['meta']['support_K']:
            model.reset_adaptation(); log = model.adapt(support.prefix(k))
            metric = one_step_metrics(model, query)
            logs.append({'K': k, 'adaptation': log, 'metric': metric})
            # Rejected validation tasks are explicit failures, not favorable skipped tasks.
            if not log['accepted'] or not metric['finite']: return float('inf'), logs
            scores.append(metric['nmse'])
    model.reset_adaptation(); base.assert_frozen()
    return sum(scores)/len(scores), logs


def meta_train(run, config, device):
    device = select_device(config['meta'].get('device', str(device)))
    if device.type == 'cpu': torch.set_num_threads(config['meta'].get('cpu_threads', 1))
    write_json(run/'hardware_meta.json', hardware_info(device))
    settings = config['meta']; deadline = Deadline(settings['max_minutes']); completed = []
    save_stage_status(run, 'meta-train', complete=False, state='running', first_order=settings['first_order'])
    try:
        for seed in config['seeds']:
            deadline.check(); base, base_path = require_base(run, seed, config, device)
            base_hash = fingerprint(base_path)
            final = run / 'checkpoints' / f'meta_seed{seed}.pt'
            if final.exists():
                require_adapter(final, base_hash, seed, config, device); completed.append(seed); continue
            seed_everything(stable_seed(seed, 'meta_training'))
            pool = make_meta_pool(config, seed, device, deadline)
            development = development_tasks(config, seed, device)
            adapter = ResidualAdapter(stable_seed(seed, 'adapter'), tuple(settings['hidden_sizes'])).to(device)
            optimizer = torch.optim.Adam(adapter.parameters(), lr=settings['learning_rate'])
            sampler = torch.Generator().manual_seed(stable_seed(seed, 'meta_batches'))
            resume = run/'checkpoints'/f'meta_resume_seed{seed}.pt'
            iteration = 0; best = float('inf'); best_state = None; outer_updates = 0
            if resume.exists():
                saved = torch.load(resume, map_location=device, weights_only=True)
                if saved['base_hash'] != base_hash or saved['config_hash'] != config_hash(config):
                    raise RuntimeError('Meta resume base/config fingerprint mismatch')
                adapter.load_state_dict(saved['current']); optimizer.load_state_dict(saved['optimizer'])
                sampler.set_state(saved['sampler'].cpu()); iteration = saved['iteration']; best = saved['best']; best_state = saved['best_state']
                outer_updates = saved.get('outer_updates', 0)
            else:
                adapter.save_checkpoint(run/'checkpoints'/f'ordinary_init_seed{seed}.pt',
                                        {'training_seed': seed, 'base_sha256': base_hash, 'internal_only': True})
                atomic_save(resume, {'current': adapter.state_dict(), 'optimizer': optimizer.state_dict(),
                            'sampler': sampler.get_state(), 'iteration': 0, 'best': best, 'best_state': None,
                            'base_hash': base_hash, 'config_hash': config_hash(config), 'outer_updates': 0})
            inner = AdaptationSettings(**config['adaptation'])
            for iteration in range(iteration + 1, settings['steps'] + 1):
                deadline.check(); optimizer.zero_grad(set_to_none=True)
                tasks = torch.randint(len(pool), (settings['tasks_per_step'],), generator=sampler).tolist()
                losses = []; logs = []
                for slot, task in enumerate(tasks):
                    support, query = pool[task]
                    k = settings['support_K'][(iteration + slot) % len(settings['support_K'])]
                    try:
                        loss, log = meta_objective(base, adapter, support.prefix(k), query, inner, first_order=settings['first_order'])
                        if log['accepted']: losses.append(loss)
                        logs.append({'task': task, 'K': k, **log})
                    except PredictionSafetyError as exc:
                        logs.append({'task': task, 'K': k, 'accepted': False, 'reason': str(exc)})
                applied = False; gradient = None
                if losses:
                    objective = torch.stack(losses).mean(); objective.backward()
                    if all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()):
                        gradient = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
                        previous = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
                        old_optimizer = copy.deepcopy(optimizer.state_dict())
                        optimizer.step(); applied = all(bool(torch.isfinite(p).all()) for p in adapter.parameters())
                        if not applied:
                            adapter.load_state_dict(previous); optimizer.load_state_dict(old_optimizer)
                    optimizer.zero_grad(set_to_none=True)
                if applied: outer_updates += 1
                append_jsonl(run/'logs/meta.jsonl', {'seed': seed, 'step': iteration, 'outer_update_applied': applied,
                              'outer_loss': float(torch.stack(losses).mean().detach()) if losses else None,
                              'outer_gradient_norm': gradient, 'tasks': logs, 'first_order': settings['first_order']})
                if iteration % settings['validation_every'] == 0 or iteration == settings['steps']:
                    score, validation_logs = validate_meta(base, adapter, development, config)
                    append_jsonl(run/'logs/meta_validation.jsonl', {'seed': seed, 'step': iteration, 'score': score, 'tasks': validation_logs})
                    if math.isfinite(score) and score < best:
                        best = score; best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
                    atomic_save(resume, {'current': adapter.state_dict(), 'optimizer': optimizer.state_dict(),
                                'sampler': sampler.get_state(), 'iteration': iteration, 'best': best, 'best_state': best_state,
                                'base_hash': base_hash, 'config_hash': config_hash(config), 'outer_updates': outer_updates})
                    base.assert_frozen()
            if best_state is None: raise RuntimeError(f'No valid meta validation checkpoint for seed {seed}')
            if outer_updates == 0: raise RuntimeError(f'No accepted outer optimizer update for seed {seed}; adapter remains untrained')
            adapter.load_state_dict(best_state); adapter.meta_trained = True
            adapter.save_checkpoint(final, {'training_seed': seed, 'base_sha256': base_hash,
                                    'config_hash': config_hash(config), 'steps_completed': iteration,
                                    'development_nmse': best, 'outer_updates': outer_updates, 'first_order': settings['first_order']})
            base.assert_frozen(); completed.append(seed)
        save_stage_status(run, 'meta-train', complete=True, completed_seeds=completed)
    except BaseException as exc:
        save_stage_status(run, 'meta-train', complete=False, completed_seeds=completed, reason=str(exc), exception=type(exc).__name__)
        raise


def require_adapter(path, base_hash, seed, config, device):
    adapter = ResidualAdapter.load_checkpoint(path, device)
    metadata = adapter.checkpoint_metadata
    if not adapter.meta_trained or metadata.get('training_seed') != seed or metadata.get('base_sha256') != base_hash or metadata.get('config_hash') != config_hash(config):
        raise RuntimeError('Meta checkpoint not trained or wrong base/seed/config fingerprint')
    return adapter


def nominal_counterfactual(query, config):
    """Evaluator-only nominal error floor at identical observable states/actions."""
    following = step(query.state, query.action, CartPolePhysics(**config['nominal']))
    return TransitionBatch(query.state, query.action, following, query.trajectory_id, query.transition_id,
                           query.provenance + '/nominal_counterfactual', {'evaluator_only': True})


def rollout_inputs(batch, horizon, trajectories):
    if len(batch) != horizon * trajectories: raise ValueError('Rollout dataset shape mismatch')
    initial = batch.state[:trajectories]
    actions = batch.action.reshape(horizon, trajectories, 1).transpose(0, 1)
    truth = torch.cat([initial[:, None], batch.next_state.reshape(horizon, trajectories, 4).transpose(0, 1)], dim=1)
    return initial, actions, truth


def metric_row(metric):
    row = {'nmse': metric['nmse'], 'finite': metric['finite'], 'failure': metric['failure']}
    for i, name in enumerate(('x', 'xdot', 'theta', 'thetadot')):
        row['rmse_' + name] = metric['component_rmse'][i]
        row['component_nmse_' + name] = metric['component_nmse'][i]
    return row


def evaluation_units(config):
    for seed in config['seeds']:
        for environment in config['test_environments']:
            for regime in config['evaluation']['regimes']:
                for trial in range(config['evaluation']['trials_per_seed']):
                    for mode in config['evaluation']['support_modes']:
                        for k in config['evaluation']['K']:
                            yield seed, environment, regime, trial, mode, k


def unit_key(seed, environment, regime, trial, mode, k):
    return hashlib.sha256(f'{seed}/{environment["name"]}/{regime}/{trial}/{mode}/{k}'.encode()).hexdigest()[:24]


def export_csv(run):
    files = sorted((run/'logs/evaluation_units').glob('*.json'))
    rows = []; rollouts = []
    for path in files:
        saved = json.loads(path.read_text()); rows.extend(saved['trials']); rollouts.extend(saved['rollouts'])
    for name, values in [('per_trial_metrics.csv', rows), ('rollout_metrics.csv', rollouts)]:
        if not values: continue
        fields = list(dict.fromkeys(key for row in values for key in row))
        with (run/name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    return len(files)


def evaluate(run, config, device):
    device = select_device(config['evaluation'].get('device', str(device)))
    if device.type == 'cpu': torch.set_num_threads(config['evaluation'].get('cpu_threads', 1))
    write_json(run/'hardware_evaluation.json', hardware_info(device))
    settings = config['evaluation']; deadline = Deadline(settings['max_minutes'])
    units = run/'logs/evaluation_units'; units.mkdir(exist_ok=True)
    expected = len(list(evaluation_units(config))); completed = 0
    save_stage_status(run, 'evaluate', complete=False, expected_units=expected)
    cache = {}; loaded_seed = None
    try:
        for seed, environment, regime, trial, mode, k in evaluation_units(config):
            deadline.check(); key = unit_key(seed, environment, regime, trial, mode, k)
            path = units/(key+'.json')
            if seed != loaded_seed:
                base, base_path = require_base(run, seed, config, device)
                base_hash = fingerprint(base_path)
                adapter_path = run/'checkpoints'/f'meta_seed{seed}.pt'
                adapter = require_adapter(adapter_path, base_hash, seed, config, device)
                adapter_hash = fingerprint(adapter_path)
                models = {'frozen': base, 'ridge': RidgeResidualModel(base, config['ridge']['regularization']),
                          'meta': MetaResidualModel(base, adapter, AdaptationSettings(**config['adaptation']))}
                loaded_seed = seed; cache = {}
            if path.exists():
                saved = json.loads(path.read_text())
                if (saved['config_hash'] != config_hash(config) or saved.get('base_sha256') != base_hash
                        or saved.get('adapter_sha256') != adapter_hash):
                    raise RuntimeError('Evaluation resume config/checkpoint fingerprint mismatch')
                completed += 1; continue
            condition = (environment['name'], regime, trial)
            if cache.get('condition') != condition:
                physics = physics_for(config, environment)
                data_seed = stable_seed(seed, 'final_evaluation', *condition)
                sequential = collect_sequential(physics, max(settings['K']), data_seed, regime, device, stream='final/support')
                iid = collect_iid(physics, max(settings['K']), data_seed, regime, device, stream='final/iid_support')
                query = collect_sequential(physics, settings['query_steps'], data_seed, regime, device,
                                           settings['query_trajectories'], stream='final/query')
                rollout_data = collect_sequential(physics, max(settings['rollout_horizons']), data_seed, regime, device,
                                                  settings['rollout_trajectories'], stream='final/rollout')
                sequential.assert_disjoint(query); iid.assert_disjoint(query)
                query.assert_disjoint(rollout_data); sequential.assert_disjoint(rollout_data); iid.assert_disjoint(rollout_data)
                inputs = rollout_inputs(rollout_data, max(settings['rollout_horizons']), settings['rollout_trajectories'])
                baseline = one_step_metrics(base, query)
                id_error = one_step_metrics(base, nominal_counterfactual(query, config))['nmse']
                frozen_rollout = rollout_metrics(base, *inputs, settings['rollout_horizons'], settings['rollout_state_scale'])
                cache = dict(condition=condition, sequential=sequential, iid=iid, query=query, inputs=inputs,
                             baseline=baseline, id_error=id_error, frozen_rollout=frozen_rollout)
            support = cache[mode].prefix(k); support_hash = support.support_hash()
            rows = []; rollout_rows = []; adaptation_logs = []
            identifiers = dict(seed=seed, trial=trial, environment=environment['name'], category=environment['category'],
                               regime=regime, support_mode=mode, K=k)
            for method, model in models.items():
                model.reset_adaptation(); log = model.adapt(support)
                metric = cache['baseline'] if method == 'frozen' else one_step_metrics(model, cache['query'])
                horizons = cache['frozen_rollout'] if method == 'frozen' else rollout_metrics(
                    model, *cache['inputs'], settings['rollout_horizons'], settings['rollout_state_scale'])
                row = {**identifiers, 'method': method, **metric_row(metric), 'E_before': cache['baseline']['nmse'],
                       'E_ID': cache['id_error'], 'E_ID_definition': 'nominal counterfactual at same heldout states/actions',
                       'error_reduction': fraction_error_removed(metric['nmse'], cache['baseline']['nmse']),
                       'recovery_ratio': recovery_ratio(metric['nmse'], cache['baseline']['nmse'], cache['id_error']),
                       'base_parameter_count': base.parameter_count, 'adapter_parameter_count': adapter.parameter_count if method=='meta' else 0,
                       'ridge_coefficient_count': 28 if method=='ridge' else 0, 'adaptation_seconds': log['adaptation_seconds'],
                       'support_hash': support_hash, 'support_count': len(support), 'query_hash': cache['query'].support_hash(),
                       'query_count': len(cache['query']), 'steps': log['gradient_steps'],
                       'stable': bool(log['accepted'] and metric['finite']), 'update_accepted': log['accepted']}
                rows.append(row); adaptation_logs.append({'method': method, **log})
                for horizon, metrics in horizons.items():
                    rollout_rows.append({**identifiers, 'method': method, 'horizon': horizon, **metric_row(metrics),
                                         'endpoint_nmse': metrics['endpoint_nmse'], 'valid_steps': metrics['valid_steps'],
                                         'stable': bool(log['accepted'] and metrics['finite']), 'support_hash': support_hash})
                base.assert_frozen(); model.reset_adaptation()
            # Commit whole matched unit atomically so resume cannot duplicate or orphan methods.
            temporary = path.with_suffix('.tmp')
            write_json(temporary, {'config_hash': config_hash(config), 'base_sha256': base_hash,
                                  'adapter_sha256': adapter_hash, 'trials': rows, 'rollouts': rollout_rows,
                                  'adaptation_logs': adaptation_logs}); temporary.replace(path)
            completed += 1
        save_stage_status(run, 'evaluate', complete=True, completed_units=completed, expected_units=expected,
                          expected_trial_rows=expected*3, expected_rollout_rows=expected*3*len(settings['rollout_horizons']))
    except BaseException as exc:
        save_stage_status(run, 'evaluate', complete=False, completed_units=completed, expected_units=expected,
                          reason=str(exc), exception=type(exc).__name__)
        raise
    finally:
        export_csv(run)
        plot(run)


def plot(run):
    from ..evaluation.report import generate_report
    status_path = run/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'complete': False}
    return generate_report(run, status)


def verify_source_snapshot(run):
    manifest = json.loads((run/'source_hashes.json').read_text())
    for relative, expected in manifest.items():
        if not relative.endswith('.py'): continue
        path = ROOT/relative
        if not path.exists() or fingerprint(path) != expected:
            raise RuntimeError(f'Source changed since this run snapshot: {relative}; restore saved source or start a new run')


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    parser = argparse.ArgumentParser(description='CartPole V1. No work happens without an explicit stage and --execute.')
    parser.add_argument('stage', choices=['train-base', 'meta-train', 'evaluate', 'plot', 'all'])
    parser.add_argument('--execute', action='store_true', help='Explicit authorization to launch the requested training/evaluation stage')
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--run-dir', type=Path, default=None, help='Continue an existing saved run; configuration remains locked')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    args = parser.parse_args(argv)
    if args.stage != 'plot' and not args.execute:
        parser.error('Training/evaluation requires --execute. No training has started.')
    if args.stage in ('meta-train', 'evaluate', 'plot') and args.run_dir is None:
        parser.error('This stage requires --run-dir from the nominal training run')
    run = args.run_dir.resolve() if args.run_dir else None
    if run is not None:
        if (run/'FROZEN_RESULT.md').exists():
            parser.error('This V1 result is immutable. Use the V2 runner and a separate output directory.')
        if not (run/'config.json').is_file(): parser.error('Run directory must contain a saved config.json')
        config = json.loads((run/'config.json').read_text())
        if args.config and config_hash(json.loads(args.config.read_text())) != config_hash(config):
            parser.error('Configuration differs from the locked run')
        if args.device and args.device != config['device']: parser.error('Device is locked in saved run configuration')
    else:
        config = json.loads((args.config or DEFAULT_CONFIG).read_text())
        if args.device: config['device'] = args.device
    validate_config(config)
    if args.stage == 'plot':
        plot(run); print(f'Saved report and figures: {run}'); return 0
    device = select_device(config['device'])
    if run is None:
        run = new_run_directory(ROOT, config)
        write_json(run/'hardware.json', hardware_info(device))
        save_stage_status(run, 'prepared', complete=True, config_hash=config_hash(config))
    else:
        verify_source_snapshot(run)
    print(f'RUN_DIRECTORY={run}', flush=True)
    try:
        if args.stage in ('train-base', 'all'): train_base(run, config, device)
        if args.stage in ('meta-train', 'all'): meta_train(run, config, device)
        if args.stage in ('evaluate', 'all'): evaluate(run, config, device)
    except (Exception, KeyboardInterrupt) as exc:
        (run/'logs/failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        print(f'INCOMPLETE: {type(exc).__name__}: {exc}. Resume outputs: {run}', flush=True)
        plot(run)
        return 1
    print(f'Stage completed: {args.stage}; artifacts: {run}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
