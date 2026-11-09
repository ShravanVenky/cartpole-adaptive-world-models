"""Predeclared nonfactorial matrix and development/confirmation separation."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
HISTORICAL = ROOT / 'results/cartpole/20261023_v2_control'
METHODS = ['frozen', 'ridge', 'meta_residual_1step', 'meta_residual_rollout', 'oracle']
ADAPTERS = METHODS[1:4]
ALPHAS = [i / 8 for i in range(9)]
BUDGETS = {'baseline': (128, 3), 'expanded': (512, 5), 'high': (1024, 5)}


def configuration():
    old = json.loads((HISTORICAL / 'config_locked.json').read_text())
    names = ['nominal', 'combined_geometry', 'strong_gain', 'near_gain', 'combined_friction_gain']
    conditions = [dict(name=n, environment=n, task='swingup', horizon=30, online=False) for n in names]
    conditions += [dict(name='online_' + n, environment=n, task='swingup', horizon=30, online=True)
                   for n in ['strong_gain', 'combined_geometry']]
    conditions += [dict(name='balance_gain', environment='strong_gain', task='balance', horizon=15, online=False)]
    return dict(schema=1, seeds=old['seeds'], conditions=conditions,
                environments=old['environments'], development_trials=[100], confirmation_trials=[200, 201, 202],
                alphas=ALPHAS, budgets={k:list(v) for k,v in BUDGETS.items()}, elites='one eighth of candidates: 16, 64, 128',
                swingup_steps=600, balance_steps=400, shift_swingup=25, shift_balance=150,
                online_updates=[5, 10, 20, 50], support_sizes=[20, 50], support_pool_steps=128,
                support_pool_trajectories=8, support_types=['consecutive', 'random_multi', 'state_diverse', 'state_action_diverse'],
                rail_boundary=2.4, safety_soft=1e5, stronger_soft=1e6, safety_margin=.3,
                near_upright_speed_penalty=4., terminal_stabilization_extra=20.,
                success_angle=.15, success_omega=.8, sustained_steps=100, recovery_window=25,
                excessive_crossing_speed=4., failure_penalty=100000., numerical_state_bound=100.,
                replay_categories=['frozen_success', 'frozen_success_meta_failure', 'ridge_failure', 'large_angle',
                                   'high_omega', 'near_rail', 'combined', 'online_shift'],
                replay_per_category_per_seed=1, replay_broad_candidates=128, replay_chunk=128,
                replay_top_k=10, replay_horizon=30, training_reference_count=8192, max_control_batch_rows=64,
                noise='none', offline_training=False, actual_termination='unchanged: rail/nonfinite/other-state bound100',
                selection='development only; highest success then lowest mean baseline-objective cost; one winner per family/method',
                confirmation='independent initial/support/random-number namespaces and trials; no confirmation tuning',
                online_support='own actual past post-change transitions only; reset/refit cumulative prefixes',
                coverage='normalized six-dimensional observable features; marginal and nearest-neighbor joint distances',
                notes='All effect comparisons use original V2 stage cost; altered planner objectives also logged separately.')


def setting(method, family='baseline', **changes):
    s = dict(method=method, family=family, alpha=1., search='baseline', support='consecutive', K=50,
             safety='original', cost='original')
    s.update(changes)
    if method in ['frozen', 'oracle']:
        s['alpha'] = 0.
        s['K'] = 0
    fields = ['family', 'method', 'alpha', 'search', 'support', 'K', 'safety', 'cost']
    s['id'] = '__'.join(str(s[k]).replace('.', 'p') for k in fields)
    return s


def development_settings(family):
    baseline = [setting(m) for m in METHODS]
    if family == 'baseline':
        return baseline
    if family == 'blend':
        return [setting(m, family, alpha=a) for m in ADAPTERS for a in ALPHAS]
    if family == 'search':
        return [setting(m, family, search=b) for b in ['expanded', 'high'] for m in METHODS]
    if family == 'support':
        return [setting(m, family, support=t, K=k) for t in ['consecutive', 'random_multi', 'state_diverse', 'state_action_diverse']
                for k in [20, 50] for m in ADAPTERS if (t, k) != ('consecutive', 50)]
    if family == 'rail':
        return [setting(m, family, safety=t) for t in ['stronger', 'margin'] for m in METHODS]
    if family == 'cost':
        return [setting(m, family, cost=t) for t in ['velocity', 'terminal'] for m in METHODS]
    raise ValueError(family)


FAMILIES = ['baseline', 'blend', 'search', 'support', 'rail', 'cost']


def applicable(condition, family):
    # Support-diversity is an independent fixed-budget intervention, not future online data.
    return not (condition['online'] and family == 'support') and not (condition['task'] == 'balance' and family == 'cost')
