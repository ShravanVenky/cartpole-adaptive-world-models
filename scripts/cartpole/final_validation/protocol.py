"""Small explicit matrix; observation, model, episode and planner seeds stay distinct."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=ROOT/'results/cartpole/20261026_000707_v2_control_diagnostics'
METHODS=['frozen','ridge','meta_residual_1step','meta_residual_rollout','oracle']
ADAPTERS=METHODS[1:4]
CHECKPOINTS=[0,1,2,3,5,10,20,50]


def spec(method,safety='original',alpha=1.,regularization='existing',support='consecutive',search='baseline'):
    s=dict(method=method,safety=safety,alpha=float(alpha),regularization=regularization,
           support=support,search=search,cost='original')
    if method in ['frozen','oracle']:s['alpha']=0.
    s['id']=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()[:16]
    return s


def live_specs():
    return [spec(m,safety) for safety in ['original','margin'] for m in METHODS]+[
        spec('ridge',safety,regularization='count_scaled') for safety in ['original','margin']]


def shadow_specs():
    # Frozen represents alpha0 for all adapters; aliases are not duplicate evaluations.
    return [spec('frozen'),spec('oracle')]+[spec(m,alpha=a) for m in ADAPTERS for a in [.25,.5,1]]+[
        spec('ridge',regularization='count_scaled')]


def fork_specs():
    rows=[]
    for k in CHECKPOINTS:
        for s in shadow_specs():
            if s['alpha'] in [0.,1.] or k in [5,20,50]:
                rows.append(dict(s,K=k))
    assert len(rows)==66
    return rows


def confirmation_specs():
    rows=[spec(m,safety) for m in ['frozen','oracle'] for safety in ['original','margin']]
    for m in ADAPTERS:
        rows += [spec(m,safety) for safety in ['original','margin']]
        rows += [spec(m,alpha=a) for a in [.25,.5]]
        rows += [spec(m,support='state_action_diverse'),spec(m,alpha=.5,support='state_action_diverse'),
                 spec(m,'margin',support='state_action_diverse')]
    assert len(rows)==25
    return rows


def configuration():
    prior=json.loads((PREVIOUS/'config_locked.json').read_text())
    shifts=['nominal','near_gain','strong_gain','combined_geometry','combined_friction_gain']
    phases=['active_swingup','upright_entry','stable_balance']
    conditions=[dict(name=f'{p}__{e}',phase=p,environment=e,task='balance' if p=='stable_balance' else 'swingup',
                     horizon=15 if p=='stable_balance' else 30) for p in phases for e in shifts]
    fixed=[dict(name=n,environment=e,task=t,horizon=15 if t=='balance' else 30) for n,e,t in [
        ('nominal_swingup','nominal','swingup'),('geometry_swingup','combined_geometry','swingup'),
        ('friction_swingup','combined_friction_gain','swingup'),('nominal_balance','nominal','balance')]]
    return dict(schema=1,model_seeds=[11,22,33,44,55],development_trials=[300],confirmation_trials=[400,401],
                fixed_confirmation_trials=[500,501],checkpoints=CHECKPOINTS,regularization_K=[5,10,20,50],
                conditions=conditions,fixed_conditions=fixed,environments=prior['environments'],
                budgets={'baseline':[128,3],'high':[1024,5]},elite_fraction=.125,
                active_change_step=10,stable_warmup_steps=100,stable_required_steps=50,entry_max_steps=450,
                swingup_steps=600,balance_steps=400,max_batch_rows=48,snapshot_conditions=4,
                recovery_cases=25,recovery_historical_cases=10,oracle_gate_success=.8,oracle_gate_poststeps=100,
                ranking_broad_candidates=128,ranking_chunk=128,multistep_horizons=[1,5,15,30,100],
                ridge_lambda_reference=.1,ridge_K_reference=20,
                ridge_objectives={'existing':'sum squared residual +0.1*||B||^2',
                    'count_scaled':'mean squared residual +(0.1/20)*||B||^2; equivalent sum loss +0.1*K/20*||B||^2'},
                support_pool_steps=128,support_pool_trajectories=8,dt=.02,rail_boundary=2.4,
                success_angle=.15,success_omega=.8,sustained_steps=100,recovery_window=25,
                failure_penalty=100000.,numerical_state_bound=100.,excessive_crossing_speed=4.,
                near_upright_speed_penalty=4.,terminal_stabilization_extra=20.,
                bootstrap_samples=2000,bootstrap_seed=4123,noise='none',offline_training=False,
                warmup='One shared Frozen nominal controller per phase/modelseed/episode; all methods branch at identical observed change state.',
                observation_rule='Only completed, non-failing, actual post-change transitions. Failure-causing transition saved but never fitted. No padding as support.',
                snapshot_rule='Common Frozen original/margin collectors; refit isolated initializations on the same actual prefix. Fixed adapters during counterfactual branches.',
                selection='After development, choose hardest Oracle-margin-recoverable nonnominal condition per phase, plus nominal stable balance; retain all15 live conditions regardless.',
                seed_rule='Distinct deterministic namespaces for model seed, episode trial, nominal prefix, support pool, MPC stream and candidate replay.',
                trust='Fixed predeclared alphas only; oracle-selected alpha is descriptive and never deployed.',
                statistics='Five seed clusters; paired episodes; censoring reported, no success inferred for unreachable K.',
                usefulness=dict(prediction_fraction=.05,decision_fraction=.05,control_success_points=.10,control_cost_fraction=.10,
                                baseline_error_floor=1e-6,sustained_checkpoints=2,minimum_reach_probability=.8),
                emergencies='No emergency or learned backup; conservative controller changes only predicted rail margin2.4 to2.1.')


def workload(cfg):
    n=len(cfg['model_seeds']);dev=n*len(cfg['development_trials']);conf=n*len(cfg['confirmation_trials'])
    live_groups=len(cfg['conditions'])*(dev+conf)
    cohorts=cfg['snapshot_conditions']*conf*2
    minimum_prefix={'active_swingup':cfg['active_change_step'],'upright_entry':1,'stable_balance':cfg['stable_warmup_steps']}
    maximum_prefix={'active_swingup':cfg['active_change_step'],'upright_entry':cfg['entry_max_steps'],'stable_balance':cfg['stable_warmup_steps']}
    live_calls=sum((dev+conf)*12*(cfg[c['task']+'_steps']-minimum_prefix[c['phase']]) for c in cfg['conditions'])
    live_batches=live_calls//12
    snap_budgets=[cfg['swingup_steps']-minimum_prefix['active_swingup'],cfg['swingup_steps']-1,
                  cfg['balance_steps']-cfg['stable_warmup_steps'],cfg['balance_steps']-cfg['stable_warmup_steps']]
    fork_calls=sum((66*b-sum(s['K'] for s in fork_specs()))*conf*2 for b in snap_budgets)
    fork_batches=sum(sum(b-min(s['K'] for s in fork_specs()[offset:offset+cfg['max_batch_rows']]) for offset in range(0,66,cfg['max_batch_rows']))*conf*2 for b in snap_budgets)
    fixed_calls=sum(n*len(cfg['fixed_confirmation_trials'])*25*cfg[c['task']+'_steps'] for c in cfg['fixed_conditions'])
    prefix_calls=(dev+conf)*sum(maximum_prefix.values())
    recovery_calls=60*cfg['swingup_steps']+15*cfg['balance_steps']
    return dict(live_conditions=len(cfg['conditions']),live_episodes=live_groups*len(live_specs()),live_groups=live_groups,
                warmup_prefixes=3*(dev+conf),snapshot_conditions=cfg['snapshot_conditions'],snapshot_cohorts=cohorts,
                snapshot_control_records=cohorts*len(fork_specs()),snapshot_ranking_records=cohorts*len(CHECKPOINTS)*len(shadow_specs()),
                fixed_confirmation_conditions=len(cfg['fixed_conditions']),fixed_confirmation_episodes=len(cfg['fixed_conditions'])*n*len(cfg['fixed_confirmation_trials'])*len(confirmation_specs()),
                oracle_failure_replay_records=cfg['recovery_cases']*3,
                planned_control_records=live_groups*12+cohorts*66+len(cfg['fixed_conditions'])*n*len(cfg['fixed_confirmation_trials'])*25+cfg['recovery_cases']*3,
                max_episode_MPC_row_calls=live_calls+fork_calls+fixed_calls+recovery_calls,max_warmup_MPC_row_calls=prefix_calls,
                max_MPC_batch_calls_including_warmup=live_batches+fork_batches+fixed_calls//25+recovery_calls*2//3+prefix_calls,
                candidate_trajectory_evaluations=cohorts*len(CHECKPOINTS)*len(shadow_specs())*(256+cfg['ranking_broad_candidates']+len(shadow_specs())),
                note='Upper-bounded records include explicit censored/unavailable rows; actual MPC calls depend on event times, failure, and reachable K. Warmup prefixes are additional shared interactions.')
