"""Predeclared bounded matrix and attribution thresholds; no choices use confirmation outcomes."""
import json
from pathlib import Path
from final_validation.protocol import configuration as previous_configuration

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'results/cartpole/20261026_223145_v2_final_validation_full'
FROZEN='frozen'; ORIGINAL='meta_residual_1step'; ROLLOUT='meta_residual_rollout'
METAS=[ORIGINAL,ROLLOUT]
METHODS=[FROZEN,ORIGINAL,ROLLOUT,'ridge','oracle','exact_residual']
SOURCES=[FROZEN,ORIGINAL,ROLLOUT,'oracle']
STAGES=['saved','replay','forks','dense','confirmation','synthesis']


def configuration():
    cfg=previous_configuration()
    names=['active_swingup__combined_geometry','active_swingup__nominal','active_swingup__combined_friction_gain',
           'active_swingup__strong_gain','stable_balance__nominal']
    cfg.update(schema='v2_model_planner_mismatch_1',source_run=str(REFERENCE.relative_to(ROOT)),
        conditions=[next(c for c in cfg['conditions'] if c['name']==name) for name in names],
        archived_trial=400,query_trial=401,confirmation_trials=[900,901],
        replay_times_geometry=[10,50,150],replay_times_other=[20,50],forecast_steps=[1,5,10,20,30],
        fixed_query_candidates=128,broad_candidates=128,candidate_chunk=2048,
        pre_failure_leads=[5,10,20,40],intervention_time=150,
        ranking_regret_threshold=.10,search_regret_threshold=.10,good_candidate_tolerance=.05,
        false_safe_threshold=0.,rank_top_k=10,
        dense_environments=['combined_geometry','nominal','combined_friction_gain','strong_gain'],
        dense_train_count=16384,dense_validation_count=4096,dense_updates=2000,dense_batch_size=1024,
        dense_learning_rates=[.001,.003],dense_check_every=100,dense_seed=7041,
        state_scale=[2.4,3.,3.141592653589793,4.],
        phase_definitions=dict(upright_angle=.15,capture_angle=.5,capture_omega=4.,high_omega=4.,
                               high_force=9.5,near_rail=1.8,large_angle=1.5707963267948966,early_steps=100),
        offline_training='Only isolated dense diagnostic copies of the same452parameter adapters',
        historical_role='All previously inspected V2 states are development/diagnostic, never new held-out confirmation',
        fixed_query_rule='Query trial401; support trial400. Independent episodes, fixed query/candidates for all observation counts.',
        replay_warm_start='Common zero CEM mean for action-ranking replay; not a reconstruction of the exact historical search.',
        full_confirmation_rule='New nominal prefixes and controller-generated online observations in trial900/901; no outcome-based tuning.',
        dense_split_rule='Synthetic development state-action resets plus development replay candidate regions; separate validation sampler; no confirmation observations.',
        dense_architecture='Same6->16->16->4 tanh,452parameters, original normalizer/output clipping/physical-delta guards.',
        attribution='Within-pool counterfactual attribution, not proof of globally optimal search or episode causality.')
    return cfg


def workload(cfg):
    n=len(cfg['model_seeds']);c=len(cfg['conditions'])
    replay=n*(2*len(cfg['replay_times_geometry'])+2*(c-1)*len(cfg['replay_times_other'])+len(cfg['replay_times_geometry']))
    pre=n*2*len(cfg['pre_failure_leads'])
    confirmation=n*len(cfg['confirmation_trials'])*c*7
    confirm_replay=n*(2*len(cfg['confirmation_trials'])*2+(c-1))
    return dict(saved_replay_slots=replay,fixed_query_groups=n*c,fixed_query_model_rows=n*c*8*6,
        pre_failure_state_slots=pre,pre_failure_control_slots=pre*6,
        development_substitution_forks=n*20,confirmation_substitution_forks=n*20,
        new_online_confirmation_control_records=confirmation,
        full_control_slots=pre*6+n*40+confirmation,
        confirmation_replay_slots=confirm_replay,
        dense_diagnostic_copies=n*len(cfg['dense_environments'])*2,
        dense_training_updates=n*len(cfg['dense_environments'])*2*len(cfg['dense_learning_rates'])*cfg['dense_updates'],
        dense_labelled_transitions_per_seed_environment=cfg['dense_train_count']+cfg['dense_validation_count'],
        max_candidate_pool=4*(128*3+3)+cfg['broad_candidates'],
        offline_scored_model_trajectories_upper_bound=(replay*6+confirm_replay*8)*(4*(128*3+3)+cfg['broad_candidates'])+n*c*8*6*128+n*c*8*128,
        notes='Counts are planned slots. Failed histories and missing archived strata are explicit unavailable records. No truncation is silently treated as task completion.')


def fork_policies(include_dense=False,pre_failure=False):
    policies=[dict(source=m,scorer=m,native=True,safety='margin',label=m) for m in [FROZEN,*METAS,'ridge','oracle']]
    policies.append(dict(source='oracle',scorer='oracle',native=True,safety='original',label='oracle_original'))
    if not pre_failure:
        seen=set()
        for meta in METAS:
            for source in [FROZEN,meta,'oracle']:
                for scorer in [FROZEN,meta,'oracle']:
                    if (source,scorer) in seen:continue
                    seen.add((source,scorer))
                    policies.append(dict(source=source,scorer=scorer,native=False,safety='margin',label=f'{source}_scored_by_{scorer}'))
        assert len(policies)==20
    if include_dense:
        policies += [dict(source='dense_'+m,scorer='dense_'+m,native=True,safety='margin',label='dense_'+m) for m in METAS]
    return policies
