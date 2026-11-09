"""Locked treatments, independent seed namespaces and explicit stage dependencies."""
from pathlib import Path
from mismatch.protocol import configuration as previous_configuration, METAS

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / 'results/cartpole/20261101_013609_468917_v2_model_planner_mismatch_full'
STAGES = ['A', 'B', 'C', 'D']
ENVIRONMENTS = ['combined_geometry', 'nominal', 'combined_friction_gain', 'strong_gain']


def optimizer(name='baseline', lr=.01, steps=10, clip=1., step_bound=.1, delta=2., algorithm='sgd'):
    return dict(name=name, learning_rate=lr, steps=steps, gradient_clip=clip,
                step_norm_bound=step_bound, max_parameter_delta=delta, algorithm=algorithm,
                isolated_diagnostic=delta > 2 or step_bound > .1)


def screen():
    rows = [optimizer(f'lr_{lr:g}', lr=lr) for lr in [.005, .01, .02, .05, .10]]
    rows += [optimizer(f'steps_{n}', steps=n) for n in [25, 50, 100, 250, 500]]
    rows[1]['name'] = 'baseline'
    return rows


def configuration():
    cfg = previous_configuration()
    cfg.update(schema='v2_final_hardening_1', reference=str(REFERENCE.relative_to(ROOT)),
        dev_trials=[1100], confirmation_trials=[1200, 1201], support_trial=900, query_trial=901,
        query_times=[50, 100, 150], adaptation_K=[1, 2, 3, 5, 10, 20, 50],
        confirmation_query_times=[50,150],
        budget_K=[5, 10, 20, 50], optimizer_screen=screen(),
        dev_environments=ENVIRONMENTS, b_environments=ENVIRONMENTS[:3],
        dependencies={'A': [], 'B': ['A'], 'C': ['A', 'B'], 'D': ['A', 'B', 'C']},
        exit_criteria=dict(prediction_nmse_ratio=.90, planning_regret_ratio=.90,
            difficult_success_gain_pp=20., nominal_regression_pp=10., maximum_observations=50,
            recovery_success=.8, generalization_cost_ratio=.90, generalization_success_gain_pp=10.,
            fitting_budget_ms=20., loop_budget_ms=20., confidence=.95),
        selection_rule='Development only: shortlist two non-relaxed settings per Meta by mean log(query/Frozen NMSE)+log(1+regret). Choose final by paired development control success, then cost, then fitting latency. Retain baseline if no control evidence of improvement.',
        noise_levels={'none': [0., 0., 0., 0.], 'mild': [.002, .02, .003, .03],
                      'strong': [.01, .10, .015, .15]},
        impulse_levels=[.25, .6, 1.0], angular_velocity_levels=[.5, 1.25, 2.5],
        force_pulse_levels=[2., 5., 8.], pulse_steps=5,
        disturbance_step=100, repeated_steps=[100, 220], combined_impulse_step=250,
        sequence_change_steps=[150,300],
        disturbance_steps=500, unseen_steps=600,
        fit_device='cpu', detector='absent in existing CartPole V2; residuals are diagnostic only',
        offline_training=False, dense_policy='Reuse immutable prior condition-specific copies only; no new dense fitting',
        reset_rule='Reset Meta to learned initialization on each K and each known persistent change; Adam state local to one fitting call.',
        coverage_rule='Diverse offline reference uses prior development pool, counts all available labelled pool interactions as privileged access.',
        checkpoint_rule='Each atomic job has JSON+NPZ content hashes; resume requires exact source/config/input hashes.',
        interpretation='Known change times; no detector, learned gate, history encoder, new architecture or control objective.')
    new = [
        ('interp_cart', 'interpolation', {'cart_mass': 1.10}),
        ('extra_cart', 'extrapolation', {'cart_mass': 2.0}),
        ('interp_pole', 'interpolation', {'pole_mass': .11}),
        ('extra_pole', 'extrapolation', {'pole_mass': .25}),
        ('interp_length', 'interpolation', {'pole_half_length': .55}),
        ('extra_length', 'extrapolation', {'pole_half_length': .85}),
        ('interp_friction', 'interpolation', {'cart_friction': .02, 'pole_friction': .001}),
        ('extra_friction', 'extrapolation', {'cart_friction': .25, 'pole_friction': .018}),
        ('interp_gain', 'interpolation', {'motor_gain': .95}),
        ('extra_gain', 'extrapolation', {'motor_gain': .45}),
        ('interp_combined', 'interpolation', {'cart_mass': 1.10, 'pole_mass': .11, 'pole_half_length': .55, 'motor_gain': .95}),
        ('extra_combined', 'extrapolation', {'cart_mass': 1.7, 'pole_mass': .2, 'pole_half_length': .8, 'motor_gain': .5}),
        ('gain_partial', 'sequence', {'motor_gain': .85}),
        ('sequence_final', 'sequence', {'motor_gain': .85,'cart_friction': .12,'pole_friction': .008,'pole_half_length': .65}),
    ]
    cfg['environments'] += [dict(name=n, category=c, **p) for n,c,p in new]
    return cfg


def disturbance_cases():
    rows=[]
    for i in range(3):
        for sign in [-1, 1]:
            rows.append(dict(name=f'cart_impulse_{sign}_{i}', kind='cart_impulse', severity=i,
                             sign=sign, environment='nominal'))
        for kind in ['angular_velocity', 'force_pulse', 'repeated_cart']:
            rows.append(dict(name=f'{kind}_{i}', kind=kind, severity=i, sign=1, environment='nominal'))
    for env in ['strong_gain', 'combined_friction_gain', 'combined_geometry']:
        rows.append(dict(name=f'persistent_{env}', kind='persistent', severity=1, sign=1, environment=env))
        rows.append(dict(name=f'adapt_then_impulse_{env}', kind='combined', severity=1, sign=1, environment=env))
    return rows


def unseen_cases():
    rows=[]
    for i,name in enumerate(['cart','pole','length','friction','gain','combined']):
        for kind in ['interp','extra']:
            phase=['active_swingup','upright_entry','stable_balance'][i%3]
            rows.append(dict(name=f'{kind}_{name}_{phase}', environment=f'{kind}_{name}', phase=phase, noise='none', sequence=False))
    for phase in ['active_swingup','upright_entry','stable_balance']:
        # Stable balance already has a no-noise extra_combined case above.
        for noise in (['none','mild','strong'] if phase!='stable_balance' else ['mild','strong']):
            rows.append(dict(name=f'noise_{phase}_{noise}', environment='extra_combined', phase=phase, noise=noise, sequence=False))
        for noise in ['none','strong']:
            rows.append(dict(name=f'sequence_{phase}_{noise}', environment='strong_gain', phase=phase, noise=noise, sequence=True))
    return rows


def workload(cfg):
    n=len(cfg['model_seeds']); e=len(cfg['dev_environments']); q=len(cfg['query_times'])
    # A1/2=10; A3=4; A4=1 per family/group. A5 up to four settings at three extra K.
    fits_a=n*e*2*(15+4*3)
    a_dev=n*e*2*15  # baseline, two shortlisted, explicit Adam/relaxed diagnostics, references
    a_conf=n*e*len(cfg['confirmation_trials'])*2*9
    b=n*len(cfg['b_environments'])*(len(cfg['dev_trials'])+len(cfg['confirmation_trials']))*3*9
    c=n*(len(disturbance_cases())*11+6*2)*(len(cfg['dev_trials'])+len(cfg['confirmation_trials']))
    d=n*len(unseen_cases())*len(cfg['confirmation_trials'])*7
    # Maximum per family: baseline+two shortlisted<=3, 7 checkpoints each.
    max_steps=500
    return dict(A=dict(offline_fits_upper=fits_a, offline_prediction_rows_upper=n*e*q*(22+12+4*10)+n*e*2*2*6,
                       control_slots_upper=a_dev+a_conf, development_control_slots=a_dev,
                       confirmation_control_slots=a_conf),
        B=dict(control_slots=b, support_collections=n*len(cfg['b_environments'])*3*3),
        C=dict(control_slots=c, scenarios=len(disturbance_cases())),
        D=dict(control_slots=d, scenarios=len(unseen_cases())),
        total_control_slots_upper=a_dev+a_conf+b+c+d,
        A12_requested_gradient_updates=n*e*2*(5*10+25+50+100+250+500),
        maximum_fit_calls=dict(A_offline=fits_a,A_control=n*e*(10*7+10)+n*e*2*(4*7+4),
            B=n*len(cfg['b_environments'])*3*3*5,
            C=n*((len(disturbance_cases())+3)*5+6*2)*3*7,
            D=n*(len(unseen_cases())+12)*2*5*7,
            collections=(n*e*2+n*len(cfg['b_environments'])*3*2)*7),
        optimizer_updates_upper=dict(A_offline=fits_a*max_steps,
            A_online=(n*e*10+n*e*2*4)*7*max_steps,
            A_confirmation_snapshot=n*e*len(cfg['confirmation_trials'])*4*max_steps,
            B_fits=n*len(cfg['b_environments'])*3*3*4*max_steps,
            C_online=n*((len(disturbance_cases())+3)*4+6*2)*3*7*max_steps,
            D_online=n*(len(unseen_cases())+12)*2*4*7*max_steps,
            support_collection=n*(e*2+len(cfg['b_environments'])*3*2)*7*10),
        note='Upper bounds include unavailable starts/support and shortlist deduplication. Conditional combined impulses require prior sustained stabilization. Warmups, support collection, and labelled reference pools are separately counted.')


def control_jobs(cfg):
    """Exact job identities; counts alone cannot certify matrix completion."""
    jobs=[]
    for seed in cfg['model_seeds']:
        for env in cfg['dev_environments']:
            for split,trials,size in [('dev',cfg['dev_trials'],15),('confirm',cfg['confirmation_trials'],9)]:
                for trial in trials:
                    for protocol in ['snapshot','online']:
                        jobs.append(dict(stage='A',key=f'A_{split}_{seed}_{trial}_{env}_{protocol}',slots=size))
        for trial in [*cfg['dev_trials'],*cfg['confirmation_trials']]:
            for env in cfg['b_environments']:
                for strategy in ['ordinary','excitation','diverse']:
                    jobs.append(dict(stage='B',key=f'B_{seed}_{trial}_{env}_{strategy}',slots=9))
            for spec in disturbance_cases():
                jobs.append(dict(stage='C',key=f'C_{seed}_{trial}_{spec["name"]}',slots=13 if spec['kind'] in ['persistent','combined'] else 11))
        for trial in cfg['confirmation_trials']:
            for spec in unseen_cases():jobs.append(dict(stage='D',key=f'D_{seed}_{trial}_{spec["name"]}',slots=7))
    return jobs
