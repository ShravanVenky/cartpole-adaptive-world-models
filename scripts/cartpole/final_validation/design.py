"""Development-only recoverability selection, independent confirmation, and failure replay."""
import json
import numpy as np
import pandas as pd
from .common import save_json,save_npz,initial
from .protocol import PREVIOUS,spec,confirmation_specs
from .simulation import run_batch


def select_snapshot_cases(ctx):
    path=ctx.run/'selection_locked.json'
    if path.exists():
        locked=json.loads(path.read_text())
        if locked['config_sha256']!=ctx.digest:raise RuntimeError('Selection config mismatch')
        return locked
    rows=[r for p in (ctx.run/'episodes').glob('development_*.json') for r in json.loads(p.read_text())['rows']]
    frame=pd.DataFrame(rows);chosen=[];assessments=[]
    for phase in ['active_swingup','upright_entry','stable_balance']:
        scores=[]
        for case in [c for c in ctx.cfg['conditions'] if c['phase']==phase and c['environment']!='nominal']:
            f=frame[frame.condition.eq(case['name'])]
            o=f[f.method.eq('oracle')&f.safety.eq('margin')]
            frozen=f[f.method.eq('frozen')&f.safety.eq('original')]
            eligible=len(o)==len(ctx.cfg['model_seeds'])*len(ctx.cfg['development_trials'])
            success=float(o.success.mean()) if len(o) else 0.
            survival=float(o.actual_postchange_transitions.median()) if len(o) else 0.
            record=dict(condition=case['name'],phase=phase,oracle_success=success,oracle_median_poststeps=survival,
                        frozen_success=float(frozen.success.mean()) if len(frozen) else 0.,
                        gate_passed=bool(eligible and success>=ctx.cfg['oracle_gate_success'] and survival>=ctx.cfg['oracle_gate_poststeps']))
            scores.append(record);assessments.append(record)
        passed=[s for s in scores if s['gate_passed']]
        if passed:best=sorted(passed,key=lambda s:(s['frozen_success'],s['oracle_success'],s['condition']))[0]
        else:best=sorted(scores,key=lambda s:(-s['oracle_success'],-s['oracle_median_poststeps'],s['condition']))[0]
        chosen.append(best['condition'])
    chosen.append('stable_balance__nominal')
    result=dict(config_sha256=ctx.digest,conditions=chosen,assessments=assessments,
                warning='A failed gate remains labeled Oracle-unvalidated; no case is silently represented as recoverable.',
                all_live_conditions_retained=True,confirmation_trials=ctx.cfg['confirmation_trials'])
    save_json(path,result);return result


def fixed_confirmation(ctx,seed,trial,case,steps_limit=None):
    settings=confirmation_specs();key=f'fixed_{seed}_{trial}_{case["name"]}'
    cached=ctx.cached(ctx.run/'episodes'/f'{key}.json')
    if cached is not None:return cached
    models=[];logs=[];hashes=[]
    for s in settings:
        support=None if s['method'] in ['frozen','oracle'] else ctx.support(seed,case,trial,s['support'],50)
        m,log=ctx.fit(seed,s,support);models.append(m);logs.append(log);hashes.append(support.support_hash() if support is not None else 'none')
    packet=run_batch(ctx,key,seed,trial,case,settings,np.repeat(initial(case['task'],seed,trial)[None].numpy(),len(settings),axis=0),
                     models,[ctx.cfg[case['task']+'_steps']]*len(settings),steps_limit=steps_limit,
                     metadata=[dict(stage='fixed_confirmation',K=50,initial_adaptation_seconds=log['adaptation_seconds'] if log else 0.,
                                    support_hash=h,pool_acquisition_interactions=1024,optimizer_support_count=50 if s['method'] not in ['frozen','oracle'] else 0)
                               for s,log,h in zip(settings,logs,hashes)])
    return packet


def recovery_manifest(ctx):
    target=ctx.run/'recovery_manifest.json'
    if target.exists():return json.loads(target.read_text())['cases']
    cases=[]
    # Reuse already recorded failure states rather than re-running the previous stress trajectories.
    previous=pd.read_csv(PREVIOUS/'control_results.csv')
    for environment in ['online_strong_gain','online_combined_geometry']:
        for seed in ctx.cfg['model_seeds']:
            f=previous[(previous.phase=='confirmation')&(previous.family=='baseline')&(previous.method=='frozen')&(previous.condition==environment)&(previous.seed==seed)&previous.failed]
            ident=f'historical_{environment}_{seed}'
            if f.empty:
                cases.append(dict(id=ident,available=False,reason='No recorded failure'));continue
            row=f.sort_values('trial').iloc[0]
            with np.load(PREVIOUS/'control'/f'{row.group}.npz') as data:state=data['states'][int(row.row),int(row.survival_steps)-1].tolist()
            cases.append(dict(id=ident,available=True,seed=seed,environment=row.environment,task='swingup',horizon=30,state=state,
                              source='previous saved state immediately before failure',source_group=row.group,source_row=int(row.row)))
    # One current developmental failure per model seed/phase. No held-out outcome selects these states.
    packets=[json.loads(p.read_text()) for p in (ctx.run/'episodes').glob('development_*.json')]
    rows=[r for p in packets for r in p['rows']]
    frame=pd.DataFrame(rows)
    for phase in ['active_swingup','upright_entry','stable_balance']:
        for seed in ctx.cfg['model_seeds']:
            ident=f'current_{phase}_{seed}'
            f=frame[frame.phase.eq(phase)&frame.seed.eq(seed)&frame.method.eq('frozen')&frame.safety.eq('original')&frame.failed]
            if f.empty:cases.append(dict(id=ident,available=False,reason='No qualifying development failure'));continue
            row=f.sort_values(['actual_postchange_transitions','condition']).iloc[0]
            cases.append(dict(id=ident,available=True,seed=seed,environment=row.environment,task=row.task,
                              horizon=15 if row.task=='balance' else 30,state=row.state_immediately_before_failure,
                              source='current development, actual state immediately before failure',source_group=row.key,source_row=int(row.row)))
    save_json(target,dict(config_sha256=ctx.digest,cases=cases,planned=len(cases)));return cases


def run_recovery(ctx,case,steps_limit=None):
    if not case['available']:
        save_json(ctx.run/'recovery_unavailable'/f'{case["id"]}.json',dict(config_sha256=ctx.digest,case=case,rows=[dict(controller=c,censored=True,reason=case['reason']) for c in ['existing','strong_search','margin']]))
        return []
    c=dict(name=case['id'],environment=case['environment'],task=case['task'],horizon=case['horizon'])
    seed=case['seed'];packets=[]
    for suffix,settings in [('baseline',[spec('oracle'),spec('oracle','margin')]),('high',[spec('oracle',search='high')])]:
        models=[ctx.make(seed,s) for s in settings]
        packets.append(run_batch(ctx,f'recovery_{case["id"]}_{suffix}',seed,700,c,settings,
                np.repeat(np.array(case['state'])[None],len(settings),axis=0),models,[ctx.cfg[case['task']+'_steps']]*len(settings),
                steps_limit=steps_limit,metadata=[dict(stage='oracle_recovery',source_case=case['id'],privileged=True) for _ in settings]))
    return packets
