"""Resumable staged execution. Smoke is the only verification mode run during implementation."""
import argparse
import datetime as dt
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .protocol import ROOT,HISTORICAL,METHODS,ADAPTERS,FAMILIES,configuration,development_settings,applicable,setting
from .context import Context,save_json,file_hash
from .audit import run_audit
from .replay import select_states,run_replay
from .coverage import run_coverage,support_rankings
from .control import run_control


def prepare(run,cfg,smoke):
    run.mkdir(parents=True,exist_ok=True)
    path=run/'config_locked.json'
    if path.exists():
        if json.loads(path.read_text())!=cfg:raise RuntimeError('Configuration mismatch; use a new directory')
    else:save_json(path,cfg)
    sources=[*sorted((Path(__file__).parent).glob('*.py')),ROOT/'scripts/cartpole/run_diagnostics.py',
             ROOT/'src/adaptive_world_model/control/evaluation_mpc.py',ROOT/'src/adaptive_world_model/control/batched_predictor.py',
             ROOT/'scripts/cartpole/control_evaluate.py']
    hashes={str(p.relative_to(ROOT)):file_hash(p) for p in sources}
    manifest=run/'source_hashes.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=hashes:raise RuntimeError('Source changed since run creation; restore snapshot or start a new run')
    else:
        for p in sources:
            dst=run/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        save_json(manifest,hashes)
    preservation=run/'preserved_artifacts.json'
    if not preservation.exists():
        # Full runs verify every prior scientific artifact, not just checkpoints.
        files=[HISTORICAL/'report.md',HISTORICAL/'config_locked.json',*sorted((ROOT/'results/cartpole/20261005_232356_821640_cartpole_v1/checkpoints').glob('*.pt')),
               *sorted((ROOT/'results/cartpole/20261006_140915_614361_cartpole_v2_factorial/checkpoints').glob('*.pt'))] if smoke else [p for folder in [
            ROOT/'results/cartpole/20261005_232356_821640_cartpole_v1',
            ROOT/'results/cartpole/20261006_140915_614361_cartpole_v2_factorial',HISTORICAL] for p in folder.rglob('*') if p.is_file()]
        save_json(preservation,{str(p.relative_to(ROOT)):file_hash(p) for p in files})
    save_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                      torch=torch.__version__,device='cuda' if torch.cuda.is_available() else 'cpu',
                                      total_memory=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None))


def verify_preserved(run):
    files=json.loads((run/'preserved_artifacts.json').read_text())
    wrong=[p for p,h in files.items() if not (ROOT/p).exists() or file_hash(ROOT/p)!=h]
    if wrong:raise RuntimeError(f'Immutable source artifacts changed: {wrong[:3]}')
    save_json(run/'integrity_verified.json',dict(files=len(files),all_unchanged=True))


def batches(settings):
    groups=defaultdict(list)
    for s in settings:groups[s['search']].append(s)
    return list(groups.values())


def select_confirmation(ctx):
    packets=[json.loads(p.read_text()) for p in (ctx.run/'control').glob('development_*.json')]
    frame=pd.DataFrame([r for p in packets for r in p['rows']])
    chosen={};selection_rows=[]
    for c in ctx.cfg['conditions']:
        cc={}
        for family in FAMILIES:
            if not applicable(c,family):continue
            if family=='baseline':cc[family]=development_settings('baseline');continue
            pool=frame[(frame.condition==c['name'])&(frame.family==family)]
            if pool.empty:raise RuntimeError(f'Missing development family {c["name"]}/{family}')
            options=development_settings(family)
            picks=[]
            for method in sorted(pool.method.unique()):
                for k in ([20,50] if family=='support' else [None]):
                    sub=pool[pool.method==method]
                    if k is not None:sub=sub[sub.K==k]
                    # Every candidate has the same five independent seeds and matched trial.
                    agg=sub.groupby('id').agg(success=('success','mean'),cost=('cost','mean'),seeds=('seed','nunique'))
                    if (agg.seeds!=len(ctx.cfg['seeds'])).any():raise RuntimeError('Incomplete development selection')
                    winner=agg.sort_values(['success','cost'],ascending=[False,True]).index[0]
                    selected=next(x for x in options if x['id']==winner)
                    picks.append(selected)
                    selection_rows.append(dict(condition=c['name'],family=family,method=method,K=k,winner=winner,
                                               development_success=float(agg.loc[winner,'success']),development_cost=float(agg.loc[winner,'cost'])))
                    if family=='support' and k==20:picks.append(setting(method,'support',K=20))
                if family=='blend':
                    picks.extend([setting(method,'blend',alpha=0),setting(method,'blend',alpha=1)])
            # Match controller interventions with a privileged reference for every selected option.
            if family in ['rail','cost','search']:
                for s in list(picks):
                    changes={k:s[k] for k in ['search','safety','cost']}
                    picks.append(setting('oracle',family,**changes))
            cc[family]=list({s['id']:s for s in picks}.values())
        chosen[c['name']]=cc
    # Interactions only after single-factor development screening: blend x search and blend x support.
    # One declared combination per adapter/condition; no unbounded Cartesian product.
    for c in ctx.cfg['conditions']:
        cc=chosen[c['name']];inter=[]
        for method in ADAPTERS:
            alpha=next(s['alpha'] for s in cc['blend'] if s['method']==method)
            search=next(s['search'] for s in cc['search'] if s['method']==method)
            inter.append(setting(method,'interaction',alpha=alpha,search=search))
            if not c['online']:
                support=next(s for s in cc['support'] if s['method']==method and s['K']==50)
                inter.append(setting(method,'interaction',alpha=alpha,support=support['support'],K=50))
        cc['interaction']=list({s['id']:s for s in inter}.values())
    result=dict(selected_using='development only',rows=selection_rows,settings=chosen)
    save_json(ctx.run/'selection_locked.json',result)
    return chosen


def stage_control(ctx,phase,chosen=None):
    trials=ctx.cfg[phase+'_trials']
    for seed in ctx.cfg['seeds']:
        for c in ctx.cfg['conditions']:
            families={f:development_settings(f) for f in FAMILIES if applicable(c,f)} if chosen is None else chosen[c['name']]
            # Different ablations with the same search budget can share one GPU batch:
            # alpha, support, rail and cost parameters remain explicit per-row inputs.
            options=[s for group in families.values() for s in group]
            width=max(1,ctx.cfg['max_control_batch_rows']//len(trials))
            for group in batches(options):
                for offset in range(0,len(group),width):
                    run_control(ctx,phase,seed,c,trials,group[offset:offset+width],
                                cache_name=f'{phase}_{seed}_{c["name"]}_joint_{group[0]["search"]}_{offset}')


def smoke(ctx):
    from .verification import contracts
    times={};started=time.perf_counter()
    times['contracts']=contracts(ctx)
    t=time.perf_counter();run_audit(ctx,smoke=True);times['termination_audit']=time.perf_counter()-t
    records=select_states(ctx,smoke=True)
    # Full-dimensional replay workload, two states only. This is not scientific validation.
    selected=[next(r for r in records if r['split']=='development'),next(r for r in records if r['split']=='confirmation')]
    t=time.perf_counter()
    for record in selected:run_replay(ctx,record,smoke=True)
    times['action_ranking_and_offline_blend']=time.perf_counter()-t
    t=time.perf_counter();run_coverage(ctx,selected[:1]);times['distribution_analysis']=time.perf_counter()-t
    t=time.perf_counter();support_rankings(ctx,selected[0]);times['support_ranking']=time.perf_counter()-t
    condition=ctx.cfg['conditions'][0]
    timing_rows=[]
    for family in FAMILIES:
        t=time.perf_counter()
        for group in batches(development_settings(family)):
            p=run_control(ctx,'smoke',11,condition,[900],group,steps=30)
            timing_rows.append(dict(family=family,search=group[0]['search'],batch=p['batch'],steps=p['steps'],
                                    setup_seconds=p['graph_and_setup_seconds'],loop_seconds=p['loop_seconds'],wall_seconds=p['wall_seconds']))
        times[family]=time.perf_counter()-t
    t=time.perf_counter()
    joint=[s for f in FAMILIES for s in development_settings(f) if s['search']=='baseline'][:64]
    p=run_control(ctx,'smoke',11,condition,[903],joint,steps=30,cache_name='smoke_joint_baseline64')
    timing_rows.append(dict(family='joint',search='baseline',batch=p['batch'],steps=p['steps'],
                            setup_seconds=p['graph_and_setup_seconds'],loop_seconds=p['loop_seconds'],wall_seconds=p['wall_seconds']))
    # Confirmation batches include three trials. Measure the largest search with that dimension too.
    for budget in ['expanded','high']:
        opts=[s for s in development_settings('search') if s['search']==budget]
        p=run_control(ctx,'smoke',11,condition,[904,905,906],opts,steps=30,cache_name='smoke_joint_'+budget+'15')
        timing_rows.append(dict(family='joint',search=budget,batch=p['batch'],steps=p['steps'],
                                setup_seconds=p['graph_and_setup_seconds'],loop_seconds=p['loop_seconds'],wall_seconds=p['wall_seconds']))
    times['joint_throughput']=time.perf_counter()-t
    online=next(c for c in ctx.cfg['conditions'] if c['online'])
    t=time.perf_counter();p=run_control(ctx,'smoke',11,online,[901],development_settings('baseline'),steps=80)
    times['online_updates']=time.perf_counter()-t
    # Balancing survives long enough to verify all four update boundaries without a full episode.
    online_balance=dict(name='smoke_online_balance',environment='strong_gain',task='balance',horizon=15,online=True)
    check=run_control(ctx,'smoke',11,online_balance,[902],development_settings('baseline'),steps=205)
    assert set(u['observations'] for u in check['updates'])=={5,10,20,50}, 'Online cumulative refit path not exercised'
    # Exercise development selection on explicitly labeled short-run smoke observations.
    # This fixture never writes into a scientific run or supplies its locked choices.
    fixture_cfg=dict(ctx.cfg,seeds=[11],conditions=[condition],development_trials=[900],confirmation_trials=[907])
    fixture=Context(ctx.run/'selection_contract',fixture_cfg,ctx.device)
    fixture.run.mkdir(exist_ok=True)
    for path in (ctx.run/'control').glob('smoke_11_nominal_*.json'):
        packet=json.loads(path.read_text())
        packet['key']='development_'+packet['key']
        for row in packet['rows']:row['phase']='development'
        save_json(fixture.run/'control'/f'{packet["key"]}.json',packet)
    choices=select_confirmation(fixture)
    for group in batches(choices[condition['name']]['interaction']):
        run_control(fixture,'confirmation',11,condition,[907],group,steps=5)
    save_json(ctx.run/'selection_smoke.json',dict(passed=True,scientific_selection=False,
                independently_namespaced_confirmation=True,resume_config_roundtrip=configuration()==json.loads((ctx.run/'config_locked.json').read_text())))
    # Validate all four cumulative update boundaries when episodes permit; explicit counts saved.
    save_json(ctx.run/'smoke_timings.json',dict(timings=times,control_timings=timing_rows,total_seconds=time.perf_counter()-started,
                                            online_updates_logged=len(p['updates']),scientific_validation=False))
    from .report import generate
    generate(ctx,smoke=True)
    estimate(ctx,timing_rows,times)
    verify_preserved(ctx.run)
    save_json(ctx.run/'status.json',dict(complete=True,mode='smoke_only',full_evaluation_started=False))


def estimate(ctx,timing_rows,times):
    estimates=[]
    for family in FAMILIES:
        seconds=0;groups=0;rows=0
        for c in ctx.cfg['conditions']:
            if not applicable(c,family):continue
            for options in batches(development_settings(family)):
                match=next(t for t in timing_rows if t['family']==family and t['search']==options[0]['search'])
                # Same candidate/search dimensions measured; horizon15 scaled by15/30.
                horizon_ratio=c['horizon']/30
                count=ctx.cfg[c['task']+'_steps']
                seconds+=len(ctx.cfg['seeds'])*(match['setup_seconds']+match['loop_seconds']/30*count*horizon_ratio)
                groups+=len(ctx.cfg['seeds']);rows+=len(ctx.cfg['seeds'])*len(options)
        # Confirmation is selected after development. Bound it by1..3trials and maximum retained winners/anchors.
        confirmation_factor={'baseline':2.2,'blend':1.0,'search':1.8,'support':1.0,'rail':2.2,'cost':2.2}[family]
        confirmation=seconds*confirmation_factor
        estimates.append(dict(investigation=family,development_groups=groups,development_episodes=rows,
                              estimated_development_seconds=seconds,estimated_confirmation_seconds=confirmation,
                              estimated_total_seconds=seconds+confirmation))
    replay_count=len(ctx.cfg['seeds'])*2*len(ctx.cfg['replay_categories'])
    offline=(times['action_ranking_and_offline_blend']/2+times['distribution_analysis']+times['support_ranking'])*replay_count
    closed=sum(e['estimated_total_seconds'] for e in estimates)
    # Estimate the real coalesced schedule rather than adding independent family timings.
    joint_seconds=0
    for c in ctx.cfg['conditions']:
        dev=[s for f in FAMILIES if applicable(c,f) for s in development_settings(f)]
        # Scheduling envelope before development winners are known. This does not select scientific settings.
        confirm=[setting(m) for m in METHODS]
        confirm += [setting(m,'blend',alpha=a) for m in ADAPTERS for a in [0,.5,1]]
        confirm += [setting(m,'search',search='high') for m in METHODS]
        if not c['online']:
            confirm += [setting(m,'support',support=t,K=k) for m in ADAPTERS for t,k in [('consecutive',20),('state_diverse',20),('state_diverse',50)]]
        confirm += [setting(m,'rail',safety='margin') for m in METHODS]
        if c['task']=='swingup':confirm += [setting(m,'cost',cost='terminal') for m in METHODS]
        confirm += [setting(m,'interaction',alpha=.5,search='high') for m in ADAPTERS]
        if not c['online']:confirm += [setting(m,'interaction',alpha=.5,support='state_diverse') for m in ADAPTERS]
        for options,trials in [(dev,1),(confirm,3)]:
            for group in batches(options):
                width=ctx.cfg['max_control_batch_rows']//trials
                for offset in range(0,len(group),width):
                    batch=len(group[offset:offset+width])*trials
                    match=next(t for t in timing_rows if t['family']=='joint' and t['search']==group[0]['search'])
                    ratio=max(.45,batch/match['batch'])
                    seconds=match['setup_seconds']+match['loop_seconds']/match['steps']*ctx.cfg[c['task']+'_steps']*(c['horizon']/30)*ratio
                    joint_seconds+=len(ctx.cfg['seeds'])*seconds
    scheduling_scale=joint_seconds/closed
    for entry in estimates:
        for key in ['estimated_development_seconds','estimated_confirmation_seconds','estimated_total_seconds']:
            entry[key]*=scheduling_scale
    central=(joint_seconds+offline)*1.15
    result=dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',families=estimates,
                estimated_offline_seconds=offline,estimated_total_seconds=central,range_seconds=[central*.75,central*1.6],
                scheduling='coalesced ablation batches, maximum64 rows; family allocations share batch time and are approximate',
                caveats=['Extrapolated from30-step full-dimensional smoke groups, not a full scientific run.',
                         'Held-out winners and retained options are unknown until development; confirmation figures are scheduling estimates.',
                         'CUDA graph capture is measured; long-run thermal/power behavior and disk compression can change speed.',
                         'No full tests or offline training were launched.'])
    save_json(ctx.run/'runtime_estimate.json',result)
    print(f'ESTIMATE {central/3600:.2f} hours; range {central*.75/3600:.2f}..{central*1.6/3600:.2f}',flush=True)


def execute(ctx,stage):
    records=None
    if stage in ['all','audit']:
        run_audit(ctx)
    if stage in ['all','offline']:
        records=select_states(ctx)
        for rec in records:run_replay(ctx,rec)
        run_coverage(ctx,records)
        for rec in records:support_rankings(ctx,rec)
    if stage in ['all','develop']:
        stage_control(ctx,'development')
        select_confirmation(ctx)
    if stage in ['all','confirm']:
        path=ctx.run/'selection_locked.json'
        if not path.exists():raise RuntimeError('Run development before confirmation')
        stage_control(ctx,'confirmation',json.loads(path.read_text())['settings'])
    if stage in ['all','report']:
        from .report import generate
        generate(ctx)
        verify_preserved(ctx.run)
        save_json(ctx.run/'status.json',dict(complete=True,mode='full',phase='complete'))


def main():
    p=argparse.ArgumentParser(description='V2 diagnostics only: no neural training. Smoke is separate from scientific output.')
    p.add_argument('stage',choices=['smoke','all','audit','offline','develop','confirm','report'])
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args();torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable')
    timestamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=args.run_dir or ROOT/('smoke tests' if args.stage=='smoke' else 'results/cartpole')/f'{timestamp}_v2_control_diagnostics'
    cfg=configuration();prepare(run,cfg,args.stage=='smoke')
    # Prevent two workers mutating the same cache or competing for the same run.
    lock=run/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run lock exists: {lock}. Check its PID before removing a stale lock.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.perf_counter();ctx=Context(run,cfg,args.device)
    save_json(run/'status.json',dict(complete=False,phase=args.stage,pid=os.getpid()))
    print(f'RUN_DIRECTORY {run}',flush=True)
    try:
        if args.stage=='smoke':smoke(ctx)
        else:execute(ctx,args.stage)
    except Exception as exc:
        save_json(run/'status.json',dict(complete=False,phase=args.stage,error=repr(exc),resumable=True))
        raise
    finally:
        save_json(run/f'execution_{args.stage}.json',dict(wall_seconds=time.perf_counter()-started,device=args.device))
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
