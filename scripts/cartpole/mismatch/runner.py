"""Explicit-mode, stage-selectable, atomic and resumable diagnostic orchestration."""
import argparse
import datetime as dt
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .protocol import ROOT,REFERENCE,STAGES,METAS,ORIGINAL,ROLLOUT,SOURCES,configuration,workload,fork_policies
from .common import Context,save_json,file_hash,source_files
from .cohorts import saved_manifest,saved_analysis,record
from .replay import analyze_case,fixed_query,evaluate_pool
from .control import control,prospective
from .dense import fit
from .report import generate


def prepare(run,cfg,mode):
    run.mkdir(parents=True,exist_ok=True)
    for name,value in [('config_locked.json',cfg),('source_hashes.json',source_files())]:
        path=run/name
        if path.exists():
            if json.loads(path.read_text())!=value:raise RuntimeError(f'Incompatible {name}; restore saved source snapshot or start a new run')
        else:save_json(path,value)
    if not (run/'source_snapshot').exists():
        for name in source_files():
            p=run/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    preserved=run/'preserved_inputs.json'
    if not preserved.exists():
        folders=[p for p in (ROOT/'results/cartpole').iterdir() if p.is_dir() and p!=run and run not in p.parents]
        if mode=='full':files=[p for folder in [*folders,ROOT/'archive'] for p in folder.rglob('*') if p.is_file()]
        else:
            files=[p for folder in folders for p in folder.rglob('*.pt')]
            files += [p for p in REFERENCE.glob('*.json') if p.name!='preserved_artifacts.json']
            files += list(REFERENCE.glob('*.csv'))+[REFERENCE/'report.md',ROOT/'docs/analysis/cartpole_v2_final_20261027/report.md']
        save_json(preserved,{str(p.relative_to(ROOT)):file_hash(p) for p in files})
    save_json(run/'planned_workload.json',workload(cfg))


def integrity(ctx):
    files=json.loads((ctx.run/'preserved_inputs.json').read_text())
    changed=[n for n,h in files.items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=h]
    if changed:raise RuntimeError(f'Preserved scientific artifacts changed: {changed[:3]}')
    if source_files()!=json.loads((ctx.run/'source_hashes.json').read_text()):raise RuntimeError('Sources changed during execution')
    save_json(ctx.run/'integrity_verified.json',dict(all_unchanged=True,files=len(files)))


def progress(ctx,stage,**info):
    save_json(ctx.run/'status.json',dict(complete=False,stage=stage,pid=os.getpid(),**info));print(f'STAGE {stage} {info}',flush=True)


def recoverability(ctx):
    rows=[]
    for path in (ctx.run/'episodes').glob('prefail_*.json'):
        packet=json.loads(path.read_text());r=packet['record']
        for row in packet['rows']:rows.append(dict(seed=row['seed'],collector=r['collector_method'],lead=r['lead'],policy=row['policy'],success=row['success'],cost=row['cost'],rail_failure=row['rail_failure']))
    d=pd.DataFrame(rows);d.to_csv(ctx.run/'earlier_recovery_replays.csv',index=False)
    if len(d):
        q=d[d.policy.isin(['oracle','oracle_original'])];out=[]
        for (seed,collector,policy),g in q.groupby(['seed','collector','policy']):
            successful=g[g.success]
            out.append(dict(seed=seed,collector=collector,policy=policy,tested=len(g),successful=len(successful),
                latest_tested_recoverable_lead=int(successful.lead.min()) if len(successful) else None,
                conclusion='Recovery demonstrated at tested lead' if len(successful) else 'No recovery demonstrated; physical impossibility not established'))
        pd.DataFrame(out).to_csv(ctx.run/'earlier_recovery_windows.csv',index=False)


def completion_audit(ctx,manifest):
    planned=workload(ctx.cfg);actual=0
    for p in (ctx.run/'episodes').glob('*.json'):
        packet=json.loads(p.read_text());assert packet['config_sha256']==ctx.digest
        assert p.with_suffix('.npz').exists();assert not any(r['incomplete_time_limit'] for r in packet['rows'])
        actual+=len(packet['rows'])
    unavailable=6*sum(r['id'].startswith('prefail_') for r in manifest['unavailable'])
    missing=[]
    for p in (ctx.run/'unavailable').glob('*.json'):
        factor=20 if p.stem.startswith(('dev_fork_','confirm_fork_')) else 7 if p.stem.startswith('confirmation_') else 0
        unavailable+=factor;missing.append(dict(key=p.stem,control_slots=factor))
    if actual+unavailable!=planned['full_control_slots']:
        raise RuntimeError(f'Completion accounting mismatch: {actual} executed +{unavailable} unavailable versus{planned["full_control_slots"]} planned')
    for r in manifest['cases']:
        assert (ctx.run/'replay'/f'{r["id"]}.json').exists()
    for r in manifest['fixed_queries']:
        p=json.loads((ctx.run/'fixed_query'/f'{r["id"]}.json').read_text())
        assert sorted(set(row['support_K'] for row in p['rows']))==ctx.cfg['checkpoints']
    assert len(list((ctx.run/'dense').glob('*_best.json')))==planned['dense_diagnostic_copies']
    save_json(ctx.run/'completion_audit.json',dict(passed=True,executed_controls=actual,unavailable_control_slots=unavailable,
        planned_controls=planned['full_control_slots'],unavailable_records=missing))


def full(ctx,stages):
    registry=json.loads((ctx.run/'completion_registry.json').read_text()) if (ctx.run/'completion_registry.json').exists() else {}
    manifest=saved_manifest(ctx)
    for stage in stages:
        if registry.get(stage,{}).get('complete'):continue
        progress(ctx,stage);tick=time.perf_counter()
        if stage=='saved':saved_analysis(ctx,manifest)
        elif stage=='replay':
            for i,r in enumerate(manifest['cases']):progress(ctx,stage,done=i,total=len(manifest['cases']));analyze_case(ctx,r)
            for r in manifest['fixed_queries']:fixed_query(ctx,r)
        elif stage=='forks':
            for r in manifest['pre_failure']:control(ctx,r['id'],r,fork_policies(pre_failure=True))
            for seed in ctx.cfg['model_seeds']:
                r=next((r for r in manifest['cases'] if r['seed']==seed and r['environment']=='combined_geometry' and r['collector_method']==ORIGINAL and r['step']==ctx.cfg['intervention_time']),None)
                if r:control(ctx,'substitution_'+r['id'],r,fork_policies())
                else:save_json(ctx.run/'unavailable'/f'dev_fork_{seed}.json',dict(reason='selected intervention state unavailable'))
            recoverability(ctx)
        elif stage=='dense':
            if not registry.get('replay',{}).get('complete'):raise RuntimeError('Dense training requires completed development replay stage; --stages saved,replay,forks,dense')
            for seed in ctx.cfg['model_seeds']:
                for env in ctx.cfg['dense_environments']:
                    for method in METAS:fit(ctx,seed,env,method)
            for r in manifest['fixed_queries']:
                p=ctx.run/'fixed_query'/f'{r["id"]}.npz'
                with np.load(p) as a:shared=(a['actions'],a['origins'],{m:int(a['native_indices'][j]) for j,m in enumerate(SOURCES)})
                analyze_case(ctx,r,dense=True,actions_override=shared)
        elif stage=='confirmation':
            if not registry.get('dense',{}).get('complete'):raise RuntimeError('Independent confirmation requires completed dense stage')
            for seed in ctx.cfg['model_seeds']:
                for trial in ctx.cfg['confirmation_trials']:
                    for case in ctx.cfg['conditions']:
                        parent=prospective(ctx,seed,trial,case)
                        if parent is None:
                            if case['environment']=='combined_geometry' and trial==ctx.cfg['confirmation_trials'][0]:
                                save_json(ctx.run/'unavailable'/f'confirm_fork_{seed}.json',dict(reason='shared nominal prefix ineligible for planned confirmation forks',control_slots=20))
                            continue
                        geometry=case['environment']=='combined_geometry'
                        if not geometry and trial!=ctx.cfg['confirmation_trials'][0]:continue
                        methods=METAS if geometry else [ORIGINAL]
                        for method in methods:
                            i=next(j for j,row in enumerate(parent['rows']) if row['method']==method)
                            for step in ([50,150] if geometry else [50]):
                                key=f'confirm_{seed}_{trial}_{case["name"]}_{method}_t{step}'
                                r=record(ctx,parent,i,step,key,ctx.run,'confirmation')
                                if r:analyze_case(ctx,r,dense=True)
                                else:save_json(ctx.run/'unavailable'/f'{key}.json',dict(reason='actual collector failed before selected confirmation state'))
                        if geometry and trial==ctx.cfg['confirmation_trials'][0]:
                            i=next(j for j,row in enumerate(parent['rows']) if row['method']==ORIGINAL)
                            r=record(ctx,parent,i,150,f'confirm_fork_{seed}',ctx.run,'confirmation')
                            if r:control(ctx,r['id'],r,fork_policies())
                            else:save_json(ctx.run/'unavailable'/f'confirm_fork_{seed}.json',dict(reason='selected confirmation intervention unavailable'))
        elif stage=='synthesis':generate(ctx)
        registry[stage]=dict(complete=True,wall_seconds=time.perf_counter()-tick)
        save_json(ctx.run/'completion_registry.json',registry)
        save_json(ctx.run/'restoration_log.json',ctx.restoration)
    integrity(ctx)
    complete=all(registry.get(stage,{}).get('complete') for stage in STAGES)
    if complete:completion_audit(ctx,manifest)
    save_json(ctx.run/'status.json',dict(complete=complete,stage='complete' if complete else 'selected_stages_complete',mode='full',scientific_validation=complete,
        remaining_stages=[s for s in STAGES if not registry.get(s,{}).get('complete')]))


def bounded(ctx,mode):
    from .verification import verify
    manifest=saved_manifest(ctx,bounded=True);saved_analysis(ctx,manifest)
    r=manifest['cases'][0];tick=time.perf_counter();packet=analyze_case(ctx,r);replay_seconds=time.perf_counter()-tick
    fixed_packet=fixed_query(ctx,manifest['fixed_queries'][0],small=True);fixed_seconds=fixed_packet['wall_seconds']
    dense=[]
    for method in METAS:dense.append(fit(ctx,11,'combined_geometry',method,small=mode=='smoke',updates_override=12 if mode=='smoke' else 100))
    f=manifest['fixed_queries'][0]
    with np.load(ctx.run/'fixed_query'/f'{f["id"]}.npz') as a:shared=(a['actions'],a['origins'],{m:int(a['native_indices'][i]) for i,m in enumerate(SOURCES)})
    analyze_case(ctx,f,dense=True,actions_override=shared)
    limit=3 if mode=='smoke' else 30
    fork=control(ctx,'bounded_substitution',r,fork_policies(),limit=limit)
    pre=manifest['pre_failure'][0];early=control(ctx,'prefail_bounded',pre,fork_policies(pre_failure=True),limit=limit)
    case=ctx.cfg['conditions'][0];live=prospective(ctx,11,899,case,include_dense=True,limit=5 if mode=='smoke' else 55)
    recoverability(ctx);checks=verify(ctx,r,packet)
    save_json(ctx.run/'restoration_log.json',ctx.restoration)
    # One shared candidate fixture benchmarks model inference batching; no scientific episode is added.
    inference=[]
    names,models=ctx.fitted_models(r)
    with np.load(ctx.run/'replay'/f'{r["id"]}.npz') as a:actions=a['actions'][:256]
    for device,n in ([('cuda',32),('cuda',128),('cuda',256),('cpu',32)] if ctx.device=='cuda' else [('cpu',32),('cpu',128)]):
        measured=Context(ctx.run,ctx.cfg,device);measured.models=ctx.models
        tick=time.perf_counter();evaluate_pool(measured,r,names,models,actions[:n])
        if device=='cuda':torch.cuda.synchronize()
        inference.append(dict(device=device,candidates=n,models=len(names),seconds=time.perf_counter()-tick))
    estimate=estimate_runtime(ctx,packet,fixed_seconds,dense,fork,early,live)
    save_json(ctx.run/'benchmark.json',dict(mode=mode,replay_seconds=replay_seconds,fixed_query_three_K_seconds=fixed_seconds,
        dense_fits=dense,substitution_setup=fork['setup_seconds'],substitution_loop_seconds=fork['loop_seconds'],substitution_steps=fork['mpc_batch_calls'],
        native_setup=live['setup_seconds'],native_loop_seconds=live['loop_seconds'],native_steps=live['mpc_batch_calls'],inference=inference))
    save_json(ctx.run/'runtime_estimate.json',estimate)
    generate(ctx,bounded=True);integrity(ctx)
    save_json(ctx.run/'status.json',dict(complete=True,mode=mode,stage='bounded_complete',full_suite_started=False,complete_dense_training_started=False,
        scientific_validation=False,contract_checks=len(checks)))


def estimate_runtime(ctx,replay,fixed_seconds,dense,fork,early,live):
    w=workload(ctx.cfg);n=len(ctx.cfg['model_seeds']);c=len(ctx.cfg['conditions'])
    step_native=live['loop_seconds']/live['mpc_batch_calls'];step_fork=fork['loop_seconds']/fork['mpc_batch_calls'];step_pre=early['loop_seconds']/early['mpc_batch_calls']
    # Full remaining episode lengths retained. Early rail termination can reduce measured work substantially.
    parts=dict(saved=30.,replay=w['saved_replay_slots']*replay['wall_seconds']+w['fixed_query_groups']*fixed_seconds*8/3,
        forks=w['pre_failure_state_slots']*(early['setup_seconds']+step_pre*500)+n*(fork['setup_seconds']+step_fork*440),
        dense=sum(sum(t['elapsed_seconds'] for t in p['optimizations'])/p['total_updates'] for p in dense)/len(dense)*w['dense_training_updates']+
              n*len(ctx.cfg['dense_environments'])*3.+w['fixed_query_groups']*replay['wall_seconds']*128/1676*8/6,
        confirmation=n*len(ctx.cfg['confirmation_trials'])*((c-1)*(live['setup_seconds']+step_native*590)+(live['setup_seconds']+step_native*300))+
            w['confirmation_replay_slots']*replay['wall_seconds']*8/6+n*(fork['setup_seconds']+step_fork*440),synthesis=60.)
    seconds=sum(parts.values())
    return dict(measured_device=torch.cuda.get_device_name(0) if ctx.device=='cuda' else 'CPU',stage_seconds=parts,
        point_hours=seconds/3600,range_hours=[seconds*.65/3600,seconds*1.65/3600],
        workload=w,measurement='Point extrapolation from this bounded run; no full suite executed',
        caveats=['Full episode horizons preserved; many pre-failure forks may terminate early.',
                 'One hard geometry workload is not every phase/physics; dense convergence and serialization add uncertainty.',
                 'GPU timings depend on laptop power/thermal load and other applications; no concurrent GPU jobs enabled.',
                 'Per-stage source preservation overhead is outside kernel timing; allow extra minutes for hashing and final report.'])


def main():
    parser=argparse.ArgumentParser(description='V2 model/planner diagnostics. Default smoke; --mode full explicitly starts science and isolated dense-copy training.')
    parser.add_argument('--mode',choices=['smoke','benchmark','full','report'],default='smoke')
    parser.add_argument('--run-dir',type=Path);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--stages',default='all',help='Comma-separated saved,replay,forks,dense,confirmation,synthesis; prerequisites must already be complete')
    args=parser.parse_args();torch.set_num_threads(1)
    stages=STAGES if args.stages=='all' else args.stages.split(',')
    if any(s not in STAGES for s in stages):parser.error('Unknown stage')
    if args.mode=='report' and args.run_dir is None:parser.error('report mode requires --run-dir')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    cfg=configuration();stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run=(args.run_dir or ROOT/('results/cartpole' if args.mode=='full' else 'smoke tests')/f'{stamp}_v2_model_planner_mismatch_{args.mode}').resolve()
    existing=[p.resolve() for p in (ROOT/'results/cartpole').iterdir() if p.is_dir() and (p.resolve()!=run or not (p/'run_kind.json').exists())]
    protected=[REFERENCE,ROOT/'archive']
    if any(run==p or p in run.parents for p in [*existing,*protected]):raise RuntimeError('Refusing to write into an existing scientific/archive directory')
    if run.exists() and any(run.iterdir()) and not (run/'config_locked.json').exists():raise RuntimeError('Refusing unrelated nonempty directory')
    kindpath=run/'run_kind.json';kind='scientific' if args.mode=='full' else 'bounded'
    if kindpath.exists():
        prior=json.loads(kindpath.read_text())
        if prior.get('suite')!='v2_mismatch' or (args.mode!='report' and prior['kind']!=kind):raise RuntimeError('Cannot mix another suite, bounded data and scientific data')
    if args.mode=='full' and (ROOT/'smoke tests') in run.parents:raise RuntimeError('Full scientific outputs cannot use smoke directory')
    prepare(run,cfg,args.mode)
    if not kindpath.exists():save_json(kindpath,dict(kind=kind,suite='v2_mismatch'))
    lock=run/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run already locked: {lock}; inspect recorded PID before removing a stale lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    ctx=Context(run,cfg,args.device);started=time.perf_counter()
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    hardware_path=run/('report_hardware.json' if args.mode=='report' else 'hardware.json')
    if not hardware_path.exists():save_json(hardware_path,dict(device=args.device,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None))
    print(f'RUN_DIRECTORY {run}',flush=True)
    try:
        if args.mode=='full':full(ctx,stages)
        elif args.mode=='report':generate(ctx,json.loads(kindpath.read_text())['kind']=='bounded');integrity(ctx)
        else:bounded(ctx,args.mode)
    except Exception as exc:
        save_json(run/'status.json',dict(complete=False,mode=args.mode,error=repr(exc),resumable=True));raise
    finally:
        save_json(run/f'execution_{args.mode}.json',dict(wall_seconds=time.perf_counter()-started,
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if args.device=='cuda' else 0,
            cache_hits=ctx.hits,cache_misses=ctx.misses))
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
