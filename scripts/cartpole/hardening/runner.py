"""One resumable entry point with explicit smoke/full separation and sequential gates."""
import argparse
import datetime as dt
import json
import os
import time
import subprocess
import traceback
from pathlib import Path
import torch
from .protocol import ROOT,STAGES,configuration,workload,control_jobs
from .common import Context,prepare,integrity,save_json,file_hash
from .storage import save_advisory_json
from . import stages


def validate(ctx,stage,persist=True):
    controls=list((ctx.run/'controls').glob(f'{stage}_*.json'))
    planned={r['key']:r['slots'] for r in control_jobs(ctx.cfg) if r['stage']==stage}
    if {p.stem for p in controls}!=set(planned):raise RuntimeError(f'{stage} job identities do not match the locked matrix')
    slots=0;unavailable=0;executed=0
    for path in controls:
        p=ctx.cached_job(path);executed+=len(p.get('rows',[]));unavailable+=p.get('unavailable_slots',0)
        if len(p.get('rows',[]))+p.get('unavailable_slots',0)!=planned[path.stem]:raise RuntimeError(f'Wrong policy count: {path}')
        if len({r['policy'] for r in p.get('rows',[])})!=len(p.get('rows',[])):raise RuntimeError(f'Duplicate policy labels: {path}')
        if any(r['incomplete_time_limit'] for r in p.get('rows',[])):raise RuntimeError(f'Truncated scientific episode {path}')
        if p.get('hidden_true_state_used_for_control'):raise RuntimeError('True state leaked to controller')
        for u in p.get('updates',[]):
            if u['K']>50 or u['step']!=u['segment']+u['K'] or u['support_count']!=u['K']:raise RuntimeError('Observation causality/count contract failed')
    expected=workload(ctx.cfg)[stage]['control_slots_upper' if stage=='A' else 'control_slots']
    if executed+unavailable!=expected:raise RuntimeError(f'{stage} completion mismatch: {executed}+{unavailable} vs {expected}')
    offline_paths=list((ctx.run/'offline').glob(f'{stage}*.json'))
    for path in offline_paths:ctx.cached_job(path)
    if stage=='A':
        groups=len(ctx.cfg['model_seeds'])*len(ctx.cfg['dev_environments'])*len(ctx.cfg['query_times'])
        for tag,count in [('A12',groups),('A34',groups),('A5',groups*len(ctx.cfg['budget_K']))]:
            actual=len(list((ctx.run/'offline').glob(f'{tag}_*.json')))
            if actual!=count:raise RuntimeError(f'{tag} rows missing: {actual} vs {count} groups')
        if not (ctx.run/'report_A_development.md').exists() or not (ctx.run/'A_selection_frozen.json').exists():raise RuntimeError('A report/selection gate missing')
        expected_queries=0
        for path in (ctx.run/'controls').glob('A_confirm_*_snapshot.json'):
            if json.loads(path.read_text())['rows']:expected_queries+=len(ctx.cfg['confirmation_query_times'])
        if len(list((ctx.run/'offline').glob('A_confirmation_*.json')))!=expected_queries:raise RuntimeError('A independent confirmation query jobs missing')
    value=dict(passed=True,executed_controls=executed,unavailable_slots=unavailable,planned_slots=expected)
    if stage=='A':
        value.update(selection_sha256=file_hash(ctx.run/'A_selection_frozen.json'),
                     development_report_sha256=file_hash(ctx.run/'report_A_development.md'))
        previous=ctx.run/'completion_A.json'
        if previous.exists():
            old=json.loads(previous.read_text())
            if old.get('selection_sha256')!=value['selection_sha256'] or old.get('development_report_sha256')!=value['development_report_sha256']:
                raise RuntimeError('Completed A selection/report changed')
    if persist:save_json(ctx.run/f'completion_{stage}.json',value)
    return value


def check_resume(ctx):
    """Read and validate cached work only. Never call a scientific stage."""
    tick=time.perf_counter()
    registry=json.loads((ctx.run/'completion_registry.json').read_text())
    completed=[s for s in STAGES if registry.get(s,{}).get('complete')]
    for stage in completed:
        if any(d not in completed for d in ctx.cfg['dependencies'][stage]):
            raise RuntimeError(f'Invalid completed-stage dependencies: {stage}')
        actual=validate(ctx,stage,persist=False)
        if actual!=registry[stage]['completion']:raise RuntimeError(f'Changed completion audit: {stage}')
    counts={};control_paths=[];orphan_arrays=[]
    for folder in ['candidate_pools','collections','controls','fits','latency','offline']:
        paths=sorted((ctx.run/folder).glob('*.json'))
        for path in paths:ctx.cached_job(path)
        counts[folder]=len(paths)
        orphan_arrays.extend(str(p.relative_to(ctx.run)) for p in (ctx.run/folder).glob('*.npz')
                             if not p.with_suffix('.json').exists())
        if folder=='controls':control_paths=paths
        print(f'RESUME_CHECK {folder}: {len(paths)} checksummed job pairs',flush=True)
    # Legacy nominal prefixes have no recorded array hash. Check their existing
    # config lock and read every array (ZIP CRC validation), without regenerating.
    import numpy as np
    prefixes=list((ctx.run/'warmup').glob('*.json'))
    for path in prefixes:
        packet=json.loads(path.read_text())
        if packet.get('config_sha256')!=ctx.digest:raise RuntimeError(f'Incompatible prefix: {path}')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as arrays:
            for name in arrays.files:arrays[name]
    planned={r['key']:r for r in control_jobs(ctx.cfg)}
    existing={p.stem:p for p in control_paths}
    if not set(existing).issubset(planned):raise RuntimeError('Unplanned cached controls in resume directory')
    per_stage={s:dict(completed_jobs=0,executed_controls=0,unavailable_slots=0,
                     planned_jobs=sum(r['stage']==s for r in planned.values())) for s in STAGES}
    for key,path in existing.items():
        packet=json.loads(path.read_text());spec=planned[key]
        n=len(packet.get('rows',[]));unavailable=packet.get('unavailable_slots',0)
        if n+unavailable!=spec['slots']:raise RuntimeError(f'Partial control packet: {path}')
        per_stage[spec['stage']]['completed_jobs']+=1
        per_stage[spec['stage']]['executed_controls']+=n
        per_stage[spec['stage']]['unavailable_slots']+=unavailable
    print('RESUME_CHECK validating protected historical inputs',flush=True)
    preserved=integrity(ctx,persist=False)
    remaining=[s for s in STAGES if s not in completed]
    missing=[key for stage in STAGES for key,spec in planned.items() if spec['stage']==stage and key not in existing]
    return dict(passed=True,check_only=True,scientific_work_executed=False,
                run_directory=str(ctx.run.resolve()),completed_stages=completed,remaining_stages=remaining,
                cache_pairs_verified=sum(counts.values()),job_counts=counts,warmup_prefixes_verified=len(prefixes),
                control_progress=per_stage,first_missing_control=missing[0] if missing else None,
                orphan_arrays_without_commit=orphan_arrays,preserved_inputs=preserved,
                source_revision_recorded=(ctx.run/'io_repair_revision.json').exists(),
                seconds=time.perf_counter()-tick)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','benchmark','full','report'],default='smoke')
    parser.add_argument('--stages',default='A,B,C,D',help='Comma-separated stages; dependencies must already be complete')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--run-dir',type=Path,help='Exact existing directory to resume; mode/config/source must match')
    parser.add_argument('--check-resume',action='store_true',help='Validate an existing run without fitting, control or stage execution')
    args=parser.parse_args();cfg=configuration();chosen=[s.strip().upper() for s in args.stages.split(',')]
    if not chosen or any(s not in STAGES for s in chosen) or chosen!=sorted(set(chosen),key=STAGES.index):parser.error('Stages must be unique and ordered A,B,C,D')
    if args.check_resume and (args.run_dir is None or args.mode!='full'):
        parser.error('--check-resume requires --mode full and an existing --run-dir; it executes no scientific work')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but not available; no silent CPU fallback')
    torch.set_num_threads(1)
    if args.mode in ['smoke','benchmark']:
        cfg.update(disturbance_step=2,repeated_steps=[2,7],combined_impulse_step=5,sequence_change_steps=[3,8])
    if args.check_resume:
        prepare(args.run_dir,cfg,args.mode,check_only=True)
        result=check_resume(Context(args.run_dir,cfg,args.device,args.mode))
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        output=args.run_dir/'resume_checks'/f'{stamp}.json'
        save_json(output,result)
        print(json.dumps(result,indent=2));print(f'RESUME_CHECK_REPORT={output}');return
    if args.mode=='report':
        if args.run_dir is None:parser.error('--mode report requires --run-dir')
        cfg=json.loads((args.run_dir/'config_locked.json').read_text());mode=json.loads((args.run_dir/'run_kind.json').read_text())['mode']
        ctx=Context(args.run_dir,cfg,args.device,mode)
        from .report import generate,plots
        c,o=generate(ctx);generate(ctx,only_A=True);plots(ctx,c,o);return
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    parent=ROOT/('results/cartpole' if args.mode=='full' else 'smoke tests')
    run=args.run_dir.resolve() if args.run_dir else parent/f'{stamp}_v2_final_hardening_{args.mode}'
    print(f'RUN_DIRECTORY={run}',flush=True);prepare(run,cfg,args.mode)
    ctx=Context(run,cfg,args.device,args.mode);save_json(run/'planned_workload.json',workload(cfg))
    hardware=dict(torch=torch.__version__,cuda_available=torch.cuda.is_available(),execution_device=args.device,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/2**20 if args.device=='cuda' else None)
    save_json(run/'hardware.json',hardware)
    tick=time.perf_counter();cpu_started=time.process_time();registry=json.loads((run/'completion_registry.json').read_text()) if (run/'completion_registry.json').exists() else {}
    from final_validation.benchmark import Monitor
    monitor=Monitor() if args.mode=='full' else None
    if monitor is not None:monitor.__enter__()
    try:
        if args.mode in ['smoke','benchmark']:
            from .verification import bounded,estimate
            measurements=bounded(ctx,args.mode);estimate(ctx,measurements)
        else:
            w=workload(cfg)
            # Estimate remaining jobs, including at most seven updates per adaptive policy/segment.
            ctx.total=sum(w['maximum_fit_calls'].values())+w['A']['offline_prediction_rows_upper']//2+sum(
                w[s]['control_slots_upper' if s=='A' else 'control_slots'] for s in STAGES)//7
            for stage in chosen:
                if registry.get(stage,{}).get('complete'):validate(ctx,stage);continue
                missing=[d for d in cfg['dependencies'][stage] if not registry.get(d,{}).get('complete')]
                if missing:raise RuntimeError(f'{stage} needs completed {missing}; run missing stages first in this directory')
                ctx.current_stage=stage
                save_json(run/'status.json',dict(complete=False,stage=stage,pid=os.getpid(),mode=args.mode))
                ts=time.perf_counter();getattr(stages,stage)(ctx);audit=validate(ctx,stage)
                registry[stage]=dict(complete=True,seconds=time.perf_counter()-ts,completion=audit)
                save_json(run/'completion_registry.json',registry)
        from .report import generate,plots
        c,o=generate(ctx);generate(ctx,only_A=True);plots(ctx,c,o)
        integrity(ctx)
        remaining=[s for s in STAGES if not registry.get(s,{}).get('complete')]
        save_json(run/'status.json',dict(complete=args.mode!='full' or not remaining,requested_stages_complete=True,
            mode=args.mode,scientific_validation=args.mode=='full' and not remaining,remaining_stages=remaining,
            full_suite_started=args.mode=='full',scientific_stages_executed=args.mode=='full'))
        save_json(run/'execution.json',dict(this_invocation_seconds=time.perf_counter()-tick,
            process_cpu_seconds=time.process_time()-cpu_started,
            process_cpu_percent_one_core=100*(time.process_time()-cpu_started)/max(time.perf_counter()-tick,1e-9),
            stage_registry=registry,cache_hits=ctx.hits,new_jobs=ctx.misses,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0.,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if args.device=='cuda' else 0.))
        print(f'DONE {run}',flush=True)
    except BaseException as error:
        details=dict(type=type(error).__name__,errno=getattr(error,'errno',None),
            winerror=getattr(error,'winerror',None),filename=getattr(error,'filename',None),
            filename2=getattr(error,'filename2',None),traceback=traceback.format_exc())
        try:
            save_json(run/'status.json',dict(complete=False,mode=args.mode,error=repr(error),error_details=details,
                resumable=True,completed_stages=list(registry)))
        except OSError as status_error:
            print(f'Could not save failure status: {status_error!r}',flush=True)
        raise
    finally:
        if monitor is not None:
            monitor.__exit__(None,None,None)
            save_advisory_json(run/'gpu_telemetry.json',dict(samples=monitor.samples,availability=bool(monitor.samples)))
