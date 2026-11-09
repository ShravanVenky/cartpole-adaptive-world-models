"""One explicit-mode runner: bounded smoke/benchmark now; full execution only when requested."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import time
import torch
from .protocol import ROOT,PREVIOUS,configuration,workload
from .common import Context,save_json,file_hash,source_hashes
from .simulation import run_live
from .snapshots import evaluate_snapshots,control_forks,unavailable_cohorts
from .design import select_snapshot_cases,fixed_confirmation,recovery_manifest,run_recovery


def prepare(run,cfg,full):
    run.mkdir(parents=True,exist_ok=True)
    config=run/'config_locked.json'
    if config.exists():
        if json.loads(config.read_text())!=cfg:raise RuntimeError('Incompatible configuration; start a new run')
    else:save_json(config,cfg)
    current=source_hashes();path=run/'source_hashes.json'
    if path.exists():
        if json.loads(path.read_text())!=current:raise RuntimeError('Source differs from the saved run. Restore its snapshot or use a new directory.')
    else:
        save_json(path,current)
        for name in current:
            target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    preservation=run/'preserved_artifacts.json'
    if not preservation.exists():
        if full:
            folders=[ROOT/'results/cartpole'/n for n in ['20261005_232356_821640_cartpole_v1','20261006_140915_614361_cartpole_v2_factorial','20261023_v2_control','20261026_000707_v2_control_diagnostics']]
            files=[p for folder in folders for p in folder.rglob('*') if p.is_file()]
        else:
            files=[p for folder in [ROOT/'results/cartpole/20261005_232356_821640_cartpole_v1/checkpoints',ROOT/'results/cartpole/20261006_140915_614361_cartpole_v2_factorial/checkpoints'] for p in folder.glob('*.pt')]
            files += [PREVIOUS/'report.md',PREVIOUS/'config_locked.json',ROOT/'scripts/cartpole/diagnostics/engine.py',ROOT/'src/adaptive_world_model/adaptation/ridge.py']
        save_json(preservation,{str(p.relative_to(ROOT)):file_hash(p) for p in files})
    save_json(run/'planned_workload.json',workload(cfg))
    save_json(run/'seed_manifest.json',dict(model_seeds=cfg['model_seeds'],development_episode_trials=cfg['development_trials'],
        confirmation_episode_trials=cfg['confirmation_trials'],fixed_confirmation_episode_trials=cfg['fixed_confirmation_trials'],
        nominal_prefix_namespace='final_v2_nominal_prefix',mpc_namespace='final_v2_mpc',snapshot_candidate_namespace='final_v2_rank',
        support_pool_namespace='diagnostic_support with new trial500/501',known_shift_time_is_privileged_protocol=True))


def verify(run):
    files=json.loads((run/'preserved_artifacts.json').read_text())
    changed=[n for n,h in files.items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=h]
    if changed:raise RuntimeError(f'Preserved inputs changed: {changed[:3]}')
    if json.loads((run/'source_hashes.json').read_text())!=source_hashes():raise RuntimeError('Evaluation sources changed during run')
    save_json(run/'integrity_verified.json',dict(files=len(files),all_unchanged=True))


def progress(ctx,stage,**extra):
    save_json(ctx.run/'status.json',dict(complete=False,stage=stage,pid=os.getpid(),**extra))
    print(f'STAGE {stage} {extra}',flush=True)


def full(ctx):
    # Benchmark first using this same run and workload, tagged as non-scientific smoke records.
    if not (ctx.run/'benchmark.json').exists():
        progress(ctx,'bounded_benchmark')
        from .benchmark import bounded_run
        bench=Context(ctx.run/'benchmark',ctx.cfg,ctx.device);bench.run.mkdir(exist_ok=True)
        bounded_run(bench,'benchmark')
        for name in ['benchmark.json','runtime_estimate.json','gpu_telemetry.json','physics_benchmark.json','planning_backend_benchmark.json']:
            shutil.copy2(bench.run/name,ctx.run/name)
    for phase,trials in [('development',ctx.cfg['development_trials']),('confirmation',ctx.cfg['confirmation_trials'])]:
        if phase=='confirmation':selected=select_snapshot_cases(ctx)
        total=len(ctx.cfg['model_seeds'])*len(trials)*len(ctx.cfg['conditions']);done=0
        for seed in ctx.cfg['model_seeds']:
            for trial in trials:
                for case in ctx.cfg['conditions']:
                    progress(ctx,phase,groups_done=done,groups_total=total)
                    parent=run_live(ctx,phase,seed,trial,case);done+=1
                    if phase=='confirmation' and case['name'] in selected['conditions'] and parent is not None:
                        for safety in ['original','margin']:
                            collector=next(i for i,r in enumerate(parent['rows']) if r['method']=='frozen' and r['safety']==safety)
                            evaluate_snapshots(ctx,parent,collector);control_forks(ctx,parent,collector)
                    elif phase=='confirmation' and case['name'] in selected['conditions']:
                        unavailable_cohorts(ctx,seed,trial,case)
        if phase=='development':select_snapshot_cases(ctx)
    progress(ctx,'independent_fixed_confirmation')
    for seed in ctx.cfg['model_seeds']:
        for trial in ctx.cfg['fixed_confirmation_trials']:
            for case in ctx.cfg['fixed_conditions']:fixed_confirmation(ctx,seed,trial,case)
    progress(ctx,'oracle_failure_recoverability')
    for case in recovery_manifest(ctx):run_recovery(ctx,case)
    progress(ctx,'statistical_reporting')
    from .report import generate
    generate(ctx,smoke=False);verify(ctx.run)
    save_json(ctx.run/'status.json',dict(complete=True,stage='complete',mode='full'))


def main():
    p=argparse.ArgumentParser(description='Final V2 validation. No offline neural training. Full suite requires explicit --mode full.')
    p.add_argument('--mode',required=True,choices=['smoke','benchmark','full','report'])
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args();torch.set_num_threads(1)
    if args.mode=='report' and args.run_dir is None:p.error('--mode report requires --run-dir')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    cfg=configuration();stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=(args.run_dir or ROOT/('results/cartpole' if args.mode=='full' else 'smoke tests')/f'{stamp}_v2_final_validation_{args.mode}').resolve()
    archived=[ROOT/'results/cartpole'/name for name in ['20261005_232356_821640_cartpole_v1','20261006_140915_614361_cartpole_v2_factorial','20261023_v2_control','20261026_000707_v2_control_diagnostics']]
    if any(run==a or a in run.parents for a in archived):raise RuntimeError('Refusing to write into an archived run')
    if args.mode=='full' and (ROOT/'smoke tests') in run.parents:raise RuntimeError('Full results must use a new scientific directory, not smoke artifacts')
    if run.exists() and any(run.iterdir()) and not (run/'config_locked.json').exists():raise RuntimeError('Refusing to reuse a nonempty unrelated directory')
    kind=run/'run_kind.json'
    if kind.exists():
        previous_kind=json.loads(kind.read_text())['kind']
        if args.mode=='full' and previous_kind!='scientific':raise RuntimeError('Cannot promote smoke data into a scientific run')
        if args.mode in ['smoke','benchmark'] and previous_kind=='scientific':raise RuntimeError('Benchmark into a new smoke directory, not an existing scientific run')
    prepare(run,cfg,args.mode=='full')
    if not kind.exists():save_json(kind,dict(kind='scientific' if args.mode=='full' else 'bounded'))
    lock=run/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run lock already exists: {lock}; inspect its PID before removing a stale lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.perf_counter();ctx=Context(run,cfg,args.device)
    save_json(run/'hardware.json',dict(device=args.device,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None))
    print(f'RUN_DIRECTORY {run}',flush=True);progress(ctx,args.mode)
    try:
        if args.mode=='full':full(ctx)
        elif args.mode=='report':
            from .report import generate
            bounded=json.loads(kind.read_text())['kind']=='bounded'
            generate(ctx,smoke=bounded);verify(run)
            save_json(run/'status.json',dict(complete=True,mode='report',scientific_validation=not bounded))
        else:
            from .benchmark import bounded_run
            from .report import generate
            bounded_run(ctx,args.mode);generate(ctx,smoke=True);verify(run)
            save_json(run/'status.json',dict(complete=True,mode=args.mode,full_suite_started=False,scientific_validation=False))
    except Exception as exc:
        save_json(run/'status.json',dict(complete=False,mode=args.mode,error=repr(exc),resumable=True));raise
    finally:
        save_json(run/f'execution_{args.mode}.json',dict(wall_seconds=time.perf_counter()-start,cache_hits=ctx.hits,cache_misses=ctx.misses))
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
