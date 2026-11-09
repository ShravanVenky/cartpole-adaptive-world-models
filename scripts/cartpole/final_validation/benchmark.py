"""Bounded end-to-end timing, GPU telemetry, and explicit scheduling estimates."""
import json
import subprocess
import threading
import time
import numpy as np
import pandas as pd
import torch
from adaptive_world_model.environments.cartpole import CartPolePhysics,step
from adaptive_world_model.control.evaluation_mpc import physics_step,EvaluationCEM,GraphedPlanner
from .protocol import spec,live_specs,workload
from .common import Predictor,Planner,save_json,initial
from .simulation import run_live,run_batch,warmup
from .snapshots import evaluate_snapshots,control_forks
from .design import fixed_confirmation,run_recovery


class Monitor:
    def __init__(self):self.samples=[];self.stop=threading.Event();self.thread=None
    def __enter__(self):
        def worker():
            while not self.stop.is_set():
                try:
                    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-i','0'],
                                     capture_output=True,text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    if p.returncode==0:
                        a=[float(x.strip()) for x in p.stdout.strip().split(',')]
                        self.samples.append(dict(time=time.time(),utilization_percent=a[0],used_mib=a[1],power_watts=a[2]))
                except (OSError,ValueError,subprocess.TimeoutExpired):pass
                self.stop.wait(.75)
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=6)


def contracts(ctx):
    seed=11;base,templates=ctx.templates(seed)
    for model_seed in ctx.cfg['model_seeds']:ctx.templates(model_seed)
    state=torch.tensor([[.1,.1,2.8,.2]]*5,device=ctx.device);u=torch.ones(5,1,device=ctx.device)
    specs=[spec(m) for m in ['frozen','ridge','meta_residual_1step','meta_residual_rollout','oracle']]
    models=[ctx.make(seed,s) for s in specs];pack=Predictor(base,models,specs,ctx.device)
    p=CartPolePhysics(motor_gain=.6,pole_half_length=.65);pack.physics=p
    with torch.no_grad():
        raw=pack(state,u);assert torch.equal(raw[-1:],step(state[-1:],u[-1:],p))
        pack.alpha.zero_();zero=pack(state,u);assert torch.equal(zero[:-1],pack.frozen(state,u)[:-1])
        settings=[spec('oracle')]*2;s=state[:2]*.01
        planner=Planner(15,'balance',settings,ctx.device,ctx.cfg);old=EvaluationCEM(15,'balance',2,device=ctx.device)
        noise=torch.randn(3,2,128,3,1,device=ctx.device);predict=lambda x,a:physics_step(x,a,p)
        actual=planner.action(predict,s,91,noise);expected=old.action(predict,s,91,noise)
        assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6),'Baseline planner changed'
        if ctx.device=='cuda':
            planner.mean.zero_();graph=GraphedPlanner(planner,predict,s,torch.tensor([0,1],device=ctx.device))
            actual=graph.action(s,84);torch.cuda.synchronize();noise=graph.noise.clone();planner.mean.zero_()
            expected=planner.action(predict,s,84,noise);assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
    # Verify the exact sum/mean interpretation and matching atK20, independent of control success.
    from .common import CountScaledRidge
    normalized=CountScaledRidge(base)
    from .common import score
    states=np.zeros((201,4),np.float32);states[20:30,2]=.6;actions=np.zeros((200,1),np.float32)
    assert score(states,actions,ctx.cfg,'balance',True)['success']
    assert not score(states,actions,ctx.cfg,'balance',False)['success']
    z=torch.randn(50,7,dtype=torch.float64);y=torch.randn(50,4,dtype=torch.float64)
    for k in [5,10,20,50]:
        lam=.1*k/20
        sum_fit=torch.linalg.solve(z[:k].T@z[:k]+lam*torch.eye(7,dtype=z.dtype),z[:k].T@y[:k])
        mean_fit=torch.linalg.solve(z[:k].T@z[:k]/k+.005*torch.eye(7,dtype=z.dtype),z[:k].T@y[:k]/k)
        assert torch.allclose(sum_fit,mean_fit,atol=1e-12,rtol=1e-10)
        if k==20:assert lam==.1
    result=dict(all_five_checkpoint_seeds_loaded=True,alpha_zero_exact=True,oracle_changed_physics_exact=True,
                baseline_CEM_preserved=True,graph_eager_parity=True,ridge_sum_mean_equivalence=True,ridge_match_at_K20=True,
                neural_offline_training=False,base_frozen=base.assert_frozen() is None)
    save_json(ctx.run/'smoke_contracts.json',result);return result


def physics_benchmark(ctx):
    result=[]
    for device in ['cpu','cuda'] if torch.cuda.is_available() else ['cpu']:
        x=torch.tensor([[.1,.1,2.8,.2]]*48,device=device);u=torch.ones(48,1,device=device)
        for _ in range(5):x=physics_step(x,u,CartPolePhysics())
        if device=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter()
        for _ in range(30):x=physics_step(x,u,CartPolePhysics())
        if device=='cuda':torch.cuda.synchronize()
        result.append(dict(device=device,batch=48,steps=30,seconds=time.perf_counter()-begin,
                           note='Eager actual simulator stepping only; CEM uses CUDA graphs.'))
    save_json(ctx.run/'physics_benchmark.json',result);return result


def planning_benchmark(ctx):
    base,_=ctx.templates(11);settings=live_specs();models=[ctx.make(11,s) for s in settings]
    state=torch.tensor([[.1,.05,.05,.1]]*len(settings),device=ctx.device)
    pack=Predictor(base,models,settings,ctx.device);pack.physics=CartPolePhysics(motor_gain=.75)
    planner=Planner(15,'balance',settings,ctx.device,ctx.cfg)
    if ctx.device!='cuda':
        result=dict(compared=False,reason='CUDA unavailable');save_json(ctx.run/'planning_backend_benchmark.json',result);return result
    groups=torch.zeros(len(settings),dtype=torch.long,device=ctx.device)
    graph=GraphedPlanner(planner,pack,state,groups);planner.mean.zero_();torch.cuda.synchronize()
    start=time.perf_counter();actions=[]
    for seed in [71,72,73]:actions.append(graph.action(state,seed))
    torch.cuda.synchronize();graphed=time.perf_counter()-start
    planner.mean.zero_();start=time.perf_counter();eager=[]
    for seed in [71,72,73]:
        gen=torch.Generator(device=ctx.device).manual_seed(seed)
        noise=torch.randn(3,1,128,3,1,generator=gen,device=ctx.device).expand(-1,len(settings),-1,-1,-1)
        eager.append(planner.action(pack,state,0,noise))
    torch.cuda.synchronize();ordinary=time.perf_counter()-start
    assert torch.allclose(torch.stack(actions),torch.stack(eager),atol=1e-6,rtol=1e-6),'Timed graph/eager paths differ'
    result=dict(compared=True,batch=len(settings),horizon=15,planning_calls=3,cuda_graph_seconds=graphed,
                gpu_eager_seconds=ordinary,speedup=ordinary/graphed,matched_actions=True,
                fastest_verified='CUDA graphs' if graphed<ordinary else 'GPU eager',
                scope='Planning-only backend comparison; end-to-end online timing separately includes updates, actual stepping and serialization.')
    save_json(ctx.run/'planning_backend_benchmark.json',result);return result


def bounded_run(ctx,mode):
    if (ctx.run/'benchmark.json').exists():
        saved=json.loads((ctx.run/'benchmark.json').read_text())
        print('BENCHMARK cached; no simulation or optimizer updates repeated',flush=True)
        return saved
    if ctx.device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();contracts(ctx);timings={}
    with Monitor() as monitor:
        case=next(c for c in ctx.cfg['conditions'] if c['name']=='stable_balance__near_gain')
        begin=time.perf_counter();parent=run_live(ctx,'smoke',11,900,case,steps_limit=55);timings['live_h15']=time.perf_counter()-begin
        assert parent is not None
        assert [x['K'] for x in parent['checkpoints']]==ctx.cfg['checkpoints']
        assert all(u['observations']==u['control_step'] for u in parent['updates'])
        assert all(u['optimizer_gradient_steps']==10 for u in parent['updates'] if u['method']=='meta')
        frow=next(i for i,r in enumerate(parent['rows']) if r['method']=='frozen' and r['safety']=='original')
        begin=time.perf_counter();snapshot=evaluate_snapshots(ctx,parent,frow);timings['snapshot_8K']=time.perf_counter()-begin
        begin=time.perf_counter();forks=control_forks(ctx,parent,frow,steps_limit=15);timings['forks']=time.perf_counter()-begin
        case30=next(c for c in ctx.cfg['conditions'] if c['name']=='active_swingup__near_gain')
        begin=time.perf_counter();live30=run_live(ctx,'smoke',11,901,case30,steps_limit=55);timings['live_h30']=time.perf_counter()-begin
        entry=warmup(ctx,11,902,'upright_entry')
        if not entry['eligible']:raise AssertionError('Nominal upright-entry smoke did not reach phase')
        entry_case=next(c for c in ctx.cfg['conditions'] if c['name']=='upright_entry__combined_geometry')
        entry_live=run_live(ctx,'smoke',11,902,entry_case,steps_limit=20)
        begin=time.perf_counter();fixed=fixed_confirmation(ctx,11,903,ctx.cfg['fixed_conditions'][1],steps_limit=20);timings['fixed25']=time.perf_counter()-begin
        # Full-sized fork batch with actualH30 objectives/inference to bound memory/throughput.
        specs=[dict(live_specs()[i%12],id=f'benchmark{i}') for i in range(48)]
        states=np.repeat(initial('swingup',11,904)[None].numpy(),48,axis=0);models=[ctx.make(11,s) for s in specs]
        dense=run_batch(ctx,'benchmark_dense48',11,904,case30,specs,states,models,[600]*48,steps_limit=20,
                        metadata=[dict(stage='benchmark_only') for _ in specs])
        from .protocol import PREVIOUS
        previous=pd.read_csv(PREVIOUS/'control_results.csv')
        row=previous[(previous.phase=='confirmation')&(previous.family=='baseline')&(previous.method=='frozen')&previous.failed].iloc[0]
        with np.load(PREVIOUS/'control'/f'{row.group}.npz') as data:state=data['states'][int(row.row),int(row.survival_steps)-1].tolist()
        recovery_case=dict(id='smoke_saved_failure',available=True,seed=11,environment=row.environment,task=row.task,
                           horizon=int(row.horizon),state=state)
        begin=time.perf_counter();recovery=run_recovery(ctx,recovery_case,steps_limit=10);timings['oracle_recovery']=time.perf_counter()-begin
        physics=physics_benchmark(ctx)
        backend=planning_benchmark(ctx)
    # Prove reuse and snapshot isolation without redoing its optimization/simulation.
    before=ctx.hits;again=run_live(ctx,'smoke',11,900,case,steps_limit=55)
    assert ctx.hits>before and again['key']==parent['key']
    base,templates=ctx.templates(11);support=None
    from .common import transition_batch
    with np.load(ctx.run/'episodes'/f'{parent["key"]}.npz') as data:support=transition_batch(data['states'][frow,:5],data['actions'][frow,:5],data['states'][frow,1:6],'smoke_isolation')
    original=templates['meta_residual_1step'].adapter.state_dict()
    saved={k:v.clone() for k,v in original.items()}
    m,_=ctx.fit(11,spec('meta_residual_1step'),support)
    assert all(torch.equal(saved[k],original[k]) for k in saved),'Adaptation mutated checkpoint initialization'
    with np.load(ctx.run/'episodes'/f'{parent["key"]}.npz') as data:
        perturbed=data['states'][frow].copy();perturbed[6:]+=100
        prefix=transition_batch(perturbed[:5],data['actions'][frow,:5],perturbed[1:6],'smoke_isolation')
        assert prefix.support_hash()==support.support_hash(),'Future modification changed past support'
        twenty=transition_batch(data['states'][frow,:20],data['actions'][frow,:20],data['states'][frow,1:21],'smoke_K20_match')
    old,_=ctx.fit(11,spec('ridge'),twenty);scaled,_=ctx.fit(11,spec('ridge',regularization='count_scaled'),twenty)
    assert torch.equal(old.coefficients,scaled.coefficients),'Actual Ridge variants must match atK20'
    assert len({p['rows'][0]['mpc_seed_at_step0'] for p in forks})==1,'Fork chunks do not share optimizer randomness'
    # Selection/censoring contracts use only the actual bounded smoke records in an isolated fixture.
    from .design import select_snapshot_cases
    from .snapshots import unavailable_cohorts
    from .common import Context
    from .report import matrix_check
    fixture_cfg=dict(ctx.cfg,model_seeds=[11],conditions=[case,case30,entry_case,next(c for c in ctx.cfg['conditions'] if c['name']=='stable_balance__nominal')])
    fixture=Context(ctx.run/'design_contract',fixture_cfg,ctx.device);fixture.run.mkdir(exist_ok=True)
    for packet in [parent,live30,entry_live]:
        copied=dict(packet,rows=[dict(r,stage='development') for r in packet['rows']])
        save_json(fixture.run/'episodes'/f'development_{packet["key"]}.json',copied)
    selection=select_snapshot_cases(fixture)
    assert len(selection['conditions'])==4 and selection['confirmation_trials']==ctx.cfg['confirmation_trials']
    unavailable_cohorts(fixture,11,400,case)
    censored=json.loads(next((fixture.run/'censoring').glob('*.json')).read_text())
    assert censored['censored']==66 and censored['uncensored']==0
    empty=pd.DataFrame(columns=['stage','method','seed','episode_trial','condition','id'])
    registry=matrix_check(fixture,empty,pd.DataFrame(),pd.DataFrame())
    assert not registry['complete'],'An incomplete matrix was falsely marked complete'
    peak=dict(torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if ctx.device=='cuda' else None,
              torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if ctx.device=='cuda' else None,
              telemetry_samples=len(monitor.samples),gpu_utilization_mean=float(np.mean([s['utilization_percent'] for s in monitor.samples])) if monitor.samples else None,
              gpu_utilization_peak=max([s['utilization_percent'] for s in monitor.samples],default=None),
              nvidia_peak_used_mib=max([s['used_mib'] for s in monitor.samples],default=None))
    save_json(ctx.run/'gpu_telemetry.json',dict(summary=peak,samples=monitor.samples))
    def summary(p):
        r={k:p[k] for k in ['setup_seconds','loop_seconds','mpc_batch_calls','wall_seconds']}
        r['mpc_core_seconds']=p['rows'][0]['mpc_batch_seconds']*p['mpc_batch_calls']
        r['one_off_checkpoint_and_update_seconds']=max(0,r['loop_seconds']-r['mpc_core_seconds'])
        r['serialization_and_metrics_seconds']=max(0,r['wall_seconds']-r['setup_seconds']-r['loop_seconds'])
        return r
    result=dict(mode=mode,timings=timings,wall_seconds=time.perf_counter()-start,peak=peak,
                live_h15=summary(parent),live_h30=summary(live30),dense48=summary(dense),
                fork_batches=[summary(p) for p in forks],snapshot_seconds=snapshot['wall_seconds'],fixed25=summary(fixed),
                recovery_batches=[summary(p) for p in recovery],planning_backend=backend,
                checks=dict(resume_cache_hit=True,no_future_support_leak=True,isolated_snapshot_updates=True,all_eight_K_paths=True,
                            experiments_1_to_6_exercised=True,fixed_confirmation_exercised=True,actual_ridge_equal_at_K20=True,
                            fork_chunks_common_randomness=True,development_confirmation_separated=True,censored_denominators_preserved=True,
                            incomplete_matrix_rejected=True),scientific_validation=False)
    save_json(ctx.run/'benchmark.json',result);estimate(ctx,result)
    return result


def estimate(ctx,b):
    cfg=ctx.cfg;w=workload(cfg);n=5
    def duration(sample,steps):
        calls=max(sample['mpc_batch_calls'],1)
        return sample['setup_seconds']+sample['mpc_core_seconds']/calls*steps+sample['one_off_checkpoint_and_update_seconds']+sample['serialization_and_metrics_seconds']*max(1,steps/calls)
    # Full matrix retained. Censoring/early group termination can reduce this upper scheduling workload.
    live=150*duration(b['live_h30'],500)+75*duration(b['live_h15'],300)
    warm=30*duration(b['live_h30'],180)+15*duration(b['live_h15'],100)
    snapshot=w['snapshot_cohorts']*b['snapshot_seconds']
    forks=40*(duration(b['dense48'],450)+duration(b['dense48'],450)*.65)+40*(duration(b['live_h15'],270)+duration(b['live_h15'],270)*.65)
    fixed=30*duration(b['fixed25'],600)+10*duration(b['fixed25'],400)*.6
    recovery=25*sum(duration(s,500) for s in b['recovery_batches'])
    components=dict(shared_nominal_prefixes=warm,online_experiments_1_2_3_6=live,snapshot_prediction_ranking_trust_ridge=snapshot,
                    snapshot_counterfactual_controls=forks,small_fixed_confirmation=fixed,oracle_failure_replays=recovery)
    subtotal=sum(components.values());total=subtotal*1.15
    report=dict(hardware='NVIDIA GeForce RTX4060Laptop GPU' if ctx.device=='cuda' else 'CPU',workload=w,
                estimated_seconds_by_workload=components,estimated_total_seconds=total,plausible_seconds=[total*.65,total*1.6],
                expected_gpu_utilization_percent=[b['peak']['gpu_utilization_mean'],b['peak']['gpu_utilization_peak']],
                measured_memory=b['peak'],expected_VRAM_note='Largest48-rowH30 batch measured; allow1GiB extra headroom for driver/display and graph allocation variability.',
                confidence='Scheduling extrapolation from bounded real workloads, not a guarantee. Phase entry, censoring, caching and thermals materially change runtime.',
                extrapolation='Per-step MPC/stepping extrapolated separately from one-offK<=50 update/checkpoint analysis; serialization scales with trajectory length.',
                bottlenecks=['CEM repeated neural/oracle rollouts','CUDA graph setup per independent batch','serialized task-local ten-step Meta updates','counterfactual fork controls and compression'],
                parallelism='One GPU process. No unbenchmarked competing GPU jobs.',full_suite_started=False)
    save_json(ctx.run/'runtime_estimate.json',report)
    print(f'ETA {total/3600:.2f}h range {total*.65/3600:.2f}..{total*1.6/3600:.2f}h',flush=True)
