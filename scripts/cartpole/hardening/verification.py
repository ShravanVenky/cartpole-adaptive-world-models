"""Bounded contracts and measured workloads; never invoke scientific stage functions."""
import copy
import json
import subprocess
import time
from dataclasses import replace
import numpy as np
import torch
from adaptive_world_model.environments.cartpole import CartPolePhysics,step
from adaptive_world_model.control.evaluation_mpc import physics_step,GraphedPlanner
from final_validation.benchmark import Monitor
from .protocol import METAS,optimizer,screen,workload
from .common import save_json, support_from_episode, query_record, Planner, setting, digest, Predictor
from .optimization import optimize,fit,audit
from .offline import evaluate
from .simulation import control,policies,diagnostic_policies,impulse,external_step,observe,recovery_metrics
from .stages import collection,diverse


def contracts(ctx):
    checks=[];seed=11;support=support_from_episode(seed,'combined_geometry');base,_=ctx.templates(seed)
    base_before={k:v.clone() for k,v in base.state_dict().items()}
    for method in METAS:
        original=ctx.make(seed,setting(method));original.adapt(support)
        candidate=ctx.make(seed,setting(method));candidate,log=optimize(candidate,support,optimizer())
        error=max(float((original._params()[n]-candidate._params()[n]).abs().max()) for n in candidate._params())
        assert error<2e-7,(method,error)
        assert log['accepted'] and log['gradient_steps']==10 and log['parameter_update_norm']<=.100001
        checks.append(dict(name='baseline_update_parity_'+method,max_abs_parameter_error=error,passed=True))
        again=ctx.make(seed,setting(method));again,second=optimize(again,support,optimizer())
        assert all(torch.equal(candidate._params()[n],again._params()[n]) for n in candidate._params())
        reject=ctx.make(seed,setting(method));reject,denied=optimize(reject,support,optimizer(delta=1e-10))
        assert not denied['accepted'] and reject._adapted_params is None
        checks.append(dict(name='reset_and_guard_rollback_'+method,passed=True))
    assert all(torch.equal(v,base_before[k]) for k,v in base.state_dict().items())
    assert not any(p.requires_grad for p in base.parameters())
    assert len(support)==50 and len(torch.unique(support.transition_id))==50
    checks.append(dict(name='frozen_buffers_weights_and_50_unique_transitions',passed=True))
    p=CartPolePhysics(motor_gain=.6);s=torch.tensor([[.1,.2,.3,.4],[-.2,-.1,-.4,-.5]],device=ctx.device);u=torch.tensor([[3.],[-4.]],device=ctx.device)
    assert torch.allclose(physics_step(s,u,p),step(s,u,p),atol=1e-7,rtol=1e-6)
    assert torch.allclose(external_step(s,u,p,0.),physics_step(s,u,p),atol=1e-7,rtol=1e-6)
    after=impulse(s,p,.6);dv=after[:,1]-s[:,1];dw=after[:,3]-s[:,3]
    horizontal=(p.cart_mass+p.pole_mass)*dv+p.pole_mass*p.pole_half_length*s[:,2].cos()*dw
    hinge=p.pole_mass*p.pole_half_length*s[:,2].cos()*dv+(4/3)*p.pole_mass*p.pole_half_length**2*dw
    assert torch.allclose(horizontal,torch.full_like(horizontal,.6),atol=1e-6) and hinge.abs().max()<1e-7
    checks.append(dict(name='RK4_and_generalized_impulse_momentum',passed=True))
    assert torch.equal(observe(s,[0,0,0,0],14),s)
    assert not torch.equal(observe(s,[.1,.1,.1,.1],14),s)
    assert torch.equal(observe(s,[.1,.1,.1,.1],14)-s,observe(s,[.1,.1,.1,.1],14)-s)
    checks.append(dict(name='matched_sensor_noise_and_separate_truth',passed=True))
    stable=np.zeros((130,4));unstable=stable.copy();unstable[20:,3]=2.
    assert recovery_metrics(stable,np.zeros((129,1)),10,ctx.cfg)['recovery_success']
    assert not recovery_metrics(unstable,np.zeros((129,1)),10,ctx.cfg)['recovery_success']
    checks.append(dict(name='100_step_angle_and_speed_recovery',passed=True))
    # Native CEM graph is tested against the exact same eager candidate noise.
    models=[base,ctx.make(seed,setting(METAS[0])),base]
    plist=[dict(method='frozen'),dict(method=METAS[0]),dict(method='oracle')]
    pack,settings=ctx.predictor(seed,plist,models,p)
    from mismatch.common import Predictor as OriginalPredictor
    original_pack=OriginalPredictor(base,models,settings,ctx.device);original_pack.physics=p
    assert all(torch.equal(pack.weights[n],original_pack.weights[n]) for n in pack.weights)
    assert torch.equal(pack.coef,original_pack.coef)
    assert torch.equal(pack(s[:1].repeat(3,1),u[:1].repeat(3,1)),original_pack(s[:1].repeat(3,1),u[:1].repeat(3,1)))
    ridge=ctx.make(seed,setting('ridge'));ridge.adapt(support)
    test_models=[*models,ridge];test_settings=[*settings,setting('ridge')]
    bulk=Predictor(base,test_models,test_settings,ctx.device);old=OriginalPredictor(base,test_models,test_settings,ctx.device)
    assert torch.equal(bulk.coef,old.coef) and all(torch.equal(bulk.weights[n],old.weights[n]) for n in bulk.weights)
    checks.append(dict(name='bulk_parameter_refresh_matches_existing_predictor',passed=True))
    planner=Planner(15,'balance',settings,ctx.device,ctx.cfg)
    state=s[:1].repeat(3,1);generator=torch.Generator(device=ctx.device).manual_seed(18)
    noise=torch.randn(planner.iterations,1,planner.n,3,1,generator=generator,device=ctx.device).expand(-1,3,-1,-1,-1)
    expected=planner.action(pack,state,0,noise).clone();planner.mean.zero_()
    if ctx.device=='cuda':
        graph=GraphedPlanner(planner,pack,state,torch.zeros(3,dtype=torch.long,device=ctx.device));planner.mean.zero_()
        got=graph.action(state,18)
        assert torch.allclose(got,expected,atol=2e-5,rtol=2e-5)
    checks.append(dict(name='existing_CEM_graph_eager_parity',passed=True))
    from .stages import selected
    try:selected(ctx)
    except RuntimeError:pass
    else:raise AssertionError('A scientific gate should not exist in bounded mode')
    checks.append(dict(name='scientific_A_gate_not_bypassed',passed=True))
    import pandas as pd
    from .stages import choose_control_setting
    from .protocol import control_jobs
    expected=workload(ctx.cfg)
    for stage in ['A','B','C','D']:
        assert sum(j['slots'] for j in control_jobs(ctx.cfg) if j['stage']==stage)==expected[stage]['control_slots_upper' if stage=='A' else 'control_slots']
    method=METAS[0];candidate=optimizer('unit_candidate',steps=100)
    fixture_rows=[dict(split='development',policy=p,environment=e,success=v,cost=c) for p,e,v,c in [
        (method,'nominal',True,10.),(method,'combined_geometry',False,100.),
        (method+'__unit_candidate','nominal',True,10.),(method+'__unit_candidate','combined_geometry',True,20.)]]
    chosen,_=choose_control_setting(pd.DataFrame(fixture_rows),method,[candidate]);assert chosen['name']=='unit_candidate'
    fixture_rows[2]['success']=False
    chosen,_=choose_control_setting(pd.DataFrame(fixture_rows),method,[candidate]);assert chosen['name']=='baseline'
    for r in fixture_rows:r['split']='confirmation'
    try:choose_control_setting(pd.DataFrame(fixture_rows),method,[candidate])
    except RuntimeError:pass
    else:raise AssertionError('Confirmation data admitted to development selection')
    checks.append(dict(name='exact_matrix_and_development_selection_firewall',passed=True))
    fixture=ctx.run/'contract_fixtures'/'cache.json'
    ctx.save_job(fixture,dict(test='integrity'),values=np.zeros(3))
    assert ctx.cached_job(fixture)['test']=='integrity'
    fixture.with_suffix('.npz').write_bytes(b'deliberate contract corruption')
    try:ctx.cached_job(fixture)
    except RuntimeError:pass
    else:raise AssertionError('Corrupted cache accepted')
    ctx.save_job(fixture,dict(test='integrity'),values=np.zeros(3))
    checks.append(dict(name='atomic_job_resume_hash_rejects_corruption',passed=True))
    save_json(ctx.run/'contracts.json',dict(passed=True,checks=checks));return checks


def bounded(ctx,mode):
    checks=contracts(ctx);save_json(ctx.run/'optimizer_audit.json',audit(ctx))
    seed=11;support=support_from_episode(seed,'combined_geometry');record=query_record(seed,'combined_geometry')
    # Exercise every optimizer branch with two updates, never the complete A matrix.
    variants=[(m,dict(c,name='smoke_'+c['name'],steps=2)) for m in METAS for c in screen()]
    variants += [(m,optimizer('smoke_adam',steps=2,algorithm='adam')) for m in METAS]
    variants += [(m,optimizer('smoke_relaxed',steps=2,clip=5.,step_bound=.5,delta=10.)) for m in METAS]
    offline=evaluate(ctx,record,support,variants,'smoke_A')
    fits=[]
    for method in METAS:
        for count in ([10,100,500] if mode=='benchmark' else [10]):
            m=ctx.make(seed,setting(method));_,log=optimize(m,support,optimizer(f'bench_{count}',steps=count))
            fits.append(dict(method=method,updates=count,device='cpu',seconds=log['adaptation_seconds'],accepted=log['accepted']))
        if ctx.device=='cuda' and mode=='benchmark':
            m=ctx.make(seed,setting(method)).to('cuda');_,log=optimize(m,support,optimizer('bench_cuda_100',steps=100))
            fits.append(dict(method=method,updates=100,device='cuda',seconds=log['adaptation_seconds'],accepted=log['accepted']))
    tiny={m:optimizer('bounded_2_steps',steps=2) for m in METAS}
    if mode=='benchmark':
        from .timing import deployment_profile
        deployment_profile(ctx,tiny,bounded=True)
    small_shortlist={m:[tiny[m],optimizer('bounded_lr_02',lr=.02,steps=2)] for m in METAS}
    extras={m:[optimizer('adam_matched_updates',steps=2,algorithm='adam'),optimizer('relaxed_guards',steps=2,clip=5.,step_bound=.5,delta=10.)] for m in METAS}
    plist=policies(shortlist=small_shortlist)+diagnostic_policies(extras)
    swing=dict(name='bounded_swing',environment='combined_geometry',phase='active_swingup',task='swingup',split='smoke',steps=55 if mode=='benchmark' else 12)
    start=dict(eligible=True,state=[.1,0.,3.1,0.],steps=10,wall_seconds=0.)
    controls=[]
    with Monitor() as monitor:
        controls.append(control(ctx,'smoke_A_snapshot',seed,1900,swing,plist,fixed_support=support,online=False,start=start))
        controls.append(control(ctx,'smoke_A_online',seed,1900,swing,plist,online=True,start=start))
        # B paths each collect only the minimum 50 observations. Privileged pool is read-only.
        for strategy in ['ordinary','excitation']:
            s,info=collection(ctx,seed,21900,'combined_geometry',strategy)
            assert s is not None and len(s)==50
            checks.append(dict(name='B_50_observation_'+strategy,passed=True))
        ds,di=diverse(ctx,seed,'combined_geometry');assert len(ds)==50 and di['privileged']
        evaluate(ctx,dict(record,id='bounded_B_query'),ds,[(m,tiny[m]) for m in METAS],'smoke_B_diverse')
        controls.append(control(ctx,'smoke_B_fixed',seed,1900,dict(swing,name='bounded_B_fixed'),policies(selected=tiny),fixed_support=ds,online=False,start=start))
        checks.append(dict(name='B_diverse_no_outcome_selection',passed=True))
        bstart=dict(eligible=True,state=[0.,0.,.02,0.],steps=100,wall_seconds=0.)
        for kind in ['cart_impulse','angular_velocity','force_pulse','repeated_cart','persistent','combined']:
            case=dict(name='bounded_'+kind,environment='combined_geometry' if kind in ['persistent','combined'] else 'nominal',
                phase='stable_balance',task='balance',split='smoke',steps=35 if mode=='benchmark' else 12,
                kind=kind,severity=1,sign=1)
            cpolicies=policies(selected=tiny,disabled=True)
            if kind in ['persistent','combined']:
                cpolicies += [dict(label='delayed_'+m,method=m,optimizer=tiny[m],online=True,delay_steps=2) for m in METAS]
            controls.append(control(ctx,'smoke_C_'+kind,seed,1901,case,cpolicies,start=bstart))
        dcase=dict(name='bounded_unseen_noise_sequence',environment='extra_combined',phase='active_swingup',task='swingup',
                   split='smoke',steps=35 if mode=='benchmark' else 12,noise='strong',sequence=True)
        controls.append(control(ctx,'smoke_D_noise_sequence',seed,1902,dcase,policies(selected=tiny,dense=False),start=start))
    for packet in controls:
        assert packet['rows'] and not any(r['numerical_failure'] for r in packet['rows'])
        for log in packet['updates']:
            assert log['support_count']==log['K'] and log['step']==log['segment']+log['K']
            assert log['K']<=50 and log['scheduled_application_step']>=log['step']
        if packet['case'].get('sequence'):assert sum(e['step']>0 and e['kind']=='persistent_change' for e in packet['events'])==2
        with np.load(ctx.run/'controls'/f'{packet["key"]}.npz') as a:
            assert np.allclose(a['states'][:,0],a['states'][0,0])
            assert np.allclose(a['predictions'],a['frozen_predictions']+a['corrections'],atol=2e-6,equal_nan=True)
            if packet['case'].get('noise')=='strong':assert not np.array_equal(a['observations'],a['states'])
        checks.append(dict(name='causal_updates_'+packet['key'],passed=True))
    # Exercise full-report numerical branches in memory, without producing synthetic scientific files.
    import pandas as pd
    from types import SimpleNamespace
    from .report import findings,paired
    samples=[]
    for stage,packet in zip(['A','B','C','D'],[controls[1],controls[2],controls[3],controls[-1]]):
        for r in packet['rows']:samples.append(dict(r,stage=stage,split='confirmation'))
    frame=pd.DataFrame(samples)
    texts,criteria=findings(SimpleNamespace(mode='full',cfg=ctx.cfg,run=ctx.run),frame,pd.DataFrame([dict(r,tag='A12') for r in offline['rows']]),dict(selected=tiny))
    assert len(texts)>3 and isinstance(criteria,pd.DataFrame)
    paired(frame)
    checks.append(dict(name='report_numeric_branches_in_memory_only',passed=True))
    # Flow-only regression: a failed ordinary support collector cannot block other B strategies.
    from unittest.mock import patch
    from . import stages
    from .common import Context,save_npz
    fixture_run=ctx.run/'contract_fixtures'/'B_missing_ordinary'
    fixture_cfg=dict(ctx.cfg,model_seeds=[11],b_environments=['combined_geometry'],dev_trials=[1100],confirmation_trials=[])
    fixture_ctx=Context(fixture_run,fixture_cfg,ctx.device,'smoke');fixture_ctx.models=ctx.models
    stem='A_dev_11_1100_combined_geometry_'
    save_json(fixture_run/'controls'/f'{stem}snapshot.json',dict(rows=[],fixture_only=True))
    sample=next(p for p in controls if p['key']=='smoke_B_fixed')
    save_json(fixture_run/'controls'/f'{stem}online.json',dict(rows=sample['rows'],fixture_only=True))
    with np.load(ctx.run/'controls'/'smoke_B_fixed.npz') as a:
        save_npz(fixture_run/'controls'/f'{stem}online.npz',states=a['states'].copy())
    calls=[]
    def collector(*args):
        return (None,dict(reason='unit collector failed',available=False)) if args[4]=='ordinary' else (support,dict(available=True))
    with patch.object(stages,'selected',return_value=tiny),patch.object(stages,'collection',side_effect=collector),\
         patch.object(stages,'diverse',return_value=(support,dict(available=True,privileged=True))),\
         patch.object(stages,'control',return_value=sample),patch.object(stages,'evaluate',side_effect=lambda *args:calls.append(args[1])):
        stages.B(fixture_ctx)
    assert len(calls)==2 and all(r['query_source'].endswith('online.json') for r in calls)
    checks.append(dict(name='B_other_strategies_survive_ordinary_support_failure_flow_fixture',passed=True))
    save_json(ctx.run/'contracts.json',dict(passed=True,checks=checks))
    result=dict(fits=fits,offline_pool_seconds=offline['wall_seconds'],
                controls=[{k:p[k] for k in ['key','steps','wall_seconds','setup_seconds','planning_seconds','inference_seconds','fit_seconds','graph_setup_seconds','initial_graph_setup_seconds','control_loop_seconds']} for p in controls],
                telemetry=monitor.samples,scientific_stages_executed=False)
    save_json(ctx.run/'benchmark_measured.json',result)
    return result


def estimate(ctx,measurements):
    w=workload(ctx.cfg);cpu=[r for r in measurements['fits'] if r['device']=='cpu']
    per_update=float(np.median([r['seconds']/r['updates'] for r in cpu]))
    controls=measurements['controls']
    swing=[p for p in controls if p['key'].startswith('smoke_A')]
    balance=[p for p in controls if p['key'].startswith('smoke_C')]
    def rate(records):return np.median([(p['control_loop_seconds']-p['fit_seconds']-max(0,p['graph_setup_seconds']-p['initial_graph_setup_seconds']))/p['steps'] for p in records])
    h30=float(rate(swing));h15=float(rate(balance));b_rate=float(rate([p for p in controls if p['key']=='smoke_B_fixed']))
    d_rate=float(rate([p for p in controls if p['key']=='smoke_D_noise_sequence']))
    # Screening has 60 query groups, up to30 fits plus shared references. Separate new fit cost from inference.
    offline_group=measurements['offline_pool_seconds']*len(ctx.cfg['model_seeds'])*len(ctx.cfg['dev_environments'])*len(ctx.cfg['query_times'])*2.5
    episodes={s:w[s]['control_slots_upper' if s=='A' else 'control_slots'] for s in ['A','B','C','D']}
    n=len(ctx.cfg['model_seeds']);e=len(ctx.cfg['dev_environments'])
    groups_by_stage=dict(A=n*e*(2*len(ctx.cfg['dev_trials'])+2*len(ctx.cfg['confirmation_trials'])),
        B=n*len(ctx.cfg['b_environments'])*3*(len(ctx.cfg['dev_trials'])+len(ctx.cfg['confirmation_trials'])),
        C=n*w['C']['scenarios']*(len(ctx.cfg['dev_trials'])+len(ctx.cfg['confirmation_trials'])),
        D=n*w['D']['scenarios']*len(ctx.cfg['confirmation_trials']))
    durations={'A':590,'B':590,'C':ctx.cfg['disturbance_steps'],'D':ctx.cfg['unseen_steps']}
    setup=float(np.median([p['setup_seconds'] for p in controls]));est={}
    # The selected update count is unknowable before A; estimate at100 and expose10..500 range.
    for stage in episodes:
        groups=groups_by_stage[stage]
        step_rate={'A':(h30+2*b_rate)/3,'B':b_rate,'C':h15,'D':(2*d_rate+h15)/3}[stage]
        plan=groups*durations[stage]*step_rate
        fits_per_group={'A':2.5*7,'B':4.,'C':4.*7+4.,'D':4.*7*1.5}[stage]
        fits=groups*fits_per_group*100*per_update
        other=groups*setup+(offline_group if stage=='A' else offline_group*.20 if stage=='B' else 0)
        # Reporting/shadow forecasts, support collection, warmups and preservation add measured but variable overhead.
        total=(plan+fits+other)*1.15
        est[stage]=dict(estimate_seconds=float(total),planning_seconds=float(plan),adaptation_seconds_at_100_updates=float(fits),
            plausible_seconds=[float((plan+fits*.1+other)*.85),float((plan+fits*5+other)*1.8)],
            assumed_selected_updates=100,control_slots=episodes[stage])
    allocated=torch.cuda.max_memory_allocated()/2**20 if ctx.device=='cuda' else 0.
    reserved=torch.cuda.max_memory_reserved()/2**20 if ctx.device=='cuda' else 0.
    result=dict(stages=est,total_estimate_seconds=sum(s['estimate_seconds'] for s in est.values()),
        total_plausible_seconds=[sum(s['plausible_seconds'][i] for s in est.values()) for i in [0,1]],
        measured_cpu_seconds_per_gradient_update=per_update,measured_H30_batch_step_seconds=h30,measured_H15_batch_step_seconds=h15,
        measured_B9_H30_batch_step_seconds=b_rate,measured_D7_H30_batch_step_seconds=d_rate,
        peak_allocated_mib=allocated,peak_reserved_mib=reserved,total_control_slots_upper=sum(episodes.values()),
        optimizer_updates_upper=w['optimizer_updates_upper'],known_A12_requested_updates=w['A12_requested_gradient_updates'],
        scope='Estimate only, not scientific results. Sequential screening and model-selected update counts make the final update total data-dependent.',
        caveats=['Short representative workloads; thermal/power variation and future selected update counts affect runtime.',
            'A count includes fixed-support and online controls; C includes unadapted Meta controls.',
            'Setup/hash/warmup costs and saved-data postprocessing add overhead; no concurrent GPU jobs enabled.',
            'Full matrix is retained; actual failed/unavailable jobs may finish earlier. Memory estimate is bounded measurement, not measured full peak.'])
    save_json(ctx.run/'runtime_estimate.json',result);return result
