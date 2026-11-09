"""Unchanged CEM, actual causal observations, explicit impulses/noise/multiple shifts."""
import time
from dataclasses import replace
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import GraphedPlanner, physics_step
from adaptive_world_model.environments.cartpole import _derivative
from adaptive_world_model.utils.randomness import stable_seed
from final_validation.simulation import warmup
from final_validation.common import pack_arrays, apply_arrays
from mismatch.cohorts import behavior
from .common import Planner, setting, transition_batch, score, state_bad, coverage
from .optimization import fit,compact
from .protocol import METAS, optimizer


def policies(selected=None,shortlist=None,dense=True,disabled=False):
    rows=[dict(label=m,method=m,optimizer=optimizer(),online=m not in ['frozen','oracle'])
          for m in ['frozen',*METAS,'ridge','oracle']]
    for method in METAS:
        configs=shortlist[method] if shortlist is not None else ([selected[method]] if selected else [])
        for c in configs: rows.append(dict(label=f'{method}__{c["name"]}',method=method,optimizer=c,online=True))
    if dense:
        rows += [dict(label='dense_'+m,method=m,optimizer=None,online=False,dense=True) for m in METAS]
    if disabled:
        rows += [dict(label='unadapted_'+m,method=m,optimizer=None,online=False) for m in METAS]
    return rows


def diagnostic_policies(extras):
    return [dict(label=f'diagnostic_{c["name"]}_{m}',method=m,optimizer=c,online=True,
                 isolated_diagnostic=c['isolated_diagnostic']) for m in METAS for c in extras[m]
            if c['name'] in ['adam_matched_updates','relaxed_guards']]


def impulse(state,physics,j):
    """Horizontal generalized impulse J[N s] on cart, no hinge impulse; coupled mass matrix."""
    cosine=state[...,2].cos(); effective=physics.cart_mass+physics.pole_mass-.75*physics.pole_mass*cosine.square()
    dv=j/effective;out=state.clone();out[...,1]+=dv;out[...,3]-=.75*cosine/physics.pole_half_length*dv
    return out


def external_step(state,command,physics,force):
    """External force [N] added after actuator gain/clipping, constant over dt."""
    total=command.squeeze(-1).clamp(-physics.force_limit,physics.force_limit)*physics.motor_gain+force
    dt=physics.dt/physics.rk4_substeps
    s=state
    for _ in range(physics.rk4_substeps):
        a=_derivative(s,total,physics);b=_derivative(s+dt*.5*a,total,physics)
        c=_derivative(s+dt*.5*b,total,physics);d=_derivative(s+dt*c,total,physics)
        s=s+dt/6*(a+2*b+2*c+d)
    return s


def observe(state,std,seed):
    generator=torch.Generator(device=state.device).manual_seed(seed)
    # One matched sensor draw across methods; true state is never passed to learned control.
    noise=torch.randn(1,4,generator=generator,device=state.device)*torch.tensor(std,device=state.device)
    return state+noise


def recovery_metrics(states,actions,event,cfg):
    angle=np.arctan2(np.sin(states[:,2]),np.cos(states[:,2]));good=(np.abs(angle)<cfg['success_angle'])&(np.abs(states[:,3])<cfg['success_omega'])
    count=0;end=None;longest=0;attempts=0
    for t in range(min(event,len(good)),len(good)):
        if good[t]:
            if count==0: attempts+=1
            count+=1;longest=max(longest,count)
            if count>=cfg['sustained_steps'] and end is None:end=t
        else:count=0
    section=states[min(event,len(states)-1):]
    return dict(event_step=event,recovery_success=end is not None,recovery_steps=end-event if end is not None else None,
        recovery_seconds=(end-event)*cfg['dt'] if end is not None else None,longest_postevent_stable=longest,
        recovery_attempts=attempts,peak_postevent_angle=float(np.max(np.abs(angle[event:]))) if event<len(angle) else None,
        peak_postevent_omega=float(np.abs(section[:,3]).max()),maximum_cart_displacement=float(np.abs(section[:,0]).max()),
        postevent_rail_clearance=float(2.4-np.abs(section[:,0]).max()))


def control(ctx,key,seed,trial,case,policy_list,fixed_support=None,online=True,limit=None,start=None):
    path=ctx.run/('collections' if case.get('collection_only') else 'controls')/f'{key}.json'; cached=ctx.cached_job(path)
    if cached:return cached
    started=time.perf_counter();task=case.get('task','balance');h=30 if task=='swingup' else 15
    phase=case.get('phase','stable_balance')
    prefix=start or warmup(ctx,seed,trial,phase)
    if not prefix['eligible']:
        return ctx.save_job(path,dict(key=key,rows=[],unavailable_slots=len(policy_list),reason='shared nominal prefix ineligible',
            case=case,seed=seed,trial=trial,split=case['split'],wall_seconds=time.perf_counter()-started),unavailable=np.ones(1))
    b=len(policy_list);models=[];initial_fits=[]
    for i,p in enumerate(policy_list):
        if p.get('dense'):m=ctx.dense_model(seed,case['environment'],p['method']);log=None
        elif fixed_support is not None and p['method'] not in ['frozen','oracle'] and p['optimizer'] is not None:
            support=fixed_support if not isinstance(fixed_support,list) else fixed_support[i]
            m,log=fit(ctx,seed,p['method'],support,p['optimizer'])
        else:m=ctx.make(seed,setting(p['method']));log=None
        models.append(m);initial_fits.append(compact(log))
    physical=ctx.physics(case['environment']); pack,settings=ctx.predictor(seed,policy_list,models,physical)
    planner=Planner(h,task,settings,ctx.device,ctx.cfg)
    state=torch.tensor(prefix['state'],device=ctx.device,dtype=torch.float32)[None].expand(b,-1).clone()
    graph=None;graph_setup=0.
    def capture():
        nonlocal graph,graph_setup
        if ctx.device=='cuda':
            ts=time.perf_counter();mean=planner.mean.clone()
            graph=GraphedPlanner(planner,pack,state,torch.zeros(b,dtype=torch.long,device=ctx.device))
            planner.mean.copy_(mean);graph_setup+=time.perf_counter()-ts
    capture();initial_graph_setup=graph_setup;planner.mean.zero_()
    if 'initial_mean' in prefix:planner.mean.copy_(torch.as_tensor(prefix['initial_mean'],device=ctx.device)[None])
    elif start is None:
        with np.load(ctx.run/'warmup'/f'{seed}_{trial}_{phase}.npz') as prefix_arrays:
            planner.mean.copy_(torch.as_tensor(prefix_arrays['cem_mean'],device=ctx.device)[None])
    std=ctx.cfg['noise_levels'][case.get('noise','none')]
    match_key=case.get('match_key',case['name'])
    observed=observe(state,std,stable_seed(seed,'hardening_sensor',trial,match_key,0))
    states=[state.clone()];observations=[observed.clone()];actions=[];predictions=[];frozen_predictions=[];corrections=[]
    dead=state_bad(state);update_logs=[];snapshots={f'at0_{n}':v for n,v in pack_arrays(pack).items()};snapshot_times=[0];events=[];impulse_eligible=torch.ones(b,dtype=torch.bool,device=ctx.device)
    pending=[[] for _ in policy_list];available=[0]*b
    kind=case.get('kind');event=0 if kind in ['persistent','combined',None] else ctx.cfg['disturbance_step']
    segment=event; total=case.get('steps',ctx.cfg['disturbance_steps'] if kind else ctx.cfg[task+'_steps']-prefix['steps'])
    if kind in ['persistent','combined'] or kind is None:
        events.append(dict(step=0,kind='persistent_change',environment=case['environment']))
    if kind=='force_pulse':events.append(dict(step=event+ctx.cfg['pulse_steps'],kind=kind,injection_start_step=event,
        external_force_N=ctx.cfg['force_pulse_levels'][case['severity']],duration_steps=ctx.cfg['pulse_steps']))
    steps=min(total,limit) if limit is not None else total; fit_seconds=0.;planning_seconds=0.;inference_seconds=0.
    warmup_seconds=prefix.get('wall_seconds',0.);setup=time.perf_counter()-started
    executed=0;loop_started=time.perf_counter();gpu_events=[]
    for t in range(steps):
        if t%10==0 and not bool((~dead).any()):break
        installed=False
        for i,queue in enumerate(pending):
            while queue and queue[0][0]<=t:
                due,models[i]=queue.pop(0);installed=True
        if installed:
            pack.refresh(models);snapshots.update({f'at{t}_{n}':v for n,v in pack_arrays(pack).items()});snapshot_times.append(t)
        if case.get('sequence') and t in ctx.cfg['sequence_change_steps']:
            first=t==ctx.cfg['sequence_change_steps'][0]
            physical=ctx.physics('gain_partial' if first else 'sequence_final');pack.physics=physical
            history=torch.stack(states[-ctx.cfg['sustained_steps']:])
            good=(torch.atan2(history[...,2].sin(),history[...,2].cos()).abs()<ctx.cfg['success_angle'])&(history[...,3].abs()<ctx.cfg['success_omega'])
            stable_before=good.all(0).cpu().tolist() if len(history)>=ctx.cfg['sustained_steps'] else [False]*b
            segment=t;events.append(dict(step=t,kind='persistent_change',environment='gain_partial' if first else 'sequence_final',stable_before=stable_before))
            for i,p in enumerate(policy_list):
                if p['online']:models[i].reset_adaptation();pending[i].clear();available[i]=t
            pack.refresh(models);capture()
            snapshots.update({f'at{t}_{n}':v for n,v in pack_arrays(pack).items()});snapshot_times.append(t)
        if kind=='repeated_cart' and t in ctx.cfg['repeated_steps']:segment=t
        active=~dead; random=stable_seed(seed,'hardening_control',trial,match_key,t)
        tick=time.perf_counter()
        with torch.no_grad():
            if ctx.device=='cuda':
                start_event=torch.cuda.Event(enable_timing=True);plan_event=torch.cuda.Event(enable_timing=True);infer_event=torch.cuda.Event(enable_timing=True)
                start_event.record()
            action=graph.action(observed,random) if graph else planner.action(pack,observed,random)
            if case.get('excitation'):
                # Fixed varied commands only away from the conservative state/velocity limits.
                varied=torch.tensor([[-8.,-4.,0.,4.,8.][(t//2)%5]],device=ctx.device).reshape(1,1)
                eligible=(observed[:,0].abs()<1.4)&(observed[:,1].abs()<2.)&(observed[:,3].abs()<8.)
                action=torch.where(eligible[:,None],(.5*action+.5*varied).clamp(-10,10),action)
            if ctx.device=='cuda':plan_event.record()
            else:planning_seconds+=time.perf_counter()-tick
            tick=time.perf_counter();pack.store_frozen=True
            pred=pack(observed,action);frozen=pack.last_frozen[:,0];pack.store_frozen=False
            if ctx.device=='cuda':infer_event.record();gpu_events.append((start_event,plan_event,infer_event))
            else:inference_seconds+=time.perf_counter()-tick
            pulse=ctx.cfg['force_pulse_levels'][case.get('severity',0)] if kind=='force_pulse' and event<=t<event+ctx.cfg['pulse_steps'] else 0.
            nxt=external_step(state,action,physical,pulse) if pulse else physics_step(state,action,physical)
            inject=(kind in ['cart_impulse','angular_velocity'] and t==event) or (kind=='repeated_cart' and t in ctx.cfg['repeated_steps']) or (kind=='combined' and t==ctx.cfg['combined_impulse_step'])
            if inject:
                if kind=='combined':
                    history=torch.stack(states[-ctx.cfg['sustained_steps']:])
                    stable=(history[...,2].sin().atan2(history[...,2].cos()).abs()<ctx.cfg['success_angle'])&(history[...,3].abs()<ctx.cfg['success_omega'])
                    impulse_eligible=stable.all(0)&active if len(history)>=ctx.cfg['sustained_steps'] else torch.zeros_like(active)
                perturbed=nxt.clone()
                if kind=='angular_velocity':perturbed[:,3]+=case.get('sign',1)*ctx.cfg['angular_velocity_levels'][case['severity']]
                else:perturbed=impulse(nxt,physical,case.get('sign',1)*ctx.cfg['impulse_levels'][case['severity']])
                nxt=torch.where(impulse_eligible[:,None],perturbed,nxt)
                events.append(dict(step=t+1,kind=kind,eligible=impulse_eligible.cpu().tolist(),severity=case['severity'],
                    horizontal_impulse_Ns=case.get('sign',1)*ctx.cfg['impulse_levels'][case['severity']] if kind!='angular_velocity' else None,
                    angular_velocity_kick_rad_s=case.get('sign',1)*ctx.cfg['angular_velocity_levels'][case['severity']] if kind=='angular_velocity' else None))
            nxt=torch.where(active[:,None],nxt,state)
            dead|=active&state_bad(nxt);state=nxt
            observed=observe(state,std,stable_seed(seed,'hardening_sensor',trial,match_key,t+1))
            states.append(state.clone());observations.append(observed.clone());actions.append(action.clone())
            predictions.append(pred.clone());corrections.append((pred-frozen).clone())
            frozen_predictions.append(frozen.clone())
        k=t-segment+1
        if online and k in ctx.cfg['adaptation_K']:
            # One causal history transfer per checkpoint, shared by the independent row fits.
            history_obs=torch.stack(observations[segment:]).cpu()
            history_actions=torch.stack(actions[segment:]).cpu()
            eligible_rows=(~dead).cpu().tolist()
            for i,p in enumerate(policy_list):
                if not p['online'] or p.get('dense') or not eligible_rows[i]:continue
                support=transition_batch(history_obs[:-1,i],history_actions[:,i],history_obs[1:,i],
                    f'hardening/actual_observed_past/{key}/row{i}/segment{segment}')
                ts=time.perf_counter();m,log=fit(ctx,seed,p['method'],support,p['optimizer']);fit_seconds+=time.perf_counter()-ts
                delay=p.get('delay_steps',0)
                applied=max(t+1,available[i])+delay
                if delay:pending[i].append((applied,m));available[i]=applied
                else:models[i]=m
                update_logs.append(dict(row=i,K=k,segment=segment,step=t+1,scheduled_application_step=applied,
                    latency_steps=delay,**compact(log)['log'],fit_path=log['fit_path'],support_hash=support.support_hash()))
            pack.refresh(models)
            snapshots.update({f'at{t+1}_{n}':v for n,v in pack_arrays(pack).items()});snapshot_times.append(t+1)
        executed=t+1
    if ctx.device=='cuda':torch.cuda.synchronize()
    loop_seconds=time.perf_counter()-loop_started
    if gpu_events:
        planning_seconds=sum(a.elapsed_time(b) for a,b,c in gpu_events)/1000
        inference_seconds=sum(b.elapsed_time(c) for a,b,c in gpu_events)/1000
    xs=torch.stack(states,1).cpu().numpy();obs=torch.stack(observations,1).cpu().numpy()
    us=torch.stack(actions,1).cpu().numpy() if actions else np.empty((b,0,1))
    ps=torch.stack(predictions,1).cpu().numpy() if actions else np.empty((b,0,4));cs=torch.stack(corrections,1).cpu().numpy() if actions else ps.copy()
    fps=torch.stack(frozen_predictions,1).cpu().numpy() if actions else ps.copy()
    # Shadow forecasts use the adapter available at the forecast start, never its later weights.
    forecast_rows=[];shadow_paths=[];frozen_shadow_paths=[];shadow_starts=[]
    with torch.no_grad():
        starts=sorted(set([*range(0,max(executed-29,0),25),*([event-5,event,event+5] if event>=5 else [0])]))
        for t in starts:
            if t<0 or t+30>executed:continue
            latest=max(k for k in snapshot_times if k<=t)
            arrays={n:snapshots[f'at{latest}_{n}'] for n in pack_arrays(pack)};apply_arrays(pack,arrays)
            c1,c2=ctx.cfg['sequence_change_steps']
            physical_at=ctx.physics('gain_partial' if case.get('sequence') and c1<=t<c2 else 'sequence_final' if case.get('sequence') and t>=c2 else case['environment'])
            pack.physics=physical_at;forecast=torch.as_tensor(obs[:,t],device=ctx.device);frozen_forecast=forecast.clone();errors=[];frozen_errors=[];predicted=[];frozen_predicted=[]
            for j in range(30):
                forecast=pack(forecast,torch.as_tensor(us[:,t+j],device=ctx.device))
                frozen_forecast=pack.frozen(frozen_forecast,torch.as_tensor(us[:,t+j],device=ctx.device))
                predicted.append(forecast.clone())
                frozen_predicted.append(frozen_forecast.clone())
                difference=forecast-torch.as_tensor(xs[:,t+j+1],device=ctx.device)
                difference[:,2]=torch.atan2(difference[:,2].sin(),difference[:,2].cos())
                errors.append((difference/torch.tensor(ctx.cfg['state_scale'],device=ctx.device)).square())
                fd=frozen_forecast-torch.as_tensor(xs[:,t+j+1],device=ctx.device)
                fd[:,2]=torch.atan2(fd[:,2].sin(),fd[:,2].cos())
                frozen_errors.append((fd/torch.tensor(ctx.cfg['state_scale'],device=ctx.device)).square())
            error=torch.stack(errors,1).cpu().numpy()
            shadow_paths.append(torch.stack(predicted,1).cpu().numpy());shadow_starts.append(t)
            frozen_shadow_paths.append(torch.stack(frozen_predicted,1).cpu().numpy());fe=torch.stack(frozen_errors,1).cpu().numpy()
            for i,p in enumerate(policy_list):
                # Forecasts after a physical failure are excluded, not padded into success metrics.
                if bool(state_bad(torch.as_tensor(xs[i,:t+31])).any()):continue
                forecast_rows.append(dict(row=i,policy=p['label'],start=t,available_snapshot_step=latest,
                    recursive_5_nmse=float(np.nan_to_num(error[i,:5],nan=1e20,posinf=1e20).mean()),
                    recursive_30_nmse=float(np.nan_to_num(error[i],nan=1e20,posinf=1e20).mean()),
                    shadow_frozen_recursive_30_nmse=float(np.nan_to_num(fe[i],nan=1e20,posinf=1e20).mean())))
    rows=[];event_recoveries=[]
    for i,p in enumerate(policy_list):
        metrics=score(xs[i],us[i],ctx.cfg,task,True);length=metrics['survival_steps'];end=length+1
        rec_event=(ctx.cfg['combined_impulse_step']+1 if kind=='combined' else event+1 if kind in ['cart_impulse','angular_velocity','repeated_cart'] else event+ctx.cfg['pulse_steps'] if kind=='force_pulse' else event)
        valid_recovery=kind!='combined' or bool(impulse_eligible[i].cpu())
        ever_stable_before=False
        if kind=='combined':
            before=xs[i,:min(length+1,ctx.cfg['combined_impulse_step']+1)]
            good_before=(np.abs(np.arctan2(np.sin(before[:,2]),np.cos(before[:,2])))<ctx.cfg['success_angle'])&(np.abs(before[:,3])<ctx.cfg['success_omega'])
            count=0
            for good in good_before:
                count=count+1 if good else 0
                if count>=ctx.cfg['sustained_steps']:ever_stable_before=True
        recovery=recovery_metrics(xs[i,:end],us[i,:length],rec_event,ctx.cfg)
        if not valid_recovery: recovery.update(recovery_success=None,recovery_steps=None,recovery_seconds=None)
        residual=(ps[i,:length]-xs[i,1:end])/pack.base.normalizer.y_scale.cpu().numpy()
        frozen_residual=(fps[i,:length]-xs[i,1:end])/pack.base.normalizer.y_scale.cpu().numpy()
        support_segments={}
        for update in update_logs:
            if update['row']==i:support_segments[str(update['segment'])]=max(support_segments.get(str(update['segment']),0),update['K'])
        fixed_count=(len(fixed_support[i]) if isinstance(fixed_support,list) else len(fixed_support)) if fixed_support is not None and p['optimizer'] is not None and p['method'] not in ['frozen','oracle'] and not p.get('dense') else 0
        used_count=max(support_segments.values(),default=fixed_count)
        combined={**metrics,**behavior(xs[i,:end],us[i,:length],ctx.cfg),**recovery}
        rows.append(dict(**combined,
            key=key,row=i,seed=seed,trial=trial,condition=case['name'],environment=case['environment'],task=task,
            phase=phase,split=case['split'],policy=p['label'],method=p['method'],optimizer=p['optimizer'],
            evaluation_protocol='online' if online else 'snapshot',
            online=online and p['online'],privileged_reference=p['method']=='oracle' or p.get('dense',False),
            application_delay_steps=p.get('delay_steps',0),
            dense_reference=p.get('dense',False),noise=case.get('noise','none'),kind=kind,
            isolated_diagnostic=p.get('isolated_diagnostic',False),
            invalid_prediction_fraction=float((~np.isfinite(ps[i,:length]).all(-1)).mean()) if length else None,
            combined_impulse_eligible=valid_recovery,never_stable_before_impulse=kind=='combined' and not ever_stable_before,
            lost_stability_before_impulse=kind=='combined' and ever_stable_before and not valid_recovery,
            support_K_max=used_count,optimizer_support_transitions_per_segment=support_segments,
            optimizer_support_unique_transitions=sum(support_segments.values()) if support_segments else fixed_count,
            one_step_nmse=float(np.nan_to_num(residual**2,nan=1e20,posinf=1e20).mean()) if length else None,
            shadow_frozen_one_step_nmse=float(np.nan_to_num(frozen_residual**2,nan=1e20,posinf=1e20).mean()) if length else None,
            correction_worsens_one_step_fraction=float((np.square(residual).mean(1)>np.square(frozen_residual).mean(1)).mean()) if length else None,
            correction_rms=float(np.nan_to_num(cs[i,:length]**2,nan=1e20).mean()**.5) if length else None,
            force_saturation=float((np.abs(us[i,:length])>=9.5).mean()) if length else None,
            actual_interactions=length,mpc_ms_per_batched_step=1000*planning_seconds/max(executed,1),
            controller_loop_ms_per_batched_step=1000*loop_seconds/max(executed,1),
            incomplete_time_limit=limit is not None and executed<total,
            numerical_residual_alarm_rate=None,detector_present=False))
        forecasts=[r for r in forecast_rows if r['row']==i]
        for metric in ['recursive_5_nmse','recursive_30_nmse','shadow_frozen_recursive_30_nmse']:
            rows[-1][metric]=float(np.mean([r[metric] for r in forecasts])) if forecasts else None
        ordered=sorted(events,key=lambda e:e['step'])
        for j,ev in enumerate(ordered):
            stop=min(length,ordered[j+1]['step']-1 if j+1<len(ordered) else length)
            if ev['step']>stop:continue
            eligible=ev.get('eligible',[True]*b)[i]
            r=recovery_metrics(xs[i,:stop+1],us[i,:stop],ev['step'],ctx.cfg)
            event_recoveries.append(dict(**r,row=i,policy=p['label'],kind=ev['kind'],eligible=eligible,
                stable_before_event=ev.get('stable_before',[None]*b)[i],window_end=stop))
    return ctx.save_job(path,dict(key=key,rows=rows,case=case,seed=seed,trial=trial,split=case['split'],
        initial_fits=initial_fits,updates=update_logs,events=events,event_recoveries=event_recoveries,forecasts=forecast_rows,snapshot_times=snapshot_times,steps=executed,planned_steps=total,
        wall_seconds=time.perf_counter()-started,setup_seconds=setup,warmup_seconds=warmup_seconds,
        planning_seconds=planning_seconds,inference_seconds=inference_seconds,fit_seconds=fit_seconds,
        control_loop_seconds=loop_seconds,optimization_wall_fraction=fit_seconds/max(loop_seconds,1e-9),
        graph_setup_seconds=graph_setup,initial_graph_setup_seconds=initial_graph_setup,actual_interactions=sum(r['actual_interactions'] for r in rows),
        observations_are_noisy=any(std),hidden_true_state_used_for_control=False,
        recovery_definition='100 consecutive angle/omega-stable states; final task success remains original V2 scoring'),
        states=xs,observations=obs,actions=us,predictions=ps,frozen_predictions=fps,corrections=cs,
        shadow_starts=np.asarray(shadow_starts),shadow_predictions=np.stack(shadow_paths,1) if shadow_paths else np.empty((b,0,30,4)),
        shadow_frozen_predictions=np.stack(frozen_shadow_paths,1) if frozen_shadow_paths else np.empty((b,0,30,4)),**snapshots)
