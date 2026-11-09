"""Shared nominal prefixes and actual online streams; failure never supplies optimizer padding."""
import time
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import GraphedPlanner,physics_step
from adaptive_world_model.environments.cartpole import CartPolePhysics,wrap_angle
from adaptive_world_model.utils.randomness import stable_seed
from .common import Predictor,Planner,rollout,score,coverage,initial,state_bad,save_json,save_npz,transition_batch,pack_arrays
from .protocol import spec,CHECKPOINTS


def warmup(ctx,seed,trial,phase,limit=None):
    path=ctx.run/'warmup'/f'{seed}_{trial}_{phase}.json'
    cached=ctx.cached(path)
    if cached is not None:return cached
    started=time.perf_counter();cfg=ctx.cfg
    task='balance' if phase=='stable_balance' else 'swingup';h=15 if task=='balance' else 30
    base,_=ctx.templates(seed);s=[spec('frozen')];pack=Predictor(base,[base],s,ctx.device)
    state=initial(task,seed,trial)[None].to(ctx.device)
    planner=Planner(h,task,s,ctx.device,cfg)
    graph=GraphedPlanner(planner,pack,state,torch.zeros(1,dtype=torch.long,device=ctx.device)) if ctx.device=='cuda' else None
    planner.mean.zero_();states=[state.clone()];actions=[];good_count=0;reached=False;failed=False
    maximum=cfg['active_change_step'] if phase=='active_swingup' else cfg['stable_warmup_steps'] if task=='balance' else cfg['entry_max_steps']
    if limit is not None:maximum=min(maximum,limit)
    for t in range(maximum):
        with torch.no_grad():
            random=stable_seed(seed,'final_v2_nominal_prefix',task,trial,t)
            action=graph.action(state,random) if graph else planner.action(pack,state,random)
            state=physics_step(state,action,CartPolePhysics());states.append(state.clone());actions.append(action.clone())
            failed=bool(state_bad(state).item())
            good=bool((wrap_angle(state[0,2]).abs()<.15)&(state[0,3].abs()<.8))
            good_count=good_count+1 if good else 0
            if phase=='upright_entry' and abs(float(wrap_angle(state[0,2])))<.15:
                reached=True;break
            if failed:break
    if phase=='active_swingup':reached=not failed and len(actions)==cfg['active_change_step']
    if phase=='stable_balance':reached=not failed and good_count>=cfg['stable_required_steps']
    arrays=dict(states=torch.stack(states,1).cpu().numpy(),actions=torch.stack(actions,1).cpu().numpy() if actions else np.empty((1,0,1)),
                state=state[0].cpu().numpy(),cem_mean=planner.mean[0].cpu().numpy())
    save_npz(path.with_suffix('.npz'),**arrays)
    packet=dict(config_sha256=ctx.digest,seed=seed,trial=trial,phase=phase,task=task,horizon=h,steps=len(actions),
                eligible=bool(reached and not failed),failed=failed,state=arrays['state'].tolist(),
                wall_seconds=time.perf_counter()-started,prefix_seed=stable_seed(seed,'final_v2_nominal_prefix',task,trial,0))
    save_json(path,packet);return packet


def run_batch(ctx,key,seed,trial,case,settings,states,models,remaining,means=None,online=False,
              steps_limit=None,metadata=None,groups=None,group_count=None,noise_key=None):
    path=ctx.run/'episodes'/f'{key}.json'
    cached=ctx.cached(path)
    if cached is not None:return cached
    start=time.perf_counter();cfg=ctx.cfg;device=ctx.device;b=len(settings)
    base,_=ctx.templates(seed)
    pack=Predictor(base,models,settings,device);physical=ctx.physics(case['environment']);pack.physics=physical
    state=torch.as_tensor(np.asarray(states),device=device,dtype=torch.float32).reshape(b,4)
    remain=torch.as_tensor(remaining,device=device,dtype=torch.long).expand(b)
    maximum=int(remain.max().item())
    if steps_limit is not None:maximum=min(maximum,steps_limit)
    planner=Planner(case['horizon'],case['task'],settings,device,cfg)
    if groups is None:groups=torch.zeros(b,dtype=torch.long,device=device)
    else:groups=torch.as_tensor(groups,dtype=torch.long,device=device)
    graph=GraphedPlanner(planner,pack,state,groups) if device=='cuda' else None
    if graph is not None and group_count is not None:
        # Fixed source-noise dimensions preserve paired randomness even across different fork chunks.
        graph.source_noise=torch.empty(planner.iterations,group_count,planner.n,(planner.h+4)//5,1,device=device)
    planner.mean.zero_()
    if means is not None:planner.mean.copy_(torch.as_tensor(np.asarray(means),device=device))
    setup=time.perf_counter()-start;loop_start=time.perf_counter()
    states_log=[state.clone()];action_log=[];prediction_log=[];correction_log=[];times=[];updates=[]
    dead=state_bad(state);failed_at=torch.zeros(b,dtype=torch.long,device=device)
    checkpoints=[];params={};coverage_rows=[]
    def checkpoint(k):
        valid=(~dead)&(remain>=k)
        checkpoints.append(dict(K=k,valid=valid.cpu().tolist(),state=state.cpu().tolist(),
                                simulated_postchange_seconds=k*cfg['dt'],elapsed_loop_wall_seconds=time.perf_counter()-loop_start))
        for name,value in pack_arrays(pack).items():params[f'K{k}_{name}']=value
        params[f'K{k}_mean']=planner.mean.cpu().numpy().copy()
        if k>0:
            # Next warm-start candidate, using only current state and current weights. No future data is fitted.
            with torch.no_grad():candidate,_=rollout(pack,state,planner.mean[:,None],planner)
            query=torch.cat([state[:,None],candidate[:,0,:-1]],1).cpu()
            query_actions=planner.mean.cpu()
            for i,s in enumerate(settings):
                if not valid[i].item():continue
                support=transition_batch(torch.stack(states_log[:-1])[:,i],torch.stack(action_log)[:,i],
                                         torch.stack(states_log[1:])[:,i],f'coverage_actual_past/{key}/row{i}/K{k}')
                cov=coverage(base,support,query[i],query_actions[i])
                early=coverage(base,support.prefix(min(5,k)));late=coverage(base,support.select(slice(max(0,k-5),k)))
                coverage_rows.append(dict(row=i,K=k,method=s['method'],regularization=s['regularization'],safety=s['safety'],
                    seed=seed,episode_trial=trial,condition=case['name'],**cov,
                    early5_feature_variance=early['feature_variance'],late5_feature_variance=late['feature_variance'],
                    early5_action_std=early['action_std'],late5_action_std=late['action_std'],query_kind='next MPC warm-start candidate'))
    if online:checkpoint(0)
    for t in range(maximum):
        active=(~dead)&(t<remain)
        if t%10==0 and not bool(active.any().item()):break
        tick=time.perf_counter()
        with torch.no_grad():
            random=stable_seed(seed,'final_v2_mpc',noise_key or key,trial,t)
            if graph:action=graph.action(state,random)
            else:
                gen=torch.Generator(device=device).manual_seed(random)
                noise=torch.randn(planner.iterations,group_count or int(groups.max())+1,planner.n,(planner.h+4)//5,1,generator=gen,device=device)
                action=planner.action(pack,state,0,noise.index_select(1,groups))
            pack.store_frozen=True;pred=pack(state,action);pack.store_frozen=False
            correction=pred-pack.last_frozen[:,0]
            nxt=physics_step(state,action,physical)
            newly=active&state_bad(nxt)
            failed_at=torch.where(newly,t+1,failed_at)
            nxt=torch.where(active[:,None],nxt,state);dead|=newly
        if device=='cuda':torch.cuda.synchronize()
        times.append(time.perf_counter()-tick)
        action_log.append(action.detach());prediction_log.append(pred.detach());correction_log.append(correction.detach());states_log.append(nxt.detach())
        state=nxt
        k=t+1
        if online and k in CHECKPOINTS:
            for i,s in enumerate(settings):
                if s['method'] in ['frozen','oracle'] or dead[i].item() or k>int(remain[i]):continue
                support=transition_batch(torch.stack(states_log[:-1])[:,i],torch.stack(action_log)[:,i],
                                         torch.stack(states_log[1:])[:,i],f'actual_past/{key}/row{i}/K{k}')
                assert len(support)==k
                models[i].reset_adaptation();log=models[i].adapt(support)
                updates.append(dict(row=i,K=k,observations=k,optimizer_gradient_steps=log.get('gradient_steps',0),
                                    control_step=t+1,update_simulated_seconds=k*.02,**log))
            pack.refresh(models);checkpoint(k)
    if device=='cuda':torch.cuda.synchronize()
    loop_seconds=time.perf_counter()-loop_start
    xs=torch.stack(states_log,1).cpu().numpy();us=torch.stack(action_log,1).cpu().numpy()
    ps=torch.stack(prediction_log,1).cpu().numpy();cs=torch.stack(correction_log,1).cpu().numpy()
    rows=[]
    for i,s in enumerate(settings):
        length=min(len(action_log),int(remain[i]));x=xs[i,:length+1];u=us[i,:length]
        is_recovery=online or bool(metadata and metadata[i].get('stage') in ['snapshot_control','oracle_recovery'])
        metrics=score(x,u,cfg,case['task'],is_recovery)
        end=metrics['survival_steps'];failed=metrics['failed'];usable=end-1 if failed else end
        errors=ps[i,:end]-x[1:end+1];errors[:,2]=np.arctan2(np.sin(errors[:,2]),np.cos(errors[:,2]))
        local=dict(s,seed=seed,model_seed=seed,episode_trial=trial,condition=case['name'],phase=case.get('phase','fixed'),task=case['task'],
                   row=i,key=key,**metrics,actual_postchange_transitions=end,usable_postchange_observations=usable,
                   maximum_optimizer_support_used=max((u['K'] for u in updates if u['row']==i),default=0),
                   mpc_seed_at_step0=stable_seed(seed,'final_v2_mpc',noise_key or key,trial,0),
                   simulated_postchange_seconds=end*.02,one_step_nmse=float(np.mean(np.square(errors/base.normalizer.y_scale.numpy()))),
                   correction_normalized_rms=float(np.sqrt(np.nanmean(np.square(cs[i,:end]/base.normalizer.y_scale.numpy())))),
                   action_saturation_fraction=float(np.mean(np.abs(u[:end])>=9.5)),
                   actual_margin_zone_fraction=float(np.mean(np.abs(x[:end,0])>2.1)),
                   conservative_controller_steps=end if s['safety']=='margin' else 0,
                   conservative_is_static_margin=True,mpc_batch_seconds=float(np.mean(times)),
                   state_immediately_before_failure=x[max(0,end-1)].tolist() if failed else None,
                   failure_state=x[end].tolist() if failed else None,environment=case['environment'],
                   incomplete_time_limit=bool(steps_limit is not None and length<int(remain[i])))
        if metadata:local.update(metadata[i])
        rows.append(local)
    save_npz(path.with_suffix('.npz'),states=xs,actions=us,predictions=ps,corrections=cs,
             failed_at=failed_at.cpu().numpy(),remaining=remain.cpu().numpy(),**params)
    packet=dict(config_sha256=ctx.digest,key=key,case=case,rows=rows,checkpoints=checkpoints,updates=updates,coverage=coverage_rows,
                setup_seconds=setup,loop_seconds=loop_seconds,wall_seconds=time.perf_counter()-start,
                mpc_batch_calls=len(action_log),mpc_row_calls=b*len(action_log),actual_interactions=sum(r['actual_postchange_transitions'] for r in rows),
                source_note='Actual streams' if online else 'Fixed-adapter counterfactual or independent fixed-support controls')
    save_json(path,packet);print(f'BATCH {key} rows={b} calls={maximum} seconds={packet["wall_seconds"]:.2f}',flush=True)
    return packet


def run_live(ctx,phase,seed,trial,case,steps_limit=None):
    from .protocol import live_specs
    prefix=warmup(ctx,seed,trial,case['phase'])
    if not prefix['eligible']:
        path=ctx.run/'ineligible'/f'{phase}_{seed}_{trial}_{case["name"]}.json'
        save_json(path,dict(config_sha256=ctx.digest,stage=phase,seed=seed,trial=trial,
                           reason='nominal phase not reached before failure/time limit',prefix=prefix,case=case))
        return None
    specs=live_specs();models=[ctx.make(seed,s) for s in specs]
    with np.load(ctx.run/'warmup'/f'{seed}_{trial}_{case["phase"]}.npz') as data:means=np.repeat(data['cem_mean'][None],len(specs),axis=0)
    total=ctx.cfg[case['task']+'_steps']-prefix['steps']
    return run_batch(ctx,f'{phase}_{seed}_{trial}_{case["name"]}',seed,trial,case,specs,
                     np.repeat(np.array(prefix['state'])[None],len(specs),axis=0),models,[total]*len(specs),means,
                     online=True,steps_limit=steps_limit,metadata=[dict(stage=phase,prefix_steps=prefix['steps'],
                     episode_seed=stable_seed(seed,'final_v2_episode_initial',case['task'],trial),
                     change_simulated_step=prefix['steps'],nominal_negative_control=case['environment']=='nominal') for _ in specs])
