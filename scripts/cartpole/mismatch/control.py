"""Unchanged V2 CEM plus explicitly privileged scoring/source counterfactual policies."""
import time
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import GraphedPlanner,physics_step
from adaptive_world_model.utils.randomness import stable_seed
from final_validation.common import pack_arrays,score,state_bad
from final_validation.simulation import warmup
from .common import Predictor,Planner,setting,save_json,save_npz,transition_batch
from .protocol import METAS


class SubstitutionPlanner:
    """Same CEM generator. Source-pool rescoring is a diagnostic intervention, not new MPC training."""
    def __init__(self,source,scorer,score_pack,policies):
        self.source=source;self.scorer=scorer;self.score_pack=score_pack
        self.mask=torch.tensor([p['native'] for p in policies],device=source.device)
        self.has_substitution=any(not p['native'] for p in policies)
        source.record_candidates=self.has_substitution

    def __getattr__(self,name):return getattr(self.source,name)

    @torch.no_grad()
    def action(self,predict,state,seed,noise=None):
        action=self.source.action(predict,state,seed,noise)
        if not self.has_substitution:return action
        actions=torch.cat([self.source.last_candidates.transpose(0,1).flatten(1,2),
                           self.source.iteration_means.transpose(0,1)],1)
        current=state[:,None].expand(-1,actions.shape[1],-1);cost=torch.zeros_like(current[...,0])
        for t in range(self.h):
            current=self.score_pack(current,actions[:,:,t]);cost+=self.scorer.objective(current,actions[:,:,t])
        cost+=self.scorer.objective(current,torch.zeros_like(actions[:,:,0]),terminal=True)
        cost=torch.nan_to_num(cost,nan=1e30,posinf=1e30,neginf=1e30)
        chosen=actions[torch.arange(self.batch,device=state.device),cost.argmin(1)]
        chosen=torch.where(self.mask[:,None,None],self.source.selected_plan,chosen)
        self.source.mean.copy_(torch.cat([chosen[:,1:],chosen[:,-1:]],1))
        return chosen[:,0]


def model_for(ctx,record,method,dense=False):
    if method.startswith('dense_'):return ctx.dense_model(record['seed'],record['environment'],method[6:])
    return ctx.restored(record,method)


def control(ctx,key,record,policies,online=False,limit=None,fresh=False):
    path=ctx.run/'episodes'/f'{key}.json';cached=ctx.cached(path)
    if cached is not None:return cached
    started=time.perf_counter();seed=record['seed'];base,_=ctx.templates(seed);b=len(policies)
    models=[];scorers=[]
    for p in policies:
        if fresh:
            make=lambda m:ctx.dense_model(seed,record['environment'],m[6:]) if m.startswith('dense_') else ctx.make(seed,setting(m))
        else:make=lambda m:model_for(ctx,record,m)
        models.append(make(p['source']));scorers.append(models[-1] if p['source']==p['scorer'] else make(p['scorer']))
    def settings(which):return [dict(setting(p[which] if not p[which].startswith('dense_') else p[which][6:],p['safety'])) for p in policies]
    sources=settings('source');scores=settings('scorer')
    pack=Predictor(base,models,sources,ctx.device);spack=Predictor(base,scorers,scores,ctx.device)
    physical=ctx.physics(record['environment']);pack.physics=physical;spack.physics=physical
    cem=Planner(record['horizon'],record['task'],sources,ctx.device,ctx.cfg)
    scoreplanner=Planner(record['horizon'],record['task'],scores,ctx.device,ctx.cfg)
    planner=SubstitutionPlanner(cem,scoreplanner,spack,policies)
    state=torch.tensor(record['state'],device=ctx.device,dtype=torch.float32)[None].repeat(b,1)
    groups=torch.zeros(b,dtype=torch.long,device=ctx.device)
    graph=GraphedPlanner(planner,pack,state,groups) if ctx.device=='cuda' else None
    planner.mean.zero_()
    if 'initial_mean' in record:planner.mean.copy_(torch.tensor(record['initial_mean'],device=ctx.device)[None].expand(b,-1,-1))
    setup=time.perf_counter()-started;xs=[state.clone()];us=[];ps=[];corrections=[];updates=[];snapshots={};checkpoints=[]
    end=record['remaining'];steps=min(end,limit) if limit is not None else end
    dead=state_bad(state);failed_at=torch.zeros(b,dtype=torch.long,device=ctx.device)
    def checkpoint(k):
        checkpoints.append(dict(K=k,valid=(~dead).cpu().tolist()))
        snapshots.update({f'K{k}_{n}':v for n,v in pack_arrays(pack).items()})
        snapshots[f'K{k}_mean']=planner.mean.cpu().numpy().copy()
    if online:checkpoint(0)
    tick=time.perf_counter()
    for t in range(steps):
        active=~dead
        if t%10==0 and not bool(active.any()):break
        random=stable_seed(seed,'mismatch_control',record['condition'],record['trial'],record['id'],t)
        with torch.no_grad():
            if graph:action=graph.action(state,random)
            else:
                gen=torch.Generator(device=ctx.device).manual_seed(random)
                noise=torch.randn(cem.iterations,1,cem.n,(cem.h+4)//5,1,generator=gen,device=ctx.device).expand(-1,b,-1,-1,-1)
                action=planner.action(pack,state,0,noise)
            pred=spack(state,action);frozen=spack.frozen(state,action);nxt=physics_step(state,action,physical)
            bad=active&state_bad(nxt);failed_at=torch.where(bad,t+1,failed_at);dead|=bad
            state=torch.where(active[:,None],nxt,state);xs.append(state.clone());us.append(action.clone());ps.append(pred);corrections.append(pred-frozen)
        k=t+1
        if online and k in ctx.cfg['checkpoints']:
            for i,p in enumerate(policies):
                if p['source'] not in [*METAS,'ridge'] or dead[i].item():continue
                support=transition_batch(torch.stack(xs[:-1])[:,i],torch.stack(us)[:,i],torch.stack(xs[1:])[:,i],f'prospective_actual_past/{key}/{i}/K{k}')
                models[i].reset_adaptation();log=models[i].adapt(support)
                if not log['accepted']:raise RuntimeError(f'Rejected prospective adaptation {key}/{i}/K{k}')
                updates.append(dict(row=i,K=k,**log))
            pack.refresh(models);spack.refresh(scorers);checkpoint(k)
    if ctx.device=='cuda':torch.cuda.synchronize()
    loop=time.perf_counter()-tick
    states=torch.stack(xs,1).cpu().numpy();actions=torch.stack(us,1).cpu().numpy();preds=torch.stack(ps,1).cpu().numpy();corr=torch.stack(corrections,1).cpu().numpy()
    rows=[]
    for i,p in enumerate(policies):
        metrics=score(states[i],actions[i],ctx.cfg,record['task'],True);length=metrics['survival_steps']
        from .cohorts import behavior
        phenotype=behavior(states[i,:length+1],actions[i,:length],ctx.cfg)
        rows.append(dict(**metrics,seed=seed,episode_trial=record['trial'],condition=record['condition'],phase=record['phase'],
            task=record['task'],environment=record['environment'],row=i,key=key,method=p['source'],policy=p['label'],
            safety=p['safety'],regularization='existing',scorer=p['scorer'],native=p['native'],
            privileged_reference=p['source']=='oracle' or p['scorer']=='oracle' or p['source'].startswith('dense_'),
            dense_reference=p['source'].startswith('dense_'),evaluation_type='actual_online' if online else 'fixed_adapter_fork',
            source_case=record['id'],support_K=record['support_K'],split=record['split'],
            usable_postchange_observations=length-1 if metrics['failed'] else length,
            incomplete_time_limit=limit is not None and len(actions[i])<end,
            correction_rms=float(np.sqrt(np.mean(corr[i,:length]**2))),online=online,**phenotype))
    save_npz(path.with_suffix('.npz'),states=states,actions=actions,predictions=preds,corrections=corr,remaining=np.repeat(end,b),
             failed_at=failed_at.cpu().numpy(),**snapshots)
    packet=dict(config_sha256=ctx.digest,key=key,rows=rows,record=record,
        case=dict(name=record['condition'],environment=record['environment'],task=record['task'],horizon=record['horizon'],phase=record['phase']),
        checkpoints=checkpoints,updates=updates,setup_seconds=setup,loop_seconds=loop,wall_seconds=time.perf_counter()-started,
        actual_interactions=sum(r['survival_steps'] for r in rows),mpc_batch_calls=len(us),
        note='Native=unchanged V2 CEM elite mean. Substitution=argmin rescoring of source CEM pool; diagnostic, explicitly privileged where Oracle used.')
    save_json(path,packet);print(f'CONTROL {key} rows={b} steps={len(us)} seconds={packet["wall_seconds"]:.2f}',flush=True)
    return packet


def prospective(ctx,seed,trial,case,include_dense=True,limit=None):
    prefix=warmup(ctx,seed,trial,case['phase'])
    if not prefix['eligible']:
        replay_slots=4 if case['environment']=='combined_geometry' else int(trial==ctx.cfg['confirmation_trials'][0])
        save_json(ctx.run/'unavailable'/f'confirmation_{seed}_{trial}_{case["name"]}.json',dict(reason='shared nominal phase not reached',prefix=prefix,
            control_slots=7 if include_dense else 5,unavailable_replay_slots=replay_slots));return None
    r=dict(id=f'prospective_{seed}_{trial}_{case["name"]}',key='',row=0,seed=seed,trial=trial,condition=case['name'],
        phase=case['phase'],task=case['task'],environment=case['environment'],horizon=case['horizon'],state=prefix['state'],
        remaining=ctx.cfg[case['task']+'_steps']-prefix['steps'],support_K=0,split='confirmation')
    with np.load(ctx.run/'warmup'/f'{seed}_{trial}_{case["phase"]}.npz') as data:r['initial_mean']=data['cem_mean'].tolist()
    policies=[dict(source=m,scorer=m,native=True,safety='margin',label=m) for m in ['frozen',*METAS,'ridge','oracle']]
    if include_dense:policies += [dict(source='dense_'+m,scorer='dense_'+m,native=True,safety='margin',label='dense_'+m) for m in METAS]
    return control(ctx,r['id'],r,policies,online=True,limit=limit,fresh=True)
