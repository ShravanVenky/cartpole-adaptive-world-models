"""As-of-K model snapshots, shared counterfactual candidates and isolated control forks."""
import time
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import physics_step
from adaptive_world_model.environments.cartpole import observable_features
from adaptive_world_model.utils.randomness import stable_seed
from diagnostics.replay import rankings
from .common import Predictor,Planner,rollout,save_json,save_npz,transition_batch,coverage
from .protocol import CHECKPOINTS,shadow_specs,fork_specs
from .simulation import run_batch


def parent_support(data,row,k,provenance):
    if k==0:return None
    return transition_batch(data['states'][row,:k],data['actions'][row,:k],data['states'][row,1:k+1],provenance)


def valid_at(parent,collector,k):
    found=next((s for s in parent['checkpoints'] if s['K']==k),None)
    return found is not None and found['valid'][collector]


@torch.no_grad()
def evaluate_snapshots(ctx,parent,collector):
    cfg=ctx.cfg;seed=parent['rows'][collector]['seed'];trial=parent['rows'][collector]['episode_trial']
    key=parent['key']+f'_collector{collector}'
    target=ctx.run/'snapshots'/f'{key}.json'
    cached=ctx.cached(target)
    if cached is not None:return cached
    started=time.perf_counter();settings=shadow_specs();base,_=ctx.templates(seed);rows=[];profiles=[];arrays={}
    case=parent['case'];physical=ctx.physics(case['environment'])
    with np.load(ctx.run/'episodes'/f'{parent["key"]}.npz') as source:
        for k in CHECKPOINTS:
            common=dict(seed=seed,episode_trial=trial,condition=case['name'],phase=case['phase'],K=k,
                        collector=parent['rows'][collector]['safety'],parent=parent['key'],collector_row=collector,
                        actual_observations=k,gradient_updates_per_meta_fit=10,nominal=case['environment']=='nominal')
            if not valid_at(parent,collector,k):
                rows.extend([dict(s,**common,censored=True,censor_reason='source controller failed before a usable checkpoint',
                                  actual_observations_available=parent['rows'][collector]['usable_postchange_observations']) for s in settings]);continue
            support=parent_support(source,collector,k,f'actual_past/{key}/K{k}')
            fits=[ctx.fit(seed,s,support) for s in settings];models=[x[0] for x in fits]
            state=torch.tensor(source['states'][collector,k],device=ctx.device)
            pack=Predictor(base,models,settings,ctx.device);pack.physics=physical
            planner=Planner(case['horizon'],case['task'],settings,ctx.device,cfg);planner.record_candidates=True
            initial=state[None].expand(len(settings),-1)
            g=torch.Generator(device=ctx.device).manual_seed(stable_seed(seed,'final_v2_rank',parent['key'],k))
            noise=torch.randn(planner.iterations,1,planner.n,(planner.h+4)//5,1,generator=g,device=ctx.device).expand(-1,len(settings),-1,-1,-1)
            planner.action(pack,initial,0,noise)
            candidates=planner.last_candidates[-1,:2].reshape(-1,planner.h,1)
            broad=(torch.rand(cfg['ranking_broad_candidates'],(planner.h+4)//5,1,generator=g,device=ctx.device)*20-10).repeat_interleave(5,1)[:,:planner.h]
            candidates=torch.cat([candidates,broad,planner.selected_plan])
            path_parts=[];cost_parts=[]
            for chunk in candidates.split(cfg['ranking_chunk']):
                paths,cost=rollout(pack,initial,chunk[None].expand(len(settings),-1,-1,-1),planner)
                path_parts.append(paths.cpu().numpy());cost_parts.append(cost.cpu().numpy())
            paths=np.concatenate(path_parts,1);costs=np.concatenate(cost_parts,1);true=paths[1];truecost=costs[1]
            array_prefix=f'K{k}_';arrays[array_prefix+'candidates']=candidates.cpu().numpy();arrays[array_prefix+'predicted_costs']=costs
            arrays[array_prefix+'true_costs']=truecost;arrays[array_prefix+'preferred_paths']=np.stack([paths[i,int(costs[i].argmin())] for i in range(len(settings))])
            arrays[array_prefix+'true_preferred_paths']=np.stack([true[int(costs[i].argmin())] for i in range(len(settings))])
            # A separate100-step, shared-action counterfactual for long recursive error. Never fitted.
            u100=(torch.rand(1,20,1,generator=g,device=ctx.device)*20-10).repeat_interleave(5,1)
            s100=initial[:,None];truth100=state[None];long_errors={h:None for h in cfg['multistep_horizons']};acc=torch.zeros(len(settings),device=ctx.device)
            scale=torch.tensor([2.4,3.,np.pi,4.],device=ctx.device)
            for h in range(1,101):
                a=u100[:,h-1:h];s100=pack(s100,a.expand(len(settings),-1,-1));truth100=physics_step(truth100,u100[:,h-1],physical)
                err=s100[:,0]-truth100;err[:,2]=torch.atan2(err[:,2].sin(),err[:,2].cos());acc+=(err/scale).square().mean(-1)
                if h in long_errors:long_errors[h]=(acc/h).cpu().numpy()
            # Local errors on identical true candidate states isolate correction from rollout accumulation.
            stride=max(1,len(candidates)//24);qpath=torch.tensor(true[::stride],device=ctx.device)
            qa=candidates[::stride];qinput=torch.cat([state[None,None].expand(len(qpath),1,4),qpath[:,:-1]],1)
            qflat=qinput.reshape(-1,4);uflat=qa.reshape(-1,1)
            predicted=pack(qflat[None].expand(len(settings),-1,-1),uflat[None].expand(len(settings),-1,-1))
            actual=qpath.reshape(-1,4);error=predicted-actual;error[...,2]=torch.atan2(error[...,2].sin(),error[...,2].cos())
            mse=(error/pack.base.normalizer.y_scale).square().mean(-1).reshape(len(settings),len(qpath),planner.h)
            correction=((predicted-predicted[0])/pack.base.normalizer.y_scale).square().mean(-1).sqrt().reshape(len(settings),len(qpath),planner.h)
            normalized=base.normalizer.encode_x(observable_features(qflat.cpu(),uflat.cpu())).double()
            distance=torch.cdist(normalized,base.normalizer.encode_x(support.features()).double()).amin(1).reshape(len(qpath),planner.h) if support is not None else torch.full((len(qpath),planner.h),float('nan'))
            cover=coverage(base,support,qflat.cpu(),uflat.cpu())
            for i,s in enumerate(settings):
                r=rankings(costs[i],truecost,paths[i],true,min(10,len(candidates)))
                log=fits[i][1];m=models[i]
                coef=m.coefficients.detach().cpu().numpy() if hasattr(m,'coefficients') else None
                if support is not None:
                    z=base.normalizer.encode_x(support.features()).double();design=torch.cat([z,torch.ones_like(z[:,:1])],1)
                    lam=getattr(m,'regularization',cfg['ridge_lambda_reference']);eigen=torch.linalg.eigvalsh(design.T@design+lam*torch.eye(7))
                    condition=float(eigen[-1]/eigen[0])
                else:condition=None
                frozen_preferred=int(costs[0].argmin());preferred=int(costs[i].argmin())
                rows.append(dict(s,**common,censored=False,**r,local_one_step_nmse=float(mse[i,:,0].mean()),
                                 preferred_action_worse_than_frozen=bool(truecost[preferred]>truecost[frozen_preferred]+1e-5),
                                 preferred_action_true_cost_difference_vs_frozen=float(truecost[preferred]-truecost[frozen_preferred]),
                                 correction_worsens_prediction_fraction=float((mse[i]>mse[0]).float().mean()),
                                 correction_rms=float(correction[i].mean()),adaptation_seconds=0 if log is None else log['adaptation_seconds'],
                                 update_accepted=True if log is None else log['accepted'],actual_gradient_steps=0 if log is None else log.get('executed_gradient_steps',log.get('gradient_steps',0)),
                                 protocol_gradient_steps=0 if log is None else log.get('gradient_steps',0),
                                 fit_cache_reused=False if log is None else log.get('cache_reused',False),
                                 uncached_adaptation_seconds=0 if log is None else log.get('uncached_adaptation_seconds',log['adaptation_seconds']),
                                 coefficient_norm=float(np.linalg.norm(coef)) if coef is not None else None,condition_number=condition,
                                 lambda_sum=getattr(m,'regularization',None),support_hash='none' if support is None else support.support_hash(),
                                 latest_support_postchange_step=k,**{f'long_recursive_{h}_nmse':float(long_errors[h][i]) for h in long_errors},
                                 long_forecast_nonfinite=bool(not np.isfinite(long_errors[100][i])),
                                 **{f'coverage_{name}':value for name,value in cover.items()}))
                for h in range(planner.h):
                    profiles.append(dict(**common,method=s['method'],alpha=s['alpha'],regularization=s['regularization'],horizon_position=h+1,
                        error=float(mse[i,:,h].mean()),correction_rms=float(correction[i,:,h].mean()),
                        correction_worsens_fraction=float((mse[i,:,h]>mse[0,:,h]).float().mean()),
                        support_distance=float(distance[:,h].median()),actual_rail_clearance=float(2.4-qinput[:,h,0].abs().max()),
                        mean_abs_omega=float(qinput[:,h,3].abs().mean()),query_within_actual_rails_fraction=float((qinput[:,h,0].abs()<=2.4).float().mean())))
            if support is not None:
                arrays[array_prefix+'support_states']=support.state.numpy();arrays[array_prefix+'support_actions']=support.action.numpy()
    save_npz(target.with_suffix('.npz'),**arrays)
    packet=dict(config_sha256=ctx.digest,key=key,rows=rows,horizon_profiles=profiles,
                wall_seconds=time.perf_counter()-started,source_parent=parent['key'],collector=collector)
    save_json(target,packet);print(f'SNAPSHOTS {key} seconds={packet["wall_seconds"]:.2f}',flush=True);return packet


def control_forks(ctx,parent,collector,steps_limit=None):
    seed=parent['rows'][collector]['seed'];trial=parent['rows'][collector]['episode_trial'];case=parent['case'];cfg=ctx.cfg
    specs=fork_specs();active=[];censored=[];metadata=[]
    source_path=ctx.run/'episodes'/f'{parent["key"]}.npz'
    with np.load(source_path) as data:
        for s in specs:
            meta=dict(stage='snapshot_control',K=s['K'],collector=parent['rows'][collector]['safety'],parent=parent['key'],collector_row=collector)
            if not valid_at(parent,collector,s['K']):
                censored.append(dict(s,**meta,seed=seed,episode_trial=trial,condition=case['name'],censored=True,success=None,
                                     reason='parent failed before usable support checkpoint'))
            else:active.append(s);metadata.append(meta)
        packets=[]
        for offset in range(0,len(active),cfg['max_batch_rows']):
            chunk=active[offset:offset+cfg['max_batch_rows']];meta=metadata[offset:offset+cfg['max_batch_rows']]
            key=f'fork_{parent["key"]}_collector{collector}_{offset}'
            cached=ctx.cached(ctx.run/'episodes'/f'{key}.json')
            if cached is not None:packets.append(cached);continue
            models=[];states=[];remaining=[];means=[]
            for s in chunk:
                k=s['K'];support=parent_support(data,collector,k,f'actual_past/{parent["key"]}/row{collector}/K{k}')
                m,_=ctx.fit(seed,s,support);models.append(m);states.append(data['states'][collector,k])
                remaining.append(int(data['remaining'][collector])-k)
                means.append(data[f'K{k}_mean'][collector])
            packets.append(run_batch(ctx,key,seed,trial,case,chunk,states,models,remaining,means,
                steps_limit=steps_limit,metadata=[dict(m,censored=False) for m in meta],
                groups=[CHECKPOINTS.index(s['K']) for s in chunk],group_count=len(CHECKPOINTS),
                noise_key=f'fork_{parent["key"]}'))
    save_json(ctx.run/'censoring'/f'{parent["key"]}_collector{collector}.json',dict(config_sha256=ctx.digest,
              rows=censored,planned=len(specs),uncensored=len(active),censored=len(censored)))
    return packets


def unavailable_cohorts(ctx,seed,trial,case):
    """Keep planned denominators when the shared nominal prefix never reaches its event."""
    for collector,safety in [(0,'original'),(5,'margin')]:
        parent=f'confirmation_{seed}_{trial}_{case["name"]}'
        rows=[dict(s,seed=seed,episode_trial=trial,condition=case['name'],phase=case['phase'],K=k,
                   collector=safety,parent=parent,collector_row=collector,censored=True,
                   censor_reason='nominal phase not reached',actual_observations_available=0)
              for k in CHECKPOINTS for s in shadow_specs()]
        path=ctx.run/'snapshots'/f'{parent}_collector{collector}.json'
        save_npz(path.with_suffix('.npz'))
        save_json(path,dict(config_sha256=ctx.digest,rows=rows,horizon_profiles=[],wall_seconds=0.,source_parent=parent))
        forks=[dict(s,seed=seed,episode_trial=trial,condition=case['name'],collector=safety,parent=parent,
                    stage='snapshot_control',censored=True,reason='nominal phase not reached') for s in fork_specs()]
        save_json(ctx.run/'censoring'/f'{parent}_collector{collector}.json',dict(config_sha256=ctx.digest,
                  rows=forks,planned=len(forks),uncensored=0,censored=len(forks)))
