"""Shared exact-deduplicated CEM pools, scoring/source substitutions and decision-critical errors."""
import json
import time
import numpy as np
import pandas as pd
import torch
from adaptive_world_model.control.evaluation_mpc import physics_step
from adaptive_world_model.environments.cartpole import observable_features
from adaptive_world_model.utils.randomness import stable_seed
from final_validation.common import coverage
from final_validation.common import pack_arrays
from .common import Planner,save_json,save_npz
from .protocol import SOURCES,METAS,METHODS


def array_error(pred,true,scale):
    error=pred-true;error=error.copy();error[...,2]=np.arctan2(np.sin(error[...,2]),np.cos(error[...,2]))
    return np.nan_to_num((error/scale)**2,nan=1e20,posinf=1e20,neginf=1e20)


def rank_correlation(p,q):
    a=pd.Series(p).rank().to_numpy();b=pd.Series(q).rank().to_numpy()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None


@torch.no_grad()
def generate_pool(ctx,record,pack,names,settings,small=False):
    planner=Planner(record['horizon'],record['task'],settings,ctx.device,ctx.cfg);planner.record_candidates=True
    state=torch.tensor(record['state'],device=ctx.device)[None].expand(len(names),-1)
    gen=torch.Generator(device=ctx.device).manual_seed(stable_seed(record['seed'],'mismatch_candidates',record['id']))
    noise=torch.randn(planner.iterations,1,planner.n,(planner.h+4)//5,1,generator=gen,device=ctx.device).expand(-1,len(names),-1,-1,-1)
    planner.action(pack,state,0,noise)
    pools=[];tags=[];native={};cursor=0
    for source in SOURCES:
        i=names.index(source);a=torch.cat([planner.last_candidates[:,i].flatten(0,1),planner.iteration_means[:,i]])
        pools.append(a.cpu().numpy());tags.extend([source]*len(a));native[source]=cursor+len(a)-1;cursor+=len(a)
    broad=(torch.rand(ctx.cfg['broad_candidates'],(planner.h+4)//5,1,generator=gen,device=ctx.device)*20-10).repeat_interleave(5,1)[:,:planner.h]
    pools.append(broad.cpu().numpy());tags.extend(['broad']*len(broad))
    raw=np.concatenate(pools);unique,first,inverse=np.unique(raw.reshape(len(raw),-1),axis=0,return_index=True,return_inverse=True)
    # np.unique is exact float32, not approximate rounding; preserve deterministic first-appearance order.
    order=np.argsort(first);old_to_new=np.empty(len(order),dtype=int);old_to_new[order]=np.arange(len(order));inverse=old_to_new[inverse]
    actions=raw[first[order]];origins=np.zeros((len(actions),len(SOURCES)+1),bool)
    for i,source in enumerate([*SOURCES,'broad']):origins[inverse[np.asarray(tags)==source],i]=True
    native={name:int(inverse[ix]) for name,ix in native.items()}
    return actions,origins,native,dict(raw_candidates=len(raw),unique_candidates=len(actions),deduplicated=len(raw)-len(actions),
        command_std=float(actions.std()),unique_first_actions=len(np.unique(actions[:,0])),
        mean_pairwise_action_distance=float(np.linalg.norm(actions[:128,None]-actions[None,:128],axis=(2,3)).mean()))


@torch.no_grad()
def evaluate_pool(ctx,record,names,models,actions):
    pack,settings=ctx.pack(record['seed'],names,models);pack.physics=ctx.physics(record['environment'])
    planner=Planner(record['horizon'],record['task'],settings,ctx.device,ctx.cfg)
    paths=[];costs=[];teachers=[];hcosts=[];terminals=[]
    scale=np.asarray(ctx.cfg['state_scale']);initial=torch.tensor(record['state'],device=ctx.device,dtype=torch.float32)
    for chunk in torch.as_tensor(actions,device=ctx.device).split(ctx.cfg['candidate_chunk']):
        if chunk.shape[1]<max(ctx.cfg['forecast_steps']):
            chunk=torch.cat([chunk,chunk[:,-1:].expand(-1,max(ctx.cfg['forecast_steps'])-chunk.shape[1],-1)],1)
        b,n,h=len(names),len(chunk),chunk.shape[1];s=initial[None,None].expand(b,n,-1)
        cumulative=torch.zeros(b,n,device=ctx.device);route=[];hc=[];tc=[]
        for t in range(h):
            s=pack(s,chunk[:,t][None].expand(b,-1,-1));route.append(s)
            cumulative+=planner.objective(s,chunk[:,t][None].expand(b,-1,-1))
            terminal=planner.objective(s,torch.zeros(b,n,1,device=ctx.device),terminal=True)
            hc.append(cumulative+terminal);tc.append(terminal)
        path=torch.stack(route,2);truth=path[names.index('oracle')]
        true_inputs=torch.cat([initial[None,None].expand(n,1,4),truth[:,:-1]],1).reshape(-1,4)
        teacher=pack(true_inputs[None].expand(b,-1,-1),chunk.reshape(-1,1)[None].expand(b,-1,-1)).reshape(b,n,h,4)
        paths.append(path.cpu().numpy());teachers.append(teacher.cpu().numpy())
        hcosts.append(torch.stack(hc,2).cpu().numpy());terminals.append(torch.stack(tc,2).cpu().numpy())
        costs.append(hc[record['horizon']-1].cpu().numpy())
    path=np.concatenate(paths,1);teacher=np.concatenate(teachers,1)
    result=dict(paths=path[:,:,:record['horizon']],costs=np.nan_to_num(np.concatenate(costs,1),nan=1e30,posinf=1e30,neginf=1e30),
        teacher=teacher[:,:,:record['horizon']],horizon_costs=np.nan_to_num(np.concatenate(hcosts,1),nan=1e30,posinf=1e30,neginf=1e30),
        terminal_costs=np.nan_to_num(np.concatenate(terminals,1),nan=1e30,posinf=1e30,neginf=1e30))
    if path.shape[2]>record['horizon']:result.update(forecast_paths=path,forecast_teacher=teacher)
    return result


def summarize(ctx,record,names,values,actions,origins,native):
    truth=values['paths'][names.index('oracle')];truecost=values['costs'][names.index('oracle')]
    finite=np.isfinite(truth).all((1,2));unsafe=(np.abs(truth[:,:,0]).max(1)>2.4)|~finite|(np.abs(truth[:,:,1:]).max((1,2))>100)
    scale=np.asarray(ctx.cfg['state_scale']);best=int(truecost.argmin());den=max(abs(float(truecost[best])),1.)
    one=[];profiles=[];substitutions=[];regimes=[]
    qstate=np.concatenate([np.tile(record['state'],(len(actions),1,1)),truth[:,:-1]],1)
    support=ctx.support(record);base,_=ctx.templates(record['seed'])
    stride=max(1,len(actions)//32)
    cov=coverage(base,support,qstate[::stride].reshape(-1,4),actions[::stride].reshape(-1,1))
    common={k:record[k] for k in ['id','seed','condition','phase','step','support_K','split','collector_method','source_success']}
    common.update(source_cost=record['source_cost'],source_rail=record['source_rail'],horizon=record['horizon'],operating_phase=record['operating_phase'])
    for i,name in enumerate(names):
        p=values['costs'][i];preferred=int(p.argmin());ownname=name if name in SOURCES else None
        own=origins[:,SOURCES.index(ownname)] if ownname else np.ones(len(p),bool)
        ix=np.flatnonzero(own);ownbest=int(ix[truecost[ix].argmin()]);ownpred=int(ix[p[ix].argmin()])
        ownplan=native.get(name,preferred)
        ranking=float((truecost[preferred]-truecost[best])/den);discovery=float((truecost[ownbest]-truecost[best])/den)
        good=float(truecost[preferred]-truecost[best])<=ctx.cfg['good_candidate_tolerance']*den
        native_regret=float((truecost[ownplan]-truecost[best])/den)
        convergence_gap=float((truecost[ownplan]-truecost[ownpred])/den)
        bad_rank=ranking>ctx.cfg['ranking_regret_threshold'];bad_search=discovery>ctx.cfg['search_regret_threshold'] or (good and convergence_gap>ctx.cfg['search_regret_threshold'])
        mechanism='mixed_failure' if bad_rank and bad_search else 'ranking_failure' if bad_rank else 'search_failure' if bad_search and good else 'no_demonstrated_candidate_failure'
        if unsafe.all():mechanism='no_safe_candidate_demonstrated'
        if ownname is None:mechanism='ranking_failure_search_not_tested' if bad_rank else 'search_not_tested'
        predclear=2.4-np.abs(values['paths'][i,:,:,0]).max(1);trueclear=2.4-np.abs(truth[:,:,0]).max(1)
        predicted_safe=(predclear>=0)&np.isfinite(values['paths'][i]).all((1,2))&(np.abs(values['paths'][i,:,:,1:]).max((1,2))<=100)
        e=array_error(values['paths'][i],truth,scale);teacher=array_error(values['teacher'][i],truth,base.normalizer.y_scale.numpy())
        k=min(ctx.cfg['rank_top_k'],len(p));top1=np.isclose(truecost[preferred],truecost[best],rtol=1e-6,atol=1e-6)
        chosen_true=truth[preferred]
        one.append(dict(**common,method=name,candidates=len(p),rank_correlation=rank_correlation(p,truecost),top1_agreement=bool(top1),
            topk_overlap=len(set(np.argsort(p)[:k])&set(np.argsort(truecost)[:k]))/k,
            normalized_planning_regret=ranking,oracle_source_coverage_regret=discovery if ownname else None,
            native_cem_regret=native_regret if ownname else None,cem_selection_gap=convergence_gap if ownname else None,
            within_own_pool_ranking_regret=float((truecost[ownpred]-truecost[ownbest])/max(abs(float(truecost[ownbest])),1.)) if ownname else None,
            true_cost_preferred=float(truecost[preferred]),true_cost_own_cem=float(truecost[ownplan]) if ownname else None,
            predicted_cost_own_cem=float(p[ownplan]) if ownname else None,best_evaluated_true_cost=float(truecost[best]),
            planning_cost_mae=float(np.abs(p-truecost).mean()),
            better_candidate_exists=bool(truecost[ownplan]>truecost[best]+1e-6*den) if ownname else None,recognizes_good_candidate=bool(good),
            generated_global_best=bool(own[best]) if ownname else None,oracle_best_predicted_percentile=float((p<p[best]).mean()),
            false_safe_fraction=float((predicted_safe&unsafe).mean()),false_unsafe_fraction=float((~predicted_safe&~unsafe).mean()),
            prediction_invalid_fraction=float((~np.isfinite(values['paths'][i]).all((1,2))).mean()),
            preferred_false_safe=bool(predicted_safe[preferred]&unsafe[preferred]),preferred_true_clearance=float(trueclear[preferred]),
            preferred_predicted_clearance=float(predclear[preferred]),local_teacher_nmse=float(teacher.mean()),
            recursive_nmse=float(e.mean()),worst_candidate_nmse=float(e.mean((1,2)).max()),
            p95_candidate_nmse=float(np.quantile(e.mean((1,2)),.95)),decision_critical_nmse=float(e[preferred].mean()),
            support_distance=cov.get('query_support_distance_median'),support_rank=cov.get('feature_rank'),
            correction_rms=float(np.sqrt(array_error(values['teacher'][i],values['teacher'][0],base.normalizer.y_scale.numpy()).mean())),
            mechanism=mechanism,source_outcome='success' if record['source_success'] else 'failure',
            exact_residual_max_error=float(np.nanmax(np.abs(values['paths'][i]-truth))) if name=='exact_residual' else None))
        forecast_path=values.get('forecast_paths',values['paths']);forecast_teacher=values.get('forecast_teacher',values['teacher'])
        forecast_truth=forecast_path[names.index('oracle')]
        pe=array_error(forecast_path[i],forecast_truth,scale)
        pt=array_error(forecast_teacher[i],forecast_truth,base.normalizer.y_scale.numpy())
        for h in ctx.cfg['forecast_steps']:
            pc=values['horizon_costs'][i,:,h-1];tc=values['horizon_costs'][names.index('oracle'),:,h-1];pick=int(pc.argmin());opt=int(tc.argmin())
            component=pe[:,:h].mean((0,1));te=pt[:,:h].mean((0,1));row=dict(**common,method=name,forecast_horizon=h,
                held_action_extension=h>record['horizon'],
                recursive_nmse=float(component.mean()),teacher_nmse=float(te.mean()),
                rank_correlation=rank_correlation(pc,tc),normalized_regret=float((tc[pick]-tc[opt])/max(abs(float(tc[opt])),1.)),
                planning_cost_mae=float(np.abs(pc-tc).mean()),
                cumulative_stage_cost_mae=float(np.abs((pc-values['terminal_costs'][i,:,h-1])-(tc-values['terminal_costs'][names.index('oracle'),:,h-1])).mean()),
                terminal_cost_mae=float(np.abs(values['terminal_costs'][i,:,h-1]-values['terminal_costs'][names.index('oracle'),:,h-1]).mean()),
                own_selected_recursive_nmse=float(pe[ownplan,:h].mean()) if ownname else None,oracle_selected_recursive_nmse=float(pe[native['oracle'],:h].mean()))
            row.update({f'{label}_nmse':float(component[j]) for j,label in enumerate(['cart_position','cart_velocity','pole_angle','angular_velocity'])});profiles.append(row)
        angle=np.abs(np.arctan2(np.sin(qstate[:,:,2]),np.cos(qstate[:,:,2])));pdefs=ctx.cfg['phase_definitions']
        masks=dict(high_angular_velocity=np.abs(qstate[:,:,3])>pdefs['high_omega'],large_angle=angle>pdefs['large_angle'],
            near_upright_capture=angle<pdefs['capture_angle'],high_force=np.abs(actions[:,:,0])>pdefs['high_force'],
            near_rail=np.abs(qstate[:,:,0])>pdefs['near_rail'],
            early_swingup=np.full(angle.shape,record['step']<pdefs['early_steps'])&(angle>.5),
            late_stabilization=np.full(angle.shape,record['step']>=pdefs['early_steps'])&(angle<=.5))
        for label,mask in masks.items():regimes.append(dict(**common,method=name,regime=label,query_steps=int(mask.sum()),
            recursive_nmse=float(e[mask].mean()) if mask.any() else None,teacher_nmse=float(teacher[mask].mean()) if mask.any() else None))
    for meta in METAS:
        for source in ['frozen',meta,'oracle']:
            candidates=np.flatnonzero(origins[:,SOURCES.index(source)])
            for scorer in ['frozen',meta,'oracle']:
                selected=int(candidates[values['costs'][names.index(scorer),candidates].argmin()])
                substitutions.append(dict(**common,meta=meta,candidate_source=source,scoring_model=scorer,
                    selected_candidate=selected,true_cost=float(truecost[selected]),predicted_cost=float(values['costs'][names.index(scorer),selected]),
                    normalized_regret=float((truecost[selected]-truecost[best])/den),unsafe=bool(unsafe[selected]),
                    privileged_reference=source=='oracle' or scorer=='oracle',
                    selection_rule='argmin rescoring over all source CEM samples plus iteration elite means; diagonal is not native elite-mean policy'))
    return dict(ranking=one,profiles=profiles,substitutions=substitutions,regimes=regimes)


def analyze_case(ctx,record,dense=False,small=False,actions_override=None):
    key=record['id']+('_dense' if dense else '');path=ctx.run/'replay'/f'{key}.json'
    cached=ctx.cached(path)
    if cached is not None:return cached
    start=time.perf_counter();names,models=ctx.fitted_models(record,dense=dense)
    pack,settings=ctx.pack(record['seed'],names,models);pack.physics=ctx.physics(record['environment'])
    cache=ctx.run/'replay'/f'{record["id"]}.npz'
    if actions_override is not None:
        actions,origins,native=actions_override;pool=dict(unique_candidates=len(actions),reused_fixed_query_pool=True)
    elif dense and cache.exists():
        with np.load(cache) as a:actions=a['actions'];origins=a['origins'];native={s:int(a['native_indices'][i]) for i,s in enumerate(SOURCES)}
        pool=json.loads(cache.with_suffix('.json').read_text())['pool'];pool['reused_base_pool']=True
    else:actions,origins,native,pool=generate_pool(ctx,record,pack,names,settings,small)
    values=evaluate_pool(ctx,record,names,models,actions)
    results=summarize(ctx,record,names,values,actions,origins,native)
    exact=next(r for r in results['ranking'] if r['method']=='exact_residual')['exact_residual_max_error']
    if exact>1e-3:raise AssertionError(f'Exact residual/Oracle recursive mismatch {exact}')
    save_npz(path.with_suffix('.npz'),actions=actions,origins=origins,native_indices=np.asarray([native[s] for s in SOURCES]),
        model_names=np.asarray(names),**{f'model_{n}':v for n,v in pack_arrays(pack).items()},**values)
    packet=dict(config_sha256=ctx.digest,key=key,record=record,pool=pool,**results,wall_seconds=time.perf_counter()-start)
    save_json(path,packet);print(f'REPLAY {key} candidates={len(actions)} seconds={packet["wall_seconds"]:.3f}',flush=True)
    return packet


def fixed_query(ctx,record,small=False):
    path=ctx.run/'fixed_query'/f'{record["id"]}.json'
    cached=ctx.cached(path)
    if cached is not None:return cached
    rows=[];arrays={};first=None;start=time.perf_counter()
    for k in ([0,5,50] if small else ctx.cfg['checkpoints']):
        r=dict(record,support_K=k,id=record['id']+f'_K{k}');names,models=ctx.fitted_models(r)
        if first is None:
            pack,settings=ctx.pack(r['seed'],names,models);pack.physics=ctx.physics(r['environment'])
            actions,origins,native,_=generate_pool(ctx,r,pack,names,settings)
            required=np.unique([*native.values(),*np.flatnonzero(origins[:,4])[:64]])
            rest=np.setdiff1d(np.arange(len(actions)),required);ix=np.r_[required,rest[:max(0,ctx.cfg['fixed_query_candidates']-len(required))]]
            inverse={int(v):j for j,v in enumerate(ix)};first=(actions[ix],origins[ix],{m:inverse[v] for m,v in native.items()})
        actions,origins,native=first;values=evaluate_pool(ctx,r,names,models,actions)
        for row in summarize(ctx,r,names,values,actions,origins,native)['ranking']:
            row.update(fixed_query=True,query_key=record['query_key']);rows.append(row)
        arrays[f'K{k}_costs']=values['costs'];arrays[f'K{k}_paths']=values['paths']
    save_npz(path.with_suffix('.npz'),actions=first[0],origins=first[1],native_indices=np.asarray([first[2][m] for m in SOURCES]),query_state=np.array(record['state']),**arrays)
    packet=dict(config_sha256=ctx.digest,key=record['id'],rows=rows,wall_seconds=time.perf_counter()-start,
        query_key=record['query_key'],support_key=record['key'],identical_queries_and_actions=True)
    save_json(path,packet);return packet
