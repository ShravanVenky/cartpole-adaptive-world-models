"""Identical support, independent queries and content-cached shared action pools."""
import time
import numpy as np
import torch
from mismatch.replay import generate_pool, evaluate_pool, array_error, rank_correlation
from .common import digest, setting, save_json, save_npz, coverage
from .optimization import fit
from .protocol import METAS, optimizer


def pool(ctx,record,support):
    path=ctx.run/'candidate_pools'/f'{record["id"]}.json'
    cached=ctx.cached_job(path)
    if cached:
        with np.load(path.with_suffix('.npz')) as a: return a['actions'].copy(),a['origins'].copy(),cached
    names=['frozen',*METAS,'oracle']; models=[]
    for method in names:
        if method in METAS: m,_=fit(ctx,record['seed'],method,support,optimizer())
        else: m=ctx.make(record['seed'],setting(method))
        models.append(m)
    pack,settings=ctx.pack(record['seed'],names,models);pack.physics=ctx.physics(record['environment'])
    tick=time.perf_counter();actions,origins,native,info=generate_pool(ctx,record,pack,names,settings)
    packet=ctx.save_job(path,dict(record=record,pool=info,native_indices=native,wall_seconds=time.perf_counter()-tick,
        support_hash=support.support_hash(),warm_start='common zero; diagnostic replay'),actions=actions,origins=origins)
    return actions,origins,packet


def evaluate(ctx,record,support,variants,tag):
    """variants=(method, optimizer settings); scoring includes common Frozen/Oracle."""
    identity=digest(dict(record=record,support=support.support_hash(),variants=variants,tag=tag))[:24]
    path=ctx.run/'offline'/f'{tag}_{identity}.json'
    cached=ctx.cached_job(path)
    if cached: return cached
    actions,origins,pool_packet=pool(ctx,record,support)
    models=[ctx.make(record['seed'],setting('frozen'))];names=['frozen']; fits=[None]
    for method,config in variants:
        m,p=fit(ctx,record['seed'],method,support,config); models.append(m);names.append(method);fits.append(p)
    models.append(ctx.make(record['seed'],setting('oracle')));names.append('oracle');fits.append(None)
    tick=time.perf_counter();values=evaluate_pool(ctx,record,names,models,actions)
    truth=values['paths'][-1]; truecost=values['costs'][-1]
    best=int(truecost.argmin());den=max(abs(float(truecost[best])),1.)
    base,_=ctx.templates(record['seed']); yscale=base.normalizer.y_scale.cpu().numpy()
    rows=[];selected=[];costs=values['costs'];physicalscale=np.asarray(ctx.cfg['state_scale'])
    for i,(name,model,packet) in enumerate(zip(names,models,fits)):
        teacher=values['teacher'][i];recursive=values['paths'][i];chosen=int(costs[i].argmin());selected.append(recursive[chosen])
        deltaerr=(teacher-truth)/yscale
        row=dict(seed=record['seed'],trial=record['trial'],environment=record['environment'],query_id=record['id'],query_step=record['step'],
            split=record['split'],method=name,optimizer=packet['optimizer']['name'] if packet else name,
            support_K=len(support),support_hash=support.support_hash(),pool_sha256=pool_packet['arrays_sha256'],
            one_step_nmse=float(np.nan_to_num(deltaerr**2,nan=1e20,posinf=1e20,neginf=1e20).mean()),
            first_step_nmse=float(np.nan_to_num(deltaerr[:,0]**2,nan=1e20,posinf=1e20).mean()),
            recursive_5_nmse=float(array_error(recursive[:,:5],truth[:,:5],physicalscale).mean()),
            recursive_30_nmse=float(array_error(recursive,truth,physicalscale).mean()),
            rank_correlation=rank_correlation(costs[i],truecost),top1_agreement=chosen==best,
            top10_overlap=len(set(np.argsort(costs[i])[:10])&set(np.argsort(truecost)[:10]))/10,
            planning_regret=float((truecost[chosen]-truecost[best])/den),
            true_plan_cost=float(truecost[chosen]),predicted_plan_cost=float(costs[i,chosen]),
            false_safe_selected=bool(np.abs(recursive[chosen,:,0]).max()<=2.4 and np.abs(truth[chosen,:,0]).max()>2.4),
            invalid_prediction_fraction=float((~np.isfinite(recursive).all(-1)).mean()),
            correction_rms=float(np.nan_to_num((teacher-values['teacher'][0])**2,nan=1e20).mean()**.5),
            fit_path=packet['fit_path'] if packet else None,
            parameter_delta=packet['log'].get('parameter_update_norm',0.) if packet else 0.,
            accepted=packet['log']['accepted'] if packet else True,
            isolated_diagnostic=packet['optimizer'].get('isolated_diagnostic',False) if packet else False,
            fit_ms=packet['log']['adaptation_seconds']*1000 if packet else 0.,
            support_loss_before=packet['log'].get('support_loss_before') if packet else None,
            support_loss_after=packet['log'].get('support_loss_after') if packet else None)
        rows.append(row)
    return ctx.save_job(path,dict(rows=rows,record=record,wall_seconds=time.perf_counter()-tick,
        coverage=coverage(base,support,np.concatenate([np.tile(record['state'],(len(actions),1,1)),truth[:,:-1]],1).reshape(-1,4)[::8],actions.reshape(-1,1)[::8]),
        tag=tag,heldout_query=True,observation_count=len(support)),
        selected_predictions=np.stack(selected),true_costs=truecost,predicted_costs=costs,
        true_selected_paths=np.stack([truth[int(c.argmin())] for c in costs]),actions=actions,
        method_names=np.asarray(names),query_state=np.asarray(record['state']))
