"""Joint training/support/candidate/visited distributions and error localization."""
import json
import time
import numpy as np
import pandas as pd
import torch
from adaptive_world_model.environments.cartpole import observable_features, CartPolePhysics
from adaptive_world_model.control.evaluation_mpc import physics_step
from .context import save_json, save_npz
from .protocol import HISTORICAL, ADAPTERS, METHODS, setting
from .engine import Predictor, Planner, rollout
from .replay import historical_support, rankings


@torch.no_grad()
def run_coverage(ctx, records):
    started=time.perf_counter();rows=[]
    for rec in records:
        target=ctx.run/'coverage'/f'{rec["id"]}.json'
        if target.exists():
            p=json.loads(target.read_text())
            if p['config_sha256']!=ctx.digest:raise RuntimeError('coverage cache mismatch')
            rows.extend(p['rows']);continue
        packet=json.loads((ctx.run/'replay'/f'{rec["id"]}.json').read_text())
        support=historical_support(rec)
        base,_=ctx.templates(rec['seed'])
        train=ctx.training_reference(rec['seed'])
        ref=base.normalizer.encode_x(train.features()).to(ctx.device)
        sr=base.normalizer.encode_x(support.features()).to(ctx.device) if support is not None else None
        settings=[setting(m) for m in METHODS]
        _,models,_=ctx.deploy(rec['seed'],settings,[support]*len(settings))
        pack=Predictor(base,models,settings,ctx.device)
        p=ctx.physics(rec['source']['environment'])
        if rec['source']['protocol']=='online' and rec['step']<25:p=CartPolePhysics()
        pack.physics=p
        with np.load(ctx.run/'replay'/f'{rec["id"]}.npz') as data:
            # Deterministic thinning; each model's own forecast distribution and common Oracle distribution.
            candidates=data['candidates'];paths=data['predicted_paths'];truth=data['true_paths']
            sources=[('training',train.state.numpy(),train.action.numpy()),
                     ('oracle_candidates',truth[:,:-1].reshape(-1,4),candidates[:,1:].reshape(-1,1))]
            if support is not None:sources.append(('support',support.state.numpy(),support.action.numpy()))
            for method in METHODS[:-1]:
                index=next(i for i,s in enumerate(packet['rows']) if s['method']==method and s['alpha']==(0 if method=='frozen' else 1))
                sources.append((method+'_candidates',paths[index,:,:-1].reshape(-1,4),candidates[:,1:].reshape(-1,1)))
        with np.load(HISTORICAL/'episodes'/f'{rec["source_key"]}.npz') as old:
            ix=rec['source_row'];end=rec['source']['survival_steps']
            sources.append(('visited',old['states'][ix,:end],old['actions'][ix,:end]))
        out=[]
        for source,s,u in sources:
            good=np.isfinite(s).all(1)&np.isfinite(u).all(1)
            s,u=s[good],u[good]
            ix=np.linspace(0,len(s)-1,min(len(s),256),dtype=int)
            s,u=torch.tensor(s[ix],device=ctx.device),torch.tensor(u[ix],device=ctx.device)
            z=observable_features(s,u);zn=pack.base.normalizer.encode_x(z)
            distance=torch.cdist(zn,ref).amin(1)
            sd=torch.cdist(zn,sr).amin(1) if sr is not None else torch.full((len(s),),float('nan'),device=ctx.device)
            sb=s[None].expand(len(settings),-1,-1);ub=u[None].expand(len(settings),-1,-1)
            pred=pack(sb,ub);frozen=pack.frozen(sb,ub);actual=physics_step(s,u,p)
            if source=='visited' and rec['source']['protocol']=='online':
                actual=torch.where(torch.tensor(ix<25,device=ctx.device)[:,None],physics_step(s,u,CartPolePhysics()),actual)
                pred[-1]=actual
            error=pred-actual[None];error[...,2]=torch.atan2(error[...,2].sin(),error[...,2].cos())
            err=(error/base.normalizer.y_scale.to(ctx.device)).square().mean(-1)
            correction=((pred-frozen)/base.normalizer.y_scale.to(ctx.device)).square().mean(-1).sqrt()
            feature=z.cpu().numpy();dist=distance.cpu().numpy();supp=sd.cpu().numpy()
            for m,method in enumerate(METHODS):
                for j in range(len(s)):
                    out.append(dict(state_id=rec['id'],split=rec['split'],seed=rec['seed'],category=rec['category'],source=source,
                                    environment=rec['source']['environment'],method=method,x=float(feature[j,0]),v=float(feature[j,1]),
                                    sin_theta=float(feature[j,2]),cos_theta=float(feature[j,3]),omega=float(feature[j,4]),
                                    commanded_force=float(feature[j,5]),applied_force=float(feature[j,5]*(1. if source=='visited' and rec['source']['protocol']=='online' and ix[j]<25 else p.motor_gain)),
                                    angle=float(np.arctan2(feature[j,2],feature[j,3])),rail_clearance=float(2.4-abs(feature[j,0])),
                                    training_distance=float(dist[j]),support_distance=float(supp[j]),
                                    one_step_nmse=float(err[m,j]),correction_normalized_rms=float(correction[m,j])))
        save_json(target,dict(config_sha256=ctx.digest,rows=out));rows.extend(out)
    frame=pd.DataFrame(rows);frame.to_csv(ctx.run/'state_action_distribution.csv',index=False)
    save_json(ctx.run/'coverage_status.json',dict(records=len(records),samples=len(rows),wall_seconds=time.perf_counter()-started,
              training_reference='8192 deterministic points from exact original80000; distances are to this finite reference, not density estimates',
              support_cost='All modes select K adaptation observations from the same precollected1024-transition passive pool; the pool acquisition cost is1024 for every mode, not K.'))
    return frame


@torch.no_grad()
def support_rankings(ctx,record):
    """Coverage intervention on the same saved candidate pool, independent of support outcomes."""
    target=ctx.run/'support_replay'/f'{record["id"]}.json'
    cached=ctx.cached(target)
    if cached is not None:return cached
    conditions=[c for c in ctx.cfg['conditions'] if c['environment']==record['source']['environment'] and not c['online'] and c['task']=='swingup']
    if not conditions or record['source']['protocol']=='online':return None
    condition=conditions[0]
    trial=ctx.cfg['development_trials'][0] if record['split']=='development' else ctx.cfg['confirmation_trials'][0]
    settings=[setting(m,'support',support=t,K=k) for t in ctx.cfg['support_types'] for k in ctx.cfg['support_sizes'] for m in ADAPTERS]
    supports=[ctx.support(record['seed'],condition,trial,s['support'],s['K']) for s in settings]
    base,models,logs=ctx.deploy(record['seed'],settings,supports)
    pack=Predictor(base,models,settings,ctx.device);pack.physics=ctx.physics(condition['environment'])
    planner=Planner(30,'swingup',settings,ctx.device,ctx.cfg)
    with np.load(ctx.run/'replay'/f'{record["id"]}.npz') as data:
        candidates=data['candidates'];truth=data['true_paths'];truecost=data['true_costs']
    state=torch.tensor(record['state'],device=ctx.device)[None].expand(len(settings),-1)
    paths,costs=[],[]
    for chunk in torch.tensor(candidates,device=ctx.device).split(ctx.cfg['replay_chunk']):
        p,c=rollout(pack,state,chunk[None].expand(len(settings),-1,-1,-1),planner)
        paths.append(p.cpu().numpy());costs.append(c.cpu().numpy())
    paths,costs=np.concatenate(paths,1),np.concatenate(costs,1)
    rows=[]
    for i,s in enumerate(settings):
        z=base.normalizer.encode_x(supports[i].features())
        design=torch.cat([z,torch.ones_like(z[:,:1])],1).double()
        eigen=torch.linalg.eigvalsh(design.T@design+.1*torch.eye(7))
        coef=models[i].coefficients.detach().numpy() if s['method']=='ridge' else np.zeros((7,4))
        rows.append(dict(s,state_id=record['id'],split=record['split'],seed=record['seed'],environment=condition['environment'],
                         support_hash=supports[i].support_hash(),ridge_coefficient_norm=float(np.linalg.norm(coef)) if s['method']=='ridge' else None,
                         ridge_design_condition=float(eigen[-1]/eigen[0]),fit_log=logs[i],
                         **rankings(costs[i],truecost,paths[i],truth,ctx.cfg['replay_top_k'])))
    save_npz(target.with_suffix('.npz'),predicted_costs=costs,true_costs=truecost,
             preferred_paths=np.stack([paths[i,int(np.argmin(costs[i]))] for i in range(len(settings))]),
             ridge_coefficients=np.stack([m.coefficients.detach().numpy() if s['method']=='ridge' else np.zeros((7,4)) for s,m in zip(settings,models)]))
    packet=dict(config_sha256=ctx.digest,rows=rows);save_json(target,packet);return packet
