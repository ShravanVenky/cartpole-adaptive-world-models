"""Isolated same-architecture capacity references. Never changes original models or checkpoints."""
import copy
import json
import time
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import physics_step
from adaptive_world_model.environments.cartpole import observable_features
from adaptive_world_model.utils.randomness import stable_seed
from .common import atomic_torch,save_json,save_npz,setting
from .protocol import METAS


def dataset(ctx,seed,environment,small=False):
    path=ctx.run/'dense'/f'{seed}_{environment}_data.npz'
    if path.exists():
        with np.load(path) as a:return {k:torch.from_numpy(a[k]).to(ctx.device) for k in a.files}
    count=(512 if small else ctx.cfg['dense_train_count']);val=(128 if small else ctx.cfg['dense_validation_count'])
    base,_=ctx.templates(seed);base=copy.deepcopy(base).to(ctx.device);base.assert_frozen()
    generator=torch.Generator(device=ctx.device).manual_seed(stable_seed(seed,'dense_development_data',environment,ctx.cfg['dense_seed']))
    region_s=[];region_a=[]
    for p in sorted((ctx.run/'replay').glob('dev_*.json')):
        packet=json.loads(p.read_text());r=packet['record']
        if r['seed']!=seed or r['environment']!=environment or p.stem.endswith('_dense'):continue
        with np.load(p.with_suffix('.npz')) as a:
            names=list(a['model_names']);truth=a['paths'][names.index('oracle')];actions=a['actions']
            states=np.concatenate([np.tile(r['state'],(len(actions),1,1)),truth[:,:-1]],1).reshape(-1,4);us=actions.reshape(-1,1)
            valid=np.isfinite(states).all(1)&(np.abs(states[:,0])<2.4)&(np.abs(states[:,1])<8)&(np.abs(states[:,3])<16)
            region_s.append(states[valid][::4]);region_a.append(us[valid][::4])
    if not region_s:raise RuntimeError(f'Dense fitting requires development replay regions for {seed}/{environment}; run replay first')
    rs=torch.as_tensor(np.concatenate(region_s),device=ctx.device,dtype=torch.float32)
    ra=torch.as_tensor(np.concatenate(region_a),device=ctx.device,dtype=torch.float32)
    def uniform(n):
        span=torch.tensor([2.2,5.,np.pi,12.],device=ctx.device)
        s=(torch.rand(n,4,generator=generator,device=ctx.device)*2-1)*span
        u=torch.rand(n,1,generator=generator,device=ctx.device)*20-10
        return s,u
    s,u=uniform(count);ix=torch.randint(len(rs),(count//2,),generator=generator,device=ctx.device)
    s[:count//2]=rs[ix];u[:count//2]=ra[ix]
    vs,vu=uniform(val)
    with torch.no_grad():
        def encode(s,u):
            x=base.normalizer.encode_x(observable_features(s,u));target=base.normalizer.encode_y(physics_step(s,u,ctx.physics(environment))-s)-base.network(x)
            return x,target
        x,y=encode(s,u);vx,vy=encode(vs,vu)
    arrays=dict(x=x,y=y,validation_x=vx,validation_y=vy,states=s,actions=u,validation_states=vs,validation_actions=vu)
    save_npz(path,**{k:v.cpu().numpy() for k,v in arrays.items()})
    save_json(path.with_suffix('.json'),dict(seed=seed,environment=environment,training_observations=count,validation_observations=val,
        mpc_region_observations=count//2,uniform_observations=count-count//2,development_region_pool=len(rs),
        output_clip_target_fraction=float((y.abs()>8).float().mean()),training_role='Extensive diagnostic data; never a few-shot baseline',
        confirmation_data_used=False,normalization='Original frozen nominal normalizer',physics_used_only_for_labels=True))
    return arrays


def fit(ctx,seed,environment,method,small=False,updates_override=None):
    target=ctx.run/'dense'/f'{seed}_{environment}_{method}_best.json'
    if target.exists():return json.loads(target.read_text())
    data=dataset(ctx,seed,environment,small);template=ctx.make(seed,setting(method));initial=copy.deepcopy(template.adapter).to(ctx.device)
    assert initial.parameter_count==452
    steps=updates_override or (12 if small else ctx.cfg['dense_updates']);check=min(ctx.cfg['dense_check_every'],steps)
    total_start=time.perf_counter();runs=[];winner=None;best_value=float('inf')
    for restart,lr in enumerate(ctx.cfg['dense_learning_rates']):
        path=ctx.run/'dense'/f'{seed}_{environment}_{method}_restart{restart}.pt'
        adapter=copy.deepcopy(initial);optimizer=torch.optim.Adam(adapter.parameters(),lr=lr)
        gen=torch.Generator(device=ctx.device).manual_seed(stable_seed(seed,'dense_updates',environment,method,restart))
        history=[];offset=0;elapsed=0.;best=None;value=float('inf');best_step=0
        if path.exists():
            prior=torch.load(path,map_location=ctx.device,weights_only=True)
            adapter.load_state_dict(prior['adapter']);optimizer.load_state_dict(prior['optimizer']);gen.set_state(prior['rng'].cpu())
            offset=prior['step'];history=prior['history'];elapsed=prior['elapsed_seconds'];best=prior['best'];value=prior['best_validation'];best_step=prior['best_step']
        def loss(x,y):return (adapter(x).clamp(-8,8)-y).square().mean()
        with torch.no_grad():initial_val=float(loss(data['validation_x'],data['validation_y']))
        if best is None:
            best={n:v.detach().cpu().clone() for n,v in adapter.state_dict().items()};value=initial_val
        tick=time.perf_counter()
        for step in range(offset,steps):
            ix=torch.randint(len(data['x']),(min(ctx.cfg['dense_batch_size'],len(data['x'])),),generator=gen,device=ctx.device)
            objective=loss(data['x'][ix],data['y'][ix]);optimizer.zero_grad(set_to_none=True);objective.backward()
            if not torch.isfinite(objective):raise RuntimeError(f'Nonfinite diagnostic training objective {path}/step{step}')
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),10.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite diagnostic gradient; original checkpoints remain untouched')
            optimizer.step()
            if (step+1)%check==0 or step+1==steps:
                with torch.no_grad():train_loss=float(loss(data['x'],data['y']));vl=float(loss(data['validation_x'],data['validation_y']))
                history.append(dict(step=step+1,train_nmse=train_loss,validation_nmse=vl))
                if vl<value:value=vl;best={n:v.detach().cpu().clone() for n,v in adapter.state_dict().items()};best_step=step+1
                atomic_torch(path,dict(adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),rng=gen.get_state(),step=step+1,
                    history=history,elapsed_seconds=elapsed+time.perf_counter()-tick,best=best,best_validation=value,best_step=best_step))
        if ctx.device=='cuda':torch.cuda.synchronize()
        runtime=elapsed+time.perf_counter()-tick
        adapter.load_state_dict(best)
        with torch.no_grad():train_best=float(loss(data['x'],data['y']))
        parameter_delta=float(torch.linalg.vector_norm(torch.cat([(best[n]-v.detach().cpu()).reshape(-1) for n,v in initial.state_dict().items()])))
        runs.append(dict(restart=restart,learning_rate=lr,updates=steps,best_step=best_step,initial_validation_nmse=initial_val,
            train_nmse=train_best,validation_nmse=value,parameter_update_norm=parameter_delta,elapsed_seconds=runtime,history=history))
        if value<best_value:best_value=value;winner=best
    save_npz(target.with_suffix('.npz'),**{n:v.numpy() for n,v in winner.items()})
    result=dict(seed=seed,environment=environment,method=method,parameter_count=452,diagnostic_only=True,
        frozen_backbone_changed=False,training_count=len(data['x']),validation_count=len(data['validation_x']),
        total_updates=steps*len(runs),validation_selected_nmse=best_value,optimizations=runs,
        wall_seconds=time.perf_counter()-total_start,selection='Development validation only, two fixed learning rates; no final-state or confirmation selection',
        limitation='Failure of both optimization attempts cannot prove representational impossibility')
    save_json(target,result);print(f'DENSE {seed} {environment} {method} updates={result["total_updates"]} seconds={result["wall_seconds"]:.2f}',flush=True)
    return result
