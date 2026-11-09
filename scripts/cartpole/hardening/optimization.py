"""Instrumented task-local SGD/Adam, preserving every existing numerical safeguard."""
import time
from collections import OrderedDict
from dataclasses import asdict
import numpy as np
import torch
from torch.func import functional_call
from adaptive_world_model.environments.cartpole import observable_features
from mismatch.common import setting
from .common import digest, restore_model
from .protocol import METAS


def optimize(model,support,config):
    """No module/checkpoint mutation. Whole-call rejection rolls back to phi_0."""
    model.reset_adaptation(); base=model.base; base.assert_frozen()
    reference=next(model.adapter.parameters()); device=reference.device
    if reference.is_cuda: torch.cuda.synchronize()
    tick=time.perf_counter(); support=support.to(device)
    anchor=OrderedDict((n,p.detach().clone()) for n,p in model.adapter.named_parameters())
    params=OrderedDict((n,p.clone().requires_grad_(True)) for n,p in anchor.items())
    settings=model.settings
    with torch.no_grad():
        x=base.normalizer.encode_x(observable_features(support.state,support.action))
        target=base.normalizer.encode_y(support.next_state-support.state)-base.network(x)
    def loss(p): return (functional_call(model.adapter,p,(x,)).clamp(-settings.residual_output_clip,settings.residual_output_clip)-target).square().mean()
    def norm(values): return torch.linalg.vector_norm(torch.cat([v.reshape(-1) for v in values]))
    losses=[float(loss(params).detach())]; gradients=[]; steps=[]; displacements=[]
    reason=None; allowed=min(settings.max_support_loss,settings.loss_growth_factor*losses[0]+1e-6)
    first={n:torch.zeros_like(p) for n,p in params.items()}; second={n:torch.zeros_like(p) for n,p in params.items()}
    clipped=0; bounded=0
    if not np.isfinite(losses[0]) or losses[0]>settings.max_support_loss: reason='initial_support_loss_bound'
    for t in range(config['steps'] if reason is None else 0):
        with torch.enable_grad():
            value=loss(params); gs=torch.autograd.grad(value,tuple(params.values()))
        if not all(bool(torch.isfinite(g).all()) for g in gs): reason='nonfinite_gradient'; break
        gn=norm(gs); gradients.append(float(gn)); factor=1.
        if config['gradient_clip'] is not None:
            factor=(gn.new_tensor(config['gradient_clip'])/gn.clamp_min(1e-12)).clamp_max(1.)
            clipped+=int(float(factor)<1.)
        ds=[]
        for (name,p),g in zip(params.items(),gs):
            g=g.detach()*factor
            if config['algorithm']=='adam':
                first[name]=.9*first[name]+.1*g; second[name]=.999*second[name]+.001*g.square()
                g=(first[name]/(1-.9**(t+1)))/((second[name]/(1-.999**(t+1))).sqrt()+1e-8)
            ds.append(config['learning_rate']*g)
        sn=norm(ds); sf=(sn.new_tensor(config['step_norm_bound'])/sn.clamp_min(1e-12)).clamp_max(1.)
        bounded+=int(float(sf)<1.); ds=[d*sf for d in ds]
        candidate=OrderedDict((n,(p-d).detach().requires_grad_(True)) for (n,p),d in zip(params.items(),ds))
        movement=float(norm([candidate[n]-anchor[n] for n in candidate]).detach())
        nextloss=float(loss(candidate).detach())
        if not np.isfinite(nextloss) or not all(bool(torch.isfinite(p).all()) for p in candidate.values()): reason='nonfinite_candidate'; break
        if movement>config['max_parameter_delta']: reason='parameter_update_norm_bound'; break
        if nextloss>allowed: reason='support_loss_growth_bound'; break
        params=candidate; losses.append(nextloss); steps.append(float(norm(ds))); displacements.append(movement)
    if reason is None:
        with torch.no_grad():
            delta=base.normalizer.decode_y(base.network(x)+functional_call(model.adapter,params,(x,)).clamp(-settings.residual_output_clip,settings.residual_output_clip))
        if not bool(torch.isfinite(delta).all()) or float(delta.abs().max())>settings.max_prediction_magnitude: reason='prediction_magnitude_bound'
    accepted=reason is None
    if accepted: model._adapted_params=OrderedDict((n,p.detach()) for n,p in params.items())
    else: model.reset_adaptation()
    if reference.is_cuda: torch.cuda.synchronize()
    log=dict(accepted=accepted,reason=reason,rolled_back=not accepted,gradient_steps=len(steps),
             steps_attempted=len(gradients),requested_updates=config['steps'],support_count=len(support),
             support_losses=losses,gradient_norms=gradients,step_norms=steps,parameter_displacements=displacements,
             parameter_update_norm=displacements[-1] if accepted and displacements else 0.,
             rejected_parameter_delta=movement if not accepted and gradients else None,
             support_loss_before=losses[0],support_loss_after=losses[-1] if accepted else losses[0],
             clipping_count=clipped,step_bound_count=bounded,adaptation_seconds=time.perf_counter()-tick,
             max_gradient_norm=max(gradients,default=0.),max_step_norm=max(steps,default=0.),
             optimizer_state_persisted=False,reset_to_initialization=True,config=config,device=str(device))
    base.assert_frozen(); return model,log


def compact(packet):
    """Full iteration traces remain in the single checksummed fit job, not duplicated per control."""
    if packet is None:return None
    return {**packet,'log':{k:v for k,v in packet['log'].items() if k not in
        ['support_losses','gradient_norms','step_norms','parameter_displacements']}}


def fit(ctx,seed,method,support,config):
    signature=dict(seed=seed,method=method,support_hash=support.support_hash(),optimizer=config)
    path=ctx.run/'fits'/f'{digest(signature)[:24]}.json'
    cached=ctx.cached_job(path)
    if cached is not None: return restore_model(ctx,seed,method,path),cached
    model=ctx.make(seed,setting(method))
    if method in ['frozen','oracle']: log=dict(accepted=True,gradient_steps=0,steps_attempted=0,adaptation_seconds=0.)
    elif method=='ridge':
        log=model.adapt(support);log.update(steps_attempted=0,gradient_steps=0,config={'algorithm':'closed_form_ridge'})
    else: model,log=optimize(model,support,config)
    arrays={'coefficients':model.coefficients.numpy().copy()} if method=='ridge' else (
        {n:p.detach().cpu().numpy().copy() for n,p in model._params().items()} if method in METAS else {'reference':np.zeros(1)})
    packet=ctx.save_job(path,dict(**signature,log=log,fit_path=str(path.relative_to(ctx.run))),**arrays)
    return model,packet


def audit(ctx):
    rows=[]
    for seed in ctx.cfg['model_seeds']:
        base,templates=ctx.templates(seed); base.assert_frozen()
        for method in METAS:
            m=templates[method]
            rows.append(dict(seed=seed,method=method,parameters=m.parameter_count,settings=asdict(m.settings)))
    expected=dict(learning_rate=.01,steps=10,gradient_clip=1.,step_norm_bound=.1,max_parameter_delta=2.)
    verified=all(all(row['settings'][k]==v for k,v in expected.items()) for row in rows)
    if not verified:raise RuntimeError('Checkpoint optimizer settings differ from requested baseline; inspect optimizer audit before continuing')
    return dict(models=rows,optimizer='functional plain gradient descent; no momentum',
        all_checkpoint_defaults_verified=verified,
        gradient_clip='one norm over all adapter gradient tensors',per_step_guard=.1,total_delta_guard=2.,
        reset='Caller resets before each cumulative K fit; MetaResidualModel.adapt by itself starts from current adapted parameters.',
        gradients='torch.autograd.grad each step; no accumulation; no outer Hessian graph at deployment',
        rejection='Entire fitting call rolls back; no partially accepted prefix installed on rejection',
        path_length_bound='10 * min(0.01*1,0.1) = 0.1; actual net displacement may be smaller',
        adam_mismatch='Pretrained Meta inner loop is SGD. Adam deployment is a labelled optimizer mismatch diagnostic.',
        detector='No active CartPole detector. Known shift schedule; residual alarms unavailable, never fabricated.')
