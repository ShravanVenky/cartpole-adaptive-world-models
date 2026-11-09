"""Bounded smoke contracts on real loaded checkpoints, not a new test suite."""
import time
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import EvaluationCEM,GraphedPlanner,physics_step
from adaptive_world_model.environments.cartpole import CartPolePhysics,step
from .protocol import METHODS,setting
from .engine import Predictor,Planner
from .context import save_json


def contracts(ctx):
    start=time.perf_counter();cfg=ctx.cfg;device=ctx.device
    base,_=ctx.templates(11)
    c=cfg['conditions'][0]
    support=ctx.support(11,c,900,'consecutive',50)
    settings=[setting(m) for m in METHODS]
    _,models,_=ctx.deploy(11,settings,[support]*len(settings))
    pack=Predictor(base,models,settings,device);pack.physics=CartPolePhysics()
    torch.manual_seed(71)
    state=torch.randn(5,4,device=device)*.1;action=torch.randn(5,1,device=device)
    with torch.no_grad():
        out=pack(state,action).cpu()
        for i,(s,m) in enumerate(zip(settings,models)):
            expected=step(state[i:i+1].cpu(),action[i:i+1].cpu(),pack.physics) if s['method']=='oracle' else state[i:i+1].cpu()+m.predict_delta(state[i:i+1].cpu(),action[i:i+1].cpu())
            if not torch.allclose(out[i:i+1],expected,atol=2e-6,rtol=2e-6):raise AssertionError('alpha1 prediction mismatch '+s['method'])
        pack.alpha.zero_();zero=pack(state,action);frozen=pack.frozen(state,action)
        if not torch.equal(zero[:-1],frozen[:-1]):raise AssertionError('alpha0 does not equal Frozen')
        if not torch.equal(physics_step(state,action,pack.physics),step(state,action,pack.physics)):raise AssertionError('RK4 differs')
    # Exact baseline planner behavior under identical noise; changed objectives remain isolated.
    opts=[setting('oracle')]*2;s=state[:2]
    planner=Planner(15,'balance',opts,device,cfg);legacy=EvaluationCEM(15,'balance',2,device=device)
    predict=lambda x,u:physics_step(x,u,CartPolePhysics())
    noise=torch.randn(3,2,128,3,1,device=device)
    got=planner.action(predict,s,1,noise);expected=legacy.action(predict,s,1,noise)
    if not torch.allclose(got,expected,atol=1e-6,rtol=1e-6):raise AssertionError('Baseline CEM drift')
    if device=='cuda':
        planner.mean.zero_();graph=GraphedPlanner(planner,predict,s,torch.tensor([0,1],device=device))
        got=graph.action(s,91);torch.cuda.synchronize();noise=graph.noise.clone();planner.mean.zero_()
        expected=planner.action(predict,s,91,noise)
        if not torch.allclose(got,expected,atol=1e-6,rtol=1e-6):raise AssertionError('Graph replay drift')
    save_json(ctx.run/'smoke_contracts.json',dict(alpha_zero_exact=True,alpha_one_matches_established_models=True,
               unchanged_baseline_cem=True,physics_reference_exact=True,graph_eager_parity=True,
               no_training=True,original_base_frozen=base.assert_frozen() is None))
    return time.perf_counter()-start
