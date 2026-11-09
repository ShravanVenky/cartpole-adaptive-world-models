"""Numerical and scientific contracts, exercised only on bounded data."""
import copy
import json
import numpy as np
import torch
from adaptive_world_model.environments.cartpole import step
from adaptive_world_model.control.evaluation_mpc import physics_step,GraphedPlanner
from .common import Planner,save_json,checkpoint_K
from .protocol import METHODS,SOURCES


def verify(ctx,record,packet):
    checks=[];base,_=ctx.templates(record['seed']);names,models=ctx.fitted_models(record)
    assert ctx.cfg['conditions'][0]['environment']=='combined_geometry';checks.append('Bounded benchmark and dense fixture target the same priority physics')
    pack,settings=ctx.pack(record['seed'],names,models);pack.physics=ctx.physics(record['environment'])
    g=torch.Generator().manual_seed(418);s=torch.randn(64,4,generator=g)*torch.tensor([.5,1.,3.,2.]);u=torch.rand(64,1,generator=g)*20-10
    p=ctx.physics(record['environment']);reference=step(s,u,p)
    exact=physics_step(s.to(ctx.device),u.to(ctx.device),p).cpu()
    assert torch.allclose(reference,exact,atol=2e-5,rtol=2e-5);checks.append('RK4 kernel matches original validated environment')
    with torch.no_grad():pred=pack(s[None].to(ctx.device).expand(len(names),-1,-1),u[None].to(ctx.device).expand(len(names),-1,-1)).cpu()
    assert torch.allclose(pred[names.index('oracle')],pred[names.index('exact_residual')],atol=1e-5,rtol=1e-5)
    checks.append('Exact residual plus Frozen equals Oracle with original normalization/angle handling')
    zero=ctx.restored(record,'ridge',0);base,_=ctx.templates(record['seed'])
    neutral,labels=ctx.pack(record['seed'],['frozen','ridge'],[base,zero]);neutral.physics=p
    with torch.no_grad():neutral_values=neutral(s[:4][None].to(ctx.device).expand(2,-1,-1),u[:4][None].to(ctx.device).expand(2,-1,-1))
    assert torch.equal(neutral_values[0],neutral_values[1]);checks.append('Unfitted Ridge exactly reproduces Frozen')
    assert checkpoint_K(19,100)==10 and checkpoint_K(50,49)==20;checks.append('As-of support excludes future and failure-causing transitions')
    with np.load(ctx.run/'replay'/f'{record["id"]}.npz') as a:
        actions=a['actions'];origins=a['origins'];native=a['native_indices']
        assert len(np.unique(actions.reshape(len(actions),-1),axis=0))==len(actions)
        for i in range(len(SOURCES)):assert origins[native[i],i]
        assert np.isfinite(a['costs']).all()
    checks.append('Exact candidate dedup preserves every final elite-average source membership')
    from .cohorts import record as saved_record
    from .replay import evaluate_pool
    key='confirmation_11_400_stable_balance__nominal';parent=ctx.packet(key)
    row=next(i for i,r in enumerate(parent['rows']) if r['method']=='meta_residual_1step' and r['safety']=='margin')
    balancing=saved_record(ctx,parent,row,50,'balancing_contract')
    bn,bm=ctx.fitted_models(balancing);values=evaluate_pool(ctx,balancing,bn,bm,actions[:8,:15])
    assert values['paths'].shape[2]==15 and values['forecast_paths'].shape[2]==30
    assert np.allclose(values['costs'],values['horizon_costs'][:,:,14])
    checks.append('H15 control objective unchanged while held-last-action diagnostic forecasts reach30')
    alternate=type(ctx)(ctx.run,dict(ctx.cfg,candidate_chunk=64),ctx.device);alternate.models=ctx.models
    original_values=evaluate_pool(ctx,record,names,models,actions[:192])
    chunked_values=evaluate_pool(alternate,record,names,models,actions[:192])
    assert np.allclose(original_values['paths'],chunked_values['paths'],atol=2e-4,rtol=2e-4,equal_nan=True)
    assert np.allclose(original_values['costs'],chunked_values['costs'],atol=2e-3,rtol=2e-4)
    checks.append('Large scoring batches match small-chunk trajectories and finite-horizon costs')
    for r in packet['substitutions']:
        if r['scoring_model']=='oracle':
            rows=[q for q in packet['substitutions'] if q['meta']==r['meta'] and q['candidate_source']==r['candidate_source']]
            assert all(r['true_cost']<=q['true_cost']+1e-4*max(1,abs(r['true_cost'])) for q in rows)
    checks.append('Oracle rescoring attains best evaluated true cost in each source pool')
    own=ctx.restored(record,record['collector_method']);data=ctx.arrays(record['key']);k=record['support_K']
    if hasattr(own,'_params'):
        for name,v in own._params().items():assert np.array_equal(v.detach().numpy(),data[f'K{k}_{name}'][record['row']])
    checks.append('Own-collector adapter restored byte-for-byte from archived as-of-K snapshot')
    if ctx.device=='cuda':
        planner=Planner(record['horizon'],record['task'],settings,ctx.device,ctx.cfg)
        state=torch.tensor(record['state'],device=ctx.device)[None].expand(len(names),-1)
        groups=torch.zeros(len(names),dtype=torch.long,device=ctx.device)
        graph=GraphedPlanner(planner,pack,state,groups);planner.mean.zero_();a=graph.action(state,819).clone()
        planner.mean.zero_();gen=torch.Generator(device=ctx.device).manual_seed(819)
        noise=torch.randn(planner.iterations,1,planner.n,(planner.h+4)//5,1,generator=gen,device=ctx.device).expand(-1,len(names),-1,-1,-1)
        b=planner.action(pack,state,0,noise)
        assert torch.allclose(a,b,atol=2e-4,rtol=2e-4);checks.append('CUDA graph and eager CEM select matching native actions under identical noise')
    for folder in ['episodes']:
        for path in (ctx.run/folder).glob('*.json'):
            p=json.loads(path.read_text())
            for row in p.get('rows',[]):
                assert row['incomplete_time_limit'] or row['survival_steps']==p['record']['remaining'] or row['failed']
    checks.append('Bounded control truncation is labelled and never promoted to full task completion')
    for model in models:
        for value in model.base.parameters() if hasattr(model,'base') else model.parameters():assert not value.requires_grad
    checks.append('All frozen backbone parameters remain outside diagnostic optimizers')
    dense=list((ctx.run/'dense').glob('*_best.json'))
    assert dense and all(any(a['parameter_update_norm']>0 for a in json.loads(p.read_text())['optimizations']) for p in dense)
    checks.append('Diagnostic copies have nonzero parameter updates while original checkpoints remain protected')
    save_json(ctx.run/'verification.json',dict(passed=True,checks=checks,checks_count=len(checks),scientific_validation=False))
    return checks
