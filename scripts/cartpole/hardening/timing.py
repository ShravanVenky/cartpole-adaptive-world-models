"""Single-controller deployment timing, separate from batched evaluation throughput."""
import time
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import GraphedPlanner
from .common import Planner,setting,digest,save_json,support_from_episode,query_record
from .optimization import fit
from .protocol import METAS,optimizer


def deployment_profile(ctx,selected,bounded=False):
    rows=[];seed=11;support=support_from_episode(seed,'combined_geometry');base,_=ctx.templates(seed)
    configs=[('frozen','frozen',None),('oracle','oracle',None),('ridge','ridge',optimizer())]
    configs += [(m,m,optimizer()) for m in METAS]
    configs += [(m+'__selected',m,selected[m]) for m in METAS]
    if bounded:configs=[r for r in configs if r[0].endswith('__selected')]
    for label,method,config in configs:
        for horizon,task in [(15,'balance'),(30,'swingup')]:
            path=ctx.run/'latency'/f'{digest([label,config,horizon])[:24]}.json'
            stored=ctx.cached_job(path)
            if stored:rows.append(stored['row']);continue
            m=ctx.make(seed,setting(method));fit_ms=0.
            if config is not None:
                m,packet=fit(ctx,seed,method,support,config);fit_ms=1000*packet['log']['adaptation_seconds']
            pack,settings=ctx.predictor(seed,[dict(method=method)],[m],ctx.physics('combined_geometry'))
            planner=Planner(horizon,task,settings,ctx.device,ctx.cfg)
            state=torch.tensor([[.1,.1,.03,.1]] if task=='balance' else [query_record(seed,'combined_geometry')['state']],device=ctx.device)
            graph=GraphedPlanner(planner,pack,state,torch.zeros(1,dtype=torch.long,device=ctx.device)) if ctx.device=='cuda' else None
            plan_times=[];inference_times=[]
            with torch.no_grad():
                for repeat in range(5 if bounded else 20):
                    if ctx.device=='cuda':torch.cuda.synchronize()
                    tick=time.perf_counter()
                    action=graph.action(state,8090+repeat) if graph else planner.action(pack,state,8090+repeat)
                    if ctx.device=='cuda':torch.cuda.synchronize()
                    plan_times.append((time.perf_counter()-tick)*1000)
                    tick=time.perf_counter();pack(state,action)
                    if ctx.device=='cuda':torch.cuda.synchronize()
                    inference_times.append((time.perf_counter()-tick)*1000)
            row=dict(label=label,method=method,horizon=horizon,batch=1,seed=seed,
                observations=50 if config is not None else 0,requested_updates=config['steps'] if method in METAS else 0,
                fit_ms=fit_ms,mpc_median_ms=float(np.median(plan_times)),mpc_p95_ms=float(np.quantile(plan_times,.95)),
                inference_median_ms=float(np.median(inference_times)),inference_p95_ms=float(np.quantile(inference_times,.95)),
                fit_frame_estimate_ms=fit_ms+float(np.quantile(plan_times,.95))+float(np.quantile(inference_times,.95)),
                sample_count=len(plan_times),sensor_actuator_IO_included=False,
                note='Representative one-controller timings, not a certified real-time deadline test')
            ctx.save_job(path,dict(row=row),planning_ms=np.asarray(plan_times),inference_ms=np.asarray(inference_times));rows.append(row)
    save_json(ctx.run/'deployment_latency.json',dict(rows=rows,bounded=bounded,scientific_validation=False))
    return rows
