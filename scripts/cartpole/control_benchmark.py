import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import torch
from control_evaluate import load_models,RUN
from adaptive_world_model.control.batched_predictor import BatchedPredictor
from adaptive_world_model.control.evaluation_mpc import EvaluationCEM,GraphedPlanner

torch.set_num_threads(1);base,models=load_models(11);records=[]
for device in ['cpu','cuda']:
    for batch in [12,48]:
        pack=BatchedPredictor(base,[models[x] for x in list(models) for _ in range(batch//4)],device)
        state=torch.zeros(batch,4,device=device);state[:,2]=.05
        cem=EvaluationCEM(15,'balance',batch,device=device)
        graph=GraphedPlanner(cem,pack,state) if device=='cuda' else None
        start=time.perf_counter()
        for i in range(20):
            action=graph.action(state,i) if graph else cem.action(pack,state,i)
        if device=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        records.append({'device':device,'batch':batch,'actions':20,'seconds':seconds,'environment_steps_per_second':batch*20/seconds})
        print(records[-1],flush=True)
import sys
(RUN/'throughput_benchmark.json').write_text(json.dumps(records,indent=2))
if '--concurrent' not in sys.argv:sys.exit(0)
pack=BatchedPredictor(base,[base]*12,'cuda');state=torch.zeros(12,4,device='cuda')
plans=[GraphedPlanner(EvaluationCEM(15,'balance',12,device='cuda'),pack,state) for _ in range(2)]
for concurrent in [False,True]:
    streams=[torch.cuda.Stream(),torch.cuda.Stream()];start=time.perf_counter()
    for i in range(20):
        for j,plan in enumerate(plans):
            if concurrent:
                with torch.cuda.stream(streams[j]):plan.action(state,i)
            else:plan.action(state,i)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    records.append({'two_jobs_concurrent_streams':concurrent,'seconds':seconds,'actions':40})
(RUN/'throughput_benchmark.json').write_text(json.dumps(records,indent=2))
