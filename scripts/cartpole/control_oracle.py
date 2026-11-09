import sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from adaptive_world_model.control.evaluation_mpc import EvaluationCEM,GraphedPlanner,physics_step,stage_cost
from adaptive_world_model.environments.cartpole import step,CartPolePhysics,wrap_angle

def episode(task,h,device='cpu',batch=8,candidates=128,steps=500,disturbance=False):
    p=CartPolePhysics();g=torch.Generator().manual_seed(873)
    s=torch.randn(batch,4,generator=g)*torch.tensor([.1,.1,.06,.1])
    if task=='swingup':s[:,2]+=math.pi
    s=s.to(device);cem=EvaluationCEM(h,task,batch,candidates=candidates,device=device)
    graph=GraphedPlanner(cem,lambda x,a:physics_step(x,a,p),s) if device=='cuda' else None
    cem.mean.zero_();states=[s];actions=[];start=time.perf_counter()
    with torch.no_grad():
        for t in range(steps):
            if disturbance and t==150:s=s+torch.tensor([0,.5,.35,.6],device=device)
            u=graph.action(s,673+t) if graph else cem.action(lambda x,a:physics_step(x,a,p),s,673+t)
            s=physics_step(s,u,p);states.append(s);actions.append(u)
    a=torch.stack(states,1).cpu();th=wrap_angle(a[:,:,2]);good=(th.abs()<.15)&(a[:,:,3].abs()<.8)&(a[:,:,0].abs()<2.4)
    success=good[:,-100:].all(1)&(a[:,:,0].abs()<2.4).all(1)
    return {'task':task,'horizon':h,'disturbance':disturbance,'success':success.tolist(),'upright_fraction':good.float().mean(1).tolist(),'max_x':a[:,:,0].abs().max(1).values.tolist(),'seconds':time.perf_counter()-start,'last_state':a[:,-1].tolist()}

if __name__=='__main__':
    torch.set_num_threads(1)
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task',default='balance');p.add_argument('--horizon',type=int,default=15);p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=8);p.add_argument('--steps',type=int,default=500);p.add_argument('--candidates',type=int,default=128);p.add_argument('--disturbance',action='store_true');args=p.parse_args()
    print(json.dumps(episode(args.task,args.horizon,args.device,args.batch,args.candidates,args.steps,args.disturbance)),flush=True)
