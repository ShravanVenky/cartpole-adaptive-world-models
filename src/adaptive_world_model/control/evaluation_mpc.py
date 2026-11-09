"""Shared batched CEM for control evaluation. No learning or physical inputs in costs."""
import math
import torch
from ..environments.cartpole import CartPolePhysics, _derivative


def physics_step(s, u, p):
    """Unchecked inner kernel, identical RK4 arithmetic to the validated simulator."""
    force = u.squeeze(-1).clamp(-p.force_limit, p.force_limit) * p.motor_gain
    dt = p.dt / p.rk4_substeps
    for _ in range(p.rk4_substeps):
        a = _derivative(s, force, p)
        b = _derivative(s + dt*.5*a, force, p)
        c = _derivative(s + dt*.5*b, force, p)
        d = _derivative(s + dt*c, force, p)
        s = s + dt/6*(a+2*b+2*c+d)
    return s


def stage_cost(s, u, task):
    x,v,t,w=s.unbind(-1)
    angle=torch.atan2(t.sin(), t.cos())
    if task=='balance':
        return 2*x.square()+v.square()+40*angle.square()+8*(w+3*angle+.25*x+.5*v).square()+.002*u.squeeze(-1).square()
    capture=torch.exp(-angle.square()/.25)
    return .2*x.square()+.03*v.square()+15*(1-t.cos())+.05*w.square()+.002*u.squeeze(-1).square()+capture*(40*angle.square()+8*(w+3*angle+.25*x+.5*v).square())


class EvaluationCEM:
    def __init__(self,horizon,task,batch,candidates=128,iterations=3,elites=16,device='cpu'):
        self.h=horizon; self.task=task; self.batch=batch; self.n=candidates
        self.iterations=iterations; self.elites=elites; self.device=device
        self.mean=torch.zeros(batch,horizon,1,device=device)
        # Correlated candidate controls: constant blocks of five simulator steps.
        self.block=min(5,horizon)

    @torch.no_grad()
    def action(self,predict,state,seed,noise=None):
        b,n,h=self.batch,self.n,self.h
        gen=torch.Generator(device=self.device).manual_seed(seed) if noise is None else None
        mean=self.mean; std=torch.full_like(mean,6.)
        for iteration in range(self.iterations):
            z=torch.randn(b,n,math.ceil(h/self.block),1,generator=gen,device=self.device) if noise is None else noise[iteration]
            z=z.repeat_interleave(self.block,dim=2)[:,:,:h]
            actions=(mean[:,None]+std[:,None]*z).clamp(-10,10)
            actions[:,0]=mean
            current=state[:,None].expand(b,n,4); cost=torch.zeros(b,n,device=self.device)
            first=None
            for t in range(h):
                current=predict(current,actions[:,:,t])
                if first is None: first=current
                cost+=stage_cost(current,actions[:,:,t],self.task)
                cost+=1e5*torch.relu(current[...,0].abs()-2.4).square()
            # Same terminal objective for all predictive models within each task.
            cost+=5*stage_cost(current,torch.zeros_like(actions[:,:,0]),self.task)
            cost=torch.nan_to_num(cost,nan=1e30,posinf=1e30,neginf=1e30)
            ix=cost.topk(self.elites,dim=1,largest=False).indices
            elite=actions.gather(1,ix[:,:,None,None].expand(b,self.elites,h,1))
            mean=elite.mean(1); std=elite.std(1,unbiased=False).clamp_min(.3)
        self.mean.copy_(torch.cat([mean[:,1:],mean[:,-1:]],dim=1))
        return mean[:,0]


class GraphedPlanner:
    """Capture unchanged CEM arithmetic; noise and states remain explicit inputs."""
    def __init__(self,cem,predict,state,groups=None):
        self.cem=cem;self.state=state.clone()
        self.noise=torch.empty(cem.iterations,cem.batch,cem.n,math.ceil(cem.h/cem.block),1,device=state.device)
        self.groups=groups
        self.source_noise=torch.empty(cem.iterations,int(groups.max().item())+1,cem.n,math.ceil(cem.h/cem.block),1,device=state.device) if groups is not None else self.noise
        self.noise.normal_();stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):cem.action(predict,self.state,0,self.noise)
        stream.synchronize();cem.mean.zero_();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=cem.action(predict,self.state,0,self.noise)
        cem.mean.zero_()

    def action(self,state,seed):
        self.state.copy_(state)
        gen=torch.Generator(device=state.device).manual_seed(seed);self.source_noise.normal_(generator=gen)
        if self.groups is not None:self.noise.copy_(self.source_noise.index_select(1,self.groups))
        self.graph.replay()
        return self.output.clone()
