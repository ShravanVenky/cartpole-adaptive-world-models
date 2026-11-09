"""Batched deployment parameters; learned base and initialization never updated."""
import copy
import torch


class BatchedPredictor:
    def __init__(self,base,models,device):
        self.base=copy.deepcopy(base).to(device); self.device=device
        self.types=torch.tensor([0 if m is base else 1 if hasattr(m,'coefficients') else 2 for m in models],device=device)
        self.coef=torch.zeros(len(models),7,4,device=device)
        self.weights={}
        for layer,(left,right) in enumerate([(6,16),(16,16),(16,4)]):
            self.weights[f'network.{layer*2}.weight']=torch.zeros(len(models),right,left,device=device)
            self.weights[f'network.{layer*2}.bias']=torch.zeros(len(models),right,device=device)
        self.refresh(models)

    def refresh(self,models):
        for i,m in enumerate(models):
            if self.types[i].item()==1:self.coef[i].copy_(m.coefficients.detach().to(self.device))
            elif self.types[i].item()==2:
                for name,value in m._params().items():self.weights[name][i].copy_(value.detach().to(self.device))

    def __call__(self,s,u):
        squeeze=s.ndim==2
        if squeeze:s=s[:,None];u=u[:,None]
        x,v,t,w=s.unbind(-1)
        z=torch.stack([x,v,t.sin(),t.cos(),w,u.squeeze(-1)],-1)
        norm=self.base.normalizer; z=(z-norm.x_mean)/norm.x_scale
        frozen=self.base.network(z)
        linear=torch.bmm(torch.cat([z,torch.ones_like(z[...,:1])],-1),self.coef)
        residual=z
        for layer in range(3):
            residual=torch.bmm(residual,self.weights[f'network.{layer*2}.weight'].transpose(1,2))+self.weights[f'network.{layer*2}.bias'][:,None]
            if layer<2:residual=residual.tanh()
        correction=torch.where(self.types[:,None,None]==1,linear,torch.where(self.types[:,None,None]==2,residual,0)).clamp(-8,8)
        delta=(frozen+correction)*norm.y_scale+norm.y_mean
        bad=(delta.abs()>50).any(-1)&(self.types[:,None]!=0)
        output=s+delta
        output=output.masked_fill(bad[...,None],float('nan'))
        return output[:,0] if squeeze else output
