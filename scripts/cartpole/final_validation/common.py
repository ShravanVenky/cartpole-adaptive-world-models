"""Reuse existing checkpoint loading, MPC, atomic outputs and transition contracts."""
import copy
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from adaptive_world_model.adaptation.ridge import RidgeResidualModel
from adaptive_world_model.environments.cartpole import CartPolePhysics,observable_features
from adaptive_world_model.utils.randomness import stable_seed
from diagnostics.context import Context as ExistingContext,save_json,save_npz,file_hash,transition_batch
from diagnostics.engine import Predictor as ExistingPredictor,Planner,rollout,episode_metrics
from adaptive_world_model.control.evaluation_mpc import physics_step
from .protocol import ROOT,PREVIOUS


class CountScaledRidge(RidgeResidualModel):
    """Same28coefficients and safeguards; fixed mean-loss penalty matched to baseline atK20."""
    def __init__(self,base,reference_lambda=.1,reference_K=20):
        super().__init__(base,regularization=reference_lambda)
        self.reference_lambda=reference_lambda;self.reference_K=reference_K

    def adapt(self,support_data):
        self.regularization=self.reference_lambda*len(support_data)/self.reference_K
        log=super().adapt(support_data)
        log.update(regularization_scheme='count_scaled',lambda_sum=self.regularization,
                   lambda_mean=self.regularization/len(support_data),reference_K=self.reference_K)
        return log


class Predictor(ExistingPredictor):
    """Avoid unused neural forwards in all-Oracle recovery batches; predictions are identical."""
    def __init__(self,base,models,settings,device):
        super().__init__(base,models,settings,device)
        self.all_oracle=all(s['method']=='oracle' for s in settings)

    def __call__(self,s,u):
        if self.all_oracle and not self.store_frozen:return physics_step(s,u,self.physics)
        return super().__call__(s,u)


class Context(ExistingContext):
    def __init__(self,run,cfg,device):
        super().__init__(run,cfg,device)
        self.hits=0;self.misses=0
        self.fit_cache={}

    def make(self,seed,s):
        base,templates=self.templates(seed)
        if s['method'] in ['frozen','oracle']:return base
        if s['method']=='ridge' and s['regularization']=='count_scaled':
            return CountScaledRidge(base,self.cfg['ridge_lambda_reference'],self.cfg['ridge_K_reference'])
        m=copy.deepcopy(templates[s['method']]);m.reset_adaptation();return m

    def fit(self,seed,s,support):
        m=self.make(seed,s);log=None
        if s['method'] not in ['frozen','oracle'] and support is not None:
            signature=hashlib.sha256()
            for tensor in [support.state,support.action,support.next_state]:signature.update(tensor.contiguous().numpy().tobytes())
            key=(seed,s['method'],s['regularization'],signature.hexdigest())
            if key in self.fit_cache:
                started=time.perf_counter();parameters,stored=self.fit_cache[key]
                if hasattr(m,'coefficients'):
                    m.coefficients.copy_(parameters['coefficients'])
                    if isinstance(m,CountScaledRidge):m.regularization=m.reference_lambda*len(support)/m.reference_K
                else:m._adapted_params=OrderedDict((name,value.clone()) for name,value in parameters.items())
                log=dict(stored,cache_reused=True,uncached_adaptation_seconds=stored['adaptation_seconds'],
                         adaptation_seconds=time.perf_counter()-started,executed_gradient_steps=0)
            else:
                m.reset_adaptation();log=m.adapt(support)
                parameters={'coefficients':m.coefficients.detach().clone()} if hasattr(m,'coefficients') else OrderedDict((k,v.detach().clone()) for k,v in m._params().items())
                self.fit_cache[key]=(parameters,dict(log));log=dict(log,cache_reused=False,executed_gradient_steps=log.get('gradient_steps',0))
        return m,log

    def cached(self,path):
        p=super().cached(path)
        if p is None:self.misses+=1
        else:self.hits+=1
        return p


def initial(task,seed,trial):
    g=torch.Generator().manual_seed(stable_seed(seed,'final_v2_episode_initial',task,trial))
    x=torch.randn(4,generator=g)*torch.tensor([.1,.1,.05,.1])
    if task=='swingup':x[2]+=np.pi
    return x


def state_bad(s):
    return ~torch.isfinite(s).all(-1)|(s[...,0].abs()>2.4)|(s[...,1:].abs()>100).any(-1)


def score(states,actions,cfg,task,recovery_protocol=False):
    """Preserve original distinction: fixed balancing forbids tipping; online recovery may tip and recover."""
    result=episode_metrics(states,actions,cfg,task,0)
    if recovery_protocol and task=='balance':
        angle=np.arctan2(np.sin(states[:,2]),np.cos(states[:,2]))
        good=(np.abs(angle)<cfg['success_angle'])&(np.abs(states[:,3])<cfg['success_omega'])&(np.abs(states[:,0])<=2.4)
        success=bool(not result['failed'] and len(good)>=cfg['sustained_steps'] and good[-cfg['sustained_steps']:].all())
        result['success']=success;result['time_limit_failure']=bool(not result['failed'] and not success)
        if success:result['failure_category']='success';result['settled_then_lost']=False
    return result


def pack_arrays(pack):
    return {'coefficients':pack.coef.detach().cpu().numpy().copy(),
            **{k:v.detach().cpu().numpy().copy() for k,v in pack.weights.items()}}


def apply_arrays(pack,arrays):
    pack.coef.copy_(torch.as_tensor(arrays['coefficients'],device=pack.device))
    for k,v in pack.weights.items():v.copy_(torch.as_tensor(arrays[k],device=pack.device))


def coverage(base,support,query_states=None,query_actions=None):
    if support is None:return dict(count=0,feature_rank=0,condition_number=None,action_std=0.,angle_span=0.,omega_span=0.)
    z=base.normalizer.encode_x(support.features()).double()
    design=torch.cat([z,torch.ones_like(z[:,:1])],1)
    singular=torch.linalg.svdvals(design)
    eigen=torch.linalg.eigvalsh(design.T@design+.1*torch.eye(7))
    theta=torch.atan2(support.state[:,2].sin(),support.state[:,2].cos())
    out=dict(count=len(support),support_hash=support.support_hash(),feature_rank=int(torch.linalg.matrix_rank(design)),
             condition_number=float(eigen[-1]/eigen[0]),unregularized_smallest_singular=float(singular[-1]),
             action_std=float(support.action.std(unbiased=False)),mean_absolute_force=float(support.action.abs().mean()),
             angle_span=float(theta.max()-theta.min()),omega_span=float(support.state[:,3].max()-support.state[:,3].min()),
             cart_velocity_span=float(support.state[:,1].max()-support.state[:,1].min()),feature_variance=float(z.var(0,unbiased=False).mean()))
    if query_states is not None:
        q=base.normalizer.encode_x(observable_features(torch.as_tensor(query_states).float(),torch.as_tensor(query_actions).float())).double()
        finite=torch.isfinite(q).all(1);out['query_invalid_fraction']=float((~finite).float().mean())
        if finite.any():
            distances=torch.cdist(q[finite],z).amin(1)
            out.update(query_support_distance_median=float(distances.median()),query_support_distance_p90=float(torch.quantile(distances,.9)))
    return out


def snapshot_model(m):
    if hasattr(m,'coefficients'):return {'coefficients':m.coefficients.detach().cpu().numpy().copy()}
    if hasattr(m,'_params'):return {k:v.detach().cpu().numpy().copy() for k,v in m._params().items()}
    return {}


def source_hashes():
    files=[*sorted(Path(__file__).parent.glob('*.py')),ROOT/'scripts/cartpole/run_final_v2.py',ROOT/'scripts/cartpole/run_final_v2.ps1',ROOT/'scripts/cartpole/control_evaluate.py',
           *sorted((ROOT/'scripts/cartpole/diagnostics').glob('*.py')),
           *sorted((ROOT/'src/adaptive_world_model').rglob('*.py'))]
    return {str(p.relative_to(ROOT)):file_hash(p) for p in files}
