"""Atomic caches, saved-as-of-time restoration, exact residual identity and unchanged models."""
import copy
import hashlib
import json
from collections import OrderedDict
import numpy as np
import torch
from adaptive_world_model.control.evaluation_mpc import physics_step
from final_validation.common import Context as PreviousContext,Predictor as PreviousPredictor,Planner,score,state_bad,coverage
from diagnostics.context import save_json,save_npz,file_hash,transition_batch
from final_validation.protocol import spec,CHECKPOINTS
from .protocol import ROOT,REFERENCE,METHODS,METAS


def setting(method,safety='margin'):
    return spec('oracle' if method=='exact_residual' else method,safety)


class Predictor(PreviousPredictor):
    def __init__(self,base,models,settings,device,exact=()):
        super().__init__(base,models,settings,device)
        self.exact_indices=torch.tensor(list(exact),device=device,dtype=torch.long)
        self.has_exact=bool(len(exact))

    def __call__(self,s,u):
        result=super().__call__(s,u)
        if self.has_exact:
            q=s.index_select(0,self.exact_indices);a=u.index_select(0,self.exact_indices)
            frozen=self.frozen(q,a);residual=physics_step(q,a,self.physics)-frozen
            result=result.index_copy(0,self.exact_indices,frozen+residual)
        return result


class Context(PreviousContext):
    def __init__(self,run,cfg,device):
        super().__init__(run,cfg,device);self.restoration=[];self.data_cache={}

    def packet(self,key,source=REFERENCE):
        return json.loads((source/'episodes'/f'{key}.json').read_text(encoding='utf-8'))

    def arrays(self,key,source=REFERENCE):
        # At most one source batch in memory; never mutate archived arrays.
        cache=(str(source),key)
        if cache not in self.data_cache:
            self.data_cache.clear()
            with np.load(source/'episodes'/f'{key}.npz') as data:self.data_cache[cache]={k:data[k] for k in data.files}
        return self.data_cache[cache]

    def support(self,record,k=None):
        data=self.arrays(record['key'],PathLike(record.get('source_dir',REFERENCE)))
        k=record['support_K'] if k is None else k
        if not k:return None
        row=record['row']
        if k>record['usable']:raise ValueError('Support prefix extends past actual usable observations')
        return transition_batch(data['states'][row,:k],data['actions'][row,:k],data['states'][row,1:k+1],
                                f'mismatch_actual_past/{record["id"]}/K{k}')

    def restored(self,record,method,k=None):
        k=record['support_K'] if k is None else k;seed=record['seed'];s=setting(method)
        base,_=self.templates(seed)
        if method in ['frozen','oracle','exact_residual']:return base
        data=self.arrays(record['key'],PathLike(record.get('source_dir',REFERENCE)))
        # Only the collector's saved parameters correspond to its actual history.
        if method==record['collector_method'] and f'K{k}_coefficients' in data:
            m=self.make(seed,s);r=record['row']
            if method=='ridge':m.coefficients.copy_(torch.from_numpy(data[f'K{k}_coefficients'][r].copy()))
            else:m._adapted_params=OrderedDict((name,torch.from_numpy(data[f'K{k}_{name}'][r].copy())) for name in m._params())
            origin='restored_saved_snapshot'
        else:
            m,log=self.fit(seed,s,self.support(record,k));origin='initialization' if k==0 else 'reconstructed_common_history'
            if log is not None and not log['accepted']:raise RuntimeError(f'Adaptation rejected: {record["id"]}/{method}')
        self.restoration.append(dict(case=record['id'],seed=seed,method=method,K=k,origin=origin))
        return m

    def fitted_models(self,record,dense=False,k=None):
        names=METHODS.copy();models=[self.restored(record,m,k) for m in names]
        if dense:
            for m in METAS:
                names.append('dense_'+m);models.append(self.dense_model(record['seed'],record['environment'],m))
        return names,models

    def dense_model(self,seed,environment,method):
        p=self.run/'dense'/f'{seed}_{environment}_{method}_best.npz'
        if not p.exists():raise RuntimeError(f'Dense stage prerequisite missing: {p}')
        m=self.make(seed,setting(method))
        with np.load(p) as a:m._adapted_params=OrderedDict((name,torch.tensor(a[name])) for name in m._params())
        return m

    def pack(self,seed,names,models,safety='margin'):
        base,_=self.templates(seed)
        settings=[setting(m if not m.startswith('dense_') else m[6:],safety) for m in names]
        pack=Predictor(base,models,settings,self.device,[i for i,m in enumerate(names) if m=='exact_residual'])
        return pack,settings


def PathLike(value):
    from pathlib import Path
    return Path(value)


def atomic_torch(path,data):
    import os
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pt.tmp');torch.save(data,tmp);os.replace(tmp,path)


def checkpoint_K(step,usable):return max(k for k in CHECKPOINTS if k<=min(step,usable))


def source_files():
    paths=[*sorted((ROOT/'scripts/cartpole/mismatch').glob('*.py')),ROOT/'scripts/cartpole/run_mismatch_v2.py',
           ROOT/'scripts/cartpole/run_mismatch_v2.ps1',*sorted((ROOT/'scripts/cartpole/final_validation').glob('*.py')),
           *sorted((ROOT/'scripts/cartpole/diagnostics').glob('*.py')),ROOT/'scripts/cartpole/control_evaluate.py',
           *sorted((ROOT/'src/adaptive_world_model').rglob('*.py'))]
    return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}
