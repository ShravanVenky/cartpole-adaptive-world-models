"""Content-verified atomic jobs, immutable inputs and shared existing components."""
import copy
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from mismatch.common import Context as PreviousContext, setting, source_files as prior_sources
from diagnostics.context import file_hash, transition_batch
from .storage import save_json, save_npz, save_advisory_json, atomic_write
from .recovery import expected_sources, verify_sources, changed_paths
from final_validation.common import coverage, score, state_bad
from mismatch.common import Predictor as ExistingPredictor, Planner
from .protocol import ROOT, REFERENCE, METAS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sources():
    paths={**prior_sources()}
    for p in [*sorted((ROOT/'scripts/cartpole/hardening').glob('*.py')),
              ROOT/'scripts/cartpole/run_hardening_v2.py', ROOT/'scripts/cartpole/run_hardening_v2.ps1']:
        paths[str(p.relative_to(ROOT))]=file_hash(p)
    return paths


class Predictor(ExistingPredictor):
    """Identical predictor arithmetic; seven bulk copies replace per-row GPU syncs."""
    def refresh(self,models):
        coefficients=[];parameters=[]
        for m in models:
            coefficients.append(m.coefficients.detach().cpu() if hasattr(m,'coefficients') else torch.zeros(7,4))
            parameters.append(m._params() if hasattr(m,'_params') else None)
        self.coef.copy_(torch.stack(coefficients))
        for name,target in self.weights.items():
            values=[p[name].detach().cpu() if p is not None else torch.zeros(target.shape[1:]) for p in parameters]
            target.copy_(torch.stack(values))


class Context(PreviousContext):
    def __init__(self, run, cfg, device, mode):
        super().__init__(run,cfg,device)
        self.mode=mode; self.start=time.perf_counter(); self.completed=0; self.total=None
        self.current_stage=None

    def cached_job(self, path):
        path=Path(path)
        if not path.exists(): self.misses+=1; return None
        packet=json.loads(path.read_text(encoding='utf-8'))
        if packet.get('config_sha256')!=self.digest: raise RuntimeError(f'Incompatible job {path}')
        target=path.with_suffix('.npz')
        if not target.exists() or file_hash(target)!=packet.get('arrays_sha256'):
            raise RuntimeError(f'Incomplete/corrupted atomic job {path}; move that job pair aside explicitly before recomputing')
        self.hits+=1
        return packet

    def save_job(self,path,packet,**arrays):
        path=Path(path); save_npz(path.with_suffix('.npz'),**arrays)
        packet=dict(packet,config_sha256=self.digest,arrays_sha256=file_hash(path.with_suffix('.npz')),
                    scientific_validation=self.mode=='full')
        save_json(path,packet)
        self.completed+=1
        elapsed=time.perf_counter()-self.start
        eta=(elapsed/self.completed*(self.total-self.completed)) if self.total and self.completed else None
        if path.parent.name!='fits' or self.completed%25==0:
            print(f'JOB {path.stem} completed={self.completed} elapsed_s={elapsed:.1f} eta_s={eta}',flush=True)
        save_advisory_json(self.run/'progress.json',dict(job=str(path.relative_to(self.run)),completed=self.completed,
                  stage=self.current_stage,wall_seconds=elapsed,eta_seconds=eta,cache_hits=self.hits,
                  eta_basis='upper-bound job count; conditional on selected settings and survival' if self.total else 'bounded workload'))
        return packet

    def dense_model(self,seed,environment,method):
        path=REFERENCE/'dense'/f'{seed}_{environment}_{method}_best.npz'
        if not path.exists(): raise RuntimeError(f'No matched dense reference: {path}')
        m=self.make(seed,setting(method))
        with np.load(path) as a:
            m._adapted_params=OrderedDict((k,torch.from_numpy(a[k].copy())) for k in m._params())
        return m

    def predictor(self,seed,policies,models,physics):
        base,_=self.templates(seed)
        settings=[setting(p['method'],'margin') for p in policies]
        pack=Predictor(base,models,settings,self.device); pack.physics=physics
        return pack,settings


def prepare(run,cfg,mode,check_only=False):
    run=Path(run)
    if check_only and not run.is_dir():raise RuntimeError(f'No existing run: {run}')
    if not check_only:run.mkdir(parents=True,exist_ok=True)
    current=sources()
    for name,value in [('config_locked.json',cfg),('run_kind.json',dict(mode=mode))]:
        p=run/name
        if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=value:
            raise RuntimeError(f'Resume mismatch in {name}; do not mix modes, source versions or configurations')
        if not p.exists():
            if check_only:raise RuntimeError(f'Missing resume lock: {p}')
            save_json(p,value)
    source_lock=run/'source_hashes.json'
    if not source_lock.exists():
        if check_only:raise RuntimeError(f'Missing resume lock: {source_lock}')
        save_json(source_lock,current)
    if expected_sources(run)!=current:raise RuntimeError('Source version mismatch; do not bypass the recorded revision gate')
    manifest=run/'preserved_inputs.json'
    if not manifest.exists():
        if check_only:raise RuntimeError(f'Missing preservation manifest: {manifest}')
        files=set(ROOT/n for n in sources() if not n.startswith('scripts/cartpole/hardening/') and 'run_hardening' not in n)
        if mode=='full':
            for folder in [ROOT/'results/cartpole', ROOT/'archive']:
                files.update(p for p in folder.rglob('*') if p.is_file() and run not in p.parents)
        else:
            files.update(p for p in (ROOT/'results/cartpole').rglob('*.pt'))
            files.update(p for p in REFERENCE.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.csv','.md'])
        save_json(manifest,{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)})
    snap=run/'source_snapshot'
    import shutil
    for name,sha in json.loads(source_lock.read_text()).items():
        p=snap/name
        if p.exists() and file_hash(p)!=sha:raise RuntimeError(f'Source snapshot changed: {p}')
        if not p.exists():
            if check_only or file_hash(ROOT/name)!=sha:raise RuntimeError(f'Original source snapshot missing: {p}')
            p.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,p)
    verify_sources(run,current)


def integrity(ctx,persist=True):
    manifest=json.loads((ctx.run/'preserved_inputs.json').read_text())
    current=sources();verify_sources(ctx.run,current)
    original=json.loads((ctx.run/'source_hashes.json').read_text())
    repaired=changed_paths(original,current)
    # The legacy Windows manifest also included the runner sources. Keep that
    # manifest unchanged: their exact old bytes were verified in source_snapshot,
    # and exact repaired bytes were verified against the explicit IO revision.
    archived=[n for n,h in manifest.items() if n in repaired and original.get(n)==h]
    changed=[n for n,h in manifest.items() if n not in archived and (not (ROOT/n).exists() or file_hash(ROOT/n)!=h)]
    if changed: raise RuntimeError(f'Preservation check failed: {changed[:5]}')
    result=dict(passed=True,unchanged_files=len(manifest)-len(archived),
                io_repaired_source_files=archived,archived_original_sources_verified=len(archived))
    if persist:save_json(ctx.run/'integrity_verified.json',result)
    return result


def support_from_episode(seed,environment,trial=900,k=50,collector=METAS[0]):
    key=f'prospective_{seed}_{trial}_active_swingup__{environment}'
    packet=json.loads((REFERENCE/'episodes'/f'{key}.json').read_text())
    row=next(i for i,r in enumerate(packet['rows']) if r['method']==collector)
    if packet['rows'][row]['usable_postchange_observations']<k:
        raise RuntimeError(f'Insufficient actual support {key}/{collector}/{k}')
    with np.load(REFERENCE/'episodes'/f'{key}.npz') as a:
        return transition_batch(a['states'][row,:k].copy(),a['actions'][row,:k].copy(),a['states'][row,1:k+1].copy(),
                                f'hardening/development/actual_support/{key}/{collector}')


def query_record(seed,environment,step=150):
    key=f'prospective_{seed}_901_active_swingup__{environment}'
    packet=json.loads((REFERENCE/'episodes'/f'{key}.json').read_text())
    row=next(i for i,r in enumerate(packet['rows']) if r['method']==METAS[1])
    with np.load(REFERENCE/'episodes'/f'{key}.npz') as a: state=a['states'][row,step].tolist()
    return dict(id=f'A_query_{seed}_{environment}_{step}',seed=seed,trial=901,state=state,
                condition=f'active_swingup__{environment}',environment=environment,task='swingup',horizon=30,
                split='development',step=step,source_key=key,query_independent_episode=True)


def restore_model(ctx,seed,method,path):
    m=ctx.make(seed,setting(method))
    with np.load(Path(path).with_suffix('.npz')) as a:
        if method=='ridge': m.coefficients.copy_(torch.from_numpy(a['coefficients'].copy()))
        elif method in METAS: m._adapted_params=OrderedDict((k,torch.from_numpy(a[k].copy())) for k in m._params())
    return m


def save_csv(path,rows):
    import pandas as pd
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload=pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
    atomic_write(path,lambda stream:stream.write(payload))
