"""Record original scientific artifact bytes before the control extension."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from adaptive_world_model.utils.artifacts import write_json,snapshot_sources
RUN=ROOT/'results/cartpole/20261023_v2_control'
paths=['results/cartpole/20261005_232356_821640_cartpole_v1','results/cartpole/20261006_140915_614361_cartpole_v2_factorial']
manifest={}
for directory in paths:
    for p in (ROOT/directory).rglob('*'):
        if p.is_file():manifest[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
write_json(RUN/'preserved_artifact_hashes.json',manifest)
cfg=json.loads((RUN/'config_locked.json').read_text());cfg['disturbance_delta']=[0,.5,.35,.6]
cfg['expected_groups']=800;cfg['expected_episode_rows']=26625
cfg['cost_source_sha256']=hashlib.sha256((ROOT/'src/adaptive_world_model/control/evaluation_mpc.py').read_bytes()).hexdigest()
cfg['execution']='single GPU process, CUDA graphs; candidates, methods, budgets and independent trials batched; matched optimizer noise per trial'
cfg['failure_penalty']=100000;cfg['failure']='rail abs(x)>2.4, nonfinite simulation, or abs(other state)>100; absorbing after first failure; actual failing transition retained'
cfg['development_changes']='Before final evaluation, oracle-only cost development resolved cart drift and stabilization. Initial disturbance [0,.8,.5,1.] exceeded rails; final recoverable disturbance [0,.5,.35,.6] passed8/8 nominal oracle. Initial checkpoints/architectures/adapter hyperparameters unchanged.'
write_json(RUN/'config_locked.json',cfg);snapshot_sources(ROOT,RUN/'source_snapshot')
print(f'Preserved {len(manifest)} original files; locked800groups/26625episodes',flush=True)
