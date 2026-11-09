import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import torch
from adaptive_world_model.experiments.cartpole_v1 import nominal_data
from adaptive_world_model.environments.cartpole import observable_features
from adaptive_world_model.utils.artifacts import write_json
cfg=json.loads((ROOT/'configs/cartpole/v1.json').read_text());torch.set_num_threads(1)
records={}
for seed in cfg['seeds']:
    data=nominal_data(cfg,seed,cfg['base']['train_count'],'cpu','base_train')
    features=observable_features(data.state,data.action)
    quantiles=torch.quantile(features,torch.tensor([0.,.01,.5,.99,1.]),dim=0)
    angle=torch.atan2(data.state[:,2].sin(),data.state[:,2].cos())
    records[seed]={'observations':len(data),'feature_order':['x','v','sin_theta','cos_theta','omega','force'],
        'quantiles':quantiles.tolist(),'quantile_levels':[0,.01,.5,.99,1],
        'angle_histogram_18_bins':torch.histc(angle,bins=18,min=-3.141592654,max=3.141592654).tolist(),
        'large_angle_fraction':float((angle.abs()>1.57).float().mean()),
        'source':'Exact deterministic nominal training dataset regenerated; no training or model updates'}
    print('COVERAGE',seed,records[seed]['large_angle_fraction'],flush=True)
write_json(ROOT/'results/cartpole/20261023_v2_control/backbone_coverage.json',records)
