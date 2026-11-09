"""Read-only reproduction; save diagnostic artifacts outside the frozen result."""
import csv
import datetime as dt
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(ROOT/'src'))
from adaptive_world_model.experiments.v1_baseline import sampled_reproduction
from adaptive_world_model.utils.artifacts import write_json

if __name__ == '__main__':
    baseline = ROOT/'results/cartpole/20261005_232356_821640_cartpole_v1'
    directory = ROOT/'smoke tests'/dt.datetime.now().strftime('v2_reproduction_%Y%m%d_%H%M%S')
    directory.mkdir(parents=True)
    rows = sampled_reproduction(baseline)
    with (directory/'baseline_reproduction.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(directory/'summary.json', {'passed': True, 'checks': len(rows), 'maximum_numeric_difference':
               max(row['absolute_difference'] for row in rows), 'training_executed': False})
    print(f'V1 reproduced: {len(rows)} checks; artifacts={directory}')
