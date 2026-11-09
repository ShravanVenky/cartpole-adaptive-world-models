import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
sys.path.insert(0,str(ROOT/'src'))
from adaptive_world_model.experiments.cartpole_v2 import main

if __name__=='__main__': raise SystemExit(main())
