"""Final V2 validation entry point. Full evaluation requires explicit --mode full."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final_validation.runner import main

if __name__=='__main__':
    main()
