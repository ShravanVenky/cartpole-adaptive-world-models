"""Final V2 hardening. Explicit --mode full required for any scientific stage."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hardening.runner import main
if __name__=='__main__':main()
