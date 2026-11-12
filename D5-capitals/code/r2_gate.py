"""D5 round 2 (CHARTER-2.md, Q5): run the frozen round-1 gate (code/gate.py, unchanged, digest checked)
on another model. The model short name is added to d5common.NAMES before gate.py runs; nothing else changes.
Usage: r2_gate.py <short> <hf name in D1-X's MODELS> [n=150]"""
import hashlib, runpy, sys
from pathlib import Path
import d5common
here = Path(__file__).resolve().parent
d5common.NAMES[sys.argv[1]] = sys.argv[2]
sys.argv = ["gate.py", sys.argv[1]] + sys.argv[3:]
runpy.run_path(str(here / "gate.py"), run_name="__main__")
