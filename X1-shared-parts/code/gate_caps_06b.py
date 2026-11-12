"""X1: D5's frozen capitals gate (D5 code/gate.py, unchanged) run on Qwen3-0.6B-Base (CHARTER.md, Models).
D5's exploration part only (150 paragraphs, as the gate fixes). The output is redirected to X1's outputs
folder by setting d5common.OUT before gate.py runs, so nothing is written into D5's folder.
Usage: gate_caps_06b.py [model short name, default qwen3-06b]"""
import sys, runpy
from pathlib import Path
X1 = Path(__file__).resolve().parent.parent
D5C = X1.parent / "D5-capitals" / "code"
sys.path.insert(0, str(D5C))
import d5common
d5common.OUT = X1 / "outputs"
short = sys.argv[1] if len(sys.argv) > 1 else "qwen3-06b"
sys.argv = [str(D5C / "gate.py"), short]
runpy.run_path(str(D5C / "gate.py"), run_name="__main__")
