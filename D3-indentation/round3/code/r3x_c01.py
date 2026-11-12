"""D3 round 3 (PROTOCOL-3.md, part X): runs X1's frozen c01_confirm.py UNCHANGED on the round-3 parts.
Only two module globals of X1's x1common are pointed elsewhere before the frozen script is executed:
  x1common.DATA -> round3/x1/data   (code/part-held.jsonl and gen/part-held.jsonl: links to the round-3 parts)
  x1common.OUT  -> round3/x1/outputs (holds a byte-identical copy of X1's frozen exploration means for IND)
x1common.load_part's own guard (X1's PROTOCOL.md read-only) still applies; this wrapper adds a second guard:
round 3's PROTOCOL-3.md must exist and be read-only. Usage: python r3x_c01.py qwen3-06b-bf16 IND [--time-limit S]"""
import runpy, stat, sys
from pathlib import Path
R3 = Path(__file__).resolve().parent.parent
X1C = R3.parent.parent / "X1-shared-parts" / "code"
p3 = R3.parent / "PROTOCOL-3.md"
if not p3.exists() or (p3.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
    raise SystemExit("refusing to run: PROTOCOL-3.md is not frozen")
if sys.argv[1:3] != ["qwen3-06b-bf16", "IND"]:
    raise SystemExit("round 3 runs X1's c01 for (qwen3-06b-bf16, IND) only")
sys.path.insert(0, str(X1C))
import x1common
x1common.DATA = R3 / "x1" / "data"; x1common.OUT = R3 / "x1" / "outputs"
sys.argv = [str(X1C / "c01_confirm.py")] + sys.argv[1:]
runpy.run_path(str(X1C / "c01_confirm.py"), run_name="__main__")
