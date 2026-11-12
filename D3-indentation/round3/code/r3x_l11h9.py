"""D3 round 3 (PROTOCOL-3.md, part E): L11H9 inside D3's sets CARRY and TERM, on indentation.
Same computation as X1's frozen c01_confirm.py for (qwen3-06b-bf16, IND) (x1common.IND, decision-position mean
ablation with X1's frozen exploration means, 20 layer-matched random draws per group from heads named in no X1
charter group, seed f"x1-c01-{group}"), for four groups that X1 did not run on indentation:
  CARRY, CARRY-L11H9, TERM, TERM-L11H9   (TERM-L11H9 and CARRY-L11H9: the sets with L11H9 left out).
Material: held (default) = the first 400 round-3 natural-code chunks and the first 400 round-3 generated
programs (as X1); refuses unless PROTOCOL-3.md is read-only. --calib = X1's exploration parts, first 400 + 400
(old material, already read; exploratory), outputs to round3/x1/outputs/calib/.
Usage: python r3x_l11h9.py [--calib] [--time-limit S]"""
import json, stat, sys, time
from pathlib import Path
R3 = Path(__file__).resolve().parent.parent
X1C = R3.parent.parent / "X1-shared-parts" / "code"
sys.path.insert(0, str(X1C))
import x1common
from x1common import *

CALIB = "--calib" in sys.argv
TL = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
NEW = {"CARRY-L11H9": "L17H4,L5H6,L17H6,L16H2,L6H5", "TERM-L11H9": "L5H6,L17H6,L16H2,L23H6,L22H8,L23H3,L25H9,L19H6"}
x1common.GROUPS.update(NEW)
groups = ["CARRY", "CARRY-L11H9", "TERM", "TERM-L11H9"]
MEANS = x1common.OUT / "means_qwen3-06b-bf16_IND_dec.pt"   # X1's frozen exploration means (read before any patch)
if CALIB:
    code, gen = load_part("code", "explore")[:400], load_part("gen", "explore")[:400]
    od = R3 / "x1" / "outputs" / "calib"
else:
    p3 = R3.parent / "PROTOCOL-3.md"
    if not p3.exists() or (p3.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
        raise SystemExit("refusing to read the round-3 parts: PROTOCOL-3.md is not frozen")
    rd = lambda p: [json.loads(l) for l in open(p)]
    code, gen = rd(R3 / "data" / "code" / "part-held.jsonl")[:400], rd(R3 / "data" / "gen" / "part-held.jsonl")[:400]
    od = R3 / "x1" / "outputs" / "conf"
od.mkdir(parents=True, exist_ok=True)
t0 = time.time()
M = Model("qwen3-06b-bf16")
task = IND(M, {"code": code, "gen": gen})
means = torch.load(MEANS)
outp = od / "r3x_l11h9_IND.json"; itp = od / "r3x_l11h9_IND_items.npz"
res = json.loads(outp.read_text()) if outp.exists() else {}
items = dict(np.load(itp)) if itp.exists() else {}
todo = [("base", [])]
for g in groups:
    todo.append((g, parse(x1common.GROUPS[g])))
    for j, r in enumerate(random_sets(parse(x1common.GROUPS[g]), 20, f"x1-c01-{g}", H=M.H)): todo.append((f"{g}~r{j}", r))
for key, heads in todo:
    if key in res: continue
    if time.time() - t0 > TL: print("time limit; resume later", flush=True); break
    if heads: M.set_ablation(heads, means, "dec")
    else: M.clear()
    out, arr = task.run(); M.clear()
    res[key] = dict(heads=nm(heads), **out)
    if "~r" not in key:
        for a, v in arr.items(): items[f"{key}|{a}"] = v
        np.savez(itp, **items)
    outp.write_text(json.dumps(res))
done = all(k_ in res for k_, _ in todo)
print(json.dumps(dict(script="r3x_l11h9", calib=CALIB, done=done, runs=sum(k_ in res for k_, _ in todo), of=len(todo),
                      secs=round(time.time() - t0))), flush=True)
