"""X1 confirmation c01 (PROTOCOL.md): decision-position mean ablation on the SEALED held parts.
Usage: c01_confirm.py <model: qwen3-06b-bf16|qwen3-17b> <task: LB|IND|ORD|CAPS> [--time-limit S] [--smoke]
  --smoke: exploration part, a quarter of the items, 2 random draws, outputs to outputs/conf_smoke/ (pipeline check only).
Same measures and code as e01 (x1common), scope 'dec' only. Head means are the exploration means saved by e01
(outputs/means_<model>_<task>_dec.pt, digests in FROZEN.md); they are not re-estimated on the held parts.
Groups per (model, task) are fixed in GROUPS_CONF below; each gets 20 layer-matched random draws
(x1common.random_sets, seed f"x1-c01-{group}"), drawn from heads not named in any charter group.
Material (fixed): LB 150 width items and 60 natural paragraphs (prose held part, in file order);
IND the first 400 natural-code chunks and the first 400 generated programs of the held parts; ORD 4 and 5 digits,
up to 96 items per class (seed 0); CAPS the first 100 paragraphs of the prose held part.
Each run is saved to outputs/conf/c01_<model>_<task>.json as it finishes (resumable); per-item arrays of the base
and named-group runs go to outputs/conf/c01_<model>_<task>_items.npz."""
import sys, time
from x1common import *

MODEL, TASK = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv
TL = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
G06 = ["L17H6", "CARRY", "CARRY-L17H6", "TERM", "TERM-L17H6", "READ", "READ-L17H6", "WRITE", "COUNT", "DED",
       "L0H2", "L0H3", "L11H9"]
GROUPS_CONF = {("qwen3-06b-bf16", "LB"): [g for g in G06 if g != "READ"],
               ("qwen3-06b-bf16", "IND"): ["L17H6", "READ-L17H6", "WRITE", "COUNT", "DED", "L0H2", "L0H3", "L11H9"],
               ("qwen3-06b-bf16", "ORD"): G06}
GROUPS_CONF[("qwen3-17b", "CAPS")] = ["L0H3", "L0H2", "L11H9", "L17H6", "READ", "CARRY", "COUNT", "DED"]
groups = GROUPS_CONF[(MODEL, TASK)]
NR = 2 if SMOKE else 20; k = 4 if SMOKE else 1
part = "explore" if SMOKE else "held"
od = OUT / ("conf_smoke" if SMOKE else "conf"); od.mkdir(exist_ok=True)
t0 = time.time()
M = Model(MODEL)
if TASK == "LB": task = LB(M, [r["text"] for r in load_part("prose", part)], n_width=150 // k, n_nat=60 // k)
elif TASK == "IND": task = IND(M, {"code": load_part("code", part)[:400 // k], "gen": load_part("gen", part)[:400 // k]})
elif TASK == "ORD": task = ORD(M, load_part("ord", part), per=96 // k, seed=0)
elif TASK == "CAPS": task = CAPS(M, [r for r in load_part("prose", part) if r["caps_ok"]][:100 // k])
means = torch.load(OUT / f"means_{MODEL}_{TASK}_dec.pt")
outp = od / f"c01_{MODEL}_{TASK}.json"; itp = od / f"c01_{MODEL}_{TASK}_items.npz"
res = json.loads(outp.read_text()) if outp.exists() else {}
items = dict(np.load(itp)) if itp.exists() else {}
todo = [("base", [])]
for g in groups:
    todo.append((g, parse(GROUPS[g])))
    for j, r in enumerate(random_sets(parse(GROUPS[g]), NR, f"x1-c01-{g}", H=M.H)): todo.append((f"{g}~r{j}", r))
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
print(json.dumps(dict(model=MODEL, task=TASK, part=part, done=done, runs=sum(k_ in res for k_, _ in todo), of=len(todo),
                      secs=round(time.time() - t0))), flush=True)
