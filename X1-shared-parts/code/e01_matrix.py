"""X1 e01 (exploratory): cross-task ablation matrix on the EXPLORATION parts.
Usage: e01_matrix.py <model: qwen3-06b-bf16|qwen3-17b> <tasks, comma list of LB,IND,ORD,CAPS> [--scope all|dec] [--rand N] [--time-limit S] [--smoke]
Scope dec: ablation only at each task's decision positions (LB: newline-marker tokens and decision tokens; IND: the
decision token; ORD: the last token; CAPS: the prediction positions; GEN not computed), means over the same positions.
For every head group in x1common.GROUPS (and N layer-matched random draws per group, random.Random(f"x1-e01-{group}")),
mean-ablate the group at all positions (per-task means over the task's exploration material, saved to
outputs/means_<model>_<task>.pt on first use) and evaluate each task's measures. Runs are saved one by one to
outputs/e01_<model>.json, so the job resumes across lock holds."""
import sys, time
from x1common import *

MODEL = sys.argv[1]; TASKS = sys.argv[2].split(","); SMOKE = "--smoke" in sys.argv
SCOPE = sys.argv[sys.argv.index("--scope") + 1] if "--scope" in sys.argv else "all"
NR = int(sys.argv[sys.argv.index("--rand") + 1]) if "--rand" in sys.argv else 10
TL = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
t0 = time.time()
M = Model(MODEL)
prose = load_part("prose", "explore") if ("LB" in TASKS or "CAPS" in TASKS) else []
caps_rows = [r for r in prose if r["caps_ok"]]
k = 4 if SMOKE else 1
T = {}
if "LB" in TASKS: T["LB"] = LB(M, [r["text"] for r in prose], n_width=120 // k, n_nat=50 // k)
if "IND" in TASKS:
    src = {"gen": load_part("gen", "explore")[:300 // k]}
    if (DATA / "code" / "part-explore.jsonl").exists(): src["code"] = load_part("code", "explore")[:300 // k]
    T["IND"] = IND(M, src)
if "ORD" in TASKS: T["ORD"] = ORD(M, load_part("ord", "explore"), per=96 // k)
if "CAPS" in TASKS: T["CAPS"] = CAPS(M, caps_rows[:60 // k])
print("built", {t: round(time.time() - t0) for t in T}, flush=True)
means = {}
for t, task in T.items():
    p = OUT / f"means_{MODEL}_{t}_{SCOPE}{'_smoke' if SMOKE else ''}.pt"
    if p.exists(): means[t] = torch.load(p)
    else:
        means[t] = task_means(M, task.seqs_for_means(), None if SCOPE == "all" else task.dec_for_means()); torch.save(means[t], p)
outp = OUT / f"e01_{MODEL}_{SCOPE}{'_smoke' if SMOKE else ''}.json"
res = json.loads(outp.read_text()) if outp.exists() else {}
todo = [("base", [])]
for g, s in GROUPS.items():
    todo.append((g, parse(s)))
    for j, r in enumerate(random_sets(parse(s), NR, f"x1-e01-{g}", H=M.H)): todo.append((f"{g}~r{j}", r))
for key, heads in todo:
    res.setdefault(key, {"heads": nm(heads)})
    for t, task in T.items():
        if t in res[key]: continue
        if time.time() - t0 > TL: break
        if heads: M.set_ablation(heads, means[t], SCOPE)
        else: M.clear()
        out, _ = task.run(); M.clear()
        res[key][t] = out
    outp.write_text(json.dumps(res))
    if time.time() - t0 > TL: print("time limit; resume later", flush=True); break
done = all(all(t in res.get(k_, {}) for t in T) for k_, _ in todo)
print(json.dumps(dict(model=MODEL, tasks=TASKS, done=done, runs=sum(all(t in res.get(k_, {}) for t in T) for k_, _ in todo), of=len(todo),
                      secs=round(time.time() - t0))), flush=True)
