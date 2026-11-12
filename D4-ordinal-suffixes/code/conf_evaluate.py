"""Evaluate the PROTOCOL.md predictions from saved outputs (no model is loaded).
Usage: conf_evaluate.py [--part conf|explore]
--part explore is a dry run on the exploration outputs (pipeline check only; those numbers are not results).
Writes outputs/<part>/EVALUATION.json with every number and each pass or fail, and prints a table.
"""
import sys, json, random, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "conf"
D = OUT / PART
NB = 2000


def src(fname):
    """conf: outputs/conf/<fname> only. explore (dry run): outputs/explore/, else the exploration outputs in outputs/."""
    f = D / fname
    return f if (f.exists() or PART != "explore") else OUT / fname


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rows(model):
    return [json.loads(l) for l in open(src(f"conf01_{model}.jsonl"))]


def cell(rs):
    t = [r for r in rs if r["cls"] == "teen"]; e = [r for r in rs if r["cls"] == "end123"]
    return dict(n_teen=len(t), n_end123=len(e), n_teen_numbers=len(set(r["n"] for r in t)),
                auroc=auroc([r["m"] for r in e], [r["m"] for r in t]),
                acc_teen=float(np.mean([r["ok"] for r in t])) if t else None,
                acc_end123=float(np.mean([r["ok"] for r in e])) if e else None)


def boot(rs, stat, seed):
    """95% interval over 2,000 resamples of distinct numbers (all items of a number move together)."""
    by = {}
    for r in rs: by.setdefault(r["n"], []).append(r)
    keys = sorted(by); rng = random.Random(seed); vals = []
    for _ in range(NB):
        s = [x for k in rng.choices(keys, k=len(keys)) for x in by[k]]
        v = stat(s)
        if v is not None: vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def with_ci(rs, name):
    c = cell(rs)
    for k in ("auroc", "acc_teen"):
        c[k + "_ci95"] = boot(rs, lambda s, k=k: cell(s)[k], f"d4-conf-boot-{name}-{k}")
    return c


E = dict(part=PART, tests={}, cells={})
T = E["tests"]

# ---- 4.1 OLMo-2 token-boundary account
R = rows("olmo2-1b")
O = {}
for dg in range(3, 8):
    for form in ("plain", "comma"):
        s = [r for r in R if r["set"] == "length" and r["digits"] == dg and r["form"] == form]
        if s: O[f"{dg}|{form}"] = with_ci(s, f"olmo-{dg}-{form}")
E["cells"]["olmo2-1b"] = O
a = lambda k: O[k]["auroc"]
T["B1"] = dict(values={"4 plain": a("4|plain"), "7 plain": a("7|plain")}, rule="each <= 0.80",
               passed=a("4|plain") <= 0.80 and a("7|plain") <= 0.80)
b2 = ["3|plain", "5|plain", "6|plain", "4|comma", "5|comma", "6|comma", "7|comma"]
T["B2"] = dict(values={k: a(k) for k in b2}, rule="each >= 0.88", passed=all(a(k) >= 0.88 for k in b2))
d4, d7 = a("4|comma") - a("4|plain"), a("7|comma") - a("7|plain")
T["B3"] = dict(values={"4 comma-plain": d4, "7 comma-plain": d7}, rule="each >= 0.10", passed=d4 >= 0.10 and d7 >= 0.10)
r54, r67 = a("5|plain") - a("4|plain"), a("6|plain") - a("7|plain")
t4, t3 = O["4|plain"]["acc_teen"], O["3|plain"]["acc_teen"]
T["B4"] = dict(values={"auroc 5-4": r54, "auroc 6-7": r67, "teen acc 4 plain": t4, "teen acc 3 plain": t3},
               rule="5-4 >= 0.10, 6-7 >= 0.10, teen acc 4 <= 0.65, teen acc 3 >= 0.85",
               passed=r54 >= 0.10 and r67 >= 0.10 and t4 <= 0.65 and t3 >= 0.85)
ok = [T[k]["passed"] for k in ("B1", "B2", "B3", "B4")]
E["verdict_boundary"] = "supported" if all(ok) else "partly supported" if (ok[0] and ok[2]) else "not supported"

# ---- B5: GPT-2 and Pythia-1B (secondary)
T["B5"] = {}
for m in ("gpt2", "pythia-1b"):
    R = rows(m)
    bd = with_ci([r for r in R if r["set"] == "boundary"], f"{m}-boundary")
    ct = with_ci([r for r in R if r["set"] == "control"], f"{m}-control")
    E["cells"][m] = dict(boundary=bd, control=ct)
    readable = bd["n_teen_numbers"] >= 3
    p = bd["acc_teen"] <= 0.50 and ct["acc_teen"] >= 0.85 and (ct["auroc"] - bd["auroc"]) >= 0.10
    T["B5"][m] = dict(values={"boundary teen acc": bd["acc_teen"], "control teen acc": ct["acc_teen"],
                              "auroc control-boundary": ct["auroc"] - bd["auroc"],
                              "distinct boundary teen numbers": bd["n_teen_numbers"]},
                      rule="boundary teen acc <= 0.50, control teen acc >= 0.85, AUROC gap >= 0.10; read only with >= 3 teen numbers",
                      read=readable, passed=bool(p) if readable else None)

# ---- 4.2 Qwen3 ablation
X = json.load(open(src("exp05_qwen3-06b_0.2-1.6.json")))["digits"]
g = lambda d: X[str(d)] if str(d) in X else X[d]
q = {}
for d in (2, 3, 4):
    x = g(d); b = x["base"]["effect"]
    q[d] = dict(kept=x["frac_effect_kept"], random_kept_min=x["random_frac_kept_min"], random_kept_max=x["random_frac_kept_max"],
                teen_drop=x["base"]["acc_teen"] - x["candidate"]["acc_teen"],
                end123_drop=x["base"]["acc_end123"] - x["candidate"]["acc_end123"],
                teen_acc_ablated=x["candidate"]["acc_teen"],
                single_kept={k: v["effect"] / b for k, v in x["singles"].items()},
                base_effect=b, base_auroc=x["base"]["auroc"])
E["cells"]["qwen3-06b_ablation"] = q
T["Q1"] = dict(values={d: dict(kept=q[d]["kept"], random_min=q[d]["random_kept_min"]) for d in (3, 4)},
               rule="kept <= 0.55 and every random pair >= 0.85, at 3 and 4 digits",
               passed=all(q[d]["kept"] <= 0.55 and q[d]["random_kept_min"] >= 0.85 for d in (3, 4)))
T["Q2"] = dict(values={d: dict(teen_drop=q[d]["teen_drop"], end123_drop=q[d]["end123_drop"]) for d in (3, 4)},
               rule="teen acc drop >= 0.20 and non-teen drop <= 0.10, at 3 and 4 digits",
               passed=all(q[d]["teen_drop"] >= 0.20 and q[d]["end123_drop"] <= 0.10 for d in (3, 4)))
T["Q3"] = dict(values=dict(kept=q[2]["kept"], teen_acc_ablated=q[2]["teen_acc_ablated"]),
               rule="kept >= 0.55 and teen acc after ablation >= 0.95 (2 digits)",
               passed=q[2]["kept"] >= 0.55 and q[2]["teen_acc_ablated"] >= 0.95)
T["Q4"] = dict(values={d: q[d]["single_kept"] for d in (3, 4)}, rule="L0H2 alone <= 0.70, L1H6 alone >= 0.80, at 3 and 4 digits",
               passed=all(q[d]["single_kept"]["0.2"] <= 0.70 and q[d]["single_kept"]["1.6"] >= 0.80 for d in (3, 4)))

# ---- Qwen3 patching
def scan(d, kind):
    s = json.load(open(src(f"exp04_qwen3-06b_{d}d_{kind}.json"))); h = np.array(s["head"])
    l, k = np.unravel_index(np.argmax(h), h.shape)
    return dict(L0H2=float(h[0, 2]), top=f"L{l}H{k}", top_value=float(h[l, k]), n=s["n"], n_valid=s.get("n_valid"))
P = {f"{d}d_{k}": scan(d, k) for d in (3, 4) for k in ("intoteen", "last")}
E["cells"]["qwen3-06b_patching"] = P
T["Q5"] = dict(values={k: P[k] for k in ("3d_intoteen", "4d_intoteen")}, rule="L0H2 is the top head and >= 0.40, at 3 and 4 digits",
               passed=all(P[k]["top"] == "L0H2" and P[k]["L0H2"] >= 0.40 for k in ("3d_intoteen", "4d_intoteen")))
T["Q6"] = dict(values={k: P[k] for k in ("3d_last", "4d_last")}, rule="|L0H2| <= 0.05 and top head >= 0.30, at 3 and 4 digits",
               passed=all(abs(P[k]["L0H2"]) <= 0.05 and P[k]["top_value"] >= 0.30 for k in ("3d_last", "4d_last")))
okq = {k: T[k]["passed"] for k in ("Q1", "Q2", "Q4", "Q5")}
E["verdict_qwen_copy"] = "supported" if all(okq.values()) else \
    "partly supported" if (okq["Q1"] and okq["Q2"]) else "not supported"

# ---- 4.3 OLMo-2 cross-token move
Y = json.load(open(src("exp05_olmo2-1b_0.5-0.0-0.9.json")))["digits"]
gy = lambda d: Y[str(d)] if str(d) in Y else Y[d]
o = {d: dict(kept=gy(d)["frac_effect_kept"], base_effect=gy(d)["base"]["effect"], base_auroc=gy(d)["base"]["auroc"],
             random_kept_min=gy(d)["random_frac_kept_min"], random_kept_max=gy(d)["random_frac_kept_max"],
             teen_acc_base=gy(d)["base"]["acc_teen"], teen_acc_ablated=gy(d)["candidate"]["acc_teen"],
             single_kept={k: v["effect"] / gy(d)["base"]["effect"] for k, v in gy(d)["singles"].items()}) for d in (3, 4, 5, 7)}
E["cells"]["olmo2-1b_ablation"] = o
o1_read = all(o[d]["base_effect"] >= 0.30 for d in (4, 7))
T["O1"] = dict(values={d: dict(kept=o[d]["kept"], random_min=o[d]["random_kept_min"], base_effect=o[d]["base_effect"]) for d in (4, 7)},
               rule="read only if base effect >= 0.30 at 4 and 7; kept <= 0.65 and every random triple >= 0.85, at 4 and 7 digits",
               read=o1_read, passed=all(o[d]["kept"] <= 0.65 and o[d]["random_kept_min"] >= 0.85 for d in (4, 7)) if o1_read else None)
T["O2"] = dict(values={d: dict(kept=o[d]["kept"]) for d in (3, 5)}, rule="kept >= 0.90 at 3 and 5 digits",
               passed=all(o[d]["kept"] >= 0.90 for d in (3, 5)))
E["verdict_olmo_move"] = "undecided" if not o1_read else \
    "supported" if (T["O1"]["passed"] and T["O2"]["passed"]) else "not supported"

for k in T:
    if isinstance(T[k], dict) and "passed" in T[k] and T[k]["passed"] is not None:
        T[k]["passed"] = bool(T[k]["passed"])
D.mkdir(exist_ok=True)
json.dump(E, open(D / "EVALUATION.json", "w"), indent=1, default=float)
for k, v in T.items():
    if k == "B5":
        for m, w in v.items(): print(f"B5 {m:10s} read={w['read']} passed={w['passed']} {w['values']}")
    else:
        print(f"{k}: {'NOT READ' if v['passed'] is None else 'PASS' if v['passed'] else 'FAIL'}  {v['values']}")
print("boundary account:", E["verdict_boundary"], "| Qwen3 early teen copy:", E["verdict_qwen_copy"], "| OLMo-2 move:", E["verdict_olmo_move"])
