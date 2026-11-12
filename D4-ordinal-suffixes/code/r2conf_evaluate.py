"""Round-2 evaluator (PROTOCOL-2.md): reads outputs/r2conf/*.json (or outputs/r2dry/ with --dry) and writes
EVALUATION-2.json with every test's values, its threshold and pass/fail, and the verdict per account.
Usage: r2conf_evaluate.py [--dry]"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DRY = "--dry" in sys.argv
D = ROOT / "outputs" / ("r2dry" if DRY else "r2conf")
load = lambda n: json.load(open(D / n)) if (D / n).exists() else None
T = {}   # test id -> dict(pass=bool or None, values=..., rule=str)


def put(tid, ok, rule, **vals):
    T[tid] = dict(passed=(None if ok is None else bool(ok)), rule=rule, values=vals)


def share(base, full, x, q="teen"):
    tot = base[q] - full[q]
    return (base[q] - x[q]) / tot if abs(tot) > 1e-9 else float("nan")


def spec(p, c, x):
    v = p[f"{c}|{x}"]; return v[x - 1] - (sum(v[:3]) - v[x - 1]) / 2


# ------------------------------------------------------------ Qwen3: the path (R1-R6)
q = load("qwen.json")
if q:
    for d in ("4", "6"):
        R = q["digits"][d]; b = R["base"]; f = R["full_L0H2"]; P = R["path_L0H2"]
        loss = b["teen"] - f["teen"]
        sh = {k: share(b, f, v) for k, v in P.items()}
        other = max(v for k, v in sh.items() if k != "mlp0")
        read = loss >= 0.30
        put(f"R1@{d}", (sh["mlp0"] >= 0.60 and other <= 0.15) if read else None,
            "L0H2 teen loss through MLP0 >= 0.60 of its full loss; every other receiver <= 0.15 (read if full loss >= 0.30)",
            full_loss=loss, via_mlp0=sh["mlp0"], max_other=other,
            max_other_name=max(((v, k) for k, v in sh.items() if k != "mlp0"))[1])
        own = b["teen"] - P["mlp0"]["teen"]
        ctl = [(b["teen"] - c["via_mlp0"]["teen"], c["head"]) for c in R["ctl_L0"]]
        put(f"R2@{d}", all(own > c for c, _ in ctl),
            "L0H2's teen loss through MLP0 exceeds that of every other layer-0 head through MLP0",
            l0h2=own, max_control=max(ctl)[0], max_control_head=max(ctl)[1])
        m = {k: share(b, f, v) for k, v in R["med_L0H2"].items()}
        f2 = R["full_L11H9"]; m2 = {k: share(b, f2, v) for k, v in R["med_L11H9"].items()}
        put(f"R3@{d}", m["mlp24"] >= 0.40 and m["mlp24"] == max(m.values()) and m2["mlp24"] >= 0.40 and m2["mlp24"] == max(m2.values()),
            "MLP24 is the largest mediator (MLP12-27) of L0H2's and of L11H9's teen loss, each >= 0.40",
            l0h2_mlp24=m["mlp24"], l0h2_next=max(v for k, v in m.items() if k != "mlp24"),
            l11h9_mlp24=m2["mlp24"], l11h9_next=max(v for k, v in m2.items() if k != "mlp24"),
            l11h9_full_loss=b["teen"] - f2["teen"])
        lens = [x["teen"] for x in R["lens"]]; rises = [lens[l] - lens[l - 1] for l in range(1, len(lens))]
        put(f"R4@{d}", rises[23] >= 1.0 and rises[23] == max(rises),
            "the largest single-layer rise of the lens teen effect is at layer 24, and >= 1.0",
            rise24=rises[23], next_rise=max(r for i, r in enumerate(rises) if i != 23),
            lens23=lens[23], lens24=lens[24])
        dla = [x[0] for x in R["dla_mlp"]]
        put(f"R5@{d}", dla[24] >= 0.50 * b["teen"] and max(abs(x) for i, x in enumerate(dla) if i != 24) <= 0.35 * b["teen"],
            "MLP24's DLA >= 0.50 of the teen effect; every other MLP's |DLA| <= 0.35 of it",
            mlp24_share=dla[24] / b["teen"], max_other_share=max(abs(x) for i, x in enumerate(dla) if i != 24) / b["teen"])
        p24, p18 = R["profile"]["mlp24"], R["profile"]["mlp18"]
        thm = [p24[f"teen|{x}"][3] - sum(p24[f"teen|{x}"][:3]) / 3 for x in (1, 2, 3)]
        s24 = [spec(p24, "teen", x) for x in (1, 2, 3)]
        s18t = [spec(p18, "teen", x) for x in (1, 2, 3)]; s18e = [spec(p18, "end123", x) for x in (1, 2, 3)]
        put(f"R6@{d}", min(thm) >= 1.5 and max(abs(x) for x in s24) <= 0.6 and min(s18t + s18e) >= 0.4
            and abs(np.mean(s18t) - np.mean(s18e)) <= 0.4,
            "MLP24 for teens: th - mean(st,nd,rd) >= 1.5 at each last digit, |own-suffix specificity| <= 0.6; "
            "MLP18: own-suffix specificity >= 0.4 for both classes and each digit, teen and non-teen means within 0.4",
            mlp24_th_minus_mean=thm, mlp24_own_spec=s24, mlp18_own_spec_teen=s18t, mlp18_own_spec_nonteen=s18e)
        put(f"base_qwen@{d}", None, "descriptive", **b)

# ------------------------------------------------------------ Qwen3: the backup route (S1-S3)
q2 = load("qwen2.json")
if q2:
    kept = lambda R, x: x["teen"] / R["base"]["teen"]
    for d in ("2", "3", "4"):
        R = q2["digits"][d]
        ab, a = kept(R, R["sets"]["A+B"]), kept(R, R["sets"]["A"])
        rnd = [kept(R, x) for x in R["random"]]
        put(f"S1@{d}", ab < min(rnd), "A+B kept below every A + random pair from layers 2-6",
            a=a, ab=ab, random_min=min(rnd), random_max=max(rnd))
        put(f"S2@{d}", kept(R, R["sets"]["B"]) >= 0.85, "B alone keeps >= 0.85 of the teen effect", b=kept(R, R["sets"]["B"]))
    drop = {d: kept(q2["digits"][d], q2["digits"][d]["sets"]["A"]) - kept(q2["digits"][d], q2["digits"][d]["sets"]["A+B"]) for d in ("2", "3", "4")}
    ab2 = kept(q2["digits"]["2"], q2["digits"]["2"]["sets"]["A+B"])
    put("S3", drop["2"] >= 0.15 and drop["2"] > drop["3"] and drop["2"] > drop["4"] and ab2 >= 0.30,
        "at 2 digits B removes >= 0.15 beyond A, more than at 3 and 4 digits, and A+B still keeps >= 0.30",
        drop=drop, ab2=ab2)

# ------------------------------------------------------------ OLMo-2 (O3-O8)
o = load("olmo.json")
if o:
    R = o["digits"]; IN, CR = ("3", "5", "8", "9"), ("4", "7", "10")
    au = {d: R[d]["base"]["auroc"] for d in R}
    put("O3", all(au[d] >= 0.88 for d in IN) and all(au[d] <= 0.82 and au[d] <= min(au[i] for i in IN) - 0.10 for d in CR),
        "AUROC >= 0.88 at 3, 5, 8, 9 digits; <= 0.82 and at least 0.10 below every in-token length at 4, 7, 10", auroc=au)
    for d in IN + CR:
        b, t = R[d]["base"], R[d]["tri"]
        k = t["teen"] / b["teen"]; rnd = [x["teen"] / b["teen"] for x in R[d]["random"]]
        if d in CR:
            ok = (k <= 0.60 and k < min(rnd)) if b["teen"] >= 0.30 else None
            put(f"O4@{d}", ok, "triple kept <= 0.60 and below every random triple (read if base effect >= 0.30)",
                kept=k, random_min=min(rnd), base=b["teen"])
        else:
            put(f"O4@{d}", k >= 0.90, "triple kept >= 0.90", kept=k, random_min=min(rnd), base=b["teen"])
    pr = {d: R[d]["probe"] for d in R}
    put("O5", all(pr[d]["2"] >= 0.90 and pr[d]["15"] <= pr[d]["2"] - 0.10 for d in CR) and all(pr[d]["15"] >= 0.93 for d in IN),
        "probe at layer 2 >= 0.90 and layer 15 <= layer 2 - 0.10 at 4, 7, 10 digits; layer 15 >= 0.93 at 3, 5, 8, 9", probe=pr)
    vr = {d: R[d]["vnorm_ratio"] for d in R}
    put("O6", all(max(vr[d]) <= 0.65 for d in CR) and min(vr["5"]) >= 0.65,
        "|v_d|/|v_3| <= 0.65 at every layer for 4, 7, 10 digits; >= 0.65 at every layer for 5 digits",
        max_ratio={d: max(vr[d]) for d in CR}, min_ratio_5=min(vr["5"]))
    st = {d: R[d]["steer_L8"] for d in ("3",) + CR}
    put("O7", all(st[d]["v3"] <= -1.5 and all(st[d]["v3"] < st[d][f"rand{j}"] for j in range(5)) and max(abs(st[d][f"rand{j}"]) for j in range(5)) <= 0.5 for d in CR),
        "steering with v_3 at layer 8 lowers non-teen mean m by >= 1.5 at 4, 7, 10 digits, more than each of 5 random "
        "directions of the same norm, whose |change| <= 0.5", steer=st)
    for d in CR:
        b, t, P = R[d]["base"], R[d]["tri"], R[d]["path_tri"]
        sh = {k: share(b, t, v) for k, v in P.items()}
        put(f"O8@{d}", (sh["mlp0"] >= 0.50 and sh["mlp0"] == max(sh.values())) if b["teen"] - t["teen"] >= 0.30 else None,
            "triple's teen loss through MLP0 >= 0.50 and the largest receiver (read if the triple's loss >= 0.30)",
            via_mlp0=sh["mlp0"], next=max(v for k, v in sh.items() if k != "mlp0"), loss=b["teen"] - t["teen"])

# ------------------------------------------------------------ other models (X)
for f in sorted(D.glob("other_*.json")):
    r = json.load(open(f)); nm = r["model"]
    for d in ("3", "4"):
        R = r["digits"][d]; b = R["base"]
        for k, v in R["sets"].items():
            tk, pk = v["teen"] / b["teen"], v["pull"] / b["pull"]
            rt = [x["teen"] / b["teen"] for x in R["random"][k]]; rp = [x["pull"] / b["pull"] for x in R["random"][k]]
            if k == "PREV":
                put(f"X:{nm}:PREV@{d}", tk <= 0.85 and tk < min(rt) and pk >= 0.55,
                    "PREV: teen kept <= 0.85 and below every random same-size set from layers 0-1; pull kept >= 0.55",
                    teen_kept=tk, random_min=min(rt), pull_kept=pk)
            else:
                put(f"X:{nm}:CUR@{d}", pk <= 0.30 and pk < min(rp) and tk >= 0.90,
                    "CUR: pull kept <= 0.30 and below every random same-size set from layers 0-1; teen kept >= 0.90",
                    pull_kept=pk, random_min=min(rp), teen_kept=tk)

# ------------------------------------------------------------ verdicts
def verdict(prefix):
    ks = [k for k in T if k.startswith(prefix) and T[k]["passed"] is not None]
    if not ks: return "not run"
    n = sum(T[k]["passed"] for k in ks)
    return "supported" if n == len(ks) else ("partly supported" if n >= len(ks) / 2 else "not supported")


V = {"qwen_path": verdict("R"), "qwen_backup": verdict("S"), "olmo_weak_route": verdict("O"), "other_models": verdict("X:")}
out = dict(dry=DRY, tests=T, verdicts=V,
           n_tests=sum(1 for t in T.values() if t["passed"] is not None),
           n_passed=sum(1 for t in T.values() if t["passed"]))
json.dump(out, open(D / "EVALUATION-2.json", "w"), indent=1, default=float)
for k, v in T.items():
    if v["passed"] is not None: print(f"{k:22s} {'PASS' if v['passed'] else 'FAIL'}  " + json.dumps(v["values"], default=lambda x: round(float(x), 3))[:200])
print(V, f"{out['n_passed']}/{out['n_tests']}")
