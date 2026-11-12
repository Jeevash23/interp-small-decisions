"""Independent check of round 2 (written 9 Nov 2026 after the confirmation ran; post hoc).
Usage: check_report2.py [REPORT-2.md or another markdown file ...]

Part A recomputes every PROTOCOL-2 test from the raw files in outputs/r2conf/ with code written separately from
code/r2conf_evaluate.py (nothing is imported from it), and compares each pass/fail and each reported value with
outputs/r2conf/EVALUATION-2.json.
Part B checks every number in the given markdown files. A number passes if it equals, at the precision it is written
with, a value in: EVALUATION-2.json, the raw base summaries in outputs/r2conf/, the values recomputed in part A
(including derived ranges and differences listed below), the exploration outputs outputs/r2/e06_*.json and
outputs/r2/e04_*.json, or a number written in PROTOCOL-2.md. Numbers in back-ticks, in lines starting with
'<!-- nocheck -->' or in list items whose text starts with it, and labels such as L0H2, MLP24, R3 or X1 are skipped.
Exits 1 if anything fails."""
import sys, re, json, bisect, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "outputs" / "r2conf"
J = lambda n: json.load(open(C / n))
E = J("EVALUATION-2.json")["tests"]
fails = []
mine = {}          # test id -> (passed, {name: value})


def close(a, b):
    if isinstance(a, dict):
        return all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, str):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


# ------------------------------------------------------------------ part A: recompute
def frac_via(base, ablated, routed):
    """share of the full teen loss that a routed ablation reproduces"""
    full = base["teen"] - ablated["teen"]
    return (base["teen"] - routed["teen"]) / full


q = J("qwen.json")
for d in ("4", "6"):
    R = q["digits"][d]; b = R["base"]
    loss = b["teen"] - R["full_L0H2"]["teen"]
    via = {k: frac_via(b, R["full_L0H2"], v) for k, v in R["path_L0H2"].items()}
    rest = sorted(((v, k) for k, v in via.items() if k != "mlp0"), reverse=True)
    ok = loss >= 0.30 and via["mlp0"] >= 0.60 and rest[0][0] <= 0.15
    mine[f"R1@{d}"] = (ok if loss >= 0.30 else None, dict(full_loss=loss, via_mlp0=via["mlp0"], max_other=rest[0][0], max_other_name=rest[0][1]))
    own = b["teen"] - R["path_L0H2"]["mlp0"]["teen"]
    ctl = sorted(((b["teen"] - c["via_mlp0"]["teen"], c["head"]) for c in R["ctl_L0"]), reverse=True)
    mine[f"R2@{d}"] = (own > ctl[0][0], dict(l0h2=own, max_control=ctl[0][0], max_control_head=ctl[0][1]))
    m1 = {k: frac_via(b, R["full_L0H2"], v) for k, v in R["med_L0H2"].items()}
    m2 = {k: frac_via(b, R["full_L11H9"], v) for k, v in R["med_L11H9"].items()}
    top1, top2 = max(m1, key=m1.get), max(m2, key=m2.get)
    nx1 = max(v for k, v in m1.items() if k != "mlp24"); nx2 = max(v for k, v in m2.items() if k != "mlp24")
    mine[f"R3@{d}"] = (top1 == "mlp24" and top2 == "mlp24" and m1["mlp24"] >= 0.40 and m2["mlp24"] >= 0.40,
                       dict(l0h2_mlp24=m1["mlp24"], l0h2_next=nx1, l11h9_mlp24=m2["mlp24"], l11h9_next=nx2,
                            l11h9_full_loss=b["teen"] - R["full_L11H9"]["teen"]))
    lens = [x["teen"] for x in R["lens"]]
    rise = {l: lens[l] - lens[l - 1] for l in range(1, len(lens))}
    best = max(rise, key=rise.get)
    mine[f"R4@{d}"] = (best == 24 and rise[24] >= 1.0, dict(rise24=rise[24], next_rise=max(v for l, v in rise.items() if l != 24),
                                                            lens23=lens[23], lens24=lens[24]))
    dla = [x[0] / b["teen"] for x in R["dla_mlp"]]
    oth = max(abs(x) for i, x in enumerate(dla) if i != 24)
    mine[f"R5@{d}"] = (dla[24] >= 0.50 and oth <= 0.35, dict(mlp24_share=dla[24], max_other_share=oth))
    P24, P18 = R["profile"]["mlp24"], R["profile"]["mlp18"]
    def own_spec(P, cls, x):
        v = P[f"{cls}|{x}"]; others = [v[i] for i in range(3) if i != x - 1]
        return v[x - 1] - (others[0] + others[1]) / 2
    thm = [P24[f"teen|{x}"][3] - (P24[f"teen|{x}"][0] + P24[f"teen|{x}"][1] + P24[f"teen|{x}"][2]) / 3 for x in (1, 2, 3)]
    s24 = [own_spec(P24, "teen", x) for x in (1, 2, 3)]
    s18t = [own_spec(P18, "teen", x) for x in (1, 2, 3)]; s18e = [own_spec(P18, "end123", x) for x in (1, 2, 3)]
    ok = min(thm) >= 1.5 and max(map(abs, s24)) <= 0.6 and min(s18t + s18e) >= 0.4 and abs(sum(s18t) / 3 - sum(s18e) / 3) <= 0.4
    mine[f"R6@{d}"] = (ok, dict(mlp24_th_minus_mean=thm, mlp24_own_spec=s24, mlp18_own_spec_teen=s18t, mlp18_own_spec_nonteen=s18e))

q2 = J("qwen2.json"); drop = {}
for d in ("2", "3", "4"):
    R = q2["digits"][d]; bt = R["base"]["teen"]
    a, ab, bb = R["sets"]["A"]["teen"] / bt, R["sets"]["A+B"]["teen"] / bt, R["sets"]["B"]["teen"] / bt
    rnd = [x["teen"] / bt for x in R["random"]]
    mine[f"S1@{d}"] = (ab < min(rnd), dict(a=a, ab=ab, random_min=min(rnd), random_max=max(rnd)))
    mine[f"S2@{d}"] = (bb >= 0.85, dict(b=bb))
    drop[d] = a - ab
ab2 = q2["digits"]["2"]["sets"]["A+B"]["teen"] / q2["digits"]["2"]["base"]["teen"]
mine["S3"] = (drop["2"] >= 0.15 and drop["2"] > max(drop["3"], drop["4"]) and ab2 >= 0.30, dict(drop=drop, ab2=ab2))

o = J("olmo.json")["digits"]; IN, CR = ["3", "5", "8", "9"], ["4", "7", "10"]
au = {d: o[d]["base"]["auroc"] for d in o}
mine["O3"] = (min(au[d] for d in IN) >= 0.88 and all(au[d] <= 0.82 and min(au[i] for i in IN) - au[d] >= 0.10 for d in CR), dict(auroc=au))
for d in IN + CR:
    bt = o[d]["base"]["teen"]; k = o[d]["tri"]["teen"] / bt; rmin = min(x["teen"] / bt for x in o[d]["random"])
    if d in CR:
        mine[f"O4@{d}"] = ((k <= 0.60 and k < rmin) if bt >= 0.30 else None, dict(kept=k, random_min=rmin, base=bt))
    else:
        mine[f"O4@{d}"] = (k >= 0.90, dict(kept=k, random_min=rmin, base=bt))
pr = {d: o[d]["probe"] for d in o}
mine["O5"] = (all(pr[d]["2"] >= 0.90 and pr[d]["2"] - pr[d]["15"] >= 0.10 for d in CR) and all(pr[d]["15"] >= 0.93 for d in IN), dict(probe=pr))
mx = {d: max(o[d]["vnorm_ratio"]) for d in CR}; mn5 = min(o["5"]["vnorm_ratio"])
mine["O6"] = (max(mx.values()) <= 0.65 and mn5 >= 0.65, dict(max_ratio=mx, min_ratio_5=mn5))
st = {d: o[d]["steer_L8"] for d in ["3"] + CR}
ok7 = True
for d in CR:
    r = [st[d][f"rand{j}"] for j in range(5)]
    ok7 &= st[d]["v3"] <= -1.5 and st[d]["v3"] < min(r) and max(map(abs, r)) <= 0.5
mine["O7"] = (ok7, dict(steer=st))
for d in CR:
    b, t = o[d]["base"], o[d]["tri"]
    sh = {k: frac_via(b, t, v) for k, v in o[d]["path_tri"].items()}
    loss = b["teen"] - t["teen"]; nxt = max(v for k, v in sh.items() if k != "mlp0")
    mine[f"O8@{d}"] = ((sh["mlp0"] >= 0.50 and sh["mlp0"] >= nxt) if loss >= 0.30 else None, dict(via_mlp0=sh["mlp0"], next=nxt, loss=loss))

for f in sorted(C.glob("other_*.json")):
    r = json.load(open(f)); nm = r["model"]
    for d in ("3", "4"):
        R = r["digits"][d]; b = R["base"]
        for k, v in R["sets"].items():
            tk, pk = v["teen"] / b["teen"], v["pull"] / b["pull"]
            rt = min(x["teen"] / b["teen"] for x in R["random"][k]); rp = min(x["pull"] / b["pull"] for x in R["random"][k])
            if k == "PREV":
                mine[f"X:{nm}:PREV@{d}"] = (tk <= 0.85 and tk < rt and pk >= 0.55, dict(teen_kept=tk, random_min=rt, pull_kept=pk))
            else:
                mine[f"X:{nm}:CUR@{d}"] = (pk <= 0.30 and pk < rp and tk >= 0.90, dict(pull_kept=pk, random_min=rp, teen_kept=tk))

tested = {k: v for k, v in E.items() if not k.startswith("base_")}
if set(tested) != set(mine):
    fails.append(f"test sets differ: only in EVALUATION {sorted(set(tested) - set(mine))}, only here {sorted(set(mine) - set(tested))}")
for k in sorted(set(tested) & set(mine)):
    ok, vals = mine[k]
    if ok != tested[k]["passed"]:
        fails.append(f"{k}: pass/fail differs (here {ok}, evaluator {tested[k]['passed']})")
    for n, v in vals.items():
        if not close(v, tested[k]["values"][n]):
            fails.append(f"{k}.{n}: value differs (here {v}, evaluator {tested[k]['values'][n]})")
n_read = sum(1 for k in mine if mine[k][0] is not None); n_pass = sum(1 for k in mine if mine[k][0])
print(f"part A: {len(mine)} tests recomputed; {n_pass} of {n_read} read tests pass; "
      f"{'agrees with EVALUATION-2.json' if not fails else str(len(fails)) + ' disagreements'}")

# ------------------------------------------------------------------ part B: numbers in the markdown
pool = set()


def walk(x):
    if isinstance(x, bool) or x is None or isinstance(x, str): return
    if isinstance(x, (int, float)):
        if math.isfinite(x): pool.update((float(x), -float(x)))
        return
    items = x.items() if isinstance(x, dict) else enumerate(x)
    for k, v in items:
        walk(v)
        if isinstance(k, str) and re.fullmatch(r"-?\d+(\.\d+)?", k): pool.add(float(k))


walk(E); walk({k: v[1] for k, v in mine.items()})
for n in ("qwen.json", "qwen2.json", "olmo.json") + tuple(p.name for p in C.glob("other_*.json")):
    r = J(n); walk({d: R["base"] for d, R in r["digits"].items()})
# derived: counts of tests, steering random maxima, MLP0-route next receivers, A-minus-(A+B) drops, E06 ratios
walk([n_pass, n_read, len(mine)])
for pre in ("R", "S", "O", "X:"):                         # tests and passes per account
    ks = [k for k in mine if k.startswith(pre) and mine[k][0] is not None]
    walk([len(ks), sum(mine[k][0] for k in ks)])
walk(112)                                                 # the worked example "112th" in the question
TOK_T = set()                                             # checkpoint sizes written in trillions of tokens
for d in CR:
    walk(max(abs(st[d][f"rand{j}"]) for j in range(5)))
for f in sorted((ROOT / "outputs" / "r2").glob("e06_*.json")):
    r = json.load(open(f)); walk(r["digits"])
    R4 = r["digits"]["4"]; walk(R4["tri"]["teen"] / R4["base"]["teen"])
    stp, tokb = re.findall(r"step(\d+)-tokens(\d+)B", r["rev"])[0]
    walk([float(stp), float(tokb)]); TOK_T.add(round(float(tokb) / 1000, 1))
for f in sorted((ROOT / "outputs" / "r2").glob("e04_*.json")):
    walk(json.load(open(f)))
for t in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", (ROOT / "PROTOCOL-2.md").read_text()):
    pool.add(float(t))
vals = sorted(pool)


def matches(tok):
    dec = len(tok.split(".")[1]) if "." in tok else 0
    v = float(tok); h = 0.5 * 10 ** -dec + 1e-9
    i = bisect.bisect_left(vals, v - h)
    return i < len(vals) and vals[i] <= v + h


files = [Path(a) for a in sys.argv[1:]] or [ROOT / "REPORT-2.md"]
for rep in files:
    n = 0; bad = []
    for ln, line in enumerate(rep.read_text().splitlines(), 1):
        s = line.lstrip("-*0123456789. ")
        if line.startswith("<!-- nocheck -->") or s.startswith("<!-- nocheck -->"): continue
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"MLP\d+[–-]\d+", "", line)                                # ranges of layers such as MLP12–27
        line = re.sub(r"\]\([^)]*\)", "]", line)                                   # link targets
        line = re.sub(r"L\d+H\d+", "", line)
        line = re.sub(r"\b(P|Q|B|O|D|R|S|X|E|C|L|H|A|T|MLP|v|v_)\d+\b", "", line)  # labels: R3, X1, E06, MLP24, v_3
        line = re.sub(r"\b(Qwen3|OLMo-2|Gemma-4|SmolLM2|GPT-2|Pythia)(-[\w.]+)*", "", line)   # model names
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)T\b", line):         # e.g. 1.0T tokens
            n += 1
            if float(m.group(1)) not in TOK_T: bad.append(f"  line {ln}: {m.group()}")
        line = re.sub(r"(?<![\w.])\d+(?:\.\d+)?T\b", "", line)
        line = re.sub(r"(?<![\w.])(\d+)B\b", r"\1", line)
        for m in re.finditer(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?", line):
            tok = m.group().replace("−", "-").replace(",", "")
            n += 1
            if not (matches(tok) or matches(tok.lstrip("-"))):
                bad.append(f"  line {ln}: {tok}")
    print(f"part B, {rep.name}: {n} numbers checked, {len(bad)} not found")
    if bad:
        print("\n".join(bad)); fails.append(f"{rep.name}: {len(bad)} numbers not found")

if fails:
    print("FAIL"); print("\n".join(fails)); sys.exit(1)
print("PASS")
