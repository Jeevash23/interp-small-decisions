"""X1 independent report check (written after the confirmation run; no model; standard library plus numpy to read
the per-item .npz files). It does not import c03_verdicts.py, x1common.py or build_report.py. From the raw
confirmation outputs (outputs/conf/c01_*.json, c01_*_items.npz, c02_attn_qwen3-06b-bf16.json) it:
 1. recomputes every test value, the random minima/maxima and every pass/fail with its own code and the thresholds
    copied from PROTOCOL.md, and compares the pass/fail list and the lead verdicts with outputs/conf/verdicts.json;
 2. recomputes the bootstrap intervals with its own resampling (random.Random(0), 2,000 resamples, units as in the
    protocol) and checks that each interval printed in REPORT.md as [lo, hi] lies within 0.02 of one of its own
    intervals at both ends (its resampling differs from c03's numpy draws, so intervals agree only approximately);
 3. scans REPORT.md: every decimal number outside lines marked <!-- nocheck --> must equal, at the printed
    precision, one of the independently recomputed values (or appear among the frozen exploration values quoted from
    PROTOCOL.md, which are checked against the protocol text itself).
Exit status 1 if anything fails. Usage: check_report.py"""
import json, re, random, sys, zipfile, io, struct
from pathlib import Path

X1 = Path(__file__).resolve().parent.parent
C = X1 / "outputs" / "conf"
R = {t: json.loads((C / f"c01_{m}_{t}.json").read_text()) for m, t in
     [("qwen3-06b-bf16", "LB"), ("qwen3-06b-bf16", "IND"), ("qwen3-06b-bf16", "ORD"), ("qwen3-17b", "CAPS")]}
ATT = json.loads((C / "c02_attn_qwen3-06b-bf16.json").read_text())
VER = json.loads((C / "verdicts.json").read_text())
fails = []

K = {"LBw": ("LB", "width_effect", 1), "LBs": ("LB", "sep", 1), "cU": ("IND", "code|dedent|lq", 0), "cT": ("IND", "code|term|lq", 0),
     "gU": ("IND", "gen|dedent|lq", 0), "gT": ("IND", "gen|term|lq", 0), "t4": ("ORD", "d4|teen_effect", 1),
     "t5": ("ORD", "d5|teen_effect", 1), "p4": ("ORD", "d4|pull", 1), "p5": ("ORD", "d5|pull", 1),
     "Du": ("CAPS", "D_up", 1), "Dl": ("CAPS", "D_low", 1), "wl": ("CAPS", "word_lp", 0)}


def f(run, t, key, ratio):
    b = R[t]["base"][key]
    return run[key] / b if ratio else run[key] - b


def V(g, k):
    t, key, ratio = K[k]
    return f(R[t][g], t, key, ratio) if g in R[t] else None


def RR(g, k):
    t, key, ratio = K[k]
    xs = [f(R[t][x], t, key, ratio) for x in R[t] if x.startswith(g + "~r")]
    return xs


def below(g, k):
    xs = RR(g, k); v = V(g, k)
    return v is not None and len(xs) == 20 and v < min(xs)


def quiet(g, a=0.03, u=0.10):
    return all(abs(V(g, k)) <= a for k in ("cU", "cT", "gT")) and abs(V(g, "gU")) <= u


mine = {
    "A1": V("L17H6", "LBw") <= 0.85 and below("L17H6", "LBw"),
    "A2": V("L17H6", "LBs") <= 0.95 and below("L17H6", "LBs"),
    "A3": V("L17H6", "gT") <= -0.010 and below("L17H6", "gT"),
    "A4": V("CARRY-L17H6", "LBw") >= 0.92 and V("TERM-L17H6", "LBw") >= 0.92 and V("CARRY", "LBw") <= 0.85 and V("TERM", "LBw") <= 0.85,
    "A5": abs(V("L17H6", "cU")) <= 0.03 and abs(V("L17H6", "gU")) <= 0.10 and 0.90 <= V("L17H6", "t4") <= 1.10 and 0.90 <= V("L17H6", "t5") <= 1.10,
    "A6": V("L17H6", "Du") <= 0.97 and below("L17H6", "Du") and abs(V("L17H6", "wl")) <= 0.03,
    "B1": V("WRITE", "LBw") <= 0.70 and quiet("WRITE"),
    "B2": V("COUNT", "LBs") <= 0.50 and quiet("COUNT"),
    "B3": V("READ-L17H6", "LBw") <= 0.75 and quiet("READ-L17H6"),
    "B4": V("DED", "cU") <= -0.40 and below("DED", "cU") and 0.95 <= V("DED", "LBw") <= 1.05 and 0.95 <= V("DED", "LBs") <= 1.05,
    "C1": V("L0H2", "t4") <= 0.75 and V("L0H2", "t5") <= 0.75 and below("L0H2", "t4") and below("L0H2", "t5"),
    "C2": V("L0H2", "LBs") <= 0.93 and below("L0H2", "LBs"),
    "C3": quiet("L0H2"),
    "C4": V("L0H3", "p4") <= 0.50 and V("L0H3", "p5") <= 0.50 and below("L0H3", "p4") and below("L0H3", "p5") and V("L0H3", "t4") >= 0.90 and V("L0H3", "t5") >= 0.90,
    "C5": V("L0H3", "LBs") <= 0.50 and below("L0H3", "LBs"),
    "C6": quiet("L0H3"),
    "C7": V("L0H3", "Dl") <= 0.60 and V("L0H3", "wl") <= -0.50 and below("L0H3", "Dl") and below("L0H3", "wl") and (1 - V("L0H3", "Dl")) >= 3 * (1 - V("L0H3", "Du")),
    "C8": all(ATT[t]["L0H3"]["dec"]["0"] >= 0.90 for t in ("LB", "IND", "ORD", "CAPS")) and all(ATT[t]["L0H2"]["dec"]["1"] >= 0.50 for t in ("LB", "ORD", "CAPS")),
    "D1": V("L11H9", "t4") <= 0.85 and V("L11H9", "t5") <= 0.90 and V("L11H9", "p4") <= 0.85 and V("L11H9", "p5") <= 0.85 and all(below("L11H9", k) for k in ("t4", "t5", "p4", "p5")),
    "D2": V("L11H9", "LBw") >= 0.93 and V("L11H9", "LBs") >= 0.93,
    "D3": quiet("L11H9", 0.05, 0.05),
    "D4": all(ATT[t]["L11H9"]["dec"]["0"] >= 0.85 for t in ("LB", "IND", "ORD", "CAPS")),
}
for n, p in mine.items():
    if VER["tests"][n]["passed"] is not bool(p): fails.append(f"verdict mismatch {n}: check {p} vs c03 {VER['tests'][n]['passed']}")
P = lambda *ns: all(mine[n] for n in ns)
lead = {"lead1_L17H6": "shared part confirmed" if P("A1", "A3", "A4") else "line-breaking reader only (indentation role not confirmed)" if P("A1", "A4") else "not confirmed",
        "lead2_separate": "confirmed" if P("B1", "B2", "B3", "B4") else "not confirmed",
        "account": "supported" if P("A1", "A4", "B1", "B4", "C1", "C2", "C4", "C5", "D1", "D2") else "partly supported"}
for k, v in lead.items():
    if VER["verdicts"][k] != v: fails.append(f"lead verdict mismatch {k}: {v} vs {VER['verdicts'][k]}")
print(f"tests recomputed: {sum(mine.values())}/{len(mine)} pass; verdicts agree: {not fails}")

# ---- pool of independently recomputed numbers
pool = set()
def add(x):
    if x is None: return
    for d in (1, 2, 3):
        pool.add(f"{x:.{d}f}".replace("-0.", "−0.") if x < 0 else f"{x:.{d}f}")
        pool.add(f"{abs(x):.{d}f}")
for t in R:
    for key, v in R[t]["base"].items():
        if isinstance(v, (int, float)): add(float(v)); pool.add(str(v))
for g in {x.split("~")[0] for t in R for x in R[t] if x != "base"}:
    for k in K:
        if g in R[K[k][0]]:
            add(V(g, k)); xs = RR(g, k)
            if xs: add(min(xs)); add(max(xs))
for t in ATT:
    for h, d in ATT[t].items():
        for sc in ("all", "dec"):
            for b, x in d[sc].items(): add(x)
        add(d["dec_last_newline"])
# the capitals gate on Qwen3-0.6B (D5's frozen gate, outputs/gate_qwen3-06b.json)
for k_, x in json.loads((X1 / "outputs" / "gate_qwen3-06b.json").read_text())["gate"].items():
    if isinstance(x, float): add(x)
# derived quantities quoted in the report
add(1 - V("L0H3", "Dl")); add(1 - V("L0H3", "Du")); add(3 * (1 - V("L0H3", "Du")))
for g in ("CARRY", "TERM"):
    add(V(g, "LBw") - V(g + "-L17H6", "LBw"))

# ---- own bootstrap intervals
import numpy as np
own_ci = []
rr = random.Random(0)
def boot(fun, n):
    out = []
    for _ in range(2000):
        idx = [rr.randrange(n) for _ in range(n)]
        try:
            x = fun(idx)
            if x == x and abs(x) != float("inf"): out.append(x)
        except ZeroDivisionError:
            pass
    out.sort(); return (out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]) if out else None
def npz(m, t): return dict(np.load(C / f"c01_{m}_{t}_items.npz", allow_pickle=True))
A = npz("qwen3-06b-bf16", "LB")
for g in ("L17H6", "L0H2", "L0H3", "WRITE", "COUNT", "READ-L17H6", "DED", "L11H9", "CARRY", "TERM", "CARRY-L17H6", "TERM-L17H6"):
    E0, E1 = A["base|E"].tolist(), A[f"{g}|E"].tolist(); n0, n1 = A["base|nat"].tolist(), A[f"{g}|nat"].tolist()
    own_ci.append(boot(lambda ix: sum(E1[i] for i in ix) / sum(E0[i] for i in ix), len(E0)))
    sep = lambda N, ix: sum(N[i][0] for i in ix) / sum(N[i][1] for i in ix) - sum(N[i][2] for i in ix) / sum(N[i][3] for i in ix)
    own_ci.append(boot(lambda ix: sep(n1, ix) / sep(n0, ix), len(n0)))
A = npz("qwen3-06b-bf16", "IND")
cl, cls, src = A["base|cluster"].tolist(), A["base|cls"].tolist(), A["base|src"].tolist()
units = sorted(set(cl)); members = {u: [i for i, c in enumerate(cl) if c == u] for u in units}
for g in ("L17H6", "DED", "L0H2", "L0H3", "L11H9", "WRITE", "COUNT", "READ-L17H6"):
    d = (A[f"{g}|lq"] - A["base|lq"]).tolist()
    for s_, c_ in (("code", "dedent"), ("code", "term"), ("gen", "dedent"), ("gen", "term")):
        def fn(ix, s_=s_, c_=c_):
            ii = [j for i in ix for j in members[units[i]] if src[j] == s_ and cls[j] == c_]
            return sum(d[j] for j in ii) / len(ii)
        own_ci.append(boot(fn, len(units)))
A = npz("qwen3-06b-bf16", "ORD")
nn, dg, oc = A["base|n"].tolist(), A["base|digits"].tolist(), A["base|cls"].tolist(); m0 = A["base|m"].tolist()
for g in ("L0H2", "L0H3", "L11H9", "L17H6"):
    m1 = A[f"{g}|m"].tolist()
    for dd in (4, 5):
        nums = sorted({x for x, y in zip(nn, dg) if y == dd}); mem = {x: [i for i in range(len(nn)) if nn[i] == x and dg[i] == dd] for x in nums}
        def eff(m, ii):
            t = [m[i] for i in ii if oc[i] == "teen"]; e = [m[i] for i in ii if oc[i] == "end123"]
            return sum(e) / len(e) - sum(t) / len(t)
        def fk(ix, m1=m1, mem=mem, nums=nums):
            ii = [j for i in ix for j in mem[nums[i]]]; return eff(m1, ii) / eff(m0, ii)
        def fp(ix, m1=m1, mem=mem, nums=nums):
            ii = [j for i in ix for j in mem[nums[i]] if oc[j] == "end123"]
            return (sum(m1[j] for j in ii) / len(ii)) / (sum(m0[j] for j in ii) / len(ii))
        own_ci.append(boot(fk, len(nums))); own_ci.append(boot(fp, len(nums)))
A = npz("qwen3-17b", "CAPS")
par = A["base|par"].tolist(); ups = sorted(set(par)); pm = {u: [i for i, p in enumerate(par) if p == u] for u in ups}
for g in ("L0H3", "L17H6", "L11H9", "L0H2"):
    for a in ("du", "dl"):
        b0, b1 = A[f"base|{a}"].tolist(), A[f"{g}|{a}"].tolist()
        own_ci.append(boot(lambda ix: sum(b1[j] for i in ix for j in pm[ups[i]]) / sum(b0[j] for i in ix for j in pm[ups[i]]), len(ups)))
    w0, w1 = A["base|wl"].tolist(), A[f"{g}|wl"].tolist()
    def fw(ix):
        jj = [j for i in ix for j in pm[ups[i]]]; return sum(w1[j] - w0[j] for j in jj) / len(jj)
    own_ci.append(boot(fw, len(ups)))
own_ci = [c for c in own_ci if c]

# ---- frozen exploration values quoted from the protocol
proto = (X1 / "PROTOCOL.md").read_text()
proto_nums = set(re.findall(r"−?\d+\.\d+", proto.replace("-", "−")))
for fn_ in (X1 / "outputs").glob("e01_summary_*.json"):
    js = json.loads(fn_.read_text())
    for g, row in js["groups"].items():
        for k_, cell in row.items():
            for kk in ("value", "rand_min", "rand_max"):
                x = cell.get(kk)
                if isinstance(x, (int, float)):
                    for d in (2, 3): proto_nums.add(f"{x:.{d}f}".replace("-", "−"))
    for k_, x in js["base"].items():
        for d in (2, 3): proto_nums.add(f"{x:.{d}f}".replace("-", "−"))

# ---- intervals (own bootstrap) -- values printed as [lo, hi]
rep = (X1 / "REPORT.md").read_text()
bad = []
ci_nums = set(); n_ci = 0
for m in re.finditer(r"\[(−?\d+\.\d+), (−?\d+\.\d+)\]", rep):
    if m.group(0) in proto: ci_nums.update(m.groups()); continue   # a threshold range quoted from PROTOCOL.md
    lo, hi = (float(x.replace("−", "-")) for x in m.groups()); n_ci += 1
    if not any(abs(lo - a) <= 0.02 and abs(hi - b) <= 0.02 for a, b in own_ci):
        fails.append(f"interval [{m.group(1)}, {m.group(2)}] not within 0.02 of any own interval")
    ci_nums.update(m.groups())
print(f"intervals checked: {n_ci}")
for i, line in enumerate(rep.splitlines(), 1):
    if "<!-- nocheck -->" in line: continue
    body = re.sub(r"`[^`]*`", "", line)                    # code spans (paths, file names)
    body = re.sub(r"\bL\d+H\d+\b", "", body)                # head names
    for m in re.finditer(r"[−-]?\d+\.\d+", body):
        s = m.group(0).replace("-", "−")
        if s in pool or s.lstrip("−") in pool or s in proto_nums or s in ci_nums: continue
        bad.append((i, s, line.strip()[:120]))
for b in bad: fails.append(f"REPORT.md line {b[0]}: {b[1]} not matched  | {b[2]}")
n_nums = sum(len(re.findall(r"[−-]?\d+\.\d+", re.sub(r"\bL\d+H\d+\b", "", re.sub(r"`[^`]*`", "", l))))
             for l in rep.splitlines() if "<!-- nocheck -->" not in l)
print(f"numbers in REPORT.md checked: {n_nums}; unmatched: {len(bad)}")

# ---- context checks: every matrix cell, and each key value next to its own label
fm = lambda x, d=3: f"{x:.{d}f}".replace("-", "−")
MT = {"LB": [("LBw"), ("LBs")], "IND": ["cU", "cT", "gU", "gT"], "ORD": ["t4", "t5", "p4", "p5"], "CAPS": ["Du", "Dl", "wl"]}
sec = rep.split("## 4.")[1].split("## 5.")[0]
titles = {"Line-breaking": "LB", "Indentation": "IND", "Ordinal suffixes": "ORD", "Capitals": "CAPS"}
n_cells = 0; cur = None
for line in sec.splitlines():
    for tt, t in titles.items():
        if line.startswith(f"**{tt}"): cur = t
    m = re.match(r"\| ([A-Z0-9a-z\-]+) \| (.*) \|$", line)
    if not m or cur is None or m.group(1) in ("Group",): continue
    g = m.group(1); cells = m.group(2).split(" | ")
    for k, cell in zip(MT[cur], cells):
        xs = RR(g, k); want = f"{fm(V(g, k))} ({fm(min(xs))} to {fm(max(xs))})"
        n_cells += 1
        if cell.strip() != want: fails.append(f"matrix {cur} {g} {k}: '{cell.strip()}' != '{want}'")
print(f"matrix cells checked: {n_cells}")
frag = [
    f"width effect kept {fm(V('L17H6', 'LBw'))}", f"separation kept {fm(V('L17H6', 'LBs'))}", f"change {fm(V('L17H6', 'gT'))}",
    f"CARRY {fm(V('CARRY', 'LBw'))}, CARRY−L17H6 {fm(V('CARRY-L17H6', 'LBw'))}", f"TERM {fm(V('TERM', 'LBw'))}, TERM−L17H6 {fm(V('TERM-L17H6', 'LBw'))}",
    f"code unmarked dedents {fm(V('L17H6', 'cU'))}", f"generated unmarked dedents {fm(V('L17H6', 'gU'))}",
    f"teen effect kept {fm(V('L17H6', 't4'))} and {fm(V('L17H6', 't5'))}",
    f"L17H6 D_up kept {fm(V('L17H6', 'Du'))}", f"D_low kept {fm(V('L17H6', 'Dl'))}, word log-prob change {fm(V('L17H6', 'wl'))}",
    f"WRITE: width kept {fm(V('WRITE', 'LBw'))}", f"COUNT: separation kept {fm(V('COUNT', 'LBs'))}", f"READ−L17H6: width kept {fm(V('READ-L17H6', 'LBw'))}",
    f"code unmarked dedents {fm(V('DED', 'cU'))}", f"line-breaking width kept {fm(V('DED', 'LBw'))}, separation kept {fm(V('DED', 'LBs'))}",
    f"Teen effect kept {fm(V('L0H2', 't4'))}", f"and {fm(V('L0H2', 't5'))}", f"Separation kept {fm(V('L0H2', 'LBs'))}",
    f"last-digit pull kept {fm(V('L0H3', 'p4'))}", f"and {fm(V('L0H3', 'p5'))}", f"teen effect kept {fm(V('L0H3', 't4'))} and {fm(V('L0H3', 't5'))}",
    f"separation kept {fm(V('L0H3', 'LBs'))}", f"width kept {fm(V('L0H3', 'LBw'))}",
    f"D_low kept {fm(V('L0H3', 'Dl'))}", f"word log-prob change {fm(V('L0H3', 'wl'))}", f"D_up kept {fm(V('L0H3', 'Du'))}",
    f"({fm(1 - V('L0H3', 'Dl'))}) is well over three times the loss on D_up (three times is {fm(3 * (1 - V('L0H3', 'Du')))})",
    f"Teen effect kept {fm(V('L11H9', 't4'))}", f"and {fm(V('L11H9', 't5'))}", f"last-digit pull kept {fm(V('L11H9', 'p4'))}",
    f"width kept {fm(V('L11H9', 'LBw'))}, separation kept {fm(V('L11H9', 'LBs'))}",
    "L0H3 on the current token: LB {:.2f}, IND {:.2f}, ORD {:.2f}, CAPS text {:.2f}".format(*[ATT[t]["L0H3"]["dec"]["0"] for t in ("LB", "IND", "ORD", "CAPS")]),
    "L0H2 on the previous token: LB {:.2f}, IND {:.2f}, ORD {:.2f}, CAPS text {:.2f}".format(*[ATT[t]["L0H2"]["dec"]["1"] for t in ("LB", "IND", "ORD", "CAPS")]),
    "Attention on the current token: LB {:.2f}, IND {:.2f}, ORD {:.2f}, CAPS text {:.2f}".format(*[ATT[t]["L11H9"]["dec"]["0"] for t in ("LB", "IND", "ORD", "CAPS")]),
    f"(width effect kept {fm(V('L17H6', 'LBw'))})", f"CARRY keeps {fm(V('CARRY', 'LBw'))} of the width effect and {fm(V('CARRY-L17H6', 'LBw'))} without L17H6",
    f"TERM keeps {fm(V('TERM', 'LBw'))}, and {fm(V('TERM-L17H6', 'LBw'))} without it", f"({fm(V('L17H6', 'gT'))} in mean log-probability",
    f"(teen effect kept {fm(V('L0H2', 't4'))} and {fm(V('L0H2', 't5'))})", f"(separation kept {fm(V('L0H2', 'LBs'))})",
    f"(last-digit pull kept {fm(V('L0H3', 'p4'))} and {fm(V('L0H3', 'p5'))})", f"(separation kept {fm(V('L0H3', 'LBs'))})",
    f"(D_low kept {fm(V('L0H3', 'Dl'))}, D_up kept {fm(V('L0H3', 'Du'))})",
    f"(teen effect kept {fm(V('L11H9', 't4'))}, last-digit pull kept {fm(V('L11H9', 'p4'))} at 4 digits)", f"line-breaking ({fm(V('L11H9', 'LBw'))})",
    f"{sum(mine.values())} of {len(mine)} frozen tests pass",
]
miss = [f for f in frag if f.replace("-", "−") not in rep.replace("-", "−")]
for f_ in miss: fails.append(f"key value not found in its context: {f_}")
print(f"key values checked in context: {len(frag)}; missing: {len(miss)}")
if fails:
    print("\n".join(fails)); sys.exit(1)
print("all checks passed")
