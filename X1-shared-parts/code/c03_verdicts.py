"""X1 confirmation c03 (PROTOCOL.md): applies the frozen rules to the saved confirmation outputs.
Reads outputs/conf/c01_<model>_<task>.json (and _items.npz for descriptive intervals) and outputs/conf/c02_attn_qwen3-06b-bf16.json.
Writes outputs/conf/verdicts.json. No model is loaded.
Conventions (PROTOCOL.md): 'kept' = ablated / base of a task measure; 'change' = ablated - base of a mean log q(true)
(indentation) or of the lower-case word log-prob (capitals). "Below every random draw" compares with the minimum over
all 20 draws of the same group (for 'change' and 'kept' alike). Decisions use point estimates; the 95% intervals
(2,000 bootstrap resamples, numpy seed 0; units: width items, paragraphs, repositories/programs, numbers) are description.
Usage: c03_verdicts.py [--smoke]"""
import sys, json
from pathlib import Path
import numpy as np

SMOKE = "--smoke" in sys.argv
D = Path(__file__).resolve().parent.parent / "outputs" / ("conf_smoke" if SMOKE else "conf")
M6, M17 = "qwen3-06b-bf16", "qwen3-17b"
R = {}
for model, task in [(M6, "LB"), (M6, "IND"), (M6, "ORD"), (M17, "CAPS")]:
    p = D / f"c01_{model}_{task}.json"
    R[task] = json.loads(p.read_text()) if p.exists() else None
ATT = json.loads((D / "c02_attn_qwen3-06b-bf16.json").read_text()) if (D / "c02_attn_qwen3-06b-bf16.json").exists() else None

KEYS = {"LBw": ("LB", "width_effect", "kept"), "LBs": ("LB", "sep", "kept"),
        "cU": ("IND", "code|dedent|lq", "change"), "cT": ("IND", "code|term|lq", "change"),
        "gU": ("IND", "gen|dedent|lq", "change"), "gT": ("IND", "gen|term|lq", "change"),
        "t4": ("ORD", "d4|teen_effect", "kept"), "t5": ("ORD", "d5|teen_effect", "kept"),
        "p4": ("ORD", "d4|pull", "kept"), "p5": ("ORD", "d5|pull", "kept"),
        "Du": ("CAPS", "D_up", "kept"), "Dl": ("CAPS", "D_low", "kept"), "wl": ("CAPS", "word_lp", "change")}


def get(run, task, key, kind):
    b = R[task]["base"][key]; v = run[key]
    return v / b if kind == "kept" else v - b


def val(g, k):
    task, key, kind = KEYS[k]
    if R[task] is None or g not in R[task]: return None
    return get(R[task][g], task, key, kind)


def rmin(g, k):
    task, key, kind = KEYS[k]
    rr = [get(R[task][x], task, key, kind) for x in R[task] if x.startswith(g + "~r")]
    return (min(rr), max(rr), len(rr)) if rr else (None, None, 0)


T, V = {}, {}


def rec(name, rule, values, passed):
    T[name] = dict(rule=rule, values=values, passed=None if passed is None else bool(passed))


def ok(*xs): return all(x is not None for x in xs)


def below(g, k):
    v = val(g, k); lo, hi, n = rmin(g, k)
    return v is not None and n == 20 and v < lo


def ind_quiet(g, b3=0.03, bu=0.10):
    vals = {k: val(g, k) for k in ("cU", "cT", "gT", "gU")}
    if not ok(*vals.values()): return vals, None
    return vals, all(abs(vals[k]) <= b3 for k in ("cU", "cT", "gT")) and abs(vals["gU"]) <= bu


# ---- A: L17H6
v = {k: val("L17H6", k) for k in ("LBw", "LBs", "gT")}
rec("A1", "L17H6: LB width kept <= 0.85 and below every random draw", dict(kept=v["LBw"], rand=rmin("L17H6", "LBw")),
    None if v["LBw"] is None else v["LBw"] <= 0.85 and below("L17H6", "LBw"))
rec("A2", "L17H6: LB separation kept <= 0.95 and below every random draw", dict(kept=v["LBs"], rand=rmin("L17H6", "LBs")),
    None if v["LBs"] is None else v["LBs"] <= 0.95 and below("L17H6", "LBs"))
rec("A3", "L17H6: generated terminal change <= -0.010 and below every random draw", dict(change=v["gT"], rand=rmin("L17H6", "gT")),
    None if v["gT"] is None else v["gT"] <= -0.010 and below("L17H6", "gT"))
a4 = {g: val(g, "LBw") for g in ("CARRY-L17H6", "TERM-L17H6", "CARRY", "TERM")}
rec("A4", "CARRY-L17H6 and TERM-L17H6: LB width kept >= 0.92; CARRY and TERM: <= 0.85", a4,
    None if not ok(*a4.values()) else a4["CARRY-L17H6"] >= 0.92 and a4["TERM-L17H6"] >= 0.92 and a4["CARRY"] <= 0.85 and a4["TERM"] <= 0.85)
a5 = dict(cU=val("L17H6", "cU"), gU=val("L17H6", "gU"), t4=val("L17H6", "t4"), t5=val("L17H6", "t5"))
rec("A5", "L17H6: |code unmarked change| <= 0.03, |generated unmarked change| <= 0.10, teen kept in [0.90, 1.10] at 4 and 5 digits", a5,
    None if not ok(*a5.values()) else abs(a5["cU"]) <= 0.03 and abs(a5["gU"]) <= 0.10 and all(0.90 <= a5[k] <= 1.10 for k in ("t4", "t5")))
a6 = dict(Du=val("L17H6", "Du"), wl=val("L17H6", "wl"), rand_Du=rmin("L17H6", "Du"))
rec("A6", "Qwen3-1.7B, L17H6: D_up kept <= 0.97 and below every random draw; |word log-prob change| <= 0.03", a6,
    None if not ok(a6["Du"], a6["wl"]) else a6["Du"] <= 0.97 and below("L17H6", "Du") and abs(a6["wl"]) <= 0.03)

# ---- B: separate parts
for name, g, k, th in (("B1", "WRITE", "LBw", 0.70), ("B2", "COUNT", "LBs", 0.50), ("B3", "READ-L17H6", "LBw", 0.75)):
    x = val(g, k); iv, q = ind_quiet(g)
    rec(name, f"{g}: LB {'width' if k == 'LBw' else 'separation'} kept <= {th}; indentation quiet (|change| <= 0.03 on code unmarked, "
              f"code terminal, generated terminal; <= 0.10 on generated unmarked)", dict(kept=x, ind=iv),
        None if x is None or q is None else x <= th and q)
b4 = dict(cU=val("DED", "cU"), rand_cU=rmin("DED", "cU"), LBw=val("DED", "LBw"), LBs=val("DED", "LBs"))
rec("B4", "DED: code unmarked change <= -0.40 and below every random draw; LB width and separation kept in [0.95, 1.05]", b4,
    None if not ok(b4["cU"], b4["LBw"], b4["LBs"]) else b4["cU"] <= -0.40 and below("DED", "cU") and 0.95 <= b4["LBw"] <= 1.05 and 0.95 <= b4["LBs"] <= 1.05)

# ---- C: early heads
c1 = dict(t4=val("L0H2", "t4"), t5=val("L0H2", "t5"), rand_t4=rmin("L0H2", "t4"), rand_t5=rmin("L0H2", "t5"))
rec("C1", "L0H2: teen kept <= 0.75 at 4 and 5 digits, each below every random draw", c1,
    None if not ok(c1["t4"], c1["t5"]) else c1["t4"] <= 0.75 and c1["t5"] <= 0.75 and below("L0H2", "t4") and below("L0H2", "t5"))
c2 = dict(LBs=val("L0H2", "LBs"), rand=rmin("L0H2", "LBs"))
rec("C2", "L0H2: LB separation kept <= 0.93 and below every random draw", c2,
    None if c2["LBs"] is None else c2["LBs"] <= 0.93 and below("L0H2", "LBs"))
iv, q = ind_quiet("L0H2"); rec("C3", "L0H2: indentation quiet (bounds as B1)", iv, q)
c4 = dict(p4=val("L0H3", "p4"), p5=val("L0H3", "p5"), t4=val("L0H3", "t4"), t5=val("L0H3", "t5"), rand_p4=rmin("L0H3", "p4"), rand_p5=rmin("L0H3", "p5"))
rec("C4", "L0H3: last-digit pull kept <= 0.50 at 4 and 5 digits, each below every random draw; teen kept >= 0.90 at both", c4,
    None if not ok(c4["p4"], c4["p5"], c4["t4"], c4["t5"]) else
    c4["p4"] <= 0.50 and c4["p5"] <= 0.50 and below("L0H3", "p4") and below("L0H3", "p5") and c4["t4"] >= 0.90 and c4["t5"] >= 0.90)
c5 = dict(LBs=val("L0H3", "LBs"), rand=rmin("L0H3", "LBs"))
rec("C5", "L0H3: LB separation kept <= 0.50 and below every random draw", c5, None if c5["LBs"] is None else c5["LBs"] <= 0.50 and below("L0H3", "LBs"))
iv, q = ind_quiet("L0H3"); rec("C6", "L0H3: indentation quiet (bounds as B1)", iv, q)
c7 = dict(Dl=val("L0H3", "Dl"), Du=val("L0H3", "Du"), wl=val("L0H3", "wl"), rand_Dl=rmin("L0H3", "Dl"), rand_wl=rmin("L0H3", "wl"))
rec("C7", "Qwen3-1.7B, L0H3: D_low kept <= 0.60 and word log-prob change <= -0.50, each below every random draw; "
          "(1 - D_low kept) >= 3 x (1 - D_up kept)", c7,
    None if not ok(c7["Dl"], c7["Du"], c7["wl"]) else c7["Dl"] <= 0.60 and c7["wl"] <= -0.50 and below("L0H3", "Dl") and below("L0H3", "wl")
    and (1 - c7["Dl"]) >= 3 * (1 - c7["Du"]))
if ATT:
    c8 = {t: dict(L0H3_self=ATT[t]["L0H3"]["dec"]["0"], L0H2_prev=ATT[t]["L0H2"]["dec"]["1"]) for t in ("LB", "IND", "ORD", "CAPS")}
    rec("C8", "attention at decision positions: L0H3 on the current token >= 0.90 in all four tasks; L0H2 on the previous token >= 0.50 in LB, ORD, CAPS",
        c8, all(c8[t]["L0H3_self"] >= 0.90 for t in c8) and all(c8[t]["L0H2_prev"] >= 0.50 for t in ("LB", "ORD", "CAPS")))
    d4 = {t: ATT[t]["L11H9"]["dec"]["0"] for t in ("LB", "IND", "ORD", "CAPS")}
    rec("D4", "attention at decision positions: L11H9 on the current token >= 0.85 in all four tasks", d4, all(x >= 0.85 for x in d4.values()))
else:
    rec("C8", "attention (missing output)", {}, None); rec("D4", "attention (missing output)", {}, None)

# ---- D: L11H9
d1 = {k: val("L11H9", k) for k in ("t4", "t5", "p4", "p5")}
d1r = {f"rand_{k}": rmin("L11H9", k) for k in d1}
rec("D1", "L11H9: teen kept <= 0.85 (4 digits) and <= 0.90 (5 digits); pull kept <= 0.85 at both; each below every random draw", dict(**d1, **d1r),
    None if not ok(*d1.values()) else d1["t4"] <= 0.85 and d1["t5"] <= 0.90 and d1["p4"] <= 0.85 and d1["p5"] <= 0.85
    and all(below("L11H9", k) for k in d1))
d2 = dict(LBw=val("L11H9", "LBw"), LBs=val("L11H9", "LBs"))
rec("D2", "L11H9: LB width and separation kept >= 0.93", d2, None if not ok(*d2.values()) else d2["LBw"] >= 0.93 and d2["LBs"] >= 0.93)
iv, q = ind_quiet("L11H9", b3=0.05, bu=0.05); rec("D3", "L11H9: |change| <= 0.05 on all four indentation measures", iv, q)

# ---- verdicts (PROTOCOL.md, Verdicts)
P = lambda *ns: all(T[n]["passed"] is True for n in ns)
V["lead1_L17H6"] = ("shared part confirmed" if P("A1", "A3", "A4") else
                    "line-breaking reader only (indentation role not confirmed)" if P("A1", "A4") else "not confirmed")
V["lead2_separate"] = "confirmed" if P("B1", "B2", "B3", "B4") else "not confirmed"
V["lead3_L0H2"] = "general previous-token mover confirmed" if P("C1", "C2", "C3") else "not confirmed"
V["lead3_L0H3"] = "current-token reader, not task- or case-specific, confirmed" if P("C4", "C5", "C7") else "not confirmed"
V["lead4_L11H9"] = "suffix head confirmed, not shared with line-breaking or indentation" if P("D1", "D2", "D3") else "not confirmed"
core = ("A1", "A4", "B1", "B4", "C1", "C2", "C4", "C5", "D1", "D2")
V["account"] = "supported" if P(*core) else ("not evaluable" if any(T[n]["passed"] is None for n in core) else "partly supported")
V["n_passed"] = sum(t["passed"] is True for t in T.values()); V["n_tests"] = len(T)


# ---- descriptive 95% intervals for the named groups' main values
def boot_ci():
    rng = np.random.default_rng(0); out = {}
    def ci(f, n_units):
        with np.errstate(divide="ignore", invalid="ignore"):
            bs = np.array([f(rng.integers(0, n_units, n_units)) for _ in range(2000)], float)
        bs = bs[np.isfinite(bs)]  # resamples with no unit of a class give no value
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else [None, None]
    for model, task in [(M6, "LB"), (M6, "IND"), (M6, "ORD"), (M17, "CAPS")]:
        p = D / f"c01_{model}_{task}_items.npz"
        if not p.exists(): continue
        A = dict(np.load(p, allow_pickle=True)); grs = sorted({k.split("|")[0] for k in A if not k.startswith("base")})
        for g in grs:
            if task == "LB":
                E0, E1 = A["base|E"], A[f"{g}|E"]; out[f"{g}|LBw"] = ci(lambda i: E1[i].mean() / E0[i].mean(), len(E0))
                n0, n1 = A["base|nat"], A[f"{g}|nat"]
                sep = lambda N, i: N[i, 0].sum() / N[i, 1].sum() - N[i, 2].sum() / N[i, 3].sum()
                out[f"{g}|LBs"] = ci(lambda i: sep(n1, i) / sep(n0, i), len(n0))
            elif task == "IND":
                cl, cls, src = A["base|cluster"], A["base|cls"], A["base|src"]; u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}
                d = A[f"{g}|lq"] - A["base|lq"]
                for k, (s_, c_) in dict(cU=("code", "dedent"), cT=("code", "term"), gU=("gen", "dedent"), gT=("gen", "term")).items():
                    msk = (src == s_) & (cls == c_)
                    def f(i, msk=msk):
                        ii = np.concatenate([idx[c] for c in u[i]]); ii = ii[msk[ii]]
                        return d[ii].mean() if len(ii) else np.nan
                    out[f"{g}|{k}"] = ci(f, len(u))
            elif task == "ORD":
                nn, dg, cls = A["base|n"], A["base|digits"], A["base|cls"]; m0, m1 = A["base|m"], A[f"{g}|m"]
                for dd in (4, 5):
                    nums = np.unique(nn[dg == dd]); ix = {x: np.where((nn == x) & (dg == dd))[0] for x in nums}
                    def eff(m, ii): t = ii[cls[ii] == "teen"]; e = ii[cls[ii] == "end123"]; return m[e].mean() - m[t].mean()
                    def f(i):
                        ii = np.concatenate([ix[x] for x in nums[i]]); return eff(m1, ii) / eff(m0, ii)
                    def fp(i):
                        ii = np.concatenate([ix[x] for x in nums[i]]); e = ii[cls[ii] == "end123"]; return m1[e].mean() / m0[e].mean()
                    out[f"{g}|t{dd}"] = ci(f, len(nums)); out[f"{g}|p{dd}"] = ci(fp, len(nums))
            elif task == "CAPS":
                par = A["base|par"]; up = np.unique(par); ix = {x: np.where(par == x)[0] for x in up}
                for k, a in (("Du", "du"), ("Dl", "dl")):
                    b0, b1 = A[f"base|{a}"], A[f"{g}|{a}"]
                    def f(i, b0=b0, b1=b1):
                        ii = np.concatenate([ix[x] for x in up[i]]); return b1[ii].mean() / b0[ii].mean()
                    out[f"{g}|{k}"] = ci(f, len(up))
                w0, w1 = A["base|wl"], A[f"{g}|wl"]
                out[f"{g}|wl"] = ci(lambda i: np.mean(np.concatenate([w1[ix[x]] - w0[ix[x]] for x in up[i]])), len(up))
    return out


CI = boot_ci()
for name in list(T):
    T[name]["passed"] = T[name]["passed"]
out = dict(tests=T, verdicts=V, ci95=CI, smoke=SMOKE,
           n_items={t: ({k: v for k, v in R[t]["base"].items() if k.startswith("n") or k.endswith("|n")} if R[t] else None) for t in R})
(D / "verdicts.json").write_text(json.dumps(out, indent=1, default=float))
for n, t in T.items(): print(n, t["passed"], json.dumps(t["values"], default=lambda x: round(x, 3) if isinstance(x, float) else str(x))[:300])
print(json.dumps(V, indent=1))
