"""D3 round 3 verdicts for parts X and E of PROTOCOL-3.md (parts R1 and R2 use the frozen c03_verdicts.py and
r2c04_verdicts.py unchanged). No model is loaded. Reads:
  round3/x1/outputs/conf/c01_qwen3-06b-bf16_IND.json   (X1's frozen c01, run on the round-3 parts)
  round3/x1/outputs/conf/r3x_l11h9_IND.json             (CARRY, CARRY-L11H9, TERM, TERM-L11H9)
  round3/outputs/conf/r3_pairs.json                     (single-head pair patching, L11H9 and L17H6)
Conventions as X1's c03_verdicts.py: change = ablated - base of mean log q(true); "below every random draw" = below
the minimum over all 20 draws of the same group; decisions use point estimates. 95% intervals (2,000 bootstrap
resamples over repositories or programs, numpy seed 0) are description only.
Writes round3/outputs/conf/verdicts_XE.json and prints a summary. Usage: python r3_verdicts.py [--calib]"""
import json, sys
from pathlib import Path
import numpy as np

R3 = Path(__file__).resolve().parent.parent
CALIB = "--calib" in sys.argv
XD = R3 / "x1" / "outputs" / ("calib" if CALIB else "conf"); PD = R3 / "outputs" / ("calib" if CALIB else "conf")
ld = lambda p: json.loads(p.read_text()) if p.exists() else None
X = ld(XD / "c01_qwen3-06b-bf16_IND.json") or {}
E = ld(XD / "r3x_l11h9_IND.json") or {}
PR = ld(PD / "r3_pairs.json")
K = {"cU": "code|dedent|lq", "cT": "code|term|lq", "gU": "gen|dedent|lq", "gT": "gen|term|lq"}
# Thresholds of the new tests, frozen in PROTOCOL-3.md (set from the calibration on old material).
E3_MARGIN = 0.010
E6_NEC = 0.05


def src(g):
    for R in (X, E):
        if g in R: return R
    return None


def val(g, k):
    R = src(g)
    return None if R is None else R[g][K[k]] - R["base"][K[k]]


def rr(g, k):
    R = src(g)
    if R is None: return (None, None, 0)
    v = [R[x][K[k]] - R["base"][K[k]] for x in R if x.startswith(g + "~r")]
    return (min(v), max(v), len(v)) if v else (None, None, 0)


def below(g, k):
    v = val(g, k); lo, _, n = rr(g, k)
    return v is not None and n == 20 and v < lo


def quiet(g, b3=0.03, bu=0.10):
    v = {k: val(g, k) for k in K}
    if any(x is None for x in v.values()): return v, None
    return v, all(abs(v[k]) <= b3 for k in ("cU", "cT", "gT")) and abs(v["gU"]) <= bu


T = {}
def rec(name, rule, values, passed): T[name] = dict(rule=rule, values=values, passed=None if passed is None else bool(passed))


# ---- part X: indentation side of X1's frozen tests (thresholds as X1's PROTOCOL.md)
v = val("L17H6", "gT")
rec("XA3", "L17H6: generated terminal change <= -0.010 and below every random draw", dict(change=v, rand=rr("L17H6", "gT")),
    None if v is None else v <= -0.010 and below("L17H6", "gT"))
a5 = dict(cU=val("L17H6", "cU"), gU=val("L17H6", "gU"))
rec("XA5", "L17H6: |code unmarked change| <= 0.03 and |generated unmarked change| <= 0.10 (indentation parts of A5)", a5,
    None if None in a5.values() else abs(a5["cU"]) <= 0.03 and abs(a5["gU"]) <= 0.10)
for t, g in (("XB1", "WRITE"), ("XB2", "COUNT"), ("XB3", "READ-L17H6"), ("XC3", "L0H2"), ("XC6", "L0H3")):
    q, p = quiet(g); rec(t, f"{g}: indentation quiet", q, p)
v = val("DED", "cU")
rec("XB4", "DED: code unmarked change <= -0.40 and below every random draw (indentation part of B4)",
    dict(change=v, rand=rr("DED", "cU")), None if v is None else v <= -0.40 and below("DED", "cU"))
q = {k: val("L11H9", k) for k in K}
rec("XD3", "L11H9: |change| <= 0.05 on all four indentation measures", q,
    None if None in q.values() else all(abs(x) <= 0.05 for x in q.values()))

# ---- part E: L11H9's second role
v = val("L11H9", "gT")
rec("E1", "L11H9: generated terminal change <= -0.010 and below every random draw (X1's A3 criterion)",
    dict(change=v, rand=rr("L11H9", "gT")), None if v is None else v <= -0.010 and below("L11H9", "gT"))
e2 = {g: dict(change=val(g, "gT"), rand=rr(g, "gT")) for g in ("CARRY", "TERM")}
rec("E2", "CARRY and TERM, each: generated terminal change <= -0.010 and below every random draw", e2,
    None if any(x["change"] is None for x in e2.values()) else all(x["change"] <= -0.010 and below(g, "gT") for g, x in e2.items()))
e3 = {g: dict(full=val(g, "gT"), without=val(f"{g}-L11H9", "gT")) for g in ("CARRY", "TERM")}
for x in e3.values(): x["diff"] = None if None in (x["full"], x["without"]) else x["without"] - x["full"]
rec("E3", f"CARRY and TERM: change(set without L11H9) - change(set) >= +{E3_MARGIN} on generated terminal dedents", e3,
    None if any(x["diff"] is None for x in e3.values()) else all(x["diff"] >= E3_MARGIN for x in e3.values()))
e4 = dict(cU=val("L11H9", "cU"), gU=val("L11H9", "gU"))
rec("E4", "L11H9: |code unmarked change| <= 0.03 and |generated unmarked change| <= 0.10", e4,
    None if None in e4.values() else abs(e4["cU"]) <= 0.03 and abs(e4["gU"]) <= 0.10)
v = val("L11H9", "cT")
rec("E5", "L11H9: natural-code terminal change below every random draw", dict(change=v, rand=rr("L11H9", "cT")),
    None if v is None else below("L11H9", "cT"))
if PR:
    c = PR["comps"]
    e6 = {}
    for h in ("L11H9", "L17H6"):
        if h not in c: e6[h] = None; continue
        rn = [c[k]["nec"] for k in c if k.startswith(h + "_rand")]; rs = [c[k]["suf"] for k in c if k.startswith(h + "_rand")]
        e6[h] = dict(nec=c[h]["nec"], suf=c[h]["suf"], rand_nec_max=max(rn) if rn else None, rand_suf_max=max(rs) if rs else None,
                     n_rand=len(rn), passed=bool(len(rn) == 20 and c[h]["nec"] >= E6_NEC and c[h]["nec"] > max(rn)))
    rec("E6", f"pairs, single head: necessity >= {E6_NEC} and above every random draw (L11H9; L17H6 reported)", e6,
        None if e6.get("L11H9") is None else e6["L11H9"]["passed"])
else:
    rec("E6", "pairs", None, None)

# ---- verdicts
P = lambda t: T[t]["passed"]
def all_(ts): return None if any(P(t) is None for t in ts) else all(P(t) for t in ts)
V = {}
e1, e3p = P("E1"), P("E3")
V["E: L11H9 second role"] = None if e1 is None else ("confirmed" if e1 and e3p else "a single-head effect only" if e1 else "not confirmed")
V["X: L17H6 indentation side (A3)"] = P("XA3")
V["X: separate parts, indentation side (B1-B4)"] = all_(["XB1", "XB2", "XB3", "XB4"])
V["X: early heads quiet on indentation (C3, C6)"] = all_(["XC3", "XC6"])
V["X: L11H9 within the frozen D3 bound"] = P("XD3")

# ---- descriptive intervals (named groups only; items saved per run)
def boot(npz, g, k, B=2000):
    if not npz.exists(): return None
    it = np.load(npz, allow_pickle=True)
    if f"{g}|lq" not in it: return None
    a, b = it["base|lq"], it[f"{g}|lq"]; s, c, cl = it["base|src"], it["base|cls"], it["base|cluster"]
    so, cc = K[k].split("|")[:2]; sel = (s == so) & (c == cc); d = (b - a)[sel]; cs = cl[sel]
    u = np.unique(cs); idx = {x: np.where(cs == x)[0] for x in u}; rng = np.random.default_rng(0); out = []
    for _ in range(B):
        pick = np.concatenate([idx[x] for x in rng.choice(u, len(u))]); out.append(d[pick].mean())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
CI = {}
for g, k in [("L11H9", "gT"), ("L17H6", "gT"), ("DED", "cU"), ("L11H9", "cT")]:
    CI[f"{g}|{k}"] = boot(XD / "c01_qwen3-06b-bf16_IND_items.npz", g, k)
for g in ("CARRY", "CARRY-L11H9", "TERM", "TERM-L11H9"):
    CI[f"{g}|gT"] = boot(XD / "r3x_l11h9_IND_items.npz", g, "gT")
n = {k: (X or E)["base"].get(f"{K[k].rsplit('|', 1)[0]}|n") for k in K} if (X or E) else None
base = {k: (X or E)["base"][K[k]] for k in K} if (X or E) else None
out = dict(calib=CALIB, tests=T, verdicts=V, intervals=CI, n_points=n, base=base)
PD.mkdir(parents=True, exist_ok=True)
(PD / ("verdicts_XE_calib.json" if CALIB else "verdicts_XE.json")).write_text(json.dumps(out, indent=1))
for t, x in T.items(): print(t, x["passed"], json.dumps(x["values"])[:220])
print(json.dumps(V, indent=1))
