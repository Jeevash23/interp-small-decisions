"""D3 round 2: independent recomputation of every round-2 number (checker code, not a producer).

  python3 -B code/r2_independent.py --records conf|calib|smoke [--out PATH]

Written without reading make_numbers.py (its round-2 part), r2c04_verdicts.py or outputs/r2/measures_*.json.
Every value is recomputed from raw records and data files:
  - c2.* from the confirmation records: r2c01 (one line per chunk), r2c02 (one line per program), r2c03 (one JSON);
    `--records conf` reads outputs/r2/conf/, `--records calib` the calibration folders (conf_calib_x1code/,
    conf_calib_gen2/, conf_calib_newcode/), `--records smoke` outputs/r2/conf_smoke/ (no r2c03 there);
  - cal.* always from the calibration records;
  - c2.v.* (pass or fail), c2.npass and c2.ntests from the PROTOCOL-2.md thresholds applied to the c2.* values;
  - x2.*, ck.*, gate.post.ol.*, x1.*, mat2.*, freeze.*2 from the raw exploratory outputs, data files, FROZEN.md and
    X1's raw run files.
Intervals: own bootstrap, 2,000 resamples, numpy seed 0, resampling repositories (natural code; key = the chunk's
repo, else the chunk index as a string) or programs (generated), 95% percentile.
Formats follow numbers.json: 3 decimals, minus sign U+2212, changes signed, steering rises 2 decimals signed,
counts with thousands commas.
Writes the flat key -> string dict to outputs/r2/independent_values.json (or --out) and a sibling
*.meta.json with the records used, completeness, unresolved keys and the verdict details.
The functions are importable (check_numbers.py and r2_check_report.py reuse them; both are checker code).
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

D3 = Path(__file__).resolve().parents[1]
OUT = D3 / "outputs"; OUT2 = OUT / "r2"; DATA = D3 / "data"
X1 = D3.parent / "X1-shared-parts"
MINUS = "\u2212"
MODELS = {"q": ("qwen3-06b-bf16", "L23H2"), "o": ("olmo1b", "L13H1")}
GROUP = ("olmo1b", "DEDO")
KWS = ("return", "raise", "break", "continue", "pass")


# ------------------------------------------------------------------ formatting
def u(x, d=3):
    """unsigned decimal, U+2212 for negatives"""
    return f"{x:.{d}f}".replace("-", MINUS)


def s(x, d=3):
    """signed decimal (changes)"""
    return f"{x:+.{d}f}".replace("-", MINUS)


def n_(k): return f"{int(k):,}"


def lines(p):
    """JSON lines. A last line that does not parse (a record being written by a run still in progress) is dropped;
    such a run is then also reported as incomplete. A bad line anywhere else is an error."""
    raw = [l for l in open(p) if l.strip()]; out = []
    for i, l in enumerate(raw):
        try: out.append(json.loads(l))
        except json.JSONDecodeError:
            if i == len(raw) - 1 and not l.endswith("\n"): break
            raise
    return out


def rj(p): return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ bootstrap
def boot_ratio(num, den, clusters, B=2000, seed=0):
    """Cluster bootstrap of sum(num)/sum(den): per-cluster sums, clusters resampled with replacement."""
    keys, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    sn = np.bincount(inv, weights=np.asarray(num, float), minlength=len(keys))
    sd = np.bincount(inv, weights=np.asarray(den, float), minlength=len(keys))
    rng = np.random.default_rng(seed); out = np.empty(B); k = len(keys)
    for b in range(B):
        w = np.bincount(rng.integers(0, k, k), minlength=k); out[b] = (w * sn).sum() / (w * sd).sum()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_mean(vals, clusters, **kw): return boot_ratio(vals, np.ones(len(vals)), clusters, **kw)


def boot_share(blk, base, neu, B=2000, seed=0):
    """Program bootstrap of (mean blk - mean base) / (mean neutral - mean base)."""
    blk, base, neu = map(lambda a: np.asarray(a, float), (blk, base, neu))
    rng = np.random.default_rng(seed); k = len(blk); out = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        out[b] = ((w * blk).sum() - (w * base).sum()) / ((w * neu).sum() - (w * base).sum())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ------------------------------------------------------------------ record locations
def rec_paths(mode):
    if mode == "conf":
        d = OUT2 / "conf"; return d, d, d
    if mode == "calib":
        return OUT2 / "conf_calib_x1code", OUT2 / "conf_calib_gen2", OUT2 / "conf_calib_newcode"
    if mode == "smoke":
        d = OUT2 / "conf_smoke"; return d, d, d
    raise SystemExit(f"unknown records mode {mode}")


# ------------------------------------------------------------------ r2c01: natural code, one head
def r2c01(path):
    """Raw (unrounded) measures from an r2c01 record file. Returns (values, extras for bootstrap)."""
    recs = lines(path); V = {}
    pts = []  # (cluster, point)
    for r in recs:
        cl = r["repo"] if r.get("repo") is not None else str(r["chunk"])
        for p in r["points"]: pts.append((cl, p))
    V["n.points"] = len(pts); V["n.chunks"] = len(recs)
    V["n.repos"] = len({r["repo"] for r in recs if r.get("repo") is not None})
    ded = [(c, p) for c, p in pts if p["cls"] == "dedent" and p["oh"]]
    same = [(c, p) for c, p in pts if p["cls"] == "same" and p["oh"]]
    term = [(c, p) for c, p in pts if p["cls"] == "term" and p["oh"]]
    V["n.ded"], V["n.same"], V["n.term"] = len(ded), len(same), len(term)
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    att = lambda P: [p["att"][-1] for _, p in P]
    dko = lambda P, k: [p[k][0] - p["base"][0] for _, p in P]
    V["att_inner_ded"], V["att_inner_same"], V["att_inner_term"] = mean(att(ded)), mean(att(same)), mean(att(term))
    V["ko_inner_ded"], V["ko_inner_same"], V["ko_inner_term"] = mean(dko(ded, "ko_inner")), mean(dko(same, "ko_inner")), mean(dko(term, "ko_inner"))
    V["base_lq_ded"] = mean([p["base"][0] for _, p in ded])
    closed = [(c, p) for c, p in ded if p["has_closed"]]
    V["n.ded_closed"] = len(closed)
    V["ko_closed_ded"], V["ko_inner_ded_closedsub"] = mean(dko(closed, "ko_closed")), mean(dko(closed, "ko_inner"))
    topen = [(c, p) for c, p in ded if p["true"] in p["oh"]]
    V["n.ded_true_open"] = len(topen); V["ko_true_ded"] = mean(dko(topen, "ko_true"))
    others = sorted({k for _, p in ded for k in p if k.startswith("ko_inner_h")}, key=lambda k: int(k[10:]))
    ov = {k: mean(dko(ded, k)) for k in others}
    V["n.others"] = len(others)
    V["others_lo"], V["others_hi"] = (min(ov.values()), max(ov.values())) if ov else (float("nan"),) * 2
    V["others_maxabs"] = max(abs(x) for x in ov.values()) if ov else float("nan")
    # picking
    multi = [(c, p) for c, p in pts if p["cls"] in ("dedent", "term") and len(p["oh"]) >= 2 and p["true"] in p["oh"]]
    V["n.multi"] = len(multi)
    pick = [int(np.argmax(p["att"])) == p["oh"].index(p["true"]) for _, p in multi]
    inner_rule = [p["oh"][-1] == p["true"] for _, p in multi]
    ni = [(c, p) for c, p in multi if p["oh"][-1] != p["true"]]
    V["n.multi_not_inner"] = len(ni)
    pick_ni = [int(np.argmax(p["att"])) == p["oh"].index(p["true"]) for _, p in ni]
    inner_att_ni = [int(np.argmax(p["att"])) == len(p["oh"]) - 1 for _, p in ni]
    V["pick_not_inner"], V["pick_inner_attended_not_inner"] = mean(pick_ni), mean(inner_att_ni)
    V["pick_all_multi"], V["innermost_rule_all_multi"] = mean(pick), mean(inner_rule)
    # steering (entries whose indent is not already the top candidate)
    for cls, nm in (("dedent", "dedent"), ("term", "term")):
        for who in ("head", "ctrl"):
            ent = [e for _, p in pts if p["cls"] == cls and "steer" in p for e in p["steer"] if e[0] == who and e[2] != p["top_base"]]
            V[f"n.steer_{nm}_{who}"] = len(ent)
            V[f"steer_{nm}_{who}_dlq"] = mean([e[3] for e in ent]); V[f"steer_{nm}_{who}_top"] = mean([e[4] == e[2] for e in ent])
    X = dict(ded=ded, ni=ni)
    return V, X


def r2c01_ci(X):
    ded, ni = X["ded"], X["ni"]
    out = {}
    out["ci.ko_inner_ded"] = boot_mean([p["ko_inner"][0] - p["base"][0] for _, p in ded], [c for c, _ in ded])
    out["ci.att_inner_ded"] = boot_mean([p["att"][-1] for _, p in ded], [c for c, _ in ded])
    out["ci.pick_not_inner"] = boot_mean([float(int(np.argmax(p["att"])) == p["oh"].index(p["true"])) for _, p in ni], [c for c, _ in ni])
    return out


# ------------------------------------------------------------------ r2c02: generated programs, one head
def r2c02(path):
    recs = lines(path); V = {}
    ok = [r for r in recs if not r.get("skip")]
    V["n.programs"] = len(ok); V["n.records"] = len(recs); V["n.skip"] = len(recs) - len(ok)
    term = [r for r in ok if r["last"] == "terminal"]; plain = [r for r in ok if r["last"] == "plain"]
    V["n.terminal"], V["n.plain"] = len(term), len(plain)
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")

    def share(rs, cond):
        rs = [r for r in rs if r.get("neutral") is not None]
        if not rs: return float("nan")
        b = mean([r["base"][0] for r in rs])
        return (mean([r[cond][0] for r in rs]) - b) / (mean([r["neutral"][0] for r in rs]) - b)
    for kw in KWS:
        rs = [r for r in term if r["kw"] == kw]; V[f"n.{kw}"] = len(rs)
        V[f"share_blk_kw_{kw}"] = share(rs, "blk_kw")
    ret = [r for r in term if r["kw"] == "return"]
    V["share_blk_kw_after_return"] = share(ret, "blk_kw_after")
    V["share_blk_kw_all"] = share(term, "blk_kw")
    L = sorted(int(k[3:]) for k in (ret[0] if ret else {}) if re.fullmatch(r"lay\d+", k))
    lay = {l: share(ret, f"lay{l}") for l in L}
    V["n.layers"] = len(L)
    if lay:
        V["share_layer_return_max"] = max(lay.values()); V["layer_return_arg"] = max(lay, key=lambda l: lay[l])
    V["l0h2_line_dqcur_term"] = mean([r["l0h2_line"][0] - r["base"][0] for r in term])
    V["l0h2_all_dqcur_term"] = mean([r["l0h2_all"][0] - r["base"][0] for r in term])
    V["qcur_base_term"] = mean([r["base"][0] for r in term])
    V["qcur_neutral_term"] = mean([r["neutral"][0] for r in term if r.get("neutral") is not None])
    for nm, rs in (("term", term), ("plain", plain)):
        for who in ("head", "ctrl"):
            ent = [e for r in rs if "steer" in r for e in r["steer"] if e[0] == who and e[2] != r["top_base"]]
            V[f"n.steer_{nm}_{who}"] = len(ent)
            V[f"steer_{nm}_{who}_dlq"] = mean([e[3] for e in ent]); V[f"steer_{nm}_{who}_top"] = mean([e[4] == e[2] for e in ent])
    X = dict(ret=[r for r in ret if r.get("neutral") is not None], term=term)
    return V, X


def r2c02_ci(X):
    ret, term = X["ret"], X["term"]
    out = {"ci.share": boot_share([r["blk_kw"][0] for r in ret], [r["base"][0] for r in ret], [r["neutral"][0] for r in ret])}
    out["ci.l0h2"] = boot_mean([r["l0h2_line"][0] - r["base"][0] for r in term], [str(r["seed"]) for r in term])
    return out


# ------------------------------------------------------------------ r2c03: group ablation
def r2c03(path, gname="DEDO"):
    R = rj(path); V = {}
    cls = np.array(R["cls"]); CL = ("header", "same", "term", "dedent")
    for c in CL: V[f"n.{c}"] = int((cls == c).sum())
    lq = lambda key, c: float(np.mean(np.array(R[key]["per_point_lq"])[cls == c]))
    for c in CL:
        V[f"d_lq_{c}"] = R[gname]["lq"][c] - R["base"]["lq"][c]
        V[f"check_perpoint_{c}"] = abs((lq(gname, c) - lq("base", c)) - V[f"d_lq_{c}"])  # run-file means vs per-point
    rk = sorted((k for k in R if re.fullmatch(r"rand\d+", k)), key=lambda k: int(k[4:]))
    V["n.rand"] = len(rk)
    rd = [R[k]["lq"]["dedent"] - R["base"]["lq"]["dedent"] for k in rk]
    rt = [R[k]["lq"]["term"] - R["base"]["lq"]["term"] for k in rk]
    V["rand_d_lq_dedent_min"], V["rand_d_lq_dedent_max"] = min(rd), max(rd); V["rand_d_lq_term_min"] = min(rt)
    V["d_psame_dedent"] = R[gname]["p_same"]["dedent"] - R["base"]["p_same"]["dedent"]
    V["p_open_term_ablated"] = R[gname]["p_open"]["term"]
    V["base_lq_dedent"], V["base_lq_term"] = R["base"]["lq"]["dedent"], R["base"]["lq"]["term"]
    V["ratio"] = abs(V["d_lq_term"]) / abs(V["d_lq_dedent"])
    V["heads"] = R[gname].get("heads", R.get("heads"))
    msk = cls == "dedent"
    dl = (np.array(R[gname]["per_point_lq"]) - np.array(R["base"]["per_point_lq"]))[msk]
    X = dict(dl=dl, cl=np.array(R["cluster"])[msk])
    return V, X


def r2c03_ci(X): return {"ci": boot_mean(X["dl"], X["cl"])}


# ------------------------------------------------------------------ formatting of c2/cal blocks
SIGNED = re.compile(r"^(ko_|others_lo|others_hi|l0h2_|d_lq_|rand_d_lq_|d_psame)")


def fmt_block(prefix, V, keys=None):
    out = {}
    for k, v in V.items():
        if keys is not None and k not in keys: continue
        if k.startswith("check_") or k in ("heads", "n.others", "n.layers", "n.records", "n.skip", "n.rand"): continue
        key = f"{prefix}.{k}"
        if k.startswith("n."): out[key] = n_(v)
        elif k == "layer_return_arg": out[key] = str(int(v))
        elif k.endswith("_dlq"): out[key] = s(v, 2)
        elif k == "ci" or k.startswith("ci."): out[key + ".lo"], out[key + ".hi"] = u(v[0]), u(v[1])
        elif SIGNED.match(k): out[key] = s(v)
        else: out[key] = u(v)
    return out


# ------------------------------------------------------------------ verdicts (PROTOCOL-2.md)
def verdicts(q, qg, o, og, g):
    """q, o: r2c01 raw values; qg, og: r2c02 raw values; g: r2c03 raw values (any may be None -> not evaluable).
    Returns {id: (verdict, detail)} for Q1-Q11, O1-O10, G1-G4 and the five questions."""
    V = {}

    def put(i, fn, *srcs):
        if any(x is None for x in srcs): V[i] = ("not evaluable", "records missing"); return
        try:
            ok, det = fn()
            if any(isinstance(x, float) and np.isnan(x) for x in det.values()): V[i] = ("not evaluable", det); return
            V[i] = ("pass" if ok else "fail", det)
        except (KeyError, ZeroDivisionError, ValueError) as e:
            V[i] = ("not evaluable", repr(e))

    fl = lambda c: max(c, 0.02)  # control floored at 0.02
    put("Q1", lambda: (q["att_inner_ded"] >= 0.30 and q["att_inner_ded"] >= 2.5 * q["att_inner_same"], dict(ded=q["att_inner_ded"], same=q["att_inner_same"])), q)
    put("Q2", lambda: (q["ko_inner_ded"] <= -0.05 and q["ko_inner_ded"] < q["others_lo"] and abs(q["ko_inner_ded"]) >= 2 * q["others_maxabs"],
                       dict(ko=q["ko_inner_ded"], lo=q["others_lo"], maxabs=q["others_maxabs"])), q)
    put("Q3", lambda: (q["ko_inner_same"] >= -0.02 and abs(q["ko_inner_term"]) <= 0.25 * abs(q["ko_inner_ded"]), dict(same=q["ko_inner_same"], term=q["ko_inner_term"])), q)
    put("Q4", lambda: (abs(q["ko_closed_ded"]) <= 0.5 * abs(q["ko_inner_ded_closedsub"]), dict(closed=q["ko_closed_ded"], inner=q["ko_inner_ded_closedsub"])), q)
    put("Q5", lambda: (q["pick_not_inner"] >= 0.60 and q["pick_all_multi"] - q["innermost_rule_all_multi"] >= 0.02,
                       dict(ni=q["pick_not_inner"], all=q["pick_all_multi"], rule=q["innermost_rule_all_multi"])), q)
    put("Q6", lambda: (q["steer_dedent_head_dlq"] >= 1.0 and q["steer_dedent_head_dlq"] >= 5 * fl(q["steer_dedent_ctrl_dlq"])
                       and q["steer_dedent_head_top"] >= q["steer_dedent_ctrl_top"] + 0.05,
                       dict(dlq=q["steer_dedent_head_dlq"], cdlq=q["steer_dedent_ctrl_dlq"], top=q["steer_dedent_head_top"], ctop=q["steer_dedent_ctrl_top"])), q)
    put("Q7", lambda: (q["ko_true_ded"] <= -0.10, dict(ko_true=q["ko_true_ded"])), q)
    put("Q8", lambda: (qg["steer_term_head_dlq"] >= 1.0 and qg["steer_term_head_dlq"] >= 5 * fl(qg["steer_term_ctrl_dlq"])
                       and qg["steer_term_head_top"] >= 0.20 and qg["steer_term_head_top"] >= qg["steer_term_ctrl_top"] + 0.10,
                       dict(dlq=qg["steer_term_head_dlq"], cdlq=qg["steer_term_ctrl_dlq"], top=qg["steer_term_head_top"], ctop=qg["steer_term_ctrl_top"])), qg)
    put("Q9", lambda: (qg["share_blk_kw_return"] >= 0.40, dict(share=qg["share_blk_kw_return"])), qg)
    put("Q10", lambda: (qg["share_layer_return_max"] <= 0.35 * qg["share_blk_kw_return"], dict(maxlayer=qg["share_layer_return_max"], share=qg["share_blk_kw_return"])), qg)
    put("Q11", lambda: (abs(qg["l0h2_line_dqcur_term"]) <= 0.05, dict(d=qg["l0h2_line_dqcur_term"])), qg)
    put("O1", lambda: (o["att_inner_ded"] >= 0.20 and o["att_inner_ded"] >= 1.4 * o["att_inner_same"], dict(ded=o["att_inner_ded"], same=o["att_inner_same"])), o)
    put("O2", lambda: (o["ko_inner_ded"] <= -0.05 and o["ko_inner_ded"] < o["others_lo"] and abs(o["ko_inner_ded"]) >= 2 * o["others_maxabs"],
                       dict(ko=o["ko_inner_ded"], lo=o["others_lo"], maxabs=o["others_maxabs"])), o)
    put("O3", lambda: (o["ko_inner_same"] >= -0.02 and abs(o["ko_inner_term"]) <= 0.25 * abs(o["ko_inner_ded"]), dict(same=o["ko_inner_same"], term=o["ko_inner_term"])), o)
    put("O4", lambda: (abs(o["ko_closed_ded"]) <= 0.5 * abs(o["ko_inner_ded_closedsub"]), dict(closed=o["ko_closed_ded"], inner=o["ko_inner_ded_closedsub"])), o)
    put("O5", lambda: (o["pick_not_inner"] >= 0.45 and o["pick_not_inner"] - o["pick_inner_attended_not_inner"] >= 0.15,
                       dict(ni=o["pick_not_inner"], inner=o["pick_inner_attended_not_inner"])), o)
    put("O6", lambda: (o["steer_dedent_head_dlq"] >= 0.8 and o["steer_dedent_head_dlq"] >= 4 * fl(o["steer_dedent_ctrl_dlq"])
                       and o["steer_dedent_head_top"] >= o["steer_dedent_ctrl_top"] + 0.05,
                       dict(dlq=o["steer_dedent_head_dlq"], cdlq=o["steer_dedent_ctrl_dlq"], top=o["steer_dedent_head_top"], ctop=o["steer_dedent_ctrl_top"])), o)
    put("O7", lambda: (o["ko_true_ded"] <= -0.08, dict(ko_true=o["ko_true_ded"])), o)
    put("O8", lambda: (og["steer_term_head_dlq"] >= 0.8 and og["steer_term_head_dlq"] >= 4 * fl(og["steer_term_ctrl_dlq"])
                       and og["steer_term_head_top"] >= 0.15 and og["steer_term_head_top"] >= og["steer_term_ctrl_top"] + 0.10,
                       dict(dlq=og["steer_term_head_dlq"], cdlq=og["steer_term_ctrl_dlq"], top=og["steer_term_head_top"], ctop=og["steer_term_ctrl_top"])), og)
    put("O9", lambda: (og["share_blk_kw_return"] >= 0.40, dict(share=og["share_blk_kw_return"])), og)
    put("O10", lambda: (og["share_layer_return_max"] <= 0.35 * og["share_blk_kw_return"], dict(maxlayer=og["share_layer_return_max"], share=og["share_blk_kw_return"])), og)
    put("G1", lambda: (g["d_lq_dedent"] <= -0.30 and g["d_lq_dedent"] < g["rand_d_lq_dedent_min"] and g["n.rand"] == 20,
                       dict(d=g["d_lq_dedent"], rmin=g["rand_d_lq_dedent_min"], nrand=g["n.rand"])), g)
    put("G2", lambda: (abs(g["d_lq_term"]) <= 0.25 * abs(g["d_lq_dedent"]) and g["p_open_term_ablated"] >= 0.90,
                       dict(term=g["d_lq_term"], ded=g["d_lq_dedent"], p_open=g["p_open_term_ablated"])), g)
    put("G3", lambda: (g["d_lq_same"] >= -0.02, dict(d=g["d_lq_same"])), g)
    put("G4", lambda: (g["d_psame_dedent"] >= 0.08, dict(d=g["d_psame_dedent"])), g)
    for qn, ids in QUESTIONS.items():
        vs = [V[i][0] for i in ids]
        V[qn] = ("confirmed" if all(v == "pass" for v in vs) else "not confirmed", dict(zip(ids, vs)))
    return V


QUESTIONS = {"question1": ["Q1", "Q2", "Q3"], "question2": ["Q5", "Q6", "Q8"], "question3": ["Q9", "Q10", "Q11"],
             "question4a": ["G1", "G2", "O9"], "question4b": ["O1", "O2", "O3", "O6", "O8"]}
PRED_IDS = [f"Q{i}" for i in range(1, 12)] + [f"O{i}" for i in range(1, 11)] + [f"G{i}" for i in range(1, 5)]


# ------------------------------------------------------------------ the confirmation block (c2.* or cal.*)
def records_block(mode):
    """Raw values for each block from the records of `mode`. Returns dict tag -> (values, extras) or None."""
    p1, p2, p3 = rec_paths(mode); B = {}
    for t, (m, h) in MODELS.items():
        f1 = p1 / f"r2c01_{m}_{h}.jsonl"; f2 = p2 / f"r2c02_{m}_{h}.jsonl"
        B[t] = r2c01(f1) if f1.exists() else None
        B[t + "g"] = r2c02(f2) if f2.exists() else None
    f3 = p3 / f"r2c03_{GROUP[0]}_{GROUP[1]}.json"
    B["g"] = r2c03(f3) if f3.exists() else None
    return B


def completeness(mode, B):
    """Expected sizes: conf = the sealed parts; calib = the run plan in r2q_calib.sh."""
    if mode == "conf":
        n1 = len(lines(DATA / "r2code" / "part-held.chunks.jsonl")); n2 = len(lines(DATA / "r2gen2" / "part-held.jsonl"))
    elif mode == "calib":
        sh = (D3 / "code" / "r2q_calib.sh").read_text()
        n1 = int(re.search(r"r2c01_header\.py \S+ \S+ --calib x1code (\d+)", sh).group(1))
        n2 = int(re.search(r"r2c02_gen2\.py \S+ \S+ --calib (\d+)", sh).group(1))
    else:
        n1, n2 = 12, 16
    st = {}
    for t in MODELS:
        st[f"r2c01_{t}"] = (B[t][0]["n.chunks"] if B[t] else 0, n1)
        st[f"r2c02_{t}"] = (B[t + "g"][0]["n.records"] if B[t + "g"] else 0, n2)
    st["r2c03_g"] = (B["g"][0]["n.rand"] + 2 if B["g"] else 0, 22)  # base + group + 20 random
    return st


def block_values(prefix, B, with_ci=True):
    out = {}
    for t in ("q", "qg", "o", "og", "g"):
        if B.get(t) is None: continue
        V, X = B[t]
        if with_ci: V = dict(V, **({"q": r2c01_ci, "o": r2c01_ci, "qg": r2c02_ci, "og": r2c02_ci, "g": r2c03_ci}[t](X)))
        out.update(fmt_block(f"{prefix}.{t}", V))
    return out


CAL_KEYS = {  # the calibration values printed in PROTOCOL-2 (cal.q.* holds Qwen3's natural and generated values)
    "r2c01": ["att_inner_ded", "att_inner_same", "ko_inner_ded", "ko_closed_ded", "ko_inner_ded_closedsub", "ko_true_ded",
              "pick_not_inner", "pick_all_multi", "innermost_rule_all_multi", "steer_dedent_head_top", "steer_dedent_ctrl_top",
              "steer_dedent_head_dlq", "steer_dedent_ctrl_dlq", "others_lo", "others_hi"],
    "r2c02": [f"share_blk_kw_{k}" for k in KWS] + ["share_layer_return_max", "l0h2_line_dqcur_term", "l0h2_all_dqcur_term",
                                                     "steer_term_head_top", "steer_term_ctrl_top", "steer_term_head_dlq", "steer_term_ctrl_dlq"],
}


def cal_values(B, completeness_st):
    out = {}
    for t in MODELS:
        if B[t]: out.update(fmt_block(f"cal.{t}", B[t][0], set(CAL_KEYS["r2c01"])))
        if B[t + "g"]: out.update(fmt_block(f"cal.{t}", B[t + "g"][0], set(CAL_KEYS["r2c02"])))
    q, qg, g = B["q"], B["qg"], B["g"]
    if q: out["cal.q.n.chunks"] = n_(q[0]["n.chunks"]); out["cal.q.n.repos"] = n_(q[0]["n.repos"])
    if qg:
        out["cal.qg.n.programs"] = n_(qg[0]["n.programs"])
        out["cal.q.qcur_base_term"] = u(qg[0]["qcur_base_term"]); out["cal.q.qcur_neutral_term"] = u(qg[0]["qcur_neutral_term"])
    if g:
        out.update(fmt_block("cal.g", g[0], {"d_lq_dedent", "d_lq_term", "d_lq_same", "rand_d_lq_dedent_min", "rand_d_lq_dedent_max",
                                              "d_psame_dedent", "p_open_term_ablated"}))
        sh = (D3 / "code" / "r2q_calib.sh").read_text()
        out["cal.g.n.chunks"] = n_(int(re.search(r"r2c03_group\.py \S+ \S+ --calib (\d+)", sh).group(1)))
    return out


# ------------------------------------------------------------------ exploratory values (raw outputs)
def x2_values():
    out = {}
    # r2e07 header scans
    for t, (m, h) in MODELS.items():
        z = np.load(OUT2 / f"r2e07_{m}_x1code_400.npz"); l, hh = map(int, h[1:].split("H"))
        out[f"x2.scan.{t}.ded"], out[f"x2.scan.{t}.same"], out[f"x2.scan.{t}.pick"] = u(z["a_ded"][l, hh]), u(z["a_same"][l, hh]), u(z["pick"][l, hh])
        for arr, nm in (("a_ded", "next"), ("pick", "nextpick")):
            a = np.array(z[arr], float).copy(); a[l, hh] = -np.inf; a = np.nan_to_num(a, nan=-np.inf)
            bl, bh = np.unravel_index(np.argmax(a), a.shape); out[f"x2.scan.{t}.{nm}"] = f"L{bl}H{bh}"
            out[f"x2.scan.{t}.{'nextded' if nm == 'next' else 'nextpickv'}"] = u(a[bl, bh])
    # r2e02 knock-outs, x1code 150 chunks
    e2 = rj(OUT2 / "r2e02_x1code_150.json"); ded = [p for p in e2 if p["cls"] == "dedent"]
    out["x2.ko.n"] = n_(len(ded))
    for k, c in (("tok", "ko23_encl"), ("line", "ko23_encl_line"), ("ded5", "koDED_encl_line"), ("abl17", "ablate17")):
        out[f"x2.ko.{k}"] = s(np.mean([p[c][0] - p["base"][0] for p in ded]))
    # attention is undefined where no enclosing header token was found (a23_encl None): those points are left out
    out["x2.ko.att"] = u(np.mean([p["a23_encl"] for p in ded if p["a23_encl"] is not None]))
    out["x2.ko.att17"] = u(np.mean([p["a23_encl_abl17"] for p in ded if p["a23_encl_abl17"] is not None]))
    trip = [t for p in ded for t in p["open_hdr"]]
    out["x2.ov.ded"] = u(np.mean([t[2] == t[0] for t in trip]), 2)
    # r2e06 query composition
    e6 = rj(OUT2 / "r2e06_qwen3-06b-bf16_L23H2_x1code_80.json"); b = e6["base"]["a_ded"]
    out["x2.qc.base"] = u(b)
    dd = sorted(((v["a_ded"] - b, k) for k, v in e6["comps"].items() if k != "base"))
    for i in (0, 1):
        nm = dd[i][1]; nm = "MLP" + nm[3:] if nm.lower().startswith("mlp") else nm
        out[f"x2.qc.top{i + 1}"] = nm; out[f"x2.qc.top{i + 1}v"] = s(dd[i][0])
    # r2e04 keyword readers, gen2 250 terminal programs (return programs)
    e4 = rj(OUT2 / "r2e04_terminal_250.json"); ret = [r for r in e4["rec"] if r["kw"] == "return" and r.get("neutral") is not None]
    out["x2.kw.n"] = n_(len(ret))

    def sh(c):
        b0 = np.mean([r["base"][0] for r in ret]); return (np.mean([r[c][0] for r in ret]) - b0) / (np.mean([r["neutral"][0] for r in ret]) - b0)
    out["x2.kw.top8"], out["x2.kw.blk"] = u(sh("ko_KWG"), 2), u(sh("blk_all"), 2)
    rs = [sh(k) for k in ret[0] if re.fullmatch(r"ko_rand\d+", k)]
    out["x2.kw.randlo"], out["x2.kw.randhi"] = u(min(rs), 2), u(max(rs), 2)
    out["x2.kw.L1H0att"] = u(dict(e4["att_top"])["L1H0"], 2)
    out["_x2.kw.nrand"] = len(rs)
    # round 1's e02 scan on OLMo-2
    eo = rj(OUT / "e02_olmo1b_newcode_160.json")
    for h in ("L13H1", "L10H12", "L11H0", "L15H4"):
        out[f"x2.oe2.{h}"] = s(eo[h]["dedent"]); out[f"x2.oe2.{h}.t"] = s(eo[h]["term"])
    heads = [k for k in eo if re.fullmatch(r"L\d+H\d+", k)]
    out["x2.oe2.maxheadterm"] = u(max(abs(eo[k]["term"]) for k in heads))
    out["_x2.oe2.nheads"] = len(heads)
    return out


def ck_values():
    # 9 Nov 12:30: extended to every scanned checkpoint (stage 1, stage-2 ingredients) and three fixed heads.
    out = {}; rows = {r["model"]: r for r in lines(OUT2 / "r2e08_200.jsonl")}
    ns = set()
    tags = {"1B": "olmo1b-1B", "21B": "olmo1b-21B", "84B": "olmo1b-84B", "420B": "olmo1b-420B", "1007B": "olmo1b-1007B",
            "4001B": "olmo1b-4001B", "iax3B": "olmo1b-s2i1-3B", "iax51B": "olmo1b-s2i1-51B", "ibx51B": "olmo1b-s2i2-51B",
            "icx51B": "olmo1b-s2i3-51B", "final": "olmo1b"}
    for tag, m in tags.items():
        if m not in rows: continue
        r = rows[m]; z = np.load(OUT2 / f"r2e08_{m}_200.npz"); ns.add(r["chunks"])
        for c, k in (("dedent", "ded"), ("term", "term"), ("same", "same")): out[f"ck.{tag}.{k}"] = u(r["behaviour"][c]["top1"])
        for (l, h) in ((11, 9), (13, 1), (15, 9)):
            out[f"ck.{tag}.L{l}H{h}.att"] = u(z["a_ded"][l, h], 2); out[f"ck.{tag}.L{l}H{h}.pick"] = u(z["pick"][l, h], 2)
        dm = np.nan_to_num(z["a_ded"] - z["a_same"], nan=-np.inf); bl, bh = np.unravel_index(np.argmax(dm), dm.shape)
        out[f"ck.{tag}.best"] = f"L{bl}H{bh}"
    # weight distances (r2e10b_dist.json, float64 sums)
    D = json.loads((OUT2 / "r2e10b_dist.json").read_text())
    for pk, nm in (("final~s1_4001B", "fin_s1end"), ("s1_4001B~s2i1_51B", "s1end_ia51"), ("s2i1_51B~s2i2_51B", "ia51_ib51"), ("s1_1007B~s1_4001B", "s1mid_s1end")):
        out[f"ck.dist.{nm}.rel"] = u(D[pk]["rel_dist"]); out[f"ck.dist.{nm}.emb"] = u(D[pk]["emb_cosine"], 4 if D[pk]["emb_cosine"] > 0.99 else 3)
    out["ck.n"] = n_(ns.pop()) if len(ns) == 1 else "inconsistent"
    return out


def gate_ol_values():
    xs = [x for x in lines(OUT / "e01_olmo1b_newcode.jsonl") if "mass" in x]
    m = lambda key, cond: float(np.mean([x[key] for x in xs if cond(x)]))
    out = {"gate.post.ol.n": n_(len(xs)), "gate.post.ol.head": u(m("top1", lambda x: x["header"])),
           "gate.post.ol.termt": u(m("top1", lambda x: x["terminal"])), "gate.post.ol.terms": u(m("p_shallow_open", lambda x: x["terminal"]))}
    out["gate.post.ol.marg"] = u(m("top1", lambda x: True) - m("copy_ok", lambda x: True))
    gs = [x for x in lines(OUT / "e01_olmo1b_gen.jsonl") if "mass" in x]
    # round 1's gate item 4: worst not-open mass at dedents over widths 2, 3 and mixed (code/gate_summary.py rule)
    out["gate.post.ol.open"] = u(max(float(np.mean([x["p_shallow_notopen"] for x in gs if str(x["width"]) == w and x["cls"] == "shallower"]))
                                     for w in ("2", "3", "mixed")))
    return out


def x1_values():
    """X1's L17H6 / CARRY / TERM / DED / L11H9 values, and all 22 X1 verdicts re-derived from X1's raw run files."""
    C = X1 / "outputs" / "conf"
    LB, IND = rj(C / "c01_qwen3-06b-bf16_LB.json"), rj(C / "c01_qwen3-06b-bf16_IND.json")
    ORD, CAPS = rj(C / "c01_qwen3-06b-bf16_ORD.json"), rj(C / "c01_qwen3-17b_CAPS.json")
    ATT = rj(C / "c02_attn_qwen3-06b-bf16.json")
    kept = lambda R, g, k: R[g][k] / R["base"][k]
    chg = lambda g, k: IND[g][k] - IND["base"][k]
    rnd = lambda R, g: [k for k in R if k.startswith(g + "~r")]
    lbw = lambda g: kept(LB, g, "width_effect"); lbs = lambda g: kept(LB, g, "sep")
    ind4 = lambda g: dict(cU=chg(g, "code|dedent|lq"), cT=chg(g, "code|term|lq"), gT=chg(g, "gen|term|lq"), gU=chg(g, "gen|dedent|lq"))
    quiet = lambda g: (lambda d: abs(d["cU"]) <= 0.03 and abs(d["cT"]) <= 0.03 and abs(d["gT"]) <= 0.03 and abs(d["gU"]) <= 0.10)(ind4(g))
    below = lambda x, R, g, f: len(rnd(R, g)) == 20 and x < min(f(k) for k in rnd(R, g))
    ok_ = lambda R, g, k: kept(R, g, k)
    T = {}
    T["A1"] = lbw("L17H6") <= 0.85 and below(lbw("L17H6"), LB, "L17H6", lbw)
    T["A2"] = lbs("L17H6") <= 0.95 and below(lbs("L17H6"), LB, "L17H6", lbs)
    gt = lambda g: chg(g, "gen|term|lq")
    T["A3"] = gt("L17H6") <= -0.010 and below(gt("L17H6"), IND, "L17H6", gt)
    T["A4"] = lbw("CARRY-L17H6") >= 0.92 and lbw("TERM-L17H6") >= 0.92 and lbw("CARRY") <= 0.85 and lbw("TERM") <= 0.85
    te = lambda g, d: ok_(ORD, g, f"d{d}|teen_effect"); pu = lambda g, d: ok_(ORD, g, f"d{d}|pull")
    T["A5"] = abs(chg("L17H6", "code|dedent|lq")) <= 0.03 and abs(chg("L17H6", "gen|dedent|lq")) <= 0.10 and all(0.90 <= te("L17H6", d) <= 1.10 for d in (4, 5))
    du = lambda g: ok_(CAPS, g, "D_up"); dl = lambda g: ok_(CAPS, g, "D_low"); wl = lambda g: CAPS[g]["word_lp"] - CAPS["base"]["word_lp"]
    T["A6"] = du("L17H6") <= 0.97 and below(du("L17H6"), CAPS, "L17H6", du) and abs(wl("L17H6")) <= 0.03
    T["B1"] = lbw("WRITE") <= 0.70 and quiet("WRITE")
    T["B2"] = lbs("COUNT") <= 0.50 and quiet("COUNT")
    T["B3"] = lbw("READ-L17H6") <= 0.75 and quiet("READ-L17H6")
    cu = lambda g: chg(g, "code|dedent|lq")
    T["B4"] = cu("DED") <= -0.40 and below(cu("DED"), IND, "DED", cu) and 0.95 <= lbw("DED") <= 1.05 and 0.95 <= lbs("DED") <= 1.05
    T["C1"] = all(te("L0H2", d) <= 0.75 and below(te("L0H2", d), ORD, "L0H2", lambda k, d=d: te(k, d)) for d in (4, 5))
    T["C2"] = lbs("L0H2") <= 0.93 and below(lbs("L0H2"), LB, "L0H2", lbs)
    T["C3"] = quiet("L0H2")
    T["C4"] = all(pu("L0H3", d) <= 0.50 and below(pu("L0H3", d), ORD, "L0H3", lambda k, d=d: pu(k, d)) and te("L0H3", d) >= 0.90 for d in (4, 5))
    T["C5"] = lbs("L0H3") <= 0.50 and below(lbs("L0H3"), LB, "L0H3", lbs)
    T["C6"] = quiet("L0H3")
    T["C7"] = (dl("L0H3") <= 0.60 and below(dl("L0H3"), CAPS, "L0H3", dl) and wl("L0H3") <= -0.50 and below(wl("L0H3"), CAPS, "L0H3", wl)
               and (1 - dl("L0H3")) >= 3 * (1 - du("L0H3")))
    tasks = ("LB", "IND", "ORD", "CAPS")
    T["C8"] = all(ATT[t]["L0H3"]["dec"]["0"] >= 0.90 for t in tasks) and all(ATT[t]["L0H2"]["dec"]["1"] >= 0.50 for t in ("LB", "ORD", "CAPS"))
    T["D1"] = (te("L11H9", 4) <= 0.85 and te("L11H9", 5) <= 0.90 and pu("L11H9", 4) <= 0.85 and pu("L11H9", 5) <= 0.85
               and all(below(f("L11H9", d), ORD, "L11H9", lambda k, f=f, d=d: f(k, d)) for f in (te, pu) for d in (4, 5)))
    T["D2"] = lbw("L11H9") >= 0.93 and lbs("L11H9") >= 0.93
    T["D3"] = max(abs(v) for v in ind4("L11H9").values()) <= 0.05
    T["D4"] = all(ATT[t]["L11H9"]["dec"]["0"] >= 0.85 for t in tasks)
    out = {"x1.L17H6.width": u(lbw("L17H6")), "x1.L17H6.term": s(gt("L17H6")),
           "x1.L17H6.termr": s(min(gt(k) for k in rnd(IND, "L17H6"))),
           "x1.carry.width": u(lbw("CARRY")), "x1.carrynoL17H6.width": u(lbw("CARRY-L17H6")),
           "x1.term.width": u(lbw("TERM")), "x1.termnoL17H6.width": u(lbw("TERM-L17H6")),
           "x1.npass": str(sum(T.values())), "x1.ntests": str(len(T)),
           "x1.L11H9.ind": u(max(abs(v) for v in ind4("L11H9").values())),
           "x1.ded.lbw": u(lbw("DED")), "x1.ded.lbs": u(lbs("DED"))}
    vj = rj(C / "verdicts.json")["tests"]
    agree = {k: (T[k] == bool(vj[k]["passed"])) for k in T}
    return out, T, agree


def mat2_freeze_values():
    out = {}
    code = lines(DATA / "r2code" / "part-held.chunks.jsonl")
    out["mat2.code.chunks"] = n_(len(code)); out["mat2.code.repos"] = n_(len({r["repo"] for r in code}))
    out["mat2.gen2.held"] = n_(len(lines(DATA / "r2gen2" / "part-held.jsonl")))
    out["mat2.gen2.explore"] = n_(len(lines(DATA / "r2gen2" / "part-explore.jsonl")))
    fz = (D3 / "FROZEN.md").read_text()
    for key, f in (("freeze.charter2", "CHARTER-2.md"), ("freeze.protocol2", "PROTOCOL-2.md")):
        m = re.findall(rf"^\| {re.escape(f)} \| [0-9a-f]{{64}} \| \d{{4}}-\d\d-\d\d (\d\d:\d\d)", fz, re.M)
        out[key] = m[0] if len(m) == 1 else f"ambiguous ({len(m)} rows)"
    return out


# ------------------------------------------------------------------ everything
def compute(mode):
    vals, meta = {}, {"records": mode}
    B = records_block(mode); st = completeness(mode, B)
    meta["completeness"] = {k: dict(done=a, expected=b, complete=a == b) for k, (a, b) in st.items()}
    vals.update(block_values("c2", B))
    raw = {t: (B[t][0] if B.get(t) else None) for t in ("q", "qg", "o", "og", "g")}
    V = verdicts(raw["q"], raw["qg"], raw["o"], raw["og"], raw["g"])
    for i in PRED_IDS: vals[f"c2.v.{i}"] = V[i][0]
    for qn in QUESTIONS: vals[f"c2.v.{qn}"] = V[qn][0]
    vals["c2.ntests"] = str(len(PRED_IDS)); vals["c2.npass"] = str(sum(V[i][0] == "pass" for i in PRED_IDS))
    meta["verdict_details"] = {k: dict(verdict=v[0], detail=v[1]) for k, v in V.items()}
    meta["consistency"] = {}
    if raw["g"]:
        meta["consistency"]["r2c03 run-file means vs per-point means (max abs diff)"] = max(raw["g"][f"check_perpoint_{c}"] for c in ("header", "same", "term", "dedent"))
        meta["consistency"]["r2c03 heads"] = raw["g"]["heads"]; meta["consistency"]["r2c03 random groups"] = raw["g"]["n.rand"]
    for t in MODELS:
        if raw[t]: meta["consistency"][f"r2c01 {t}: other heads in layer"] = raw[t]["n.others"]
        if raw[t + "g"]: meta["consistency"][f"r2c02 {t}: skipped programs"] = raw[t + "g"]["n.skip"]
    Bc = B if mode == "calib" else records_block("calib")
    vals.update(cal_values(Bc, None))
    x2 = x2_values(); meta["consistency"]["x2.kw random sets"] = x2.pop("_x2.kw.nrand"); meta["consistency"]["x2.oe2 heads scanned"] = x2.pop("_x2.oe2.nheads")
    vals.update(x2); vals.update(ck_values()); vals.update(gate_ol_values())
    x1, T, agree = x1_values(); vals.update(x1)
    meta["x1_verdicts"] = {k: dict(rederived="pass" if T[k] else "fail", matches_verdicts_json=agree[k]) for k in T}
    vals.update(mat2_freeze_values())
    return vals, meta, B


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", choices=["conf", "calib", "smoke"], required=True)
    ap.add_argument("--out", default=str(OUT2 / "independent_values.json"))
    a = ap.parse_args()
    vals, meta, _ = compute(a.records)
    out = Path(a.out); out.write_text(json.dumps(vals, indent=1, ensure_ascii=False) + "\n")
    mp = out.with_suffix(".meta.json"); mp.write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str) + "\n")
    inc = [k for k, v in meta["completeness"].items() if not v["complete"]]
    print(f"records: {a.records}; {len(vals)} values written to {out}")
    if inc: print("INCOMPLETE records:", ", ".join(f"{k} {meta['completeness'][k]['done']}/{meta['completeness'][k]['expected']}" for k in inc))
    bad = [k for k, v in meta["x1_verdicts"].items() if not v["matches_verdicts_json"]]
    print("X1 verdicts re-derived:", sum(v["rederived"] == "pass" for v in meta["x1_verdicts"].values()), "pass of", len(meta["x1_verdicts"]),
          "; disagreements with X1 verdicts.json:", bad or "none")


if __name__ == "__main__":
    main()
