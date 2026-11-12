"""D3 round 3: independent check of every number and verdict in REPORT-3.md (checker code, not a producer).

  python3 -B round3/code/r3_check_independent.py [--json OUT]

Written 11 Nov 2026 without importing or running r3_build_report.py, r3_verdicts.py, c03_verdicts.py,
r2c04_verdicts.py or gate_summary.py, and without reading verdicts.json, verdicts_XE.json, measures_conf.json or
gate_summary.json. Every value is recomputed from raw run records:
  R1  round3/outputs/conf/c01_newcode.json, c01_gen.json, c02_gen.json      (original: outputs/conf/)
  R2  round3/outputs/r2/conf/r2c01_*.jsonl, r2c02_*.jsonl, r2c03_olmo1b_DEDO.json   (original: outputs/r2/conf/)
      (record readers copied from code/r2_independent.py, the round-2 checker, with paths changed)
  X,E round3/x1/outputs/conf/c01_qwen3-06b-bf16_IND.json, r3x_l11h9_IND.json, round3/outputs/conf/r3_pairs.json
      (original: X1-shared-parts/outputs/conf/c01_qwen3-06b-bf16_IND.json)
  G   round3/outputs/e01_<model>_{newcode,gen}.jsonl   (original: outputs/e01_*.jsonl, cross-checked against
      outputs/gate_summary_postfix_final.json)
  Material: round3/data/code/{fetch_summary,build_summary}.json, part-held.jsonl, fetched.jsonl, gen/gen2 parts.
Each report number is compared with the recomputed value rounded to the decimals the report prints.
Bootstrap intervals (square brackets in part E) are descriptive and are not recomputed; they are stripped first.
"""
import argparse, hashlib, json, os, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np

D3 = Path(__file__).resolve().parents[2]
R3 = D3 / "round3"; O3 = R3 / "outputs"; X1 = D3.parent / "X1-shared-parts"
REPORT = D3 / "REPORT-3.md"
MINUS = "−"
KWS = ("return", "raise", "break", "continue", "pass")
READ = []  # every round-3 file read, for the provenance check


def rj(p):
    p = Path(p); READ.append(p); return json.loads(p.read_text())


def lines(p):
    p = Path(p); READ.append(p)
    raw = [l for l in open(p) if l.strip()]; out = []
    for i, l in enumerate(raw):
        try: out.append(json.loads(l))
        except json.JSONDecodeError:
            if i == len(raw) - 1 and not l.endswith("\n"): break
            raise
    return out


# ================================================================== R1
def r1(dirp, c02name="c02_gen.json"):
    nc, gn, pr = rj(dirp / "c01_newcode.json"), rj(dirp / "c01_gen.json"), rj(dirp / c02name)
    ch = lambda R, g, m, c: R[g][m][c] - R["base"][m][c]
    rk = lambda R, g: [k for k in R if re.fullmatch(rf"{g}_rand\d+", k)]
    V = {}
    V["H1"] = ch(nc, "DED", "lq", "dedent"); V["H1.r"] = [ch(nc, k, "lq", "dedent") for k in rk(nc, "DED")]
    V["H2"] = ch(nc, "DED", "p_same", "dedent"); V["H2.r"] = [ch(nc, k, "p_same", "dedent") for k in rk(nc, "DED")]
    V["H3.open"] = nc["DED"]["p_open"]["term"]; V["H3.term"] = ch(nc, "DED", "lq", "term")
    V["H3.ratio"] = abs(V["H3.term"]) / abs(V["H1"])
    V["H4"] = ch(nc, "DED", "lq", "same")
    V["H5"] = ch(nc, "TERM", "lq", "term"); V["H5.r"] = [ch(nc, k, "lq", "term") for k in rk(nc, "TERM")]
    V["H6"] = ch(nc, "TERM", "lq", "dedent")
    V["H7"] = ch(nc, "TERMDED", "lq", "term"); V["H7.sum"] = V["H3.term"] + V["H5"]
    V["H8"] = ch(gn, "DED", "lq", "dedent"); V["H8.r"] = [ch(gn, k, "lq", "dedent") for k in rk(gn, "DED")]
    C = pr["comps"]; cr = [k for k in C if re.fullmatch(r"CARRY_rand\d+", k)]
    V["H9.nec"], V["H9.suf"] = C["CARRY"]["nec"], C["CARRY"]["suf"]
    V["H9.rnec"], V["H9.rsuf"] = [C[k]["nec"] for k in cr], [C[k]["suf"] for k in cr]
    V["n.chunks.nc"], V["n.chunks.gen"] = nc["n_chunks_used"], gn["n_chunks_used"]
    V["nrand"] = dict(DEDnc=len(V["H1.r"]), TERMnc=len(V["H5.r"]), DEDgen=len(V["H8.r"]), CARRY=len(cr))
    T = {}
    T["H1"] = V["H1"] <= -0.40 and V["H1"] < min(V["H1.r"]) and len(V["H1.r"]) == 20
    T["H2"] = V["H2"] >= 0.10 and V["H2"] > max(V["H2.r"]) and len(V["H2.r"]) == 20
    T["H3"] = V["H3.open"] >= 0.95 and V["H3.ratio"] <= 0.25
    T["H4"] = V["H4"] >= -0.01
    T["H5"] = V["H5"] <= -0.03 and V["H5"] < min(V["H5.r"]) and len(V["H5.r"]) == 20
    T["H6"] = abs(V["H6"]) <= 0.10
    T["H7"] = V["H7"] < V["H7.sum"]
    T["H8"] = V["H8"] <= -0.25 and V["H8"] < min(V["H8.r"]) and len(V["H8.r"]) == 20
    T["H9"] = (V["H9.nec"] >= 0.40 and V["H9.suf"] >= 0.20 and V["H9.nec"] > max(V["H9.rnec"])
               and V["H9.suf"] > max(V["H9.rsuf"]) and len(cr) == 20)
    core = all(T[k] for k in ("H1", "H2", "H3", "H5", "H6"))
    T["verdict"] = "confirmed" if core else ("partly confirmed" if all(T[k] for k in ("H1", "H2", "H3")) else "not confirmed")
    return V, T


# ================================================================== R2 (readers copied from code/r2_independent.py)
def r2c01(path):
    recs = lines(path); V = {}
    pts = []
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
    V["att_inner_ded"], V["att_inner_same"] = mean(att(ded)), mean(att(same))
    V["ko_inner_ded"], V["ko_inner_same"], V["ko_inner_term"] = mean(dko(ded, "ko_inner")), mean(dko(same, "ko_inner")), mean(dko(term, "ko_inner"))
    closed = [(c, p) for c, p in ded if p["has_closed"]]
    V["ko_closed_ded"], V["ko_inner_ded_closedsub"] = mean(dko(closed, "ko_closed")), mean(dko(closed, "ko_inner"))
    topen = [(c, p) for c, p in ded if p["true"] in p["oh"]]
    V["ko_true_ded"] = mean(dko(topen, "ko_true"))
    others = sorted({k for _, p in ded for k in p if k.startswith("ko_inner_h")}, key=lambda k: int(k[10:]))
    ov = {k: mean(dko(ded, k)) for k in others}
    V["n.others"] = len(others)
    V["others_lo"], V["others_hi"] = min(ov.values()), max(ov.values())
    V["others_maxabs"] = max(abs(x) for x in ov.values())
    multi = [(c, p) for c, p in pts if p["cls"] in ("dedent", "term") and len(p["oh"]) >= 2 and p["true"] in p["oh"]]
    pick = [int(np.argmax(p["att"])) == p["oh"].index(p["true"]) for _, p in multi]
    inner_rule = [p["oh"][-1] == p["true"] for _, p in multi]
    ni = [(c, p) for c, p in multi if p["oh"][-1] != p["true"]]
    V["pick_not_inner"] = mean([int(np.argmax(p["att"])) == p["oh"].index(p["true"]) for _, p in ni])
    V["pick_inner_attended_not_inner"] = mean([int(np.argmax(p["att"])) == len(p["oh"]) - 1 for _, p in ni])
    V["pick_all_multi"], V["innermost_rule_all_multi"] = mean(pick), mean(inner_rule)
    for who in ("head", "ctrl"):
        ent = [e for _, p in pts if p["cls"] == "dedent" and "steer" in p for e in p["steer"] if e[0] == who and e[2] != p["top_base"]]
        V[f"steer_dedent_{who}_dlq"] = mean([e[3] for e in ent]); V[f"steer_dedent_{who}_top"] = mean([e[4] == e[2] for e in ent])
    return V


def r2c02(path):
    recs = lines(path); V = {}
    ok = [r for r in recs if not r.get("skip")]
    V["n.programs"] = len(ok); V["n.records"] = len(recs)
    term = [r for r in ok if r["last"] == "terminal"]
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")

    def share(rs, cond):
        rs = [r for r in rs if r.get("neutral") is not None]
        b = mean([r["base"][0] for r in rs])
        return (mean([r[cond][0] for r in rs]) - b) / (mean([r["neutral"][0] for r in rs]) - b)
    ret = [r for r in term if r["kw"] == "return"]
    V["n.return"] = len(ret)
    V["share_blk_kw_return"] = share(ret, "blk_kw")
    L = sorted(int(k[3:]) for k in ret[0] if re.fullmatch(r"lay\d+", k))
    lay = {l: share(ret, f"lay{l}") for l in L}
    V["share_layer_return_max"] = max(lay.values())
    V["l0h2_line_dqcur_term"] = mean([r["l0h2_line"][0] - r["base"][0] for r in term])
    for who in ("head", "ctrl"):
        ent = [e for r in term if "steer" in r for e in r["steer"] if e[0] == who and e[2] != r["top_base"]]
        V[f"steer_term_{who}_dlq"] = mean([e[3] for e in ent]); V[f"steer_term_{who}_top"] = mean([e[4] == e[2] for e in ent])
    return V


def r2c03(path, g="DEDO"):
    R = rj(path); V = {}
    rk = sorted((k for k in R if re.fullmatch(r"rand\d+", k)), key=lambda k: int(k[4:]))
    V["n.rand"] = len(rk)
    for c in ("same", "term", "dedent"): V[f"d_lq_{c}"] = R[g]["lq"][c] - R["base"]["lq"][c]
    rd = [R[k]["lq"]["dedent"] - R["base"]["lq"]["dedent"] for k in rk]
    V["rand_lo"], V["rand_hi"] = min(rd), max(rd)
    V["d_psame_dedent"] = R[g]["p_same"]["dedent"] - R["base"]["p_same"]["dedent"]
    V["p_open_term_ablated"] = R[g]["p_open"]["term"]
    # per-point means against the run-file means (consistency)
    cls = np.array(R["cls"])
    V["perpoint_maxdiff"] = max(abs((np.mean(np.array(R[g]["per_point_lq"])[cls == c]) - np.mean(np.array(R["base"]["per_point_lq"])[cls == c]))
                                    - V[f"d_lq_{c}"]) for c in ("same", "term", "dedent"))
    return V


def r2(d):
    B = dict(q=r2c01(d / "r2c01_qwen3-06b-bf16_L23H2.jsonl"), qg=r2c02(d / "r2c02_qwen3-06b-bf16_L23H2.jsonl"),
             o=r2c01(d / "r2c01_olmo1b_L13H1.jsonl"), og=r2c02(d / "r2c02_olmo1b_L13H1.jsonl"), g=r2c03(d / "r2c03_olmo1b_DEDO.json"))
    q, qg, o, og, g = B["q"], B["qg"], B["o"], B["og"], B["g"]
    fl = lambda c: max(c, 0.02)
    T = {}
    T["Q1"] = q["att_inner_ded"] >= 0.30 and q["att_inner_ded"] >= 2.5 * q["att_inner_same"]
    T["Q2"] = q["ko_inner_ded"] <= -0.05 and q["ko_inner_ded"] < q["others_lo"] and abs(q["ko_inner_ded"]) >= 2 * q["others_maxabs"] and q["n.others"] == 15
    T["Q3"] = q["ko_inner_same"] >= -0.02 and abs(q["ko_inner_term"]) <= 0.25 * abs(q["ko_inner_ded"])
    T["Q4"] = abs(q["ko_closed_ded"]) <= 0.5 * abs(q["ko_inner_ded_closedsub"])
    T["Q5"] = q["pick_not_inner"] >= 0.60 and q["pick_all_multi"] - q["innermost_rule_all_multi"] >= 0.02
    T["Q6"] = (q["steer_dedent_head_dlq"] >= 1.0 and q["steer_dedent_head_dlq"] >= 5 * fl(q["steer_dedent_ctrl_dlq"])
               and q["steer_dedent_head_top"] >= q["steer_dedent_ctrl_top"] + 0.05)
    T["Q7"] = q["ko_true_ded"] <= -0.10
    T["Q8"] = (qg["steer_term_head_dlq"] >= 1.0 and qg["steer_term_head_dlq"] >= 5 * fl(qg["steer_term_ctrl_dlq"])
               and qg["steer_term_head_top"] >= 0.20 and qg["steer_term_head_top"] >= qg["steer_term_ctrl_top"] + 0.10)
    T["Q9"] = qg["share_blk_kw_return"] >= 0.40
    T["Q10"] = qg["share_layer_return_max"] <= 0.35 * qg["share_blk_kw_return"]
    T["Q11"] = abs(qg["l0h2_line_dqcur_term"]) <= 0.05
    T["O1"] = o["att_inner_ded"] >= 0.20 and o["att_inner_ded"] >= 1.4 * o["att_inner_same"]
    T["O2"] = o["ko_inner_ded"] <= -0.05 and o["ko_inner_ded"] < o["others_lo"] and abs(o["ko_inner_ded"]) >= 2 * o["others_maxabs"]
    T["O3"] = o["ko_inner_same"] >= -0.02 and abs(o["ko_inner_term"]) <= 0.25 * abs(o["ko_inner_ded"])
    T["O4"] = abs(o["ko_closed_ded"]) <= 0.5 * abs(o["ko_inner_ded_closedsub"])
    T["O5"] = o["pick_not_inner"] >= 0.45 and o["pick_not_inner"] - o["pick_inner_attended_not_inner"] >= 0.15
    T["O6"] = (o["steer_dedent_head_dlq"] >= 0.8 and o["steer_dedent_head_dlq"] >= 4 * fl(o["steer_dedent_ctrl_dlq"])
               and o["steer_dedent_head_top"] >= o["steer_dedent_ctrl_top"] + 0.05)
    T["O7"] = o["ko_true_ded"] <= -0.08
    T["O8"] = (og["steer_term_head_dlq"] >= 0.8 and og["steer_term_head_dlq"] >= 4 * fl(og["steer_term_ctrl_dlq"])
               and og["steer_term_head_top"] >= 0.15 and og["steer_term_head_top"] >= og["steer_term_ctrl_top"] + 0.10)
    T["O9"] = og["share_blk_kw_return"] >= 0.40
    T["O10"] = og["share_layer_return_max"] <= 0.35 * og["share_blk_kw_return"]
    T["G1"] = g["d_lq_dedent"] <= -0.30 and g["d_lq_dedent"] < g["rand_lo"] and g["n.rand"] == 20
    T["G2"] = abs(g["d_lq_term"]) <= 0.25 * abs(g["d_lq_dedent"]) and g["p_open_term_ablated"] >= 0.90
    T["G3"] = g["d_lq_same"] >= -0.02
    T["G4"] = g["d_psame_dedent"] >= 0.08
    QS = {"Question 1": ["Q1", "Q2", "Q3"], "Question 2": ["Q5", "Q6", "Q8"], "Question 3": ["Q9", "Q10", "Q11"],
          "Question 4a": ["G1", "G2", "O9"], "Question 4b": ["O1", "O2", "O3", "O6", "O8"]}
    Qv = {k: ("confirmed" if all(T[i] for i in v) else "not confirmed") for k, v in QS.items()}
    return B, T, Qv


def r2_cells(B):
    """Numbers printed in the R2 table, in printed order, per id."""
    q, qg, o, og, g = B["q"], B["qg"], B["o"], B["og"], B["g"]
    C = {}
    for P, h, hg in (("Q", q, qg), ("O", o, og)):
        C[P + "1"] = [h["att_inner_ded"], h["att_inner_same"]]
        C[P + "2"] = [h["ko_inner_ded"], h["others_lo"], h["others_hi"]]
        C[P + "3"] = [h["ko_inner_same"], h["ko_inner_term"]]
        C[P + "4"] = [h["ko_closed_ded"], h["ko_inner_ded_closedsub"]]
        C[P + "5"] = [h["pick_not_inner"], h["pick_all_multi"], h["innermost_rule_all_multi"]]
        C[P + "6"] = [h["steer_dedent_head_dlq"], h["steer_dedent_ctrl_dlq"], h["steer_dedent_head_top"], h["steer_dedent_ctrl_top"]]
        C[P + "7"] = [h["ko_true_ded"]]
        C[P + "8"] = [hg["steer_term_head_dlq"], hg["steer_term_ctrl_dlq"], hg["steer_term_head_top"], hg["steer_term_ctrl_top"]]
        C[P + "9"] = [hg["share_blk_kw_return"]]
        C[P + "10"] = [hg["share_layer_return_max"]]
    C["Q11"] = [qg["l0h2_line_dqcur_term"]]
    C["G1"] = [g["d_lq_dedent"], g["rand_lo"], g["rand_hi"]]
    C["G2"] = [g["d_lq_term"], g["p_open_term_ablated"]]
    C["G3"] = [g["d_lq_same"]]
    C["G4"] = [g["d_psame_dedent"]]
    return C


# ================================================================== X and E
def ind(R):
    chg = lambda g, k: R[g][k] - R["base"][k]
    four = lambda g: [chg(g, "code|dedent|lq"), chg(g, "code|term|lq"), chg(g, "gen|dedent|lq"), chg(g, "gen|term|lq")]
    rnd = lambda g, k: [chg(r, k) for r in R if r.startswith(g + "~r")]
    return chg, four, rnd


def xpart(R):
    chg, four, rnd = ind(R)
    quiet = lambda g: (lambda d: abs(d[0]) <= 0.03 and abs(d[1]) <= 0.03 and abs(d[3]) <= 0.03 and abs(d[2]) <= 0.10)(four(g))
    C, T = {}, {}
    gT = rnd("L17H6", "gen|term|lq"); C["XA3"] = [chg("L17H6", "gen|term|lq"), min(gT), max(gT)]
    T["XA3"] = C["XA3"][0] <= -0.010 and C["XA3"][0] < min(gT) and len(gT) == 20
    C["XA5"] = [chg("L17H6", "code|dedent|lq"), chg("L17H6", "gen|dedent|lq")]
    T["XA5"] = abs(C["XA5"][0]) <= 0.03 and abs(C["XA5"][1]) <= 0.10
    for i, g in (("XB1", "WRITE"), ("XB2", "COUNT"), ("XB3", "READ-L17H6"), ("XC3", "L0H2"), ("XC6", "L0H3")):
        C[i] = four(g); T[i] = quiet(g)
    C["XD3"] = four("L11H9"); T["XD3"] = max(abs(v) for v in C["XD3"]) <= 0.05
    cU = rnd("DED", "code|dedent|lq"); C["XB4"] = [chg("DED", "code|dedent|lq"), min(cU), max(cU)]
    T["XB4"] = C["XB4"][0] <= -0.40 and C["XB4"][0] < min(cU) and len(cU) == 20
    return C, T


def epart(I, G, P):
    """I: X1 c01 IND run (L11H9 alone); G: r3x_l11h9 run (CARRY, TERM); P: r3_pairs."""
    chI, _, rI = ind(I); chG, _, rG = ind(G)
    C, T, X = {}, {}, {}
    e1 = chI("L11H9", "gen|term|lq"); r1 = rI("L11H9", "gen|term|lq")
    C["E1"] = [e1, min(r1), max(r1)]; T["E1"] = e1 <= -0.010 and e1 < min(r1) and len(r1) == 20
    ca, te = chG("CARRY", "gen|term|lq"), chG("TERM", "gen|term|lq")
    rc, rt = rG("CARRY", "gen|term|lq"), rG("TERM", "gen|term|lq")
    C["E2"] = [ca, min(rc), max(rc), te, min(rt), max(rt)]
    T["E2"] = all(v <= -0.010 and v < min(r) and len(r) == 20 for v, r in ((ca, rc), (te, rt)))
    cn, tn = chG("CARRY-L11H9", "gen|term|lq"), chG("TERM-L11H9", "gen|term|lq")
    C["E3"] = [cn, ca, cn - ca, tn, te, tn - te]; T["E3"] = (cn - ca) >= 0.010 and (tn - te) >= 0.010
    C["E4"] = [chI("L11H9", "code|dedent|lq"), chI("L11H9", "gen|dedent|lq")]
    T["E4"] = abs(C["E4"][0]) <= 0.03 and abs(C["E4"][1]) <= 0.10
    e5 = chI("L11H9", "code|term|lq"); r5 = rI("L11H9", "code|term|lq")
    C["E5"] = [e5, min(r5), max(r5)]; T["E5"] = e5 < min(r5) and len(r5) == 20
    Cm = P["comps"]
    rk = lambda h: [k for k in Cm if re.fullmatch(rf"{h}_rand\d+", k)]
    l, lr = Cm["L11H9"], rk("L11H9"); s, sr = Cm["L17H6"], rk("L17H6")
    C["E6"] = [l["nec"], max(Cm[k]["nec"] for k in lr), l["suf"], max(Cm[k]["suf"] for k in lr),
               s["nec"], max(Cm[k]["nec"] for k in sr), s["suf"], max(Cm[k]["suf"] for k in sr)]
    T["E6"] = l["nec"] >= 0.05 and l["nec"] > max(Cm[k]["nec"] for k in lr) and len(lr) == 20
    # L17H6's draws: how many it beats, and which draw beats it
    above = sum(s["nec"] > Cm[k]["nec"] for k in sr)
    top = max(sr, key=lambda k: Cm[k]["nec"])
    X["L17H6.above"], X["L17H6.ndraws"], X["L17H6.topdraw.heads"], X["L17H6.topdraw.nec"] = above, len(sr), Cm[top]["heads"], Cm[top]["nec"]
    X["E6.L11H9.heads.layer11"] = all(str(h).startswith("L11H") for k in lr for h in Cm[k]["heads"])
    X["E6.L17H6.heads.layer17"] = all(str(h).startswith("L17H") for k in sr for h in Cm[k]["heads"])
    X["E1.rand.n"] = len(r1)
    T["verdict"] = "confirmed" if (T["E1"] and T["E3"]) else ("a single-head effect only" if T["E1"] else "not confirmed")
    return C, T, X


# ================================================================== gate
def gate(dirp, model, tag=False):
    rows = lambda src: [x for x in lines(dirp / f"e01_{model}_{src}.jsonl") if "mass" in x]
    nc, gn = rows("newcode"), rows("gen")
    m = lambda xs, k, f=lambda x: True: float(np.mean([x[k] for x in xs if f(x)]))
    V = dict(n=len(nc), head=m(nc, "top1", lambda x: x["header"]), termt=m(nc, "top1", lambda x: x["terminal"]),
             terms=m(nc, "p_shallow_open", lambda x: x["terminal"]), marg=m(nc, "top1") - m(nc, "copy_ok"))
    V["open"] = max(m(gn, "p_shallow_notopen", lambda x, w=w: str(x["width"]) == w and x["cls"] == "shallower") for w in ("2", "3", "mixed"))
    V["items"] = dict(item1=V["head"] >= 0.90, item2=V["termt"] >= 0.5 and V["terms"] >= 0.7, item3=V["marg"] >= 0.10, open=V["open"] <= 0.10)
    return V


# ================================================================== comparison machinery
NUM = re.compile(r"[−+\-]?\d[\d,]*(?:\.\d+)?")
RESULTS = []  # (section, label, report text, report number, computed, ok)


def nums(text):
    text = re.sub(r"\[[^\]]*\]", " ", text)        # bootstrap intervals: not checked
    text = re.sub(r"L\d+H\d+", " ", text)           # head names
    return NUM.findall(text)


def cmp_num(rep, val):
    r = rep.replace(MINUS, "-").replace(",", "")
    d = len(r.split(".")[1]) if "." in r else 0
    if isinstance(val, bool): val = int(val)
    if isinstance(val, str): return r.lstrip("+") == val, val
    mine = f"{val:.{d}f}"
    return abs(float(r) - float(mine)) < 1e-12, mine


def check(section, label, text, vals, report):
    if text not in report:
        RESULTS.append((section, label, text, "(text not found in REPORT-3.md)", "", False)); return
    ns = nums(text)
    if len(ns) != len(vals):
        RESULTS.append((section, label, text, f"{len(ns)} numbers in text", f"{len(vals)} values supplied", False)); return
    for rn, v in zip(ns, vals):
        ok, mine = cmp_num(rn, v)
        RESULTS.append((section, label, text, rn, mine, ok))


def check_word(section, label, text, mine_word, report):
    ok = text in report and mine_word.lower() in text.lower()
    RESULTS.append((section, label, text, "verdict", mine_word, ok))


def rows_of(report, ids):
    out = {}
    for l in report.splitlines():
        if l.startswith("| "):
            c = [x.strip() for x in l.strip().strip("|").split(" | ")]
            if c[0] in ids: out[c[0]] = (l, c)
    return out


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=None); a = ap.parse_args()
    rep = REPORT.read_text()
    C3 = O3 / "conf"
    # ---------- R1
    V3, T3 = r1(C3); V0, _ = r1(D3 / "outputs" / "conf")
    rows = rows_of(rep, [f"H{i}" for i in range(1, 10)])
    for i in range(1, 10):
        h = f"H{i}"; line, c = rows[h]
        main_ = {"H3": "H3.open", "H9": "H9.nec"}.get(h, h)
        check("R1", f"{h} original", c[2], [V0[main_]], rep)
        check("R1", f"{h} round 3", c[3], [V3[main_]], rep)
        ctx = {"H1": [min(V3["H1.r"])], "H2": [max(V3["H2.r"])], "H3": [V3["H3.term"], V3["H3.ratio"]], "H5": [min(V3["H5.r"])],
               "H7": [V3["H7.sum"]], "H8": [min(V3["H8.r"])], "H9": [V3["H9.suf"], max(V3["H9.rnec"]), max(V3["H9.rsuf"])]}.get(h, [])
        if ctx: check("R1", f"{h} context", c[4], ctx, rep)
        check_word("R1", f"{h} verdict", line, "pass" if T3[h] else "fail", rep)
    n1 = sum(T3[f"H{i}"] for i in range(1, 10))
    check("Result", "R1 pass count", f"(H1–H9): {n1} of 9 tests pass", [1, 9, n1, 9], rep)
    check_word("Result", "R1 verdict", f"by the frozen rule the two-route account is **{T3['verdict']}**", T3["verdict"], rep)

    # ---------- R2
    B3, T2, Qv = r2(O3 / "r2" / "conf"); B0, _, _ = r2(D3 / "outputs" / "r2" / "conf")
    c3, c0 = r2_cells(B3), r2_cells(B0)
    ids = [f"Q{i}" for i in range(1, 12)] + [f"O{i}" for i in range(1, 11)] + [f"G{i}" for i in range(1, 5)]
    rows = rows_of(rep, ids)
    for i in ids:
        line, c = rows[i]
        check("R2", f"{i} original", c[2], c0[i], rep); check("R2", f"{i} round 3", c[3], c3[i], rep)
        check_word("R2", f"{i} verdict", line, "pass" if T2[i] else "fail", rep)
    n2 = sum(T2[i] for i in ids)
    check("Result", "R2 pass count", f"G1–G4): {n2} of 25 tests pass", [1, 4, n2, 25], rep)
    for qn, v in Qv.items():
        m = re.search(rf"{qn} \([^)]*(?:\([^)]*\))?[^)]*\): \*\*([a-z ]+)\*\*", rep)
        RESULTS.append(("Result", f"{qn} verdict", m.group(0) if m else qn, m.group(1) if m else "(not found)", v, bool(m) and m.group(1) == v))
    q = B3["q"]; qg = B3["qg"]
    check("R2", "R2 counts", "Natural code: 2,764 chunks from 225 repositories (3,780 unmarked dedents, 19,383 same-level lines and 3,855 terminal dedents",
          [q["n.chunks"], q["n.repos"], q["n.ded"], q["n.same"], q["n.term"]], rep)
    check("R2", "R2 programs", "Generated: 1,500 targeted programs (149 ending in", [qg["n.programs"], qg["n.return"]], rep)

    # ---------- X
    IND3 = rj(R3 / "x1" / "outputs" / "conf" / "c01_qwen3-06b-bf16_IND.json")
    IND0 = rj(X1 / "outputs" / "conf" / "c01_qwen3-06b-bf16_IND.json")
    Cx3, Tx3 = xpart(IND3); Cx0, _ = xpart(IND0)
    xids = ["XA3", "XA5", "XB1", "XB2", "XB3", "XC3", "XC6", "XD3", "XB4"]
    rows = rows_of(rep, xids)
    for i in xids:
        line, c = rows[i]
        check("X", f"{i} original", c[2], Cx0[i], rep); check("X", f"{i} round 3", c[3], Cx3[i], rep)
        check_word("X", f"{i} verdict", line, "pass" if Tx3[i] else "fail", rep)
    nx = sum(Tx3.values())
    check("Result", "X pass count", f"indentation side: {nx} of 9 tests pass", [nx, 9], rep)
    b = IND3["base"]
    check("X", "X points", f"Points (round 3): code unmarked 548, code terminal 540, generated unmarked 1,547, generated terminal 1,288.",
          [3, b["code|dedent|n"], b["code|term|n"], b["gen|dedent|n"], b["gen|term|n"]], rep)
    check("X", "X base", "Unablated mean log q(true): cU −0.304, cT −0.039, gU −1.846, gT −0.566.",
          [b["code|dedent|lq"], b["code|term|lq"], b["gen|dedent|lq"], b["gen|term|lq"]], rep)

    # ---------- E
    G3 = rj(R3 / "x1" / "outputs" / "conf" / "r3x_l11h9_IND.json"); P3 = rj(C3 / "r3_pairs.json")
    Ce, Te, Xe = epart(IND3, G3, P3)
    # same base in both runs (same points)
    base_same = all(IND3["base"][k] == G3["base"][k] for k in IND3["base"])
    eids = ["E1", "E2", "E3", "E4", "E5", "E6"]
    rows = rows_of(rep, eids)
    for i in eids:
        line, c = rows[i]
        check("E", f"{i} result", c[2], Ce[i], rep)
        check_word("E", f"{i} verdict", line, "pass" if Te[i] else "fail", rep)
    check_word("E", "E verdict", f"**Verdict: {Te['verdict']}.**", Te["verdict"], rep)
    res = (f"**L11H9's second role (part E): {Te['verdict']}** (E1 {'pass' if Te['E1'] else 'fail'}, E3 {'pass' if Te['E3'] else 'fail'}; "
           f"qualifiers: E2 {'pass' if Te['E2'] else 'fail'}, E4 {'pass' if Te['E4'] else 'fail'}, E5 {'pass' if Te['E5'] else 'fail'}, "
           f"E6 {'pass' if Te['E6'] else 'fail'})")
    RESULTS.append(("Result", "E summary line", res, "verdicts", "as recomputed", res in rep))

    # ---------- gate
    models = [("Qwen3-0.6B-Base", "qwen3-06b"), ("OLMo-2-0425-1B", "olmo1b"), ("Pythia-410M", "pythia-410m"),
              ("Gemma-4-E2B", "gemma4-e2b"), ("GPT-2", "gpt2")]
    PF = json.loads((D3 / "outputs" / "gate_summary_postfix_final.json").read_text())
    rows = rows_of(rep, [m for m, _ in models]); G = {}
    gate_orig_agree = {}
    for disp, mk in models:
        g3 = gate(O3, mk); G[mk] = g3; line, c = rows[disp]
        check("G", f"{disp} points", c[1], [g3["n"]], rep)
        keys = ["head", "termt", "terms", "marg", "open"]
        if mk == "olmo1b":
            for k, cell in zip(keys, c[2:]): check("G", f"{disp} {k}", cell, [g3[k]], rep)
        else:
            g0 = gate(D3 / "outputs", mk)
            pf = PF[mk]; nc = pf["newcode"]
            pfv = dict(head=nc["header_top1"], termt=nc["terminal_top1"], terms=nc["terminal_shopen"], marg=nc["top1"] - nc["copy"],
                       open=max(pf[f"gen_w{w}"]["dedent_shnotopen"] for w in ("2", "3", "mixed")))
            gate_orig_agree[mk] = max(abs(g0[k] - pfv[k]) for k in keys)
            for k, cell in zip(keys, c[2:]): check("G", f"{disp} {k}", cell, [g3[k], g0[k]], rep)

    # ---------- Reading (hand-written)
    I1 = IND3; chI, _, rI = ind(I1)
    rd = [
        ("chunks, programs", "passes again on 2,764 chunks of permissively licensed code and 2,000 new generated programs.",
         [V3["n.chunks.nc"], V3["n.chunks.gen"]]),
        ("H1", "costs unmarked dedents −0.994 in mean log q(true) (original −1.091; random groups at most −0.028)", [V3["H1"], V0["H1"], min(V3["H1.r"])]),
        ("H5", "only just past its frozen bar: −0.032 against a threshold of −0.03 (original −0.044)", [V3["H5"], -0.03, V0["H5"]]),
        ("R2 count", "All 25 round-2 tests pass on the same new code and on 1,500 new targeted programs", [n2, -2, qg["n.programs"]]),
        ("Q2", "costs unmarked dedents −0.117 (original −0.148), while the other 15 heads of layer 23 move it by −0.004 to +0.035",
         [q["ko_inner_ded"], B0["q"]["ko_inner_ded"], q["n.others"], 23, q["others_lo"], q["others_hi"]]),
        ("Q6", "raises that header's indent by +2.74 nats on average (original +2.72)", [q["steer_dedent_head_dlq"], B0["q"]["steer_dedent_head_dlq"]]),
        ("Q9", "(0.710; original 0.716)", [qg["share_blk_kw_return"], B0["qg"]["share_blk_kw_return"]]),
        ("G1", "costs unmarked dedents −0.776 against random sets from −0.109 to +0.098", [B3["g"]["d_lq_dedent"], B3["g"]["rand_lo"], B3["g"]["rand_hi"]]),
        ("O2", "again reads open headers (knock-out −0.115)", [B3["o"]["ko_inner_ded"]]),
        ("XA3", "(−0.025, random −0.010 to 0.004; original −0.022)", [Cx3["XA3"][0], Cx3["XA3"][1], Cx3["XA3"][2], Cx0["XA3"][0]]),
        ("XB4", "(code unmarked dedents −0.801 on the first 400 chunks; original −1.112)", [Cx3["XB4"][0], 400, Cx0["XB4"][0]]),
        ("E1", "at generated terminal-statement dedents by 0.027, below all 20 random layer-11 heads (−0.004 to 0.003)",
         [abs(Ce["E1"][0]), Xe["E1.rand.n"], -11, Ce["E1"][1], Ce["E1"][2]]),
        ("E3", "leaving L11H9 out weakens CARRY by 0.039 and TERM by 0.052", [Ce["E3"][2], Ce["E3"][5]]),
        ("E5", "(−0.003, random −0.001 to 0.001; E5)", [Ce["E5"][0], Ce["E5"][1], Ce["E5"][2], 5]),
        ("E6", "it carries 0.122 of the clean-minus-corrupt gap by necessity, against at most 0.033 for random layer-11 heads (E6)",
         [Ce["E6"][0], Ce["E6"][1], -11, 6]),
        ("E6 L17H6", "carries 0.069; it lies above 19 of its 20 random layer-17 heads", [Ce["E6"][4], Xe["L17H6.above"], Xe["L17H6.ndraws"], -17]),
        ("E6 L17H4", "itself a CARRY head and the strongest terminal carrier (0.261)", [Xe["L17H6.topdraw.nec"]]),
        ("Pythia gate", "(0.874 against the 0.90 bar; original 0.870)", [G["pythia-410m"]["head"], 0.90, gate(D3 / "outputs", "pythia-410m")["head"]]),
    ]
    for lab, txt, vals in rd: check("Reading", lab, txt, vals, rep)
    # L17H4 is the draw that beats L17H6
    RESULTS.append(("Reading", "E6 top draw is L17H4", "which is L17H4", "L17H4", str(Xe["L17H6.topdraw.heads"]),
                    "which is L17H4" in rep and Xe["L17H6.topdraw.heads"] == ["L17H4"]))
    # larger than L17H6
    RESULTS.append(("Reading", "L11H9 larger than L17H6", "meets it again with a larger value than L17H6", f"{Ce['E1'][0]:.3f} vs {Cx3['XA3'][0]:.3f}",
                    str(abs(Ce["E1"][0]) > abs(Cx3["XA3"][0])), abs(Ce["E1"][0]) > abs(Cx3["XA3"][0])))
    # calibration 0.239 (round3/outputs/calib/r3_pairs.json, L17H6's random draws)
    Pc = rj(O3 / "calib" / "r3_pairs.json")["comps"]
    sr = [k for k in Pc if re.fullmatch(r"L17H6_rand\d+", k)]
    calib_l17h4 = [Pc[k]["nec"] for k in sr if Pc[k]["heads"] == ["L17H4"]]
    calib_top = max(sr, key=lambda k: Pc[k]["nec"]) if sr else None
    check("Reading", "E6 calibration", "the same draw appeared in calibration (0.239)", [max(calib_l17h4) if calib_l17h4 else float("nan")], rep)
    RESULTS.append(("Reading", "calibration top draw is L17H4", "same draw appeared in calibration", "L17H4",
                    str(Pc[calib_top]["heads"]) if calib_top else "none", bool(calib_top) and Pc[calib_top]["heads"] == ["L17H4"]))
    # gate pass statements
    gi = {mk: G[mk]["items"] for mk in G}
    allpass = [mk for mk in ("qwen3-06b", "olmo1b", "gemma4-e2b") if all(gi[mk].values())]
    RESULTS.append(("Reading", "gate: Qwen3, OLMo-2, Gemma pass every item", "pass every gate item", "3 models", f"{len(allpass)} models",
                    len(allpass) == 3))
    py = gi["pythia-410m"]
    RESULTS.append(("Reading", "gate: Pythia misses only header", "Pythia-410M again misses only the header item", "item1 only",
                    str({k: v for k, v in py.items()}), (not py["item1"]) and py["item2"] and py["item3"] and py["open"]))
    gp = gi["gpt2"]
    RESULTS.append(("Reading", "gate: GPT-2 fails every item", "GPT-2 again fails every item", "all fail", str(gp), not any(gp.values())))
    py0 = gate(D3 / "outputs", "pythia-410m")["items"]; gp0 = gate(D3 / "outputs", "gpt2")["items"]
    RESULTS.append(("Reading", "gate: 'again' (original Pythia header-only, GPT-2 all)", "again misses / again fails", "same pattern originally",
                    f"pythia {py0}; gpt2 {gp0}", (not py0["item1"]) and py0["item2"] and py0["item3"] and py0["open"] and not any(gp0.values())))

    # ---------- material
    fs = rj(R3 / "data" / "code" / "fetch_summary.json"); bs = rj(R3 / "data" / "code" / "build_summary.json")
    fe = lines(R3 / "data" / "code" / "fetched.jsonl"); ph = lines(R3 / "data" / "code" / "part-held.jsonl")
    gen = lines(R3 / "data" / "gen" / "part-held.jsonl"); gen2 = lines(R3 / "data" / "gen2" / "part-held.jsonl")
    lic_ok = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "Unlicense"}
    lc = {}
    for r in ph: lc[r["license"]] = lc.get(r["license"], 0) + 1
    check("Material", "files, bytes", "1,000 Python files (9,460,326 bytes) fetched on 10 November 2026 from 274 GitHub repositories created on or after 1 September 2026",
          [len(fe), sum(r["bytes"] for r in fe), 10, 2026, len({r["repo"] for r in fe}), 1, 2026], rep)
    check("Material", "listing (fetch_summary only)", "(1,404 listed, 1,068 after removing", [fs["repos_listed"], fs["repos_after_exclusion"]], rep)
    check("Material", "build", "652 files survived normalisation and gave 2,764 chunks from 225 repositories (9 chunks that repeat an earlier part and 83 repeats",
          [bs["normalised"], len(ph), len({r["repo"] for r in ph}), bs["dup_old"], bs["dup"]], rep)
    lictxt = "Chunks by licence: Apache-2.0 1,192, MIT 573, BSD-3-Clause 471, CC0-1.0 203, Unlicense 169, BSD-2-Clause 156."
    # numbers in licence names are part of the names: check the counts directly
    for nm in ("Apache-2.0", "MIT", "BSD-3-Clause", "CC0-1.0", "Unlicense", "BSD-2-Clause"):
        mm = re.search(rf"{re.escape(nm)} ([\d,]*\d)", lictxt)
        RESULTS.append(("Material", f"licence {nm}", lictxt, mm.group(1), f"{lc.get(nm, 0):,}", lictxt in rep and mm.group(1) == f"{lc.get(nm, 0):,}"))
    gs, g2s = sorted(r["seed"] for r in gen), sorted(r["seed"] for r in gen2)
    check("Material", "generated", "2,000 from round 1's generator (seeds 300000–301999) and 1,500 from round 2's targeted generator (seeds 10000–11499)",
          [len(gen), 1, gs[0], gs[-1], len(gen2), 2, g2s[0], g2s[-1]], rep)
    mat_extra = dict(
        fetched_utc=fs["fetched_utc"], fetch_summary_matches=(fs["files"] == len(fe) and fs["bytes"] == sum(r["bytes"] for r in fe) and fs["repos_used"] == len({r["repo"] for r in fe})),
        all_created_after_1_sep=all(r["created_at"] >= "2026-09-01" for r in fe), all_licences_permissive=all(r["license"] in lic_ok for r in fe),
        all_commits_pinned=all(re.fullmatch(r"[0-9a-f]{40}", r["commit"]) for r in fe),
        chunks_commit_pinned=all(re.fullmatch(r"[0-9a-f]{40}", r.get("commit", "")) for r in ph),
        gen_seeds_contiguous=gs == list(range(gs[0], gs[0] + len(gs))), gen2_seeds_contiguous=g2s == list(range(g2s[0], g2s[0] + len(g2s))),
        build_summary_matches=(bs["held_chunks"] == len(ph) and bs["held_repos"] == len({r["repo"] for r in ph}) and bs["licences"] == lc),
        distinct_files_in_chunks=len({(r["repo"], r["path"]) for r in ph}),
        licence_dir_entries=len(os.listdir(R3 / "data" / "code" / "licences")))
    for k in ("fetch_summary_matches", "all_created_after_1_sep", "all_licences_permissive", "all_commits_pinned", "chunks_commit_pinned",
              "gen_seeds_contiguous", "gen2_seeds_contiguous", "build_summary_matches"):
        RESULTS.append(("Material", k, "(material claim)", "true", str(mat_extra[k]), bool(mat_extra[k])))
    RESULTS.append(("Material", "fetch date", "fetched on 10 November 2026", "2026-11-10", mat_extra["fetched_utc"], mat_extra["fetched_utc"].startswith("2026-11-10")))
    # X1 parts: first 400 chunks / programs of the round-3 parts
    x1c = lines(R3 / "x1" / "data" / "code" / "part-held.jsonl"); x1g = lines(R3 / "x1" / "data" / "gen" / "part-held.jsonl")
    mat_extra["x1_code_is_round3_code"] = x1c == ph; mat_extra["x1_gen_is_round3_gen"] = x1g == gen

    # ---------- provenance
    fz = (D3 / "FROZEN.md").read_text()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    digests = {}
    for rel in ("PROTOCOL-3.md", "round3/data/code/part-held.jsonl", "round3/data/code/fetched.jsonl", "round3/data/code/fetch_summary.json",
                "round3/data/code/build_summary.json", "round3/data/gen/part-held.jsonl", "round3/data/gen2/part-held.jsonl"):
        m = re.search(rf"^\| {re.escape(rel)} \| ([0-9a-f]{{64}}) \|", fz, re.M)
        digests[rel] = bool(m) and m.group(1) == sha(D3 / rel)
    for rel, ok in digests.items():
        RESULTS.append(("Provenance", f"digest {rel}", "FROZEN.md", "frozen digest", "match" if ok else "MISMATCH", ok))
    t0 = os.stat(D3 / "PROTOCOL-3.md").st_mtime
    prov = []
    for p in sorted(set(READ)):
        rp = p.resolve()
        under = str(p).startswith(str(R3))
        if not under: continue
        mt = os.stat(p).st_mtime
        kind = "data (sealed before the freeze)" if "/round3/data/" in str(p) or "/round3/x1/data/" in str(p) else (
            "calibration (old material)" if "/calib/" in str(p) else "run output")
        prov.append(dict(path=str(p.relative_to(D3)), mtime=datetime.fromtimestamp(mt).isoformat(timespec="seconds"), after_freeze=mt > t0, kind=kind))
    for r in prov:
        ok = r["after_freeze"] if r["kind"] == "run output" else True
        RESULTS.append(("Provenance", f"mtime {r['path']}", r["kind"], r["mtime"], "after 21:37:43" if r["after_freeze"] else "before 21:37:43", ok))

    # ---------- summary
    nok = sum(r[5] for r in RESULTS); nbad = len(RESULTS) - nok
    kind = lambda r: "provenance" if r[0] == "Provenance" else ("number" if NUM.fullmatch(str(r[3]).replace(",", "") or "x") or re.fullmatch(r"[\u2212+\-]?[\d,]*\d(\.\d+)?", str(r[3])) else "verdict or claim")
    from collections import Counter
    kc = Counter(kind(r) for r in RESULTS); kb = Counter(kind(r) for r in RESULTS if not r[5])
    print(f"checks: {len(RESULTS)}; passed {nok}; failed {nbad}; by kind: {dict(kc)}; failed by kind: {dict(kb)}")
    for sec in ("Result", "R1", "R2", "X", "E", "G", "Reading", "Material", "Provenance"):
        rs = [r for r in RESULTS if r[0] == sec]
        print(f"  {sec}: {len(rs)} checks, {sum(not r[5] for r in rs)} failed")
    print("verdicts recomputed: R1", sum(T3[f'H{i}'] for i in range(1, 10)), "/9", T3["verdict"], "| R2", n2, "/25", Qv,
          "| X", nx, "/9 | E", {k: Te[k] for k in eids}, Te["verdict"])
    print("gate original (e01 files) vs gate_summary_postfix_final.json, max abs diff:", gate_orig_agree)
    print("r2c03 per-point vs run-file means, max abs diff:", B3["g"]["perpoint_maxdiff"], "| X/E base identical:", base_same)
    print("material extras:", mat_extra)
    print("E6 random heads layer-matched:", Xe["E6.L11H9.heads.layer11"], Xe["E6.L17H6.heads.layer17"])
    for r in RESULTS:
        if not r[5]: print(f"MISMATCH [{r[0]}] {r[1]}: report {r[3]!r} vs mine {r[4]!r} :: {r[2][:160]}")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(results=[dict(zip(("section", "label", "text", "report", "mine", "ok"), r)) for r in RESULTS],
                                                provenance=prov, material=mat_extra, gate_orig_agree=gate_orig_agree), indent=1, ensure_ascii=False, default=str))
    return nbad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
