"""D3 round 2: measures and verdicts for PROTOCOL-2.md, from the saved records of r2c01, r2c02 and r2c03.
No model is loaded. Usage:
  python r2c04_verdicts.py conf        -> measures + verdicts on the sealed parts, outputs/r2/conf/verdicts.json
  python r2c04_verdicts.py calib       -> measures on the calibration (exploration) runs, no verdicts
  python r2c04_verdicts.py smoke       -> measures on the smoke tests (pipeline check only)
Bootstrap intervals (description only, not used for verdicts): 2,000 resamples of repositories (natural code) or
programs (generated code), seed 0."""
import sys, json
from pathlib import Path
import numpy as np

R2 = Path(__file__).resolve().parent.parent; O = R2 / "outputs" / "r2"
mode = sys.argv[1]
RUNS = [("qwen3-06b-bf16", "L23H2"), ("olmo1b", "L13H1")]
D1 = {"conf": O / "conf", "calib": O / "conf_calib_x1code", "smoke": O / "conf_smoke"}[mode]
D2 = {"conf": O / "conf", "calib": O / "conf_calib_gen2", "smoke": O / "conf_smoke"}[mode]
D3 = {"conf": O / "conf", "calib": O / "conf_calib_newcode", "smoke": O / "conf_smoke"}[mode]
rng0 = lambda: np.random.default_rng(0)
mean = lambda v: float(np.mean(v)) if len(v) else None


def boot(groups, f, n=2000):
    """groups: list of per-cluster lists of items; f(list of items) -> float. 95% percentile interval."""
    rng = rng0(); g = [x for x in groups if x]; vals = []
    for _ in range(n):
        s = [it for i in rng.integers(0, len(g), len(g)) for it in g[i]]
        v = f(s)
        if v is not None: vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def m_header(model, head):
    f = D1 / f"r2c01_{model}_{head}.jsonl"
    if not f.exists(): return None
    recs = [json.loads(l) for l in open(f)]
    P = [(c.get("repo") or str(c["chunk"]), p) for c in recs for p in c["points"]]
    others = sorted({k for _, p in P for k in p if k.startswith("ko_inner_h")}, key=lambda k: int(k[10:]))
    sel = lambda pred: [(r, p) for r, p in P if pred(p)]
    ded = sel(lambda p: p["cls"] == "dedent" and p["oh"]); same = sel(lambda p: p["cls"] == "same" and p["oh"])
    term = sel(lambda p: p["cls"] == "term" and p["oh"])
    d = lambda k, pts: mean([p[k][0] - p["base"][0] for _, p in pts])
    out = dict(n=dict(points=len(P), chunks=len(recs), repos=len({r for r, _ in P}), ded=len(ded), same=len(same), term=len(term)))
    # attention (H-A)
    out["att_inner_ded"] = mean([p["att"][-1] for _, p in ded]); out["att_inner_same"] = mean([p["att"][-1] for _, p in same])
    out["att_inner_term"] = mean([p["att"][-1] for _, p in term])
    # knock-out of the innermost header end (H-K)
    out["base_lq_ded"] = mean([p["base"][0] for _, p in ded])
    out["ko_inner_ded"] = d("ko_inner", ded); out["ko_inner_same"] = d("ko_inner", same); out["ko_inner_term"] = d("ko_inner", term)
    out["ko_inner_others_ded"] = {k[10:]: d(k, ded) for k in others}
    ov = list(out["ko_inner_others_ded"].values())
    out["ko_inner_others_min"] = min(ov) if ov else None; out["ko_inner_others_maxabs"] = max(abs(x) for x in ov) if ov else None
    clo = [(r, p) for r, p in ded if p["has_closed"]]
    out["n"]["ded_closed"] = len(clo); out["ko_closed_ded"] = d("ko_closed", clo); out["ko_inner_ded_closedsub"] = d("ko_inner", clo)
    tr = [(r, p) for r, p in ded if p["true"] in p["oh"]]
    out["n"]["ded_true_open"] = len(tr); out["ko_true_ded"] = d("ko_true", tr)
    # picking (H-P)
    mult = sel(lambda p: p["cls"] in ("dedent", "term") and len(p["oh"]) >= 2 and p["true"] in p["oh"])
    pick = lambda p: int(int(np.argmax(p["att"])) == p["oh"].index(p["true"]))
    nin = [(r, p) for r, p in mult if p["oh"][-1] != p["true"]]
    out["n"]["multi"] = len(mult); out["n"]["multi_not_inner"] = len(nin)
    out["pick_not_inner"] = mean([pick(p) for _, p in nin])
    out["pick_inner_attended_not_inner"] = mean([int(int(np.argmax(p["att"])) == len(p["oh"]) - 1) for _, p in nin])
    out["pick_all_multi"] = mean([pick(p) for _, p in mult]); out["innermost_rule_all_multi"] = mean([int(p["oh"][-1] == p["true"]) for _, p in mult])
    # steering (H-S)
    for c in ("dedent", "term"):
        S = {"head": [], "ctrl": []}
        for r, p in P:
            if p["cls"] != c or "steer" not in p: continue
            for who, j, ind, dlq, top in p["steer"]:
                if ind != p["top_base"]: S[who].append((r, dlq, int(top == ind)))
        for who in S:
            out[f"steer_{c}_{who}_dlq"] = mean([x[1] for x in S[who]]); out[f"steer_{c}_{who}_top"] = mean([x[2] for x in S[who]])
            out["n"][f"steer_{c}_{who}"] = len(S[who])
    # bootstrap intervals by repository (description only)
    byrepo = lambda pts: [[p for rr, p in pts if rr == r] for r in sorted({rr for rr, _ in pts})]
    out["ci"] = dict(ko_inner_ded=boot(byrepo(ded), lambda s: mean([p["ko_inner"][0] - p["base"][0] for p in s])),
                     att_inner_ded=boot(byrepo(ded), lambda s: mean([p["att"][-1] for p in s])),
                     pick_not_inner=boot(byrepo(nin), lambda s: mean([pick(p) for p in s])))
    return out


def m_gen2(model, head):
    f = D2 / f"r2c02_{model}_{head}.jsonl"
    if not f.exists(): return None
    R = [json.loads(l) for l in open(f)]; R = [r for r in R if not r.get("skip")]
    T = [r for r in R if r["last"] == "terminal"]; Pl = [r for r in R if r["last"] == "plain"]
    out = dict(n=dict(programs=len(R), terminal=len(T), plain=len(Pl)))
    pc = lambda rs, k: mean([r[k][0] for r in rs])
    def share(rs, k):
        rs = [r for r in rs if r.get("neutral") is not None]
        b, a, n = pc(rs, "base"), pc(rs, k), pc(rs, "neutral")
        return None if not rs or n is None or abs(n - b) < 1e-9 else (a - b) / (n - b)
    out["qcur_base_term"] = pc(T, "base"); out["qcur_neutral_term"] = mean([r["neutral"][0] for r in T if r.get("neutral")])
    L = len([k for k in R[0] if k.startswith("lay")]) if R else 0
    for kw in ["return", "raise", "break", "continue", "pass"]:
        rs = [r for r in T if r["kw"] == kw]; out["n"][kw] = len(rs)
        out[f"share_blk_kw_{kw}"] = share(rs, "blk_kw"); out[f"share_blk_kw_after_{kw}"] = share(rs, "blk_kw_after")
        out[f"share_layer_{kw}"] = [share(rs, f"lay{l}") for l in range(L)]
    sl = [x for x in out["share_layer_return"] if x is not None]; out["share_layer_return_max"] = max(sl) if sl else None
    out["share_blk_kw_all"] = share(T, "blk_kw")
    out["l0h2_line_dqcur_term"] = mean([r["l0h2_line"][0] - r["base"][0] for r in T])
    out["l0h2_all_dqcur_term"] = mean([r["l0h2_all"][0] - r["base"][0] for r in T])
    for nm, rs in (("term", T), ("plain", Pl)):
        S = {"head": [], "ctrl": []}
        for r in rs:
            for who, j, ind, dlq, top in r.get("steer", []):
                if ind != r["top_base"]: S[who].append((dlq, int(top == ind)))
        for who in S:
            out[f"steer_{nm}_{who}_dlq"] = mean([x[0] for x in S[who]]); out[f"steer_{nm}_{who}_top"] = mean([x[1] for x in S[who]])
            out["n"][f"steer_{nm}_{who}"] = len(S[who])
    rets = [r for r in T if r["kw"] == "return" and r.get("neutral") is not None]
    out["ci"] = dict(share_blk_kw_return=boot([[r] for r in rets], lambda s: share(s, "blk_kw")),
                     l0h2_line_dqcur_term=boot([[r] for r in T], lambda s: mean([r["l0h2_line"][0] - r["base"][0] for r in s])))
    return out


def m_group(model, g):
    f = D3 / f"r2c03_{model}_{g}.json"
    if not f.exists(): return None
    r = json.loads(f.read_text()); C = ["header", "same", "term", "dedent"]
    if "base" not in r or g not in r: return dict(incomplete=True)
    rnd = [r[k] for k in r if k.startswith("rand")]
    out = dict(n=r["n"], heads=r["heads"], n_random=len(rnd))
    for c in C:
        b = r["base"]["lq"][c]; out[f"base_lq_{c}"] = b; out[f"d_lq_{c}"] = r[g]["lq"][c] - b
        rd = [x["lq"][c] - b for x in rnd]; out[f"rand_d_lq_{c}_min"] = min(rd) if rd else None; out[f"rand_d_lq_{c}_max"] = max(rd) if rd else None
        out[f"d_psame_{c}"] = r[g]["p_same"][c] - r["base"]["p_same"][c]; out[f"p_open_{c}_ablated"] = r[g]["p_open"][c]
    cls = np.array(r["cls"]); cl = np.array(r["cluster"]); dd = np.array(r[g]["per_point_lq"]) - np.array(r["base"]["per_point_lq"])
    idx = np.where(cls == "dedent")[0]
    out["ci"] = dict(d_lq_dedent=boot([[float(dd[i]) for i in idx if cl[i] == c] for c in sorted(set(cl[idx]))], lambda s: mean(s)))
    return out


M = {}
for model, head in RUNS:
    M[f"{model}/{head}/natural"] = m_header(model, head); M[f"{model}/{head}/gen2"] = m_gen2(model, head)
M["olmo1b/DEDO/group"] = m_group("olmo1b", "DEDO")


def show(d, ind=""):
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("ko_inner_others_ded",): print(f"{ind}{k}:"); show(v, ind + "  ")
        elif isinstance(v, list) and len(v) > 4: print(f"{ind}{k}: [" + ", ".join("—" if x is None else f"{x:.3f}" for x in v) + "]")
        elif isinstance(v, float): print(f"{ind}{k}: {v:.4f}")
        elif k == "ko_inner_others_ded": print(f"{ind}{k}: " + " ".join(f"h{h}:{x:+.3f}" for h, x in v.items()))
        else: print(f"{ind}{k}: {v}")


for k, v in M.items():
    print(f"== {k}"); show(v) if v else print("  (no records)")
(O / f"measures_{mode}.json").write_text(json.dumps(M, indent=1))
if mode != "conf": sys.exit(0)
# ---------------- VERDICTS (PROTOCOL-2.md, "Predictions") ----------------
QN, QG = M["qwen3-06b-bf16/L23H2/natural"], M["qwen3-06b-bf16/L23H2/gen2"]
ON, OG, GR = M["olmo1b/L13H1/natural"], M["olmo1b/L13H1/gen2"], M["olmo1b/DEDO/group"]


def ratio_ok(head, ctrl, k): return head >= k * max(ctrl, 0.02)


def preds_head(N, G, att_min, att_ratio, steer_min, steer_k, top_gen_min, pick_rule):
    oth = list(N["ko_inner_others_ded"].values())
    P = {}
    P["attention"] = N["att_inner_ded"] >= att_min and N["att_inner_ded"] >= att_ratio * N["att_inner_same"]
    P["knockout_need"] = N["ko_inner_ded"] <= -0.05 and all(N["ko_inner_ded"] < x for x in oth) and abs(N["ko_inner_ded"]) >= 2 * N["ko_inner_others_maxabs"]
    P["specificity"] = N["ko_inner_same"] >= -0.02 and abs(N["ko_inner_term"]) <= 0.25 * abs(N["ko_inner_ded"])
    P["closed_control"] = abs(N["ko_closed_ded"]) <= 0.5 * abs(N["ko_inner_ded_closedsub"])
    P["picking"] = pick_rule(N)
    P["steer_natural"] = (N["steer_dedent_head_dlq"] >= steer_min and ratio_ok(N["steer_dedent_head_dlq"], N["steer_dedent_ctrl_dlq"], steer_k)
                          and N["steer_dedent_head_top"] >= N["steer_dedent_ctrl_top"] + 0.05)
    P["ko_true"] = N["ko_true_ded"] <= (-0.10 if att_min >= 0.3 else -0.08)
    P["steer_generated"] = (G["steer_term_head_dlq"] >= steer_min and ratio_ok(G["steer_term_head_dlq"], G["steer_term_ctrl_dlq"], steer_k)
                            and G["steer_term_head_top"] >= top_gen_min and G["steer_term_head_top"] >= G["steer_term_ctrl_top"] + 0.10)
    P["keyword_read"] = G["share_blk_kw_return"] is not None and G["share_blk_kw_return"] >= 0.40
    P["spread_layers"] = G["share_layer_return_max"] is not None and G["share_blk_kw_return"] is not None and G["share_layer_return_max"] <= 0.35 * G["share_blk_kw_return"]
    return P


PQ = preds_head(QN, QG, 0.30, 2.5, 1.0, 5, 0.20,
                lambda N: N["pick_not_inner"] >= 0.60 and N["pick_all_multi"] >= N["innermost_rule_all_multi"] + 0.02)
PQ["l0h2_line"] = abs(QG["l0h2_line_dqcur_term"]) <= 0.05
PO = preds_head(ON, OG, 0.20, 1.4, 0.8, 4, 0.15,
                lambda N: N["pick_not_inner"] >= 0.45 and N["pick_not_inner"] >= N["pick_inner_attended_not_inner"] + 0.15)
PG = dict(group_need=GR["d_lq_dedent"] <= -0.30 and GR["d_lq_dedent"] < GR["rand_d_lq_dedent_min"],
          group_spares_terminal=abs(GR["d_lq_term"]) <= 0.25 * abs(GR["d_lq_dedent"]) and GR["p_open_term_ablated"] >= 0.90,
          group_same=GR["d_lq_same"] >= -0.02, group_current=GR["d_psame_dedent"] >= 0.08)
NAMES = {"attention": "1", "knockout_need": "2", "specificity": "3", "closed_control": "4", "picking": "5", "steer_natural": "6",
         "ko_true": "7", "steer_generated": "8", "keyword_read": "9", "spread_layers": "10", "l0h2_line": "11"}
V = {f"Q{NAMES[k]}": bool(v) for k, v in PQ.items()}
V.update({f"O{NAMES[k]}": bool(v) for k, v in PO.items()})
V.update({f"G{i + 1}": bool(v) for i, v in enumerate(PG.values())})
Q = {"question 1 (what the block-end head reads)": ["Q1", "Q2", "Q3"],
     "question 2 (how far to dedent)": ["Q5", "Q6", "Q8"],
     "question 3 (terminal route: keyword read directly, spread, not via L0H2 on the line)": ["Q9", "Q10", "Q11"],
     "question 4a (two routes in OLMo-2)": ["G1", "G2", "O9"],
     "question 4b (L13H1 works like L23H2)": ["O1", "O2", "O3", "O6", "O8"]}
QV = {q: "confirmed" if all(V[x] for x in xs) else "not confirmed" for q, xs in Q.items()}
print("\nVERDICTS"); [print(f"  {k}: {'pass' if v else 'FAIL'}") for k, v in V.items()]
[print(f"  {q}: {v}") for q, v in QV.items()]
(O / "conf" / "verdicts.json").write_text(json.dumps(dict(predictions=V, questions=QV, question_parts=Q), indent=1))
