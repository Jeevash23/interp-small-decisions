"""D5 round 2 verdicts (PROTOCOL-2.md). Reads only saved per-position records; loads no model.
Usage: r2_verdict.py [conf|explore]
  conf    (default): the held-out runs on the reserve part, outputs/r2/*_reserve_*; writes outputs/r2/conf_verdicts2.json
  explore: the exploration runs named in EXPLORE below; writes outputs/r2/explore_verdicts2_check.json
           (used before the freeze to test this script).
Every measure is recomputed here from the records. Decisions use point estimates; 95% intervals come from
2,000 bootstrap resamples of articles (pageid), seed 'd5-r2-boot-<key>'."""
import json, random, sys
from pathlib import Path
import numpy as np
import torch

R2 = Path(__file__).resolve().parent.parent / "outputs" / "r2"
MODE = sys.argv[1] if len(sys.argv) > 1 else "conf"
B = 2000
MODELS = ["qwen3-17b", "gemma"]
if MODE == "conf":
    FILES = {m: {"sets": [f"exp03_{m}_reserve_0_conf.pt"], "attn": [f"exp06_{m}_reserve_0.json"],
                 "edit": [f"exp05_{m}_reserve_0.json"], "other": [f"exp04_{m}_reserve_0.json", f"exp04_{m}_reserve_264.json"]} for m in MODELS}
    for m in MODELS: FILES[m]["single"] = [f"exp03_{m}_reserve_0_l17h6.pt"] if m == "qwen3-17b" else []
else:
    FILES = {"qwen3-17b": {"sets": ["exp03_qwen3-17b_conf_440_a.pt"], "attn": ["exp06_qwen3-17b_conf_0.json"],
                           "edit": ["exp05_qwen3-17b_conf_100_pieces.json"], "other": ["exp04_qwen3-17b_explore_0.json", "exp04_qwen3-17b_conf_0.json"],
                           "single": ["exp03_qwen3-17b_conf_440_l17h6.pt"]},
             "gemma": {"sets": ["exp03_gemma_conf_440_a.pt"], "attn": ["exp06_gemma_conf_0.json"],
                       "edit": ["exp05_gemma_conf_100.json"], "other": ["exp04_gemma_conf_0.json"], "single": []}}


def case_of(s):
    up = any(c.isupper() for c in s); lo = any(c.islower() for c in s)
    return "caseless" if not (up or lo) else "upper" if up and not lo else "lower" if lo and not up else "mixed"


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


def cls(s):
    s = s.strip(); let = [c for c in s if c.isalpha()]
    if not let: return "none"
    if len(let) >= 2 and all(c.isupper() for c in let): return "ALL"
    if let[0].isupper() and any(c.islower() for c in let): return "Init"
    if len(let) == 1 and let[0].isupper(): return "One"
    if all(c.islower() for c in let): return "low"
    return "mix"


def boot(items, stat, key):
    """items: list of (article, x); stat: function of a list of x. Point, 95% interval, n, articles."""
    arts = sorted({a for a, _ in items}, key=str); by = {a: [] for a in arts}
    for a, x in items: by[a].append(x)
    point = stat([x for _, x in items]); rng = random.Random(f"d5-r2-boot-{key}"); vals = []
    for _ in range(B):
        xs = [x for a in (rng.choice(arts) for _ in arts) for x in by[a]]
        v = stat(xs)
        if v == v: vals.append(v)
    return {"point": point, "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "n": len(items), "n_art": len(arts)}


def share(xs): return float(np.mean(xs)) if xs else float("nan")
def ratio(xs):  # xs: (numerator, denominator) pairs; ratio of means
    if not xs: return float("nan")
    a = np.array(xs, float); return float(a[:, 0].mean() / a[:, 1].mean())


def loadall(kind, m):
    out = []
    for f in FILES[m][kind]:
        p = R2 / f
        if not p.exists(): continue
        out.append(torch.load(p, weights_only=False) if f.endswith(".pt") else json.load(open(p)))
    return out


LOCAL = {"qwen3-17b": "L17H6", "gemma": "L14H7"}; BROAD = {"qwen3-17b": "L19H0", "gemma": "L24H6"}
res = {"mode": MODE, "models": {}}
for m in MODELS:
    M, P = {}, {}

    # ---- sets: writers (W), carriers (CA in A runs, CU in U runs), readers (R)
    for x in loadall("sets", m):
        recs = x["records"]; nrand = len(x["random"]["writers"])
        def rows(key, base_key, want_base):
            return [rc for rc in recs if case_of(rc[base_key]["top"]) == want_base]
        # W: upper share among U positions with an upper-case top-1, per row
        el = rows("W", "U", "upper")
        W_up = [boot([(rc["pageid"], float(case_of(rc["W"][j][0]) == "upper")) for rc in el], share, f"{m}-W-up-{j}") if j < 2
                else share([float(case_of(rc["W"][j][0]) == "upper") for rc in el]) for j in range(2 + nrand)]
        M["W_none_upper"] = W_up[0]; M["W_named_upper"] = W_up[1]; M["W_random_upper"] = W_up[2:]
        M["W_named_same_word"] = boot([(rc["pageid"], float(same_word(rc["U"]["top"], rc["W"][1][0]))) for rc in el], share, f"{m}-W-sw")
        M["W_named_proj_removed"] = boot([(rc["pageid"], (rc["W"][0][2] - rc["W"][1][2], rc["W"][0][2] - rc["L"]["proj"])) for rc in recs], ratio, f"{m}-W-proj")
        P["W1"] = (M["W_named_upper"]["point"] <= 0.30 and min(M["W_random_upper"]) >= 0.85 and M["W_named_same_word"]["point"] >= 0.55)
        # CA: projection excess removed in A runs (relative to L), per row; CU: upper share in U runs
        def fr(key, j, comp):  # comp 2 = projection, 1 = D
            return [(rc["pageid"], (rc[key][0][comp] - rc[key][j][comp], rc[key][0][comp] - rc["L"]["proj" if comp == 2 else "D"])) for rc in recs]
        M["CA_named_proj_removed"] = boot(fr("CA", 1, 2), ratio, f"{m}-CA-proj")
        M["CA_named_D_removed"] = boot(fr("CA", 1, 1), ratio, f"{m}-CA-D")
        M["CA_random_proj_removed"] = [ratio([v for _, v in fr("CA", j, 2)]) for j in range(2, 2 + nrand)]
        M["CA_none_upper"] = share([float(case_of(rc["CA"][0][0]) == "upper") for rc in recs])
        M["CA_named_upper"] = boot([(rc["pageid"], float(case_of(rc["CA"][1][0]) == "upper")) for rc in recs], share, f"{m}-CA-up")
        el = rows("CU", "U", "upper")
        M["CU_named_upper"] = boot([(rc["pageid"], float(case_of(rc["CU"][1][0]) == "upper")) for rc in el], share, f"{m}-CU-up")
        M["CU_random_upper"] = [share([float(case_of(rc["CU"][j][0]) == "upper") for rc in el]) for j in range(2, 2 + nrand)]
        P["C1"] = (M["CA_named_proj_removed"]["point"] >= 0.60 and max(M["CA_random_proj_removed"]) <= M["CA_named_proj_removed"]["point"] - 0.25
                   and M["CA_named_D_removed"]["point"] >= 0.30)
        P["C2"] = M["CU_named_upper"]["point"] <= 0.60 and min(M["CU_random_upper"]) >= 0.90
        # R: flip rate among L positions with a lower-case top-1
        el = rows("R", "L", "lower")
        M["R_none_flip"] = boot([(rc["pageid"], float(case_of(rc["R"][0][0]) == "upper")) for rc in el], share, f"{m}-R-none")
        M["R_named_flip"] = boot([(rc["pageid"], float(case_of(rc["R"][1][0]) == "upper")) for rc in el], share, f"{m}-R-named")
        M["R_random_flip"] = [share([float(case_of(rc["R"][j][0]) == "upper") for rc in el]) for j in range(2, 2 + nrand)]
        P["R1"] = (M["R_none_flip"]["point"] >= 0.80 and M["R_named_flip"]["point"] <= 0.20
                   and min(M["R_random_flip"]) >= M["R_none_flip"]["point"] - 0.15)

    # ---- single head (Qwen3 L17H6), context-only runs
    for x in loadall("single", m):
        recs = x["records"]; nrand = len(x["random"]["carriers"])
        f2 = lambda j: [(rc["pageid"], (rc["CA"][0][2] - rc["CA"][j][2], rc["CA"][0][2] - rc["L"]["proj"])) for rc in recs]
        M["L17H6_proj_removed"] = boot(f2(1), ratio, f"{m}-L17H6")
        M["L17H6_random_proj_removed"] = [ratio([v for _, v in f2(j)]) for j in range(2, 2 + nrand)]
        P["C3"] = M["L17H6_proj_removed"]["point"] >= 0.20 and max(M["L17H6_random_proj_removed"]) < M["L17H6_proj_removed"]["point"]

    # ---- attention by distance (descriptive except C4)
    for x in loadall("attn", m):
        recs = x["records"]; heads = [k for k in recs[0] if k.startswith("L")]
        M["attn_bins"] = x["bins"]
        for h in heads:
            M[f"attn_{h}"] = [float(np.mean([rc[h][b] for rc in recs])) for b in range(len(x["bins"]))]
            # share of the attention on text tokens other than the first that falls on the current word and the
            # seven words before it (bins 0, 1, 2-3, 4-7 over bins 0 .. 8+; the first text token and BOS are left out)
            M[f"attn_{h}_recent8"] = boot([(rc["pageid"], (sum(rc[h][:4]), sum(rc[h][:5]))) for rc in recs], ratio, f"{m}-attn-{h}")

        if f"attn_{LOCAL[m]}_recent8" in M and f"attn_{BROAD[m]}_recent8" in M:
            P["C4"] = M[f"attn_{LOCAL[m]}_recent8"]["point"] >= 0.60 and M[f"attn_{BROAD[m]}_recent8"]["point"] <= 0.50

    # ---- Q3: edits
    for x in loadall("edit", m):
        recs = x["records"]
        el = [rc for rc in recs if case_of(rc["lo"]) == "lower" and case_of(rc["edit"]) == "upper"]
        upch = [rc for rc in el if not same_word(rc["lo"], rc["up"])]; upkp = [rc for rc in el if same_word(rc["lo"], rc["up"])]
        M["E_flips"] = len(el)
        M["E_change"] = boot([(rc["pageid"], float(not same_word(rc["lo"], rc["edit"]))) for rc in el], share, f"{m}-E-ch")
        M["E_change_if_up_changes"] = boot([(rc["pageid"], float(not same_word(rc["lo"], rc["edit"]))) for rc in upch], share, f"{m}-E-uc")
        M["E_change_if_up_keeps"] = boot([(rc["pageid"], float(not same_word(rc["lo"], rc["edit"]))) for rc in upkp], share, f"{m}-E-uk")
        M["E_change_not_by_up"] = boot([(rc["pageid"], float(not same_word(rc["lo"], rc["edit"]))) for rc in el if rc in upkp] +
                                       [(rc["pageid"], 0.0) for rc in upch], share, f"{m}-E-nu")
        ch = [rc for rc in el if not same_word(rc["lo"], rc["edit"])]
        M["E_changed_in_lower_top5"] = boot([(rc["pageid"], float(rc["edit_lower_rank"] is not None and rc["edit_lower_rank"] <= 5)) for rc in ch], share, f"{m}-E-t5")
        M["E_random_flip"] = share([float(case_of(rc["rand"]) == "upper") for rc in el])
        P["E1"] = (M["E_change_if_up_changes"]["point"] - M["E_change_if_up_keeps"]["point"] >= 0.30
                   and M["E_changed_in_lower_top5"]["point"] >= 0.40)
    # extra pieces need the tokeniser: counts are stored by the run (pieces fields), see below
        if el and "pieces_lo" in el[0]:
            ex = lambda rc: rc["pieces_up"] - rc["pieces_lo"]
            M["E_change_extra0"] = boot([(rc["pageid"], float(not same_word(rc["lo"], rc["edit"]))) for rc in el if ex(rc) == 0], share, f"{m}-E-x0")
            M["E_change_extra1p"] = boot([(rc["pageid"], float(not same_word(rc["lo"], rc["edit"]))) for rc in el if ex(rc) >= 1], share, f"{m}-E-x1")
            P["E2"] = M["E_change_extra1p"]["point"] - M["E_change_extra0"]["point"] >= 0.10

    # ---- Q4: acronyms, initial capitals, headings
    paras, heads = [], []
    for x in loadall("other", m):
        paras += x["para"]; heads += x["head"]
        M["proj_upper_mean"], M["proj_lower_mean"] = x["proj_upper_mean"], x["proj_lower_mean"]
    if paras or heads:
        gap = M["proj_upper_mean"] - M["proj_lower_mean"]
        def keep(recs, kind, base, j, want):
            g = [rc for rc in recs if rc["kind"] == kind and cls(rc["tops"][0]) == base]
            return [(rc["pageid"], float(cls(rc["tops"][j]) == want)) for rc in g]
        M["A_acr_rm_keeps"] = boot(keep(paras, "acr", "ALL", 1, "ALL"), share, f"{m}-A-acr")
        M["A_acr_rm_rand_keeps"] = boot(keep(paras, "acr", "ALL", 2, "ALL"), share, f"{m}-A-acr-r")
        M["A_init_rm_keeps"] = boot(keep(paras, "init", "Init", 1, "Init"), share, f"{m}-A-init")
        M["A_init_rm_rand_keeps"] = boot(keep(paras, "init", "Init", 2, "Init"), share, f"{m}-A-init-r")
        hin = [rc for rc in heads if rc["caps"]]
        M["A_head_rm_keeps"] = boot(keep(hin, "hin", "ALL", 1, "ALL"), share, f"{m}-A-head")
        M["A_head_rm_rand_keeps"] = boot(keep(hin, "hin", "ALL", 2, "ALL"), share, f"{m}-A-head-r")
        # projections inside the heading and at the first word after it, capitalised heading minus original heading,
        # as a share of the gap between the round-1 upper and lower means (paired by item and word)
        def pair(kind):
            a = {(rc["item"], rc["word"].lower()): rc for rc in heads if rc["kind"] == kind and rc["caps"]}
            b = {(rc["item"], rc["word"].lower()): rc for rc in heads if rc["kind"] == kind and not rc["caps"]}
            return [(a[k]["pageid"], (a[k]["proj"] - b[k]["proj"]) / gap) for k in a if k in b]
        M["A_head_inside_excess"] = boot(pair("hin"), share, f"{m}-A-hin")
        M["A_head_after_excess"] = boot(pair("hbody"), share, f"{m}-A-hbody")
        hb = [rc for rc in heads if rc["kind"] == "hbody" and rc["caps"]]
        M["A_head_after_allcaps_top"] = boot([(rc["pageid"], float(cls(rc["tops"][0]) == "ALL")) for rc in hb], share, f"{m}-A-hb-all")
        P["A1"] = M["A_acr_rm_keeps"]["point"] >= 0.85 and M["A_acr_rm_rand_keeps"]["point"] >= 0.90
        P["A2"] = M["A_init_rm_keeps"]["point"] >= 0.90 and M["A_init_rm_rand_keeps"]["point"] >= 0.95
        P["A3"] = M["A_head_rm_keeps"]["point"] <= 0.85 and M["A_head_rm_rand_keeps"]["point"] >= 0.95
        P["A4"] = (M["A_head_inside_excess"]["point"] >= 0.40 and M["A_head_after_excess"]["point"] <= 0.25
                   and M["A_head_after_allcaps_top"]["point"] <= 0.10)
    V = {}
    if "W1" in P and "R1" in P: V["Q1_parts"] = "supported" if P["W1"] and P["R1"] else ("partly supported" if P["W1"] or P["R1"] else "not supported")
    if "C1" in P and "C2" in P: V["Q2_carriers"] = "supported" if P["C1"] and P["C2"] else ("partly supported" if P["C1"] or P["C2"] else "not supported")
    if "E1" in P and "E2" in P: V["Q3_word_changes"] = "supported" if P["E1"] and P["E2"] else ("partly supported" if P["E1"] or P["E2"] else "not supported")
    if "A1" in P and "A2" in P: V["Q4_acronyms_initials_separate"] = "supported" if P["A1"] and P["A2"] else "not supported"
    if "A3" in P and "A4" in P: V["Q4_headings_same_direction_reset"] = "supported" if P["A3"] and P["A4"] else "not supported"
    V["n_pass"] = sum(bool(v) for v in P.values()); V["n_tests"] = len(P)
    res["models"][m] = {"measures": M, "predictions": P, "verdicts": V}

out = R2 / ("conf_verdicts2.json" if MODE == "conf" else "explore_verdicts2_check.json")
json.dump(res, open(out, "w"), indent=1)
res["parts_account_both_models"] = all(res["models"][m]["verdicts"].get(q) == "supported" for m in MODELS for q in ("Q1_parts", "Q2_carriers"))
json.dump(res, open(out, "w"), indent=1)
print("parts account supported in both models:", res["parts_account_both_models"])
for m in MODELS:
    print(m, res["models"][m]["predictions"], res["models"][m]["verdicts"])
    for k, v in res["models"][m]["measures"].items():
        if isinstance(v, dict): print("  ", k, round(v["point"], 3), [round(v["lo"], 3), round(v["hi"], 3)], v["n"])
        elif isinstance(v, list) and v and isinstance(v[0], float): print("  ", k, "min %.3f max %.3f" % (min(v), max(v)) if len(v) > 8 else [round(z, 3) for z in v])
        else: print("  ", k, v)
