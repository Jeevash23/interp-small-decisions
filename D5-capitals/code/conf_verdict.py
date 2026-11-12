"""D5 confirmation verdicts (PROTOCOL.md, Sections 4-5). Reads only saved outputs; loads no model.
Usage: conf_verdict.py [prefix=conf_]
With the default prefix it reads the confirmation outputs (conf_*.json) and writes
outputs/conf_verdicts.json. With prefix "" it reads the exploration outputs instead and writes
outputs/explore_verdicts_check.json (used before the freeze to test this script; layer choices and
thresholds are the same).
Every measure is recomputed here from the per-position records, not taken from the scripts' own
summaries. Decisions use point estimates; 95% intervals come from 2,000 bootstrap resamples of
paragraphs (seed 'd5-conf-boot')."""
import json, random, sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs"
PRE = sys.argv[1] if len(sys.argv) > 1 else "conf_"
MODELS = {"gemma": {"layer": 24, "early": 16, "L": 35, "late_from": 30},
          "qwen3-17b": {"layer": 22, "early": 16, "L": 28, "late_from": 23}}
B = 2000


def case_of(s):
    letters = [c for c in s if c.isalpha() and (c.isupper() or c.islower())]
    if not letters: return "caseless"
    up = any(c.isupper() for c in letters); lo = any(c.islower() for c in letters)
    return "upper" if up and not lo else "lower" if lo and not up else "mixed"


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


def boot(items, stat, key):
    """items: list of (para, x). stat: function of a list of x. Returns point, lo, hi."""
    paras = sorted({p for p, _ in items}); by = {p: [] for p in paras}
    for p, x in items: by[p].append(x)
    point = stat([x for _, x in items])
    rng = random.Random(f"d5-conf-boot-{key}"); vals = []
    if stat is share:  # fast path: per-paragraph sums and counts
        sm = np.array([sum(by[p]) for p in paras], float); ct = np.array([len(by[p]) for p in paras], float)
        for _ in range(B):
            ix = [rng.randrange(len(paras)) for _ in paras]; vals.append(float(sm[ix].sum() / ct[ix].sum()))
    else:
        for _ in range(B):
            xs = [x for p in (rng.choice(paras) for _ in paras) for x in by[p]]
            vals.append(stat(xs))
    vals = np.array([v for v in vals if v == v])
    return {"point": point, "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "n": len(items), "n_para": len(paras)}


def share(xs): return float(np.mean(xs)) if xs else float("nan")


def load(name):
    p = OUT / f"{PRE}{name}"
    return json.load(open(p)) if p.exists() else None


res = {"prefix": PRE, "models": {}}
for mdl, cfg in MODELS.items():
    l, le, L = cfg["layer"], cfg["early"], cfg["L"]; r = {"measures": {}, "predictions": {}}; M = r["measures"]; P = r["predictions"]

    # gate replication
    g = load(f"gate_{mdl}.json")
    if g:
        M["gate"] = {k: g["gate"][k] for k in ("c1_case_kept", "c2_same_word", "c3_inside_word", "pass")}
        P["G1"] = bool(g["gate"]["pass"])

    # L1, L2: logit lens
    e1 = load(f"exp01_{mdl}.json")
    if e1:
        recs = e1["records"]
        def lens_share(run, layer, a, b):
            # comparisons with U leave out positions where U and Lc have equal lens values at that layer
            # (as in exp01's own summary; about 2% of Gemma's positions, mostly short function words)
            keep = (lambda rc: rc[run]["lens"][layer][0] != rc[run]["lens"][layer][1]) if 0 in (a, b) else (lambda rc: True)
            return [(rc["para"], float(rc[run]["lens"][layer][a] > rc[run]["lens"][layer][b])) for rc in recs if keep(rc)]
        M["lens_Lc_gt_U_at_l"] = boot(lens_share("upper", l, 1, 0), share, f"{mdl}-L1a")
        M["lens_Lr_gt_U_at_l"] = boot(lens_share("upper", l, 2, 0), share, f"{mdl}-L1b")
        M["lens_Lc_gt_U_last"] = boot(lens_share("upper", L - 1, 1, 0), share, f"{mdl}-L1c")
        prof = [share([x for _, x in lens_share("upper", k, 1, 0)]) for k in range(L)]
        M["lens_Lc_gt_U_by_layer"] = prof
        first_below = next((k for k in range(L) if prof[k] < 0.5 and all(p < 0.5 for p in prof[k:])), None)
        M["first_layer_Lc_below_half_for_good"] = first_below
        M["lens_Lc_gt_Lr_upper_at_l"] = boot(lens_share("upper", l, 1, 2), share, f"{mdl}-L2a")
        M["lens_Lc_gt_Lr_lower_at_l"] = boot(lens_share("lower", l, 1, 2), share, f"{mdl}-L2b")
        P["L1"] = (M["lens_Lc_gt_U_at_l"]["point"] >= 0.60 and M["lens_Lr_gt_U_at_l"]["point"] <= 0.30
                   and M["lens_Lc_gt_U_last"]["point"] <= 0.05 and first_below is not None and first_below >= cfg["late_from"])
        a, b = M["lens_Lc_gt_Lr_upper_at_l"]["point"], M["lens_Lc_gt_Lr_lower_at_l"]["point"]
        P["L2"] = abs(a - b) <= 0.05 and a >= 0.75 and b >= 0.75

    # D1, D4: adding / subtracting the direction (exp02)
    e2 = load(f"exp02_{mdl}.json")
    if e2:
        def dir_stats(layer, v, j):
            want = "upper" if v == "lower" else "lower"
            el = [rc for rc in e2["records"] if case_of(rc[v][str(layer)][0]) == v]
            fl = [(rc["para"], float(case_of(rc[v][str(layer)][j]) == want)) for rc in el]
            kp = [(rc["para"], float(case_of(rc[v][str(layer)][j]) == want and same_word(rc[v][str(layer)][0], rc[v][str(layer)][j]))) for rc in el]
            return fl, kp
        for layer in (l, le):
            for v, sgn in (("lower", "add"), ("upper", "sub")):
                for j, kind in ((1, "dir"), (2, "rand")):
                    fl, kp = dir_stats(layer, v, j)
                    M[f"{sgn}_{kind}_L{layer}_flip"] = boot(fl, share, f"{mdl}-D-{sgn}-{kind}-{layer}-f")
                    M[f"{sgn}_{kind}_L{layer}_same_word"] = boot(kp, share, f"{mdl}-D-{sgn}-{kind}-{layer}-k")
        P["D1"] = all(M[f"{s}_dir_L{l}_flip"]["point"] >= 0.80 and M[f"{s}_dir_L{l}_same_word"]["point"] >= 0.45
                      and M[f"{s}_rand_L{l}_flip"]["point"] <= 0.05 for s in ("add", "sub"))
        P["D4"] = M[f"add_dir_L{le}_flip"]["point"] <= 0.15 and M[f"sub_dir_L{le}_flip"]["point"] >= 0.40

    # D2, D3: removing the direction; adding it to ordinary text (exp07)
    e7 = load(f"exp07_{mdl}.json")
    if e7:
        el = [rc for rc in e7["records"] if case_of(rc["upper"]["top0"]) == "upper"]
        for kind in ("dir", "rand"):
            M[f"rm_{kind}_upper_share"] = boot([(rc["para"], float(case_of(rc["upper"][f"rm_{kind}_{l}"]) == "upper")) for rc in el], share, f"{mdl}-D2-{kind}-u")
            M[f"rm_{kind}_same_word_any_case"] = boot([(rc["para"], float(same_word(rc["upper"]["top0"], rc["upper"][f"rm_{kind}_{l}"]))) for rc in el], share, f"{mdl}-D2-{kind}-w")
        P["D2"] = (M["rm_dir_upper_share"]["point"] <= 0.70 and M["rm_dir_same_word_any_case"]["point"] >= 0.60
                   and M["rm_rand_upper_share"]["point"] >= 0.95)
        elo = [rc for rc in e7["records"] if case_of(rc["orig"]["top0"]) == "lower"]
        for kind in ("dir", "rand"):
            x = [rc["orig"][f"add_{kind}_a1_{l}"] for rc in elo]
            M[f"orig_add_{kind}_to_upper"] = boot([(rc["para"], float(case_of(s) == "upper")) for rc, s in zip(elo, x)], share, f"{mdl}-D3-{kind}-u")
            M[f"orig_add_{kind}_to_upper_same_word"] = boot([(rc["para"], float(case_of(s) == "upper" and same_word(rc["orig"]["top0"], s))) for rc, s in zip(elo, x)], share, f"{mdl}-D3-{kind}-w")
        P["D3"] = (M["orig_add_dir_to_upper"]["point"] >= 0.75 and M["orig_add_dir_to_upper_same_word"]["point"] >= 0.50
                   and M["orig_add_rand_to_upper"]["point"] <= 0.05)

    # R1: recency (exp04)
    e4 = load(f"exp04_{mdl}.json")
    if e4:
        for f in ("L+u1", "L+u2", "L+u8", "U+l1", "U+l2", "U+l8"):
            M[f"upper_top1_{f}"] = boot([(rc["para"], float(rc[f]["case"] == "upper")) for rc in e4["records"] if rc[f]["case"] in ("upper", "lower", "mixed")], share, f"{mdl}-R1-{f}")
        P["R1"] = (M["upper_top1_L+u2"]["point"] >= 0.50 and M["upper_top1_L+u8"]["point"] >= 0.85
                   and M["upper_top1_U+l2"]["point"] <= 0.25 and M["upper_top1_U+l8"]["point"] <= 0.10)

    # E1: where changed words come from (exp08)
    e8 = load(f"exp08_{mdl}.json")
    if e8:
        x = [o for o in e8["records"] if o["kind"] == "upper"]
        xd = [o for o in x if not o["same_first"]]
        M["changed_word_top5"] = boot([(o["para"], float(o["rank_word"] <= 5)) for o in x], share, f"{mdl}-E1-w")
        M["changed_word_top5_diff_first"] = boot([(o["para"], float(o["rank_word"] <= 5)) for o in xd], share, f"{mdl}-E1-wd")
        M["ctrl_word_top5"] = boot([(o["para"], float(o["rank_ctrl"] <= 5)) for o in x if o["rank_ctrl"] is not None], share, f"{mdl}-E1-c")
        P["E1"] = (M["changed_word_top5_diff_first"]["point"] >= 0.50 and M["ctrl_word_top5"]["point"] <= 0.10)

    # H1: Qwen3 L0H3 (exp06)
    if mdl == "qwen3-17b":
        e6 = load(f"exp06_{mdl}_L0H3.json")
        if e6:
            items, recs6 = e6["items"], e6["records"]
            for v in ("upper", "lower"):
                M[f"L0H3_self_attention_{v}"] = boot([(it["para"], it[v]["att_off"]["0"]) for it in items], share, f"{mdl}-H1-att-{v}")
            idx = [n for n, it in enumerate(items) if case_of(it["upper"]["top0"]) == "upper"]
            def moved(name):
                def st(ns):
                    D0 = np.mean([items[n]["upper"]["D0"] for n in ns]); Dw = np.mean([items[n]["lower"]["D0"] for n in ns])
                    Dp = np.mean([recs6[n]["patched"]["upper"][name][0] for n in ns]); return float((D0 - Dp) / (D0 - Dw))
                return st
            for name in ("same_lower", "other_lower", "other_upper", "ctrl_L0H2_same_lower", "ctrl_L0H0_same_lower"):
                M[f"L0H3_{name}_moved"] = boot([(items[n]["para"], n) for n in idx], moved(name), f"{mdl}-H1-{name}")
                M[f"L0H3_{name}_flip"] = boot([(items[n]["para"], float(case_of(recs6[n]["patched"]["upper"][name][1]) == "lower")) for n in idx], share, f"{mdl}-H1-{name}-f")
            P["H1"] = (M["L0H3_self_attention_upper"]["point"] >= 0.90 and M["L0H3_self_attention_lower"]["point"] >= 0.90
                       and M["L0H3_other_lower_moved"]["point"] >= 0.20 and M["L0H3_other_lower_flip"]["point"] >= 0.25
                       and M["L0H3_other_upper_moved"]["point"] <= 0.08 and M["L0H3_other_upper_flip"]["point"] <= 0.05
                       and M["L0H3_ctrl_L0H2_same_lower_moved"]["point"] <= 0.08 and M["L0H3_ctrl_L0H0_same_lower_moved"]["point"] <= 0.08)

    core = ["L1", "L2", "D1", "D2", "D3"]
    if all(k in P for k in core):
        P["account"] = ("supported" if all(P[k] for k in core) else
                        "partly supported" if P["D1"] and P["D3"] else "not supported")
    res["models"][mdl] = r

name = "conf_verdicts.json" if PRE == "conf_" else "explore_verdicts_check.json"
json.dump(res, open(OUT / name, "w"), indent=1)
for mdl, r in res["models"].items():
    print(mdl, json.dumps(r["predictions"]))
