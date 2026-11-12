"""Write numbers.tex, the figure data files and checks/numbers.json for the D5 write-up, from saved outputs only.

  python3 -B scripts/make_numbers.py

Sources (all under ../../outputs/ or ../../data/, read only):
  conf_verdicts.json           held-out measures and verdicts (written by the frozen conf_verdict.py)
  explore_verdicts_check.json  the same measures on the exploration outputs (exploratory)
  gate_<model>.json            exploration gate, four models
  conf_exp01_<model>.json      per-layer lens summaries on held-out text (Figure 2, descriptive)
  exp02_<model>.json           exploration layer sweep of the capitals direction (Figure 1)
  conf_exp08_<model>.json      median ranks of changed and control words
  part-explore.jsonl, part-conf.jsonl, part-reserve.jsonl, headings-reserve.jsonl   counts only
  r2/conf_verdicts2.json        round-2 held-out measures, predictions and verdicts (frozen r2_verdict.py)
  r2/explore_verdicts2_check.json  the same measures on the round-2 exploration outputs (exploratory)
  gate_olmo1b.json             round-2 gate of OLMo-2-0425-1B (exploratory, Q5)
  ../X1-shared-parts/outputs/conf/verdicts.json   the cross-study test X1 (held-out values quoted)
Every number in main.tex is a macro \\nv{key} defined here; scripts/check_numbers.py recomputes them independently.
"""
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
D5 = PAPER.parents[1]
OUT = D5 / "outputs"
NV = {}
MODELS = {"g": ("gemma", 24, 35), "q": ("qwen3-17b", 22, 28)}


def f3(x):
    v = round(float(x), 3)
    return "0.000" if v == 0 else f"{v:.3f}".replace("-", "−")


def f2(x):
    v = round(float(x), 2)
    return "0.00" if v == 0 else f"{v:.2f}"


def put(key, val):
    assert key not in NV, key
    NV[key] = val


def load(name):
    with open(OUT / name) as f:
        return json.load(f)


# ---------------------------------------------------------------- material
for part in ("explore", "conf", "reserve"):
    rows = [json.loads(s) for s in open(D5 / "data" / f"part-{part}.jsonl")]
    put(f"mat.{part}.paras", f"{len(rows):,}")
    put(f"mat.{part}.articles", f"{len({r['pageid'] for r in rows}):,}")

# ---------------------------------------------------------------- exploration gate (four models)
for k, m in (("g", "gemma"), ("q", "qwen3-17b"), ("s", "smollm2"), ("p", "pythia-1b")):
    g = load(f"gate_{m}.json")
    put(f"gate.{k}.c1", f3(g["gate"]["c1_case_kept"]))
    put(f"gate.{k}.c2", f3(g["gate"]["c2_same_word"]))
    put(f"gate.{k}.c3", f3(g["gate"]["c3_inside_word"]))
    put(f"gate.{k}.ceil", f3(g["decode"]["orig_matches_lower"]))
    put(f"gate.{k}.n", str(g["n_paragraphs"]))

# ---------------------------------------------------------------- held-out and exploration measures
C = load("conf_verdicts.json"); E = load("explore_verdicts_check.json")
assert C["prefix"] == "conf_"
npred = {}
for k, (m, l, L) in MODELS.items():
    cm, em = C["models"][m]["measures"], E["models"][m]["measures"]
    for key, v in cm.items():
        if isinstance(v, dict) and "point" in v:
            put(f"c.{k}.{key}", f3(v["point"])); put(f"c.{k}.{key}.lo", f3(v["lo"])); put(f"c.{k}.{key}.hi", f3(v["hi"]))
            put(f"c.{k}.{key}.n", f"{v['n']:,}"); put(f"c.{k}.{key}.np", str(v["n_para"]))
            if isinstance(em.get(key), dict): put(f"e.{k}.{key}", f3(em[key]["point"]))
    for c in ("c1_case_kept", "c2_same_word", "c3_inside_word"):
        put(f"c.{k}.gate.{c[:2]}", f3(cm["gate"][c]))
    put(f"c.{k}.firstlow", str(cm["first_layer_Lc_below_half_for_good"]))
    put(f"e.{k}.firstlow", str(em["first_layer_Lc_below_half_for_good"]))
    P = C["models"][m]["predictions"]
    preds = [p for p in P if p != "account"]
    npred[k] = (sum(P[p] for p in preds), len(preds))
    put(f"c.{k}.npass", str(npred[k][0])); put(f"c.{k}.ntests", str(npred[k][1]))
    for p in preds: put(f"c.{k}.{p}", "passed" if P[p] else "failed")
    put(f"c.{k}.account", P["account"])
put("c.ntests", str(npred["g"][1] + npred["q"][1])); put("c.npass", str(npred["g"][0] + npred["q"][0]))

# exploration-to-held-out differences over every point measure computed on both
gaps = []
for k, (m, _, _) in MODELS.items():
    cm, em = C["models"][m]["measures"], E["models"][m]["measures"]
    for key, v in cm.items():
        if isinstance(v, dict) and "point" in v and isinstance(em.get(key), dict):
            gaps.append((abs(v["point"] - em[key]["point"]), k, key))
gaps.sort(reverse=True)
put("gap.n", str(len(gaps))); put("gap.max", f3(gaps[0][0])); put("gap.over05", str(sum(g > 0.05 for g, _, _ in gaps)))
put("gap.second", f3(gaps[1][0]))
TEXT = {"gap.max.model": gaps[0][1], "gap.max.key": gaps[0][2]}

# changed words: median ranks (conf_exp08 summaries)
for k, (m, _, _) in MODELS.items():
    s = load(f"conf_exp08_{m}.json")["summary"]["upper_diff_first"]
    for nm, x in (("medrank", s["word_median_rank"]), ("ctrlrank", s["ctrl_median_rank"])):
        put(f"c.{k}.e1.{nm}", f"{x:,.0f}" if x == int(x) else f"{x:,.1f}")

# copying (route R3), descriptive: top-1 accuracy of the true first piece for words already seen in the passage and new ones
for k, (m, _, _) in MODELS.items():
    x = load(f"conf_exp01_{m}.json")
    for grp in ("seen", "new", "all"):
        for v in ("upper", "lower"):
            put(f"c.{k}.{grp}.{v}", f3(x[grp][v]["top1"]))
        put(f"c.{k}.{grp}.n", f"{x[grp]['upper']['n']:,}")

# ---------------------------------------------------------------- Figure 2 (held-out lens by layer, descriptive)
lines = ["model layer depth LcU LrU LcLr"]
for k, (m, l, L) in MODELS.items():
    s = load(f"conf_exp01_{m}.json")["lens_upper"]
    for i in range(L):
        lines.append(f"{k} {i} {i / (L - 1):.4f} {s['share_Lc_gt_U'][i]:.4f} {s['share_Lr_gt_U'][i]:.4f} {s['share_Lc_gt_Lr'][i]:.4f}")
    for i in (0, 16, 19, 20, l):
        for nm, key in (("LcU", "share_Lc_gt_U"), ("LrU", "share_Lr_gt_U"), ("LcLr", "share_Lc_gt_Lr")):
            NV.setdefault(f"fig2.{k}.{nm}.{i}", f2(s[key][i]))
for k in MODELS:
    rows = [r.split() for r in lines[1:] if r.startswith(k + " ")]
    (PAPER / f"fig-lens-{k}.dat").write_text("layer depth LcU LrU LcLr\n" + "\n".join(" ".join(r[1:]) for r in rows) + "\n")

# ---------------------------------------------------------------- Figure 1 (exploration sweep of the direction)
for k, (m, l, L) in MODELS.items():
    e = load(f"exp02_{m}.json")
    put(f"sweep.{k}.rows", f"{e['rows'][0]}--{e['rows'][1]}")
    put(f"sweep.{k}.npara", str(e["rows"][1] - e["rows"][0]))
    out = ["layer depth add sub addrand subrand"]
    for lay in e["layers"]:
        s = e["summary"][str(lay)]
        out.append(f"{lay} {lay / (L - 1):.4f} {s['lower_dir_a1']['flip']:.4f} {s['upper_dir_a1']['flip']:.4f} "
                   f"{s['lower_rand_a1']['flip']:.4f} {s['upper_rand_a1']['flip']:.4f}")
        put(f"sweep.{k}.add.{lay}", f3(s["lower_dir_a1"]["flip"])); put(f"sweep.{k}.sub.{lay}", f3(s["upper_dir_a1"]["flip"]))
    (PAPER / f"fig-sweep-{k}.dat").write_text("\n".join(out) + "\n")

# ================================================================ round 2
_SEALED = json.loads((D5.parent / "SEALED-MATERIAL.json").read_text())["files"]["D5-capitals/data/headings-reserve.jsonl"]  # release: the unread sealed file is not shipped; its line count is in SEALED-MATERIAL.json
put("mat.head.reserve.items", f"{_SEALED['lines']:,}")
put("mat.head.reserve.articles", f"{_SEALED['distinct_pageid']:,}")
g = load("gate_olmo1b.json")
put("gate.o.c1", f3(g["gate"]["c1_case_kept"])); put("gate.o.c2", f3(g["gate"]["c2_same_word"])); put("gate.o.c3", f3(g["gate"]["c3_inside_word"]))
C2 = load("r2/conf_verdicts2.json"); E2 = load("r2/explore_verdicts2_check.json")
assert C2["mode"] == "conf" and E2["mode"] == "explore"
R2M = {"q": "qwen3-17b", "g": "gemma"}; tot = [0, 0]
for k, m in R2M.items():
    cm, em = C2["models"][m]["measures"], E2["models"][m]["measures"]
    for pre, mm in (("r2", cm), ("r2e", em)):
        for key, v in mm.items():
            if isinstance(v, dict) and "point" in v:
                put(f"{pre}.{k}.{key}", f3(v["point"]))
                if pre == "r2":
                    put(f"{pre}.{k}.{key}.lo", f3(v["lo"])); put(f"{pre}.{k}.{key}.hi", f3(v["hi"])); put(f"{pre}.{k}.{key}.n", f"{v['n']:,}")
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)) and not key.startswith("attn_"):
                put(f"{pre}.{k}.{key}.min", f3(min(v))); put(f"{pre}.{k}.{key}.max", f3(max(v)))
            elif key.startswith("attn_L") and isinstance(v, list):
                for i, x in enumerate(v): put(f"{pre}.{k}.{key}.{i}", f2(x))
    put(f"r2.{k}.E_flips", f"{cm['E_flips']:,}")
    put(f"r2.{k}.proj_upper_mean", f"{cm['proj_upper_mean']:.1f}".replace("-", "−")); put(f"r2.{k}.proj_lower_mean", f"{cm['proj_lower_mean']:.1f}".replace("-", "−"))
    P = C2["models"][m]["predictions"]; V = C2["models"][m]["verdicts"]
    for pk, pv in P.items(): put(f"r2.{k}.{pk}", "passed" if pv else "failed")
    for vk, vv in V.items(): put(f"r2.{k}.v.{vk}", str(vv))
    tot[0] += V["n_pass"]; tot[1] += V["n_tests"]
put("r2.npass", str(tot[0])); put("r2.ntests", str(tot[1]))
gaps2 = []
for k, m in R2M.items():
    cm, em = C2["models"][m]["measures"], E2["models"][m]["measures"]
    for key, v in cm.items():
        if isinstance(v, dict) and "point" in v and isinstance(em.get(key), dict):
            gaps2.append((abs(v["point"] - em[key]["point"]), k, key))
gaps2.sort(reverse=True)
put("r2gap.n", str(len(gaps2))); put("r2gap.max", f3(gaps2[0][0])); put("r2gap.over05", str(sum(x > 0.05 for x, _, _ in gaps2)))
put("r2gap.second", f3(gaps2[1][0]))
TEXT.update({"r2gap.max.model": gaps2[0][1], "r2gap.max.key": gaps2[0][2], "r2gap.second.key": gaps2[1][2], "r2gap.second.model": gaps2[1][1]})
# attention figure data (held out): one row per distance bin (current word .. first token), one column per head
for k, m in R2M.items():
    cm = C2["models"][m]["measures"]; heads = [h[5:] for h in cm if h.startswith("attn_L") and not h.endswith("_recent8")]
    rows = ["bin " + " ".join(heads)]
    for b in range(6): rows.append(f"{b} " + " ".join(f"{cm['attn_' + h][b]:.4f}" for h in heads))
    (PAPER / f"fig-attn-{k}.dat").write_text("\n".join(rows) + "\n")
# X1, held-out values quoted (cross-study test)
X1 = json.load(open(D5.parent / "X1-shared-parts" / "outputs" / "conf" / "verdicts.json"))["tests"]
put("x1.l0h3.dlow", f3(X1["C7"]["values"]["Dl"])); put("x1.l0h3.dup", f3(X1["C7"]["values"]["Du"])); put("x1.l0h3.wl", f2(X1["C7"]["values"]["wl"]).replace("-", "−"))
self_ = [X1["C8"]["values"][t]["L0H3_self"] for t in ("LB", "IND", "ORD", "CAPS")]
put("x1.l0h3.self.min", f2(min(self_))); put("x1.l0h3.self.max", f2(max(self_)))
put("x1.l17h6.dup", f3(X1["A6"]["values"]["Du"])); put("x1.l17h6.wl", f3(X1["A6"]["values"]["wl"]))
put("x1.ntests", str(len(X1))); put("x1.npass", str(sum(bool(t["passed"]) for t in X1.values())))

# ---------------------------------------------------------------- write
with open(PAPER / "numbers.tex", "w") as f:
    f.write("% Generated by scripts/make_numbers.py from saved outputs. Do not edit.\n% Use in the text as \\nv{key}; an unknown key stops the build.\n")
    for key in sorted(NV):
        f.write(f"\\expandafter\\def\\csname nv@{key}\\endcsname{{{NV[key]}}}\n")
(PAPER / "checks").mkdir(exist_ok=True)
json.dump({"numbers": NV, "text": TEXT}, open(PAPER / "checks" / "numbers.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(NV)} numbers written; largest exploration-to-held-out gap {gaps[0]}")
