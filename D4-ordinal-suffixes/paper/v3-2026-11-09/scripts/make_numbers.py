"""Every number in the D4 write-up, generated from saved outputs. Standard library only.

  python3 -B scripts/make_numbers.py

Reads (never writes), all under ../../ (the D4 study folder):
  outputs/conf/EVALUATION.json        the prospective test, scored by the frozen code/conf_evaluate.py (run once)
  outputs/conf/exp04_*.json           held-out patching scans (attention of L0H2 to the tens digit)
  outputs/conf/chain_conf.started/.done, FROZEN.md, data/MANIFEST.md
  outputs/gate_summary_*.json         the gate (exploration)
  outputs/teen_effect_*.json          exp03 length sweep (exploratory)
  outputs/exp04_*.json, outputs/exp05_*.json, outputs/exp06_qwen3-06b.jsonl   exploration mechanism runs
Writes numbers.tex (one macro per value, used in the text as \\nv{key}), fig-auroc.dat and checks/numbers.json.
"""
import json, re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
D4 = PAPER.parents[1]
OUT = D4 / "outputs"
MINUS = "\u2212"
V = {}


def load(p):
    return json.loads(Path(p).read_text())


def f3(x):
    s = f"{x:.3f}"
    return MINUS + s[1:] if s.startswith("-") else s


def f2(x):
    s = f"{x:.2f}"
    if s == "-0.00": s = "0.00"
    return MINUS + s[1:] if s.startswith("-") else s


def put(k, v):
    assert k not in V, k
    V[k] = v


# ---------------- material, freezes, run window
man = (D4 / "data" / "MANIFEST.md").read_text()
for part in ("explore", "conf", "reserve"):
    m = re.search(rf"\| {part} \| (\d+) \| (\d+) \|", man)
    put(f"mat.{part}.frames", m.group(1)); put(f"mat.{part}.items", f"{int(m.group(2)):,}")
put("mat.frames", str(sum(int(re.search(rf"\| {p} \| (\d+) \|", man).group(1)) for p in ("explore", "conf", "reserve"))))
fz = (D4 / "FROZEN.md").read_text()
put("freeze.charter", re.search(r"\| CHARTER\.md \| [0-9a-f]{64} \| 2026-11-07 (\d\d:\d\d)", fz).group(1))
put("freeze.protocol", re.search(r"\| PROTOCOL\.md \| [0-9a-f]{64} \| 2026-11-08 (\d\d:\d\d)", fz).group(1))
put("freeze.protocol.files", str(len(re.findall(r"\| (?:PROTOCOL\.md|code/\S+) \| [0-9a-f]{64} \| 2026-11-08", fz))))
put("run.start", (OUT / "conf" / "chain_conf.started").read_text().split()[1][:5])
put("run.end", (OUT / "conf" / "chain_conf.done").read_text().split()[1][:5])

# ---------------- gate (exploration)
GM = {"qwen3-06b": "q3", "pythia-410m": "p410", "pythia-1b": "p1b", "olmo2-1b": "olmo", "smollm2-17b": "smol",
      "gemma4-e2b": "gemma", "gpt2": "gpt2"}
for m, k in GM.items():
    g = load(OUT / f"gate_summary_{m}.json")["_gate"]
    put(f"gate.{k}.G1", f3(g["G1"])); put(f"gate.{k}.G2", f3(g["G2"]))
put("gate.p410.miss", f3(0.80 - load(OUT / "gate_summary_pythia-410m.json")["_gate"]["G2"]))
put("gate.n.qualify", str(sum(load(OUT / f"gate_summary_{m}.json")["_gate"]["qualifies"] for m in GM)))
put("gate.n.models", str(len(GM)))

# ---------------- exp03 length sweep (exploratory)
# Post hoc (8 Nov): AUROC recomputed from the per-item exp03 records with tied values counted as one half.
# The exploratory helper code/teen_effect.py ranked ties arbitrarily; with bf16 logits that moves some cells by up to ~0.002.
def auroc_ties(pos, neg):
    allv = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    ranks, i = {}, 0
    r = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]: j += 1
        for q in range(i, j + 1): r[q] = (i + j) / 2 + 1
        i = j + 1
    sp = sum(rr for rr, (v, g) in zip(r, allv) if g == 0)
    return (sp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


TE = {"qwen3-06b": "q3", "smollm2-17b": "smol", "gemma4-e2b": "gemma", "olmo2-1b": "olmo"}
dat_explore = {}
for m, k in TE.items():
    rows = [json.loads(l) for l in open(OUT / f"exp03_{m}.jsonl")]
    au = {}
    for d in range(2, 8):
        s = [r for r in rows if r["digits"] == d and r["form"] == "plain"]
        mm = lambda r: r["logits"][r["n"] % 10 - 1] - r["logits"][3]
        au[d] = auroc_ties([mm(r) for r in s if r["cls"] == "end123"], [mm(r) for r in s if r["cls"] == "teen"])
    dat_explore[k] = au
    lo, hi = min(au[d] for d in range(3, 8)), max(au[d] for d in range(3, 8))
    put(f"x3.{k}.d2", f3(au[2])); put(f"x3.{k}.min37", f3(lo)); put(f"x3.{k}.max37", f3(hi))

# ---------------- exploration mechanism runs
x5 = load(OUT / "exp05_qwen3-06b_0.2-1.6.json")["digits"]
for d in ("2", "3", "4"):
    put(f"x5.q3.kept{d}", f2(x5[d]["frac_effect_kept"]))
    put(f"x5.q3.rmin{d}", f2(x5[d]["random_frac_kept_min"]))
x5o = load(OUT / "exp05_olmo2-1b_0.5-0.0-0.9.json")["digits"]
for d in ("3", "4", "5", "7"):
    put(f"x5.olmo.kept{d}", f2(x5o[d]["frac_effect_kept"]))
put("x5.olmo.rmin", f2(min(x5o[d]["random_frac_kept_min"] for d in x5o)))
s4 = load(OUT / "exp04_qwen3-06b_4d_intoteen.json")
put("x4.q3.4d.L0H2", f2(s4["head"][0][2])); put("x4.q3.4d.att", f2(s4["att"][0][2][1]))
s2 = load(OUT / "exp04_qwen3-06b_2d_intoteen.json")
put("x4.q3.2d.L0H2", f2(s2["head"][0][2]))
o3 = [load(OUT / f"exp04_olmo2-1b_3d_{k}.json") for k in ("intoteen", "outteen")]
put("x4.olmo.3d.maxhead", f2(max(max(max(r) for r in s["head"]) for s in o3)))
put("x4.olmo.3d.mlp13", f2(min(s["mlp"][13] for s in o3)))
o4 = [load(OUT / f"exp04_olmo2-1b_4d_{k}.json") for k in ("intoteen", "outteen")]
for (l, h) in ((0, 5), (0, 0), (0, 9)):
    put(f"x4.olmo.4d.L{l}H{h}.lo", f2(min(s["head"][l][h] for s in o4)))
    put(f"x4.olmo.4d.L{l}H{h}.hi", f2(max(s["head"][l][h] for s in o4)))
put("x4.olmo.4d.att.L0H5", f2(o4[0]["att"][0][5][1]))
R6 = [json.loads(l) for l in open(OUT / "exp06_qwen3-06b.jsonl")]
T6 = [r for r in R6 if r["teen"]]
mean = lambda xs: sum(xs) / len(xs)
put("x6.att.ok", f3(mean([r["att"]["0.2"] for r in T6 if r["ok"]])))
put("x6.att.bad", f3(mean([r["att"]["0.2"] for r in T6 if not r["ok"]])))
for fam in ("rankings", "anniversaries"):
    put(f"x6.acc.{fam}", f3(mean([r["ok"] for r in T6 if r["family"] == fam])))
put("x6.n", f"{len(T6):,}")

# ---------------- the prospective test (EVALUATION.json)
E = load(OUT / "conf" / "EVALUATION.json")
C, T = E["cells"], E["tests"]
O = C["olmo2-1b"]
for key, c in O.items():
    d, form = key.split("|")
    k = f"c.olmo.{d}{form[0]}"
    put(k + ".auroc", f3(c["auroc"])); put(k + ".lo", f3(c["auroc_ci95"][0])); put(k + ".hi", f3(c["auroc_ci95"][1]))
    put(k + ".acct", f3(c["acc_teen"])); put(k + ".acce", f3(c["acc_end123"])); put(k + ".nnum", str(c["n_teen_numbers"]))
put("c.olmo.nteen", str(O["4|plain"]["n_teen"]))
put("c.olmo.B3.d4", f3(T["B3"]["values"]["4 comma-plain"])); put("c.olmo.B3.d7", f3(T["B3"]["values"]["7 comma-plain"]))
put("c.olmo.B4.r54", f3(T["B4"]["values"]["auroc 5-4"])); put("c.olmo.B4.r67", f3(T["B4"]["values"]["auroc 6-7"]))
put("c.olmo.B2.min", f3(min(T["B2"]["values"].values())))
for m, k in (("gpt2", "gpt2"), ("pythia-1b", "p1b")):
    for s in ("boundary", "control"):
        c = C[m][s]
        put(f"c.{k}.{s}.auroc", f3(c["auroc"])); put(f"c.{k}.{s}.lo", f3(c["auroc_ci95"][0])); put(f"c.{k}.{s}.hi", f3(c["auroc_ci95"][1]))
        put(f"c.{k}.{s}.acct", f3(c["acc_teen"])); put(f"c.{k}.{s}.nteen", f"{c['n_teen']:,}"); put(f"c.{k}.{s}.nend", f"{c['n_end123']:,}")
        put(f"c.{k}.{s}.nnum", str(c["n_teen_numbers"]))
    put(f"c.{k}.gap", f3(T["B5"][m]["values"]["auroc control-boundary"]))
Q = C["qwen3-06b_ablation"]
for d in ("2", "3", "4"):
    q = Q[d]
    put(f"c.q3.{d}.base", f2(q["base_effect"])); put(f"c.q3.{d}.bauroc", f3(q["base_auroc"]))
    put(f"c.q3.{d}.kept", f3(q["kept"])); put(f"c.q3.{d}.rmin", f3(q["random_kept_min"])); put(f"c.q3.{d}.rmax", f3(q["random_kept_max"]))
    put(f"c.q3.{d}.tdrop", f3(q["teen_drop"])); put(f"c.q3.{d}.edrop", f3(q["end123_drop"]))
    put(f"c.q3.{d}.L0H2", f3(q["single_kept"]["0.2"])); put(f"c.q3.{d}.L1H6", f3(q["single_kept"]["1.6"]))
    put(f"c.q3.{d}.tacc", f3(q["teen_acc_ablated"]))
P = C["qwen3-06b_patching"]
for k, v in P.items():
    put(f"c.q3.p{k}.L0H2", f3(v["L0H2"])); put(f"c.q3.p{k}.top", v["top"]); put(f"c.q3.p{k}.topv", f3(v["top_value"]))
    put(f"c.q3.p{k}.nvalid", str(v["n_valid"])); put(f"c.q3.p{k}.n", str(v["n"]))
for d in (3, 4):
    put(f"c.q3.att{d}", f2(load(OUT / "conf" / f"exp04_qwen3-06b_{d}d_intoteen.json")["att"][0][2][1]))
Y = C["olmo2-1b_ablation"]
for d in ("3", "4", "5", "7"):
    y = Y[d]
    put(f"c.ol.{d}.base", f2(y["base_effect"])); put(f"c.ol.{d}.bauroc", f3(y["base_auroc"])); put(f"c.ol.{d}.kept", f3(y["kept"]))
    put(f"c.ol.{d}.rmin", f3(y["random_kept_min"])); put(f"c.ol.{d}.rmax", f3(y["random_kept_max"]))
    for h, v in y["single_kept"].items(): put(f"c.ol.{d}.s{h.replace('.', '')}", f3(v))
npass = sum(1 for k, t in T.items() if k != "B5" and t["passed"]) + sum(1 for t in T["B5"].values() if t["passed"])
put("c.npass", str(npass)); put("c.ntests", str(len([k for k in T if k != "B5"]) + len(T["B5"])))
put("c.verdict.boundary", E["verdict_boundary"]); put("c.verdict.qwen", E["verdict_qwen_copy"]); put("c.verdict.olmo", E["verdict_olmo_move"])

# ---------------- prespecified thresholds (PROTOCOL.md section 4; check_numbers.py confirms each appears there)
THR = {"B1": "0.80", "B2": "0.88", "B3": "0.10", "B4.acc4": "0.65", "B4.acc3": "0.85", "B5.bd": "0.50", "B5.ct": "0.85",
       "Q1.kept": "0.55", "Q1.rand": "0.85", "Q2.teen": "0.20", "Q2.end": "0.10", "Q3.kept": "0.55", "Q3.acc": "0.95",
       "Q4.L0H2": "0.70", "Q4.L1H6": "0.80", "Q5": "0.40", "Q6.abs": "0.05", "Q6.top": "0.30",
       "O1.kept": "0.65", "O1.rand": "0.85", "O1.read": "0.30", "O2": "0.90"}
for k, v in THR.items(): put("thr." + k, v)

# ---------------- round 2 (depth): PROTOCOL-2.md, run once 9 Nov; tests scored by the frozen code/r2conf_evaluate.py
R2C = OUT / "r2conf"
E2 = load(R2C / "EVALUATION-2.json"); T2 = E2["tests"]
tv = lambda k: T2[k]["values"]
put("r2.freeze.charter", re.search(r"\| CHARTER-2\.md \| [0-9a-f]{64} \| 2026-11-08 (\d\d:\d\d)", fz).group(1))
put("r2.freeze.protocol", re.search(r"frozen 9 Nov 2026, (\d\d:\d\d) IST", fz).group(1))
put("r2.run.start", (R2C / "chain_conf.started").read_text().split()[1][:5])
put("r2.run.end", (R2C / "chain_conf.done").read_text().split()[1][:5])
read2 = {k: t for k, t in T2.items() if t["passed"] is not None}
put("r2.ntests", str(len(read2))); put("r2.npass", str(sum(t["passed"] for t in read2.values())))
for pre, nm in (("R", "path"), ("S", "backup"), ("O", "olmo"), ("X:", "other")):
    ks = [k for k in read2 if k.startswith(pre)]
    put(f"r2.{nm}.n", str(len(ks))); put(f"r2.{nm}.pass", str(sum(read2[k]["passed"] for k in ks)))
    put(f"r2.{nm}.verdict", {"R": E2["verdicts"]["qwen_path"], "S": E2["verdicts"]["qwen_backup"],
                              "O": E2["verdicts"]["olmo_weak_route"], "X:": E2["verdicts"]["other_models"]}[pre])
# Qwen3-0.6B: the path
q = load(R2C / "qwen.json")["digits"]
for d in ("4", "6"):
    b = tv(f"base_qwen@{d}")
    put(f"r2.q.{d}.base", f3(b["teen"])); put(f"r2.q.{d}.bauroc", f3(b["auroc"]))
    v = tv(f"R1@{d}"); put(f"r2.q.{d}.R1.via", f3(v["via_mlp0"])); put(f"r2.q.{d}.R1.other", f3(v["max_other"]))
    v = tv(f"R2@{d}"); put(f"r2.q.{d}.R2.own", f3(v["l0h2"])); put(f"r2.q.{d}.R2.ctl", f3(v["max_control"]))
    v = tv(f"R3@{d}")
    for k in ("l0h2_mlp24", "l0h2_next", "l11h9_mlp24", "l11h9_next", "l11h9_full_loss"): put(f"r2.q.{d}.R3.{k}", f3(v[k]))
    v = tv(f"R4@{d}"); put(f"r2.q.{d}.R4.rise", f3(v["rise24"])); put(f"r2.q.{d}.R4.next", f3(v["next_rise"]))
    v = tv(f"R5@{d}"); put(f"r2.q.{d}.R5.share", f3(v["mlp24_share"])); put(f"r2.q.{d}.R5.other", f3(v["max_other_share"]))
    v = tv(f"R6@{d}")
    put(f"r2.q.{d}.R6.thlo", f3(min(v["mlp24_th_minus_mean"]))); put(f"r2.q.{d}.R6.thhi", f3(max(v["mlp24_th_minus_mean"])))
    put(f"r2.q.{d}.R6.spec24", f3(max(abs(x) for x in v["mlp24_own_spec"])))
    s18 = v["mlp18_own_spec_teen"] + v["mlp18_own_spec_nonteen"]
    put(f"r2.q.{d}.R6.s18lo", f3(min(s18))); put(f"r2.q.{d}.R6.s18hi", f3(max(s18)))
    for c, key in (("teen", "mlp18_own_spec_teen"), ("non", "mlp18_own_spec_nonteen")):
        for i, x in enumerate(v[key], 1): put(f"r2.q.{d}.R6.s18.{c}{i}", f3(x))
with open(PAPER / "fig-lens.dat", "w") as f:
    f.write("layer d4 d6\n")
    for l in range(len(q["4"]["lens"])):
        f.write(f"{l} {q['4']['lens'][l]['teen']:.4f} {q['6']['lens'][l]['teen']:.4f}\n")
# Qwen3-0.6B: the backup route
q2 = load(R2C / "qwen2.json")["digits"]
for d in ("2", "3", "4"):
    v = tv(f"S1@{d}")
    put(f"r2.s.{d}.base", f3(q2[d]["base"]["teen"]))
    for k in ("a", "ab", "random_min", "random_max"): put(f"r2.s.{d}.{k}", f3(v[k]))
    put(f"r2.s.{d}.b", f3(tv(f"S2@{d}")["b"])); put(f"r2.s.{d}.drop", f3(tv("S3")["drop"][d]))
# OLMo-2-1B: the weak route
o = load(R2C / "olmo.json")["digits"]
for d in ("3", "4", "5", "7", "8", "9", "10"):
    put(f"r2.o.{d}.base", f3(o[d]["base"]["teen"])); put(f"r2.o.{d}.auroc", f3(tv("O3")["auroc"][d]))
    put(f"r2.o.{d}.acct", f3(o[d]["base"]["acc_teen"]))
    v = tv(f"O4@{d}"); put(f"r2.o.{d}.kept", f3(v["kept"])); put(f"r2.o.{d}.rmin", f3(v["random_min"]))
    pr = tv("O5")["probe"][d]; put(f"r2.o.{d}.p2", f3(pr["2"])); put(f"r2.o.{d}.p15", f3(pr["15"]))
    put(f"r2.o.{d}.pieces", "+".join(str(len(x)) for x in o[d]["pieces"]))
for d in ("4", "7", "10"):
    put(f"r2.o.{d}.vmax", f3(tv("O6")["max_ratio"][d]))
    st = tv("O7")["steer"][d]; put(f"r2.o.{d}.steer", f3(-st["v3"])); put(f"r2.o.{d}.rand", f3(max(abs(st[f"rand{j}"]) for j in range(5))))
    v = tv(f"O8@{d}"); put(f"r2.o.{d}.via", f3(v["via_mlp0"])); put(f"r2.o.{d}.next", f3(v["next"]))
put("r2.o.5.vmin", f3(tv("O6")["min_ratio_5"]))
# other one-token-per-digit models
OM = {"qwen3-17b": "q17", "smollm2-17b": "smol", "gemma4-e2b": "gemma"}
for m, k in OM.items():
    r = load(R2C / f"other_{m}.json")
    for d in ("3", "4"):
        put(f"r2.x.{k}.{d}.base", f3(r["digits"][d]["base"]["teen"])); put(f"r2.x.{k}.{d}.bauroc", f3(r["digits"][d]["base"]["auroc"]))
        for sname in r["sets"]:
            v = tv(f"X:{m}:{sname}@{d}"); kk = f"r2.x.{k}.{d}.{sname}"
            put(kk + ".teen", f3(v["teen_kept"])); put(kk + ".pull", f3(v["pull_kept"])); put(kk + ".rmin", f3(v["random_min"]))
# exploration E06: OLMo-2 stage-1 checkpoints (exploratory, descriptive)
for f in sorted((OUT / "r2").glob("e06_olmo2-1b_*.json")):
    r = load(f); tok = re.search(r"tokens(\d+)B", r["rev"]).group(1); k = f"r2.e6.{tok}"
    for d in ("3", "4", "5", "7"): put(f"{k}.auroc{d}", f3(r["digits"][d]["base"]["auroc"]))
    R4 = r["digits"]["4"]; put(f"{k}.base4", f2(R4["base"]["teen"]))
    put(f"{k}.tri", f3(R4["tri"]["teen"] / R4["base"]["teen"]) if R4["base"]["teen"] >= 0.30 else "--")
    put(f"{k}.p2", f2(R4["probe_last"]["2"])); put(f"{k}.p14", f2(R4["probe_last"]["14"]))
# post hoc (9 Nov): released OLMo-2 weights vs the published checkpoints (../../outputs/r2post/olmo_lineage.json)
LIN = load(OUT / "r2post" / "olmo_lineage.json"); EMB = "model.embed_tokens.weight"
put("post.lin.emb", f3(LIN["main_vs_stage1end"][EMB]["cos"]))
put("post.lin.s2min", f"{min(v[EMB]['cos'] for k, v in LIN.items() if k.startswith('stage2-') and k.endswith('_vs_stage1end')):.4f}")
# round-2 thresholds (PROTOCOL-2.md section 4; check_numbers.py confirms each appears there)
THR2 = {"R1.via": "0.60", "R1.other": "0.15", "R1.read": "0.30", "R3": "0.40", "R4": "1.0", "R5.share": "0.50", "R5.other": "0.35",
        "R6.th": "1.5", "R6.spec": "0.6", "R6.s18": "0.4", "S2": "0.85", "S3.drop": "0.15", "S3.ab": "0.30",
        "O3.in": "0.88", "O3.cr": "0.82", "O3.gap": "0.10", "O4.kept": "0.60", "O4.in": "0.90", "O5.p2": "0.90", "O5.in": "0.93",
        "O6": "0.65", "O7.steer": "1.5", "O7.rand": "0.5", "O8": "0.50", "X.teen": "0.85", "X.pull": "0.55", "X.cur": "0.30", "X.curteen": "0.90"}
for k, v in THR2.items(): put("thr2." + k, v)

# ---------------- figure data: OLMo-2 teen AUROC by length (held out) and exploration curves
with open(PAPER / "fig-auroc.dat", "w") as f:
    f.write("digits olmo_plain olmo_comma q3 smol gemma\n")
    for d in range(2, 8):
        op = O.get(f"{d}|plain", {}).get("auroc"); oc = O.get(f"{d}|comma", {}).get("auroc")
        row = [op, oc] + [dat_explore[k].get(d) for k in ("q3", "smol", "gemma")]
        f.write(f"{d} " + " ".join("nan" if x is None else f"{x:.4f}" for x in row) + "\n")

with open(PAPER / "numbers.tex", "w") as f:
    f.write("% Generated by scripts/make_numbers.py from saved outputs. Do not edit.\n")
    f.write("% Use in the text as \\nv{key}; an unknown key stops the build.\n")
    for k in sorted(V):
        f.write(f"\\expandafter\\def\\csname nv@{k}\\endcsname{{{V[k]}}}\n")
(PAPER / "checks").mkdir(exist_ok=True)
json.dump(V, open(PAPER / "checks" / "numbers.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(V)} numbers written")
