"""Every number in the combined D3-D4-D5-X1 paper (version 5), generated from the studies' saved outputs (read only).

  python -B scripts/make_numbers.py

How it works.
  D3, D4, D5: each source paper's own number builder (D3 v2, D4 v3, D5 v2, file scripts/make_numbers.py) is run
  unchanged except for two lines: where it writes (redirected to build/src-d3/ etc. inside this folder, so nothing in
  the source paper folders is touched) and where its study folder is (the real one, read only). Each builder reads
  only its study's saved outputs. Every key it writes is carried over with a prefix: d3.*, d4.*, d5.*.
  X1: read here from X1-shared-parts/outputs/conf/ (verdicts.json for the tested values and verdicts; the raw run files
  c01_*.json for the full ablation matrix; c02_attn_*.json for attention), under the prefix x1m.* (to keep it apart
  from the x1.* keys that D3 and D5 already carry).
  dx.*: totals across the studies, from the carried-over values.
Writes numbers.tex (macros used as \\nv{key}), the figure data fig-d3-*.dat, fig-d4-*.dat, fig-d5-*.dat, and
checks/numbers.json. scripts/check_numbers.py checks every value without this code.
"""
import json, re, shutil, sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
AREA = PAPER.parent                          # the release root, which holds the study folders
BUILD = PAPER / "build"
SOURCES = {
    "d3": ("D3-indentation", "paper/v2-2026-11-09", "D3"),
    "d4": ("D4-ordinal-suffixes", "paper/v3-2026-11-09", "D4"),
    "d5": ("D5-capitals", "paper/v2-2026-11-08", "D5"),
}
MINUS = "−"
OUT = {}


def put(k, v):
    assert k not in OUT, k
    OUT[k] = v


def read_numbers(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\\expandafter\\def\\csname nv@(.+?)\\endcsname\{(.*)\}$", line)
        if m: d[m.group(1)] = m.group(2)
    return d


# ---------------------------------------------------------------- D3, D4, D5 via their own builders
for pre, (study, paper, var) in SOURCES.items():
    sdir = AREA / study
    src = sdir / paper / "scripts" / "make_numbers.py"
    text = src.read_text()
    scratch = BUILD / f"src-{pre}"
    if scratch.exists(): shutil.rmtree(scratch)
    (scratch / "checks").mkdir(parents=True)
    a = "PAPER = Path(__file__).resolve().parents[1]\n"
    b = f"{var} = PAPER.parents[1]\n"
    assert text.count(a) == 1 and text.count(b) == 1, (pre, "path lines not found exactly once")
    text = text.replace(a, f"PAPER = Path({str(scratch)!r})\n").replace(b, f"{var} = Path({str(sdir)!r})\n")
    ns = {"__file__": str(src), "__name__": f"dx_source_{pre}"}
    exec(compile(text, str(src), "exec"), ns)
    for k, v in read_numbers(scratch / "numbers.tex").items():
        put(f"{pre}.{k}", v)
    for f in sorted(scratch.glob("fig-*.dat")):
        shutil.copyfile(f, PAPER / f"fig-{pre}-{f.name[4:]}")

# ---------------------------------------------------------------- X1
X1 = AREA / "X1-shared-parts" / "outputs" / "conf"
VER = json.loads((X1 / "verdicts.json").read_text())


def f3(x):
    s = f"{x:.3f}"
    if s in ("-0.000",): s = "0.000"
    return MINUS + s[1:] if s.startswith("-") else s


def f2(x):
    s = f"{x:.2f}"
    return MINUS + s[1:] if s.startswith("-") else s


T = VER["tests"]
put("x1m.ntests", str(VER["verdicts"]["n_tests"]))
put("x1m.npass", str(VER["verdicts"]["n_passed"]))
assert sum(t["passed"] for t in T.values()) == VER["verdicts"]["n_passed"]
for name, t in T.items():
    put(f"x1m.v.{name}", "pass" if t["passed"] else "fail")
put("x1m.account", VER["verdicts"]["account"])

# raw matrix: every named group against the unablated run; random range over its 20 draws
RAW = {"LB": "c01_qwen3-06b-bf16_LB.json", "IND": "c01_qwen3-06b-bf16_IND.json",
       "ORD": "c01_qwen3-06b-bf16_ORD.json", "CAPS": "c01_qwen3-17b_CAPS.json"}
MEAS = {  # measure code: (task, field, kind) where kind 'kept' = ablated/base and 'chg' = ablated - base
    "LBw": ("LB", "width_effect", "kept"), "LBs": ("LB", "sep", "kept"),
    "cU": ("IND", "code|dedent|lq", "chg"), "cT": ("IND", "code|term|lq", "chg"),
    "gU": ("IND", "gen|dedent|lq", "chg"), "gT": ("IND", "gen|term|lq", "chg"),
    "t4": ("ORD", "d4|teen_effect", "kept"), "t5": ("ORD", "d5|teen_effect", "kept"),
    "p4": ("ORD", "d4|pull", "kept"), "p5": ("ORD", "d5|pull", "kept"),
    "Du": ("CAPS", "D_up", "kept"), "Dl": ("CAPS", "D_low", "kept"), "wl": ("CAPS", "word_lp", "chg"),
}
runs = {t: json.loads((X1 / f).read_text()) for t, f in RAW.items()}


def val(run, field, kind, base):
    return run[field] / base[field] if kind == "kept" else run[field] - base[field]


for code, (task, field, kind) in MEAS.items():
    R = runs[task]; base = R["base"]
    groups = [g for g in R if g != "base" and "~" not in g]
    for g in groups:
        gk = g.replace("-L17H6", "noL17H6")
        put(f"x1m.m.{gk}.{code}", f3(val(R[g], field, kind, base)))
        rv = [val(R[r], field, kind, base) for r in R if r.startswith(g + "~r")]
        assert len(rv) == 20, (task, g, len(rv))
        put(f"x1m.m.{gk}.{code}.rlo", f3(min(rv))); put(f"x1m.m.{gk}.{code}.rhi", f3(max(rv)))
        # v4, post hoc: where the named group's value lies against the range of its 20 random groups (unrounded values)
        v = val(R[g], field, kind, base)
        put(f"x1m.o.{gk}.{code}", "below" if v < min(rv) else "above" if v > max(rv) else "in")
    if task == "LB":
        put(f"x1m.base.{code}", f3(base[field]))
for task, keys in (("IND", ["code|dedent|lq", "code|term|lq", "gen|dedent|lq", "gen|term|lq"]),):
    pass
put("x1m.n.LB.width", f"{runs['LB']['base']['n_width']:,}"); put("x1m.n.LB.par", f"{runs['LB']['base']['n_nat']:,}")
for cls, key in (("cU", "code|dedent|n"), ("cT", "code|term|n"), ("gU", "gen|dedent|n"), ("gT", "gen|term|n")):
    put(f"x1m.n.IND.{cls}", f"{runs['IND']['base'][key]:,}")
put("x1m.n.ORD.teen4", str(runs["ORD"]["base"]["d4|n_teen"])); put("x1m.n.ORD.non4", str(runs["ORD"]["base"]["d4|n_end123"]))
put("x1m.n.CAPS.pos", f"{runs['CAPS']['base']['n_pos']:,}"); put("x1m.n.CAPS.par", f"{runs['CAPS']['base']['n_par']:,}")
# the C7 comparison: loss on D_low against three times the loss on D_up
c7 = T["C7"]["values"]
put("x1m.C7.lossDl", f3(1 - c7["Dl"])); put("x1m.C7.threeLossDu", f3(3 * (1 - c7["Du"])))

# v4, post hoc: the criterion the protocol froze for L17H6's shared role (test A3: change on generated terminal
# dedents <= -0.010 and below every random draw), applied to every single head run on the indentation task
for h in ("L17H6", "L11H9", "L0H2", "L0H3"):
    R = runs["IND"]; b = R["base"]
    v = R[h]["gen|term|lq"] - b["gen|term|lq"]
    rv = [R[r]["gen|term|lq"] - b["gen|term|lq"] for r in R if r.startswith(h + "~r")]
    put(f"x1m.ph.A3rule.{h}", "met" if (v <= -0.010 and v < min(rv)) else "not met")

# Qwen3-0.6B on the capitals gate (why capitals used Qwen3-1.7B)
G06 = json.loads((AREA / "X1-shared-parts" / "outputs" / "gate_qwen3-06b.json").read_text())
put("x1m.gate06b.c2", f3(G06["gate"]["c2_same_word"]))

# attention at decision positions (held parts, Qwen3-0.6B)
ATT = json.loads((X1 / "c02_attn_qwen3-06b-bf16.json").read_text())
for task in ("LB", "IND", "ORD", "CAPS"):
    put(f"x1m.att.L0H3.self.{task}", f2(ATT[task]["L0H3"]["dec"]["0"]))
    put(f"x1m.att.L0H2.prev.{task}", f2(ATT[task]["L0H2"]["dec"]["1"]))
    put(f"x1m.att.L11H9.self.{task}", f2(ATT[task]["L11H9"]["dec"]["0"]))
self3 = [ATT[t]["L0H3"]["dec"]["0"] for t in ATT]; self119 = [ATT[t]["L11H9"]["dec"]["0"] for t in ATT]
put("x1m.att.L0H3.self.min", f2(min(self3))); put("x1m.att.L0H3.self.max", f2(max(self3)))
put("x1m.att.L11H9.self.min", f2(min(self119))); put("x1m.att.L11H9.self.max", f2(max(self119)))

# ---------------------------------------------------------------- dx: models in the ordinal study beyond the two main ones
D4O = AREA / "D4-ordinal-suffixes" / "outputs"
ord_models = {f.name[len("gate_summary_"):-5] for f in D4O.glob("gate_summary_*.json")
              if json.loads(f.read_text())["_gate"]["qualifies"]}          # passed the round-1 gate and were studied
ord_models |= {f.name[len("other_"):-5] for f in (D4O / "r2conf").glob("other_*.json")}  # round-2 further models
further = sorted(ord_models - {"qwen3-06b", "olmo2-1b"})
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
put("dx.ord.nfurther", str(len(further))); put("dx.ord.nfurther.word", WORDS[len(further)])
put("dx.d4r2.nfail", str(int(OUT["d4.r2.ntests"]) - int(OUT["d4.r2.npass"])))

# ---------------------------------------------------------------- dx (v4): units of the ordinal counts, gate counts, Gemma size
# round 1: numbered predictions in the frozen record, and how many models the per-model one (B5) is read in
E1 = json.loads((D4O / "conf" / "EVALUATION.json").read_text())["tests"]
put("dx.d4r1.npred", str(len(E1)))
permodel = [k for k, t in E1.items() if "passed" not in t]
assert permodel == ["B5"], permodel
put("dx.d4r1.B5.nmodels.word", WORDS[len(E1["B5"])])
assert len(E1) - 1 + len(E1["B5"]) == int(OUT["d4.c.ntests"])
# round 2: the failed tests, and how many distinct predictions (test name without its number length) they are
E2 = json.loads((D4O / "r2conf" / "EVALUATION-2.json").read_text())["tests"]
failed = [k for k, t in E2.items() if t["passed"] is False]
assert len(failed) == int(OUT["dx.d4r2.nfail"])
put("dx.d4r2.nfail.word", WORDS[len(failed)])
put("dx.d4r2.nfailpred.word", WORDS[len({k.split("@")[0] for k in failed})])
put("dx.d4r2.nfaillen.word", WORDS[len({k.split("@")[1] for k in failed})])
# ordinal gate: the seven round-1 gate records, plus Qwen3-1.7B-Base, gated by the same rule before round 2's analysis
g1 = [json.loads(f.read_text())["_gate"]["qualifies"] for f in sorted(D4O.glob("gate_summary_*.json"))]
g17 = json.loads((D4O / "r2" / "e04_qwen3-17b.json").read_text())["gate"]["qualifies"]
put("dx.ordgate.q17", "passed" if g17 else "failed")
put("dx.ordgate.ngated", str(len(g1) + 1)); put("dx.ordgate.npass", str(sum(g1) + int(g17)))
# capitals gate: the capitals study's own gate records (four charter models, then OLMo-2) and the cross-study test's
D5O = AREA / "D5-capitals" / "outputs"
cg = [json.loads(f.read_text())["gate"]["pass"] for f in sorted(D5O.glob("gate_*.json"))] + [G06["gate"]["pass"]]
put("dx.capsgate.ngated.word", WORDS[len(cg)]); put("dx.capsgate.npass.word", WORDS[sum(cg)])
put("dx.capsgate.ncharter.word", WORDS[len(cg) - 2])
# Gemma-4-E2B's size, from the saved excerpt of its model card (billions of parameters)
card = (PAPER / "sources" / "gemma-4-E2B-model-card.txt").read_text()
m = re.search(r"\*\*Total Parameters\*\* \| ([\d.]+)B effective <br> \(([\d.]+)B with embeddings\)", card)
put("dx.gemma.effB", m.group(1)); put("dx.gemma.totB", m.group(2))

# ---------------------------------------------------------------- r3 (v5): the indentation study's third round, a
# replication of the code-based tests on permissively licensed and newly generated material, and the prospective
# test of L11H9's second role. Read from the frozen verdict scripts' outputs in D3-indentation/round3/ (and, for the
# original column of the appendix tables, the same files of rounds 1 and 2).
R3D = AREA / "D3-indentation" / "round3"; D3O = AREA / "D3-indentation" / "outputs"
def sg3(x): return ("+" if round(x, 3) > 0 else "") + f3(x)
def sg2(x): s = f2(x); return s if s.startswith(MINUS) else "+" + s
FS = json.loads((R3D / "data" / "code" / "fetch_summary.json").read_text())
BS = json.loads((R3D / "data" / "code" / "build_summary.json").read_text())
put("r3.mat.nfiles", f"{FS['files']:,}"); put("r3.mat.nrepos", f"{FS['repos_used']:,}")
put("r3.mat.nchunks", f"{BS['held_chunks']:,}"); put("r3.mat.nchunkrepos", f"{BS['held_repos']:,}")
put("r3.mat.ngen", f"{sum(1 for _ in open(R3D / 'data' / 'gen' / 'part-held.jsonl')):,}")
put("r3.mat.ngen2", f"{sum(1 for _ in open(R3D / 'data' / 'gen2' / 'part-held.jsonl')):,}")
# R1: c03_verdicts.py output (value; context numbers) for round 3 and round 1
for tag, d in (("r3.r1", R3D / "outputs" / "conf"), ("r3.o1", D3O / "conf")):
    V1 = {x["id"]: x for x in json.loads((d / "verdicts.json").read_text())["verdicts"]}
    for i in range(1, 10):
        h = f"H{i}"; v = V1[h]["value"]
        put(f"{tag}.{h}", f3(v) if h in ("H3", "H9") else sg3(v))
        if tag == "r3.r1": put(f"{tag}.v.{h}", V1[h]["verdict"])
    if tag == "r3.r1":
        put("r3.r1.npass", str(sum(x["verdict"] == "pass" for x in V1.values()))); put("r3.r1.ntests", str(len(V1)))
        ctx = lambda h: [float(x) for x in re.findall(r"-?\d+\.\d+", V1[h]["context"])]
        put("r3.r1.H1.r", sg3(ctx("H1")[0])); put("r3.r1.H5.r", sg3(ctx("H5")[0])); put("r3.r1.H8.r", sg3(ctx("H8")[0]))
        put("r3.r1.H9.suf", f3(ctx("H9")[0])); put("r3.r1.H9.rnec", f3(ctx("H9")[1]))
# R2: r2c04_verdicts.py outputs (measures and verdicts)
R2KEYS = {"1": ("n", "att_inner_ded", f3), "2": ("n", "ko_inner_ded", sg3), "3": ("n", "ko_inner_same", sg3), "4": ("n", "ko_closed_ded", sg3),
          "5": ("n", "pick_not_inner", f3), "6": ("n", "steer_dedent_head_dlq", sg2), "7": ("n", "ko_true_ded", sg3),
          "8": ("g", "steer_term_head_dlq", sg2), "9": ("g", "share_blk_kw_return", f3), "10": ("g", "share_layer_return_max", f3)}
for tag, d in (("r3.r2", R3D / "outputs" / "r2"), ("r3.o2", D3O / "r2")):
    M2 = json.loads((d / "measures_conf.json").read_text())
    for P, mod in (("Q", "qwen3-06b-bf16/L23H2"), ("O", "olmo1b/L13H1")):
        for i, (src, k, fm) in R2KEYS.items():
            put(f"{tag}.{P}{i}", fm(M2[f"{mod}/{'natural' if src == 'n' else 'gen2'}"][k]))
    put(f"{tag}.Q11", sg3(M2["qwen3-06b-bf16/L23H2/gen2"]["l0h2_line_dqcur_term"]))
    G = M2["olmo1b/DEDO/group"]
    for i, k in (("1", "d_lq_dedent"), ("2", "d_lq_term"), ("3", "d_lq_same"), ("4", "d_psame_dedent")):
        put(f"{tag}.G{i}", sg3(G[k]))
    if tag == "r3.r2":
        put("r3.r2.G1.rlo", sg3(G["rand_d_lq_dedent_min"])); put("r3.r2.G1.rhi", sg3(G["rand_d_lq_dedent_max"]))
        put("r3.r2.q.others_maxabs", f3(M2["qwen3-06b-bf16/L23H2/natural"]["ko_inner_others_maxabs"]))
        V2 = json.loads((d / "conf" / "verdicts.json").read_text())
        put("r3.r2.npass", str(sum(V2["predictions"].values()))); put("r3.r2.ntests", str(len(V2["predictions"])))
        for k, v in V2["predictions"].items(): put(f"r3.r2.v.{k}", "pass" if v else "fail")
        put("r3.r2.nqconf", str(sum(v == "confirmed" for v in V2["questions"].values()))); put("r3.r2.nq", str(len(V2["questions"])))
        put("r3.r2.nreturn", f"{M2['qwen3-06b-bf16/L23H2/gen2']['n']['return']:,}")
# X and E: r3_verdicts.py output
XE = json.loads((R3D / "outputs" / "conf" / "verdicts_XE.json").read_text()); TX = XE["tests"]
XK = ["XA3", "XA5", "XB1", "XB2", "XB3", "XB4", "XC3", "XC6", "XD3"]; EK = ["E1", "E2", "E3", "E4", "E5", "E6"]
for k in XK + EK: put(f"r3.v.{k}", "pass" if TX[k]["passed"] else "fail")
put("r3.x.npass", str(sum(TX[k]["passed"] for k in XK))); put("r3.x.ntests", str(len(XK)))
put("r3.e.npass", str(sum(TX[k]["passed"] for k in EK))); put("r3.e.ntests", str(len(EK)))
put("r3.e.verdict", XE["verdicts"]["E: L11H9 second role"])
v = TX["XA3"]["values"]; put("r3.x.XA3", sg3(v["change"])); put("r3.x.XA3.rlo", sg3(v["rand"][0])); put("r3.x.XA3.rhi", sg3(v["rand"][1]))
v = TX["XB4"]["values"]; put("r3.x.XB4", sg3(v["change"])); put("r3.x.XB4.rlo", sg3(v["rand"][0])); put("r3.x.XB4.rhi", sg3(v["rand"][1]))
put("r3.x.XD3.max", f3(max(abs(x) for x in TX["XD3"]["values"].values())))
v = TX["E1"]["values"]; put("r3.e.E1", sg3(v["change"])); put("r3.e.E1.rlo", sg3(v["rand"][0])); put("r3.e.E1.rhi", sg3(v["rand"][1]))
for g in ("CARRY", "TERM"):
    v = TX["E2"]["values"][g]; put(f"r3.e.E2.{g}", sg3(v["change"])); put(f"r3.e.E2.{g}.rlo", sg3(v["rand"][0])); put(f"r3.e.E2.{g}.rhi", sg3(v["rand"][1]))
    v = TX["E3"]["values"][g]; put(f"r3.e.E3.{g}.without", sg3(v["without"])); put(f"r3.e.E3.{g}.diff", sg3(v["diff"]))
v = TX["E4"]["values"]; put("r3.e.E4.cU", sg3(v["cU"])); put("r3.e.E4.gU", sg3(v["gU"]))
v = TX["E5"]["values"]; put("r3.e.E5", sg3(v["change"])); put("r3.e.E5.rlo", sg3(v["rand"][0])); put("r3.e.E5.rhi", sg3(v["rand"][1]))
v = TX["E6"]["values"]
put("r3.e.E6.nec", f3(v["L11H9"]["nec"])); put("r3.e.E6.rnec", f3(v["L11H9"]["rand_nec_max"]))
put("r3.e.E6.L17H6.nec", f3(v["L17H6"]["nec"])); put("r3.e.E6.L17H6.rnec", f3(v["L17H6"]["rand_nec_max"]))
# the gate on the round-3 code (descriptive): Pythia-410M's header item
GS = json.loads((R3D / "outputs" / "gate_summary.json").read_text())
put("r3.gate.p410.head", f3(GS["pythia-410m"]["newcode"]["header_top1"]))
put("r3.gate.npassall", str(sum(all(GS[m]["gate"].values()) for m in GS))); put("r3.gate.nmodels", str(len(GS)))
put("r3.ntests", str(int(OUT["r3.r1.ntests"]) + int(OUT["r3.r2.ntests"]) + int(OUT["r3.x.ntests"]) + int(OUT["r3.e.ntests"])))
put("r3.npass", str(int(OUT["r3.r1.npass"]) + int(OUT["r3.r2.npass"]) + int(OUT["r3.x.npass"]) + int(OUT["r3.e.npass"])))
put("r3.ind.npass", str(int(OUT["r3.r1.npass"]) + int(OUT["r3.r2.npass"]))); put("r3.ind.ntests", str(int(OUT["r3.r1.ntests"]) + int(OUT["r3.r2.ntests"])))
put("r3.xe.npass", str(int(OUT["r3.x.npass"]) + int(OUT["r3.e.npass"]))); put("r3.xe.ntests", str(int(OUT["r3.x.ntests"]) + int(OUT["r3.e.ntests"])))

# ---------------------------------------------------------------- dx: totals across the studies
def ni(k): return int(OUT[k].replace(",", "").replace("{,}", ""))


studies = {  # (tests, passed) keys per round, as each source paper names them
    "d3r1": ("d3.c.ntests", "d3.c.npass"), "d3r2": ("d3.c2.ntests", "d3.c2.npass"),
    "d4r1": ("d4.c.ntests", "d4.c.npass"), "d4r2": ("d4.r2.ntests", "d4.r2.npass"),
    "d5r1": ("d5.c.ntests", "d5.c.npass"), "d5r2": ("d5.r2.ntests", "d5.r2.npass"),
    "x1": ("x1m.ntests", "x1m.npass"),
}
tot_t = tot_p = 0
for s, (kt, kp) in studies.items():
    tot_t += ni(kt); tot_p += ni(kp)
put("dx.ntests", f"{tot_t:,}"); put("dx.npass", f"{tot_p:,}"); put("dx.nfail", f"{tot_t - tot_p:,}")

# ---------------------------------------------------------------- write
def esc(v): return v.replace("%", "\\%")


lines = ["% Generated by scripts/make_numbers.py from the studies' saved outputs. Do not edit.",
         "% Use in the text as \\nv{key}; an unknown key stops the build."]
for k in sorted(OUT):
    lines.append(f"\\expandafter\\def\\csname nv@{k}\\endcsname{{{OUT[k]}}}")
(PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
(PAPER / "checks").mkdir(exist_ok=True)
(PAPER / "checks" / "numbers.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
print(f"{len(OUT)} values written ({sum(k.startswith('d3.') for k in OUT)} d3, {sum(k.startswith('d4.') for k in OUT)} d4, "
      f"{sum(k.startswith('d5.') for k in OUT)} d5, {sum(k.startswith('x1m.') for k in OUT)} x1m, {sum(k.startswith('dx.') for k in OUT)} dx, {sum(k.startswith('r3.') for k in OUT)} r3)")
