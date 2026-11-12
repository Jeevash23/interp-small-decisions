"""Every number in the D3 write-up (v2: rounds 1 and 2), generated from the study's saved outputs (read only).

  python3 -B scripts/make_numbers.py

Reads, under ../../ (the D3 study folder): data/*/ (counts and summaries), FROZEN.md, outputs/gate_summary_*.json,
outputs/v1-before-2230-fix/gate_summary.json, outputs/e01_pythia-410m_newcode.jsonl (first-indent split),
exploration JSON and arrays (e02, e03, e04, e05, e06, e07b, e08, e09, e10), and the confirmation summaries
outputs/conf/c01_*.json, c02_gen.json, verdicts.json. Writes numbers.tex (macros used as \\nv{key}),
fig-resid.dat and checks/numbers.json. scripts/check_numbers.py recomputes these values without this code.
Round 2 (added for v2) reads outputs/r2/ (exploration JSON/NPZ, calibration records, the r2e08 checkpoint scan,
gate_olmo1b.json), outputs/e02_olmo1b_newcode_160.json, the confirmation summary outputs/r2/measures_conf.json
and outputs/r2/conf/verdicts.json, FROZEN.md, and X1's verdicts (../X1-shared-parts/outputs/conf/verdicts.json).
D3_R2_MEASURES=calib (drafting only) reads measures_calib.json in place of measures_conf.json.
"""
import json, re
from pathlib import Path
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
D3 = PAPER.parents[1]
O = D3 / "outputs"; C = O / "conf"
MINUS = "\u2212"
V = {}


def J(p): return json.loads(Path(p).read_text())


def sgn(s): return MINUS + s[1:] if s.startswith("-") else s


def f3(x): return sgn(f"{x:.3f}")


def f2(x): return sgn(f"{x:.2f}")
def f4(x): return sgn(f"{x:.4f}")


def pf3(x): return ("+" if x >= 0 else "") + f3(x)


def ni(x): return f"{int(x):,}"


def put(k, v):
    assert k not in V, k
    V[k] = v


# ---------------- material and freezes
for part in ("explore", "conf"):
    put(f"mat.gen.{part}", ni(sum(1 for _ in open(D3 / "data" / "gen" / f"part-{part}.jsonl"))))
fs = J(D3 / "data" / "newcode" / "fetch_summary.json"); bs = J(D3 / "data" / "newcode" / "build_summary.json")
put("mat.nc.files", ni(fs["files"])); put("mat.nc.repos", ni(fs["repos_used"])); put("mat.nc.listed", ni(fs["repos_listed"]))
put("mat.nc.mb", f"{fs['bytes'] / 1e6:.1f}"); put("mat.nc.norm", ni(bs["normalised"])); put("mat.nc.chunks", ni(bs["chunks"]))
for part in ("explore", "conf"):
    put(f"mat.nc.{part}.chunks", ni(bs[f"{part}_chunks"])); put(f"mat.nc.{part}.repos", ni(bs[f"{part}_repos"]))
fz = (D3 / "FROZEN.md").read_text()
put("freeze.charter", re.search(r"\| CHARTER\.md \| [0-9a-f]{64} \| 2026-11-07 (\d\d:\d\d)", fz).group(1))
put("freeze.protocol", re.search(r"\| PROTOCOL\.md \| [0-9a-f]{64} \| 2026-11-08 (\d\d:\d\d)", fz).group(1))

# ---------------- gate (exploration parts)
GM = {"pythia-410m": "py", "qwen3-06b": "qw", "gemma4-e2b": "ge", "gpt2": "g2"}
for tag, path in (("post", O / "gate_summary_postfix_final.json"), ("pre", O / "v1-before-2230-fix" / "gate_summary.json")):
    G = J(path)
    for m, k in GM.items():
        if m not in G or not G[m].get("newcode"): continue
        s = G[m]["newcode"]; p = f"gate.{tag}.{k}"
        put(f"{p}.n", ni(s["n"])); put(f"{p}.head", f3(s["header_top1"])); put(f"{p}.termt", f3(s["terminal_top1"]))
        put(f"{p}.terms", f3(s["terminal_shopen"])); put(f"{p}.top1", f3(s["top1"])); put(f"{p}.copy", f3(s["copy"]))
        put(f"{p}.marg", f3(s["top1"] - s["copy"])); put(f"{p}.other", f3(s["other"]))
        ol = [G[m].get(f"gen_w{w}") for w in ("2", "3", "mixed")]
        if all(ol): put(f"{p}.open", f3(max(o["dedent_shnotopen"] for o in ol)))
G = J(O / "gate_summary_postfix_final.json")
put("gate.post.py.same", f3(G["pythia-410m"]["newcode"]["same_top1"])); put("gate.post.qw.same", f3(G["qwen3-06b"]["newcode"]["same_top1"]))
put("gate.post.qw.ntermpts", ni(G["qwen3-06b"]["newcode"]["n_terminal"])); put("gate.post.qw.nheadpts", ni(G["qwen3-06b"]["newcode"]["n_header"]))
put("gate.post.qw.ded", f3(G["qwen3-06b"]["newcode"]["dedent_top1"])); put("gate.post.qw.ntded", f3(G["qwen3-06b"]["newcode"]["nonterm_dedent_top1"]))
put("gate.v1.py.termpts", ni(J(O / "v1-before-2230-fix" / "gate_summary.json")["pythia-410m"]["newcode"]["n_terminal"]))
# Pythia's first indented line of a chunk (post-fix per-point records)
xs = [json.loads(l) for l in open(O / "e01_pythia-410m_newcode.jsonl") if '"mass"' in l]
seen, first, rest = set(), [], []
for x in xs:
    if x["header"]: (first if (x["prev"] == 0 and x["chunk"] not in seen) else rest).append(x)
    if x["true"] > 0: seen.add(x["chunk"])
put("py.first.n", ni(len(first))); put("py.first.top1", f3(np.mean([x["top1"] for x in first])))
put("py.rest.n", ni(len(rest))); put("py.rest.top1", f3(np.mean([x["top1"] for x in rest])))
q = lambda x, k: x["mass"][x["K"].index(k)] / sum(x["mass"])
put("py.first.q4", f2(np.mean([q(x, 4) for x in first]))); put("py.first.q8", f2(np.mean([q(x, 8) for x in first])))

# ---------------- exploration (Qwen3-0.6B-Base unless stated)
e3 = lambda cls, cat, l, h: float(np.load(O / f"e03_qwen3-06b_newcode_{cls}_{cat}.npy")[l, h])
put("x.att.L20H14.term.open", f2(e3("term", "openstarts", 20, 14))); put("x.att.L19H3.term.open", f2(e3("term", "openstarts", 19, 3)))
put("x.att.L22H9.term.open", f2(e3("term", "openstarts", 22, 9))); put("x.att.L22H9.same.prev", f2(e3("same", "prevstart", 22, 9)))
put("x.att.L1H0.head.word", f2(e3("header", "prevword", 1, 0))); put("x.att.L5H6.term.word", f2(e3("term", "prevword", 5, 6)))
e2 = J(O / "e02_qwen3-06b-bf16_newcode_160.json")
comp = {k: v for k, v in e2.items() if k not in ("base", "n")}
put("x.e2.ncomp", ni(len(comp))); put("x.e2.maxterm", f3(max(-v["term"] for v in comp.values())))
put("x.e2.L23H2.ded", f3(comp["L23H2"]["dedent"]))
e2f = J(O / "e02_qwen3-06b_newcode_160.json")  # float32, first 103 components
both = [k for k in e2f if k in e2 and k not in ("base", "n")]
put("x.bf16.ncomp", ni(len(both)))
put("x.bf16.r", f3(min(float(np.corrcoef([e2f[k][m] for k in both], [e2[k][m] for k in both])[0, 1]) for m in ("header", "same", "term", "dedent", "term_shopen"))))
put("x.bf16.base", f3(max(abs(e2f["base"][m] - e2["base"][m]) for m in e2["base"])))
e6 = J(O / "e06_qwen3-06b-bf16_newcode_120_OPEN_PREV_KW_DED.json")
b6, d6 = e6["base"], e6["groups"]["DED"]
r6 = [r["lq"]["dedent"] for r in d6["random"]]
put("x.e6.base.ded", f3(b6["lq"]["dedent"])); put("x.e6.ded.ded", f3(d6["obs"]["lq"]["dedent"]))
put("x.e6.rand.lo", f3(min(r6))); put("x.e6.rand.hi", f3(max(r6)))
put("x.e6.base.psame", f3(b6["p_same"]["dedent"])); put("x.e6.ded.psame", f3(d6["obs"]["p_same"]["dedent"]))
put("x.e6.ded.termopen", f3(d6["obs"]["p_open"]["term"])); put("x.e6.n.ded", ni(e6["n"]["dedent"]))
e6b = J(O / "e06_qwen3-06b-bf16_newcode_120_DEDno23H2_DEDno17H3_DEDno17H4_DEDno17H5_DEDno18H6.json")
put("x.loo.no23", f3(e6b["groups"]["DEDno23H2"]["obs"]["lq"]["dedent"]))
others = [e6b["groups"][g]["obs"]["lq"]["dedent"] for g in ("DEDno17H3", "DEDno17H4", "DEDno17H5", "DEDno18H6")]
put("x.loo.lo", f3(min(others))); put("x.loo.hi", f3(max(others)))
e4 = J(O / "e04_qwen3-06b-bf16_120.json")
put("x.e4.gap", f2(e4["gap"])); put("x.e4.n", ni(e4["n"]))
for l in (4, 5, 16, 17, 20): put(f"x.e4.r{l}.suf", f2(e4["comps"][f"resid{l}"]["suf"])); put(f"x.e4.r{l}.nec", f2(e4["comps"][f"resid{l}"]["nec"]))
put("x.e4.L17H4.suf", f3(e4["comps"]["L17H4"]["suf"])); put("x.e4.L17H4.nec", f3(e4["comps"]["L17H4"]["nec"]))
with open(PAPER / "fig-resid.dat", "w") as fh:
    fh.write("layer suf nec\n")
    for l in range(28): fh.write(f"{l} {e4['comps'][f'resid{l}']['suf']:.4f} {e4['comps'][f'resid{l}']['nec']:.4f}\n")
e8 = J(O / "e08_qwen3-06b-bf16_120_TOP6_TOP12_EARLY_OPEN.json")["comps"]
put("x.e8.top6.nec", f3(e8["TOP6"]["nec"])); put("x.e8.top6.suf", f3(e8["TOP6"]["suf"]))
put("x.e8.top6.rnec", f3(max(e8["TOP6"]["rand_nec"]))); put("x.e8.top6.rsuf", f3(max(e8["TOP6"]["rand_suf"])))
e7 = J(O / "e07b_qwen3-06b-bf16_newcode_250.json")
for cls in ("term", "dedent"):
    rows = {r[0]: r for r in e7[cls]["top"] + e7[cls]["bottom"]}
    for h in ("L23H2", "L22H8"):
        put(f"x.e7.{cls}.{h}.t", f2(rows[h][2])); put(f"x.e7.{cls}.{h}.c", f2(rows[h][3]))
    put(f"x.e7.{cls}.mlp27", f2(rows["mlp27"][1]))
e5 = J(O / "e05_qwen3-06b_newcode_5005.json")
put("x.e5.n", ni(e5["n"])); put("x.e5.prev3", f2(e5["3"]["prev"])); put("x.e5.cls0", f2(e5["0"]["cls_acc"]))
put("x.e5.ndedmax", f2(max(e5[str(l)]["nded"] for l in range(29)))); put("x.e5.ndedarg", str(max(range(29), key=lambda l: e5[str(l)]["nded"])))
put("x.e5.true28", f2(e5["28"]["true"]))
e9 = J(O / "e09_qwen3-06b-bf16_250_READ.json")["summary"]
put("x.e9.n", ni(e9["base"]["n"]))
for w in ("return", "pass"):
    put(f"x.e9.{w}.kw", f3(e9[f"kw:{w}"]["mean"])); put(f"x.e9.{w}.ctl", f3(e9[f"ctl:{w}"]["mean"])); put(f"x.e9.{w}.real", f3(e9[f"real:{w}"]["mean"]))
put("x.e9.strret", pf3(e9["strdiff:return"]["mean"]))
ratios = []
for f in ("e09_qwen3-06b-bf16_250_READ.json", "e09_pythia-410m_250.json", "e09_gemma4-e2b_250.json"):
    s = J(O / f)["summary"]
    ratios += [s[f"diff:{w}"]["mean"] / s[f"real:{w}"]["mean"] for w in ("return", "pass", "break", "continue", "raise")]
put("x.e9.maxratio", f2(max(ratios)))
e10 = J(O / "e10_qwen3-06b-bf16_H25_STEP_WRITE_MLP26_MLPS.json")
b10 = e10["base"]; h25 = e10["groups"]["H25"]["obs"]
for w in ("2", "4"): put(f"x.e10.h25.ded.w{w}", pf3(h25[f"dedent|{w}"]["lq"] - b10[f"dedent|{w}"]["lq"]))
put("x.e10.h25.ded.nat", pf3(h25["dedent|nat"]["lq"] - b10["dedent|nat"]["lq"]))
e6g = J(O / "e06_qwen3-06b-bf16_gen_200_DED_TERM.json")
put("x.e6g.ded.term", f3(e6g["groups"]["DED"]["obs"]["lq"]["term"] - e6g["base"]["lq"]["term"]))
put("x.e6g.ded.termr", f3(min(r["lq"]["term"] for r in e6g["groups"]["DED"]["random"]) - e6g["base"]["lq"]["term"]))

# ---------------- confirmation (held out)
n = J(C / "c01_newcode.json"); g = J(C / "c01_gen.json"); p = J(C / "c02_gen.json"); VV = J(C / "verdicts.json")
vt = {v["id"]: v for v in VV["verdicts"]}
d = lambda r, grp, meas, c: r[grp][meas][c] - r["base"][meas][c]
rlo = lambda r, grp, meas, c: min(v[meas][c] for k, v in r.items() if k.startswith(grp + "_rand")) - r["base"][meas][c]
rhi = lambda r, grp, meas, c: max(v[meas][c] for k, v in r.items() if k.startswith(grp + "_rand")) - r["base"][meas][c]
put("c.nchunks.nc", ni(n["n_chunks_used"])); put("c.nchunks.gen", ni(g["n_chunks_used"]))
for src, r in (("nc", n), ("gen", g)):
    for cl in ("header", "same", "term", "dedent"):
        put(f"c.n.{src}.{cl}", ni(r["n_points"][cl])); put(f"c.base.{src}.{cl}", f3(r["base"]["lq"][cl]))
    put(f"c.base.{src}.psame", f3(r["base"]["p_same"]["dedent"]))
put("c.H1", f3(d(n, "DED", "lq", "dedent"))); put("c.H1.r", f3(rlo(n, "DED", "lq", "dedent")))
put("c.H1.abs", f2(-d(n, "DED", "lq", "dedent")))
put("c.H2", pf3(d(n, "DED", "p_same", "dedent"))); put("c.H2.r", pf3(rhi(n, "DED", "p_same", "dedent")))
put("c.H2.after", f3(n["DED"]["p_same"]["dedent"]))
put("c.H3.open", f3(n["DED"]["p_open"]["term"])); put("c.H3.d", f3(d(n, "DED", "lq", "term")))
put("c.H3.ratio", f3(abs(d(n, "DED", "lq", "term")) / abs(d(n, "DED", "lq", "dedent"))))
put("c.H4", pf3(d(n, "DED", "lq", "same")))
put("c.H5", f3(d(n, "TERM", "lq", "term"))); put("c.H5.r", f3(rlo(n, "TERM", "lq", "term")))
put("c.H6", f3(d(n, "TERM", "lq", "dedent")))
put("c.H7", f3(d(n, "TERMDED", "lq", "term"))); put("c.H7.sum", f3(d(n, "TERM", "lq", "term") + d(n, "DED", "lq", "term")))
put("c.H7.psame", f3(n["TERMDED"]["p_same"]["term"]))
put("c.H8", f3(d(g, "DED", "lq", "dedent"))); put("c.H8.r", f3(rlo(g, "DED", "lq", "dedent")))
cm = p["comps"]
put("c.H9.nec", f3(cm["CARRY"]["nec"])); put("c.H9.suf", f3(cm["CARRY"]["suf"]))
put("c.H9.rnec", f3(max(v["nec"] for k, v in cm.items() if k.startswith("CARRY_rand"))))
put("c.H9.rsuf", f3(max(v["suf"] for k, v in cm.items() if k.startswith("CARRY_rand"))))
put("c.pairs", ni(p["n"])); put("c.gap", f2(p["gap"]))
put("c.dedpairs.nec", f3(cm["DED"]["nec"])); put("c.dedpairs.suf", f3(cm["DED"]["suf"]))
for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"):
    lo, hi = vt[h]["ci95"]; put(f"c.{h}.lo", f3(lo)); put(f"c.{h}.hi", f3(hi))
put("c.npass", str(sum(v["verdict"] == "pass" for v in VV["verdicts"]))); put("c.ntests", str(len(VV["verdicts"])))


# ================= ROUND 2 (v2) =================
import os
O2 = O / "r2"


def pf2(x): return ("+" if x >= 0 else "") + f2(x)


def fk(k, x):  # changes carry a sign; shares and attention do not
    if k.endswith("_dlq"): return pf2(x)
    if k.startswith(("ko_", "d_lq", "d_psame", "l0h2", "rand_d")): return pf3(x)
    return f3(x)


fz = (D3 / "FROZEN.md").read_text()
put("freeze.charter2", re.search(r"\| CHARTER-2\.md \| [0-9a-f]{64} \| 2026-11-08 (\d\d:\d\d)", fz).group(1))
put("freeze.protocol2", re.search(r"\| PROTOCOL-2\.md \| [0-9a-f]{64} \| 2026-11-09 (\d\d:\d\d)", fz).group(1))
cnt = lambda f: sum(1 for _ in open(f))
put("mat2.code.chunks", ni(cnt(D3 / "data" / "r2code" / "part-held.chunks.jsonl")))
put("mat2.code.repos", ni(len({json.loads(l).get("repo") for l in open(D3 / "data" / "r2code" / "part-held.chunks.jsonl")})))
put("mat2.gen2.held", ni(cnt(D3 / "data" / "r2gen2" / "part-held.jsonl"))); put("mat2.gen2.explore", ni(cnt(D3 / "data" / "r2gen2" / "part-explore.jsonl")))
# OLMo-2 gate (round 1's rules)
go = J(O2 / "gate_olmo1b.json")["olmo1b"]; s = go["newcode"]
put("gate.post.ol.n", ni(s["n"])); put("gate.post.ol.head", f3(s["header_top1"])); put("gate.post.ol.termt", f3(s["terminal_top1"]))
put("gate.post.ol.terms", f3(s["terminal_shopen"])); put("gate.post.ol.marg", f3(s["top1"] - s["copy"]))
put("gate.post.ol.open", f3(max(go[f"gen_w{w}"]["dedent_shnotopen"] for w in ("2", "3", "mixed"))))
# r2e07 head scans (x1code, 400 chunks)
for mk, f, hd in (("q", "r2e07_qwen3-06b-bf16_x1code_400.npz", (23, 2)), ("o", "r2e07_olmo1b_x1code_400.npz", (13, 1))):
    z = np.load(O2 / f)
    put(f"x2.scan.{mk}.ded", f3(z["a_ded"][hd])); put(f"x2.scan.{mk}.same", f3(z["a_same"][hd])); put(f"x2.scan.{mk}.pick", f3(z["pick"][hd]))
    a = z["a_ded"].copy(); a[hd] = -1; l, h = np.unravel_index(np.argmax(a), a.shape)
    put(f"x2.scan.{mk}.next", f"L{l}H{h}"); put(f"x2.scan.{mk}.nextded", f3(a[l, h]))
    pk = z["pick"].copy(); pk[hd] = -1; l, h = np.unravel_index(np.argmax(pk), pk.shape)
    put(f"x2.scan.{mk}.nextpick", f"L{l}H{h}"); put(f"x2.scan.{mk}.nextpickv", f3(pk[l, h]))
# r2e02 knock-outs (Qwen3, x1code 150)
rec = J(O2 / "r2e02_x1code_150.json"); D = [r for r in rec if r["cls"] == "dedent"]
dm = lambda k: float(np.mean([r[k][0] - r["base"][0] for r in D]))
put("x2.ko.n", ni(len(D))); put("x2.ko.tok", f3(dm("ko23_encl"))); put("x2.ko.line", f3(dm("ko23_encl_line")))
put("x2.ko.ded5", f3(dm("koDED_encl_line"))); put("x2.ko.abl17", f3(dm("ablate17")))
put("x2.ko.att", f3(np.mean([r["a23_encl"] for r in D if r["a23_encl"] is not None])))
put("x2.ko.att17", f3(np.mean([r["a23_encl_abl17"] for r in D if r["a23_encl_abl17"] is not None])))
ovs = [int(fav == ind) for r in D for ind, att, fav in r["open_hdr"]]; put("x2.ov.ded", f2(np.mean(ovs)))
# r2e06 query composition (largest drops in L23H2's attention to the enclosing ':\n' at unmarked dedents)
q6 = J(O2 / "r2e06_qwen3-06b-bf16_L23H2_x1code_80.json"); b6 = q6["base"]["a_ded"]
drops = sorted(((v["a_ded"] - b6, k) for k, v in q6["comps"].items() if k != "base"))
put("x2.qc.base", f3(b6)); put("x2.qc.top1", drops[0][1].replace("mlp", "MLP")); put("x2.qc.top1v", f3(drops[0][0]))
put("x2.qc.top2", drops[1][1].replace("mlp", "MLP")); put("x2.qc.top2v", f3(drops[1][0]))
# r2e04 keyword readers (gen2 terminal, 250)
k4 = J(O2 / "r2e04_terminal_250.json")
R4 = [r for r in k4["rec"] if r["kw"] == "return" and r.get("neutral")]
sh4 = lambda k: (np.mean([r[k][0] for r in R4]) - np.mean([r["base"][0] for r in R4])) / (np.mean([r["neutral"][0] for r in R4]) - np.mean([r["base"][0] for r in R4]))
put("x2.kw.n", ni(len(R4))); put("x2.kw.top8", f2(sh4("ko_KWG"))); put("x2.kw.blk", f2(sh4("blk_all")))
r4 = [sh4(f"ko_rand{k}") for k in range(8)]; put("x2.kw.randlo", f2(min(r4))); put("x2.kw.randhi", f2(max(r4)))
put("x2.kw.L1H0att", f2(dict(k4["att_top"])["L1H0"]))
# OLMo-2 single ablations (round 1's e02 on round-1 exploration code)
e2o = J(O / "e02_olmo1b_newcode_160.json")
for h in ("L13H1", "L10H12", "L11H0", "L15H4"): put(f"x2.oe2.{h}", f3(e2o[h]["dedent"])); put(f"x2.oe2.{h}.t", f3(e2o[h]["term"]))
put("x2.oe2.maxheadterm", f3(max(-v["term"] for k, v in e2o.items() if k.startswith("L"))))
# calibration runs (exploration material, same scripts as the confirmation)
MC = J(O2 / "measures_calib.json")
for mk, key in (("q", "qwen3-06b-bf16/L23H2"), ("o", "olmo1b/L13H1")):
    N, G2 = MC[key + "/natural"], MC[key + "/gen2"]; p_ = f"cal.{mk}"
    for k in ("att_inner_ded", "att_inner_same", "ko_inner_ded", "ko_closed_ded", "ko_inner_ded_closedsub", "ko_true_ded",
              "pick_not_inner", "pick_all_multi", "innermost_rule_all_multi", "steer_dedent_head_top", "steer_dedent_ctrl_top"):
        put(f"{p_}.{k}", fk(k, N[k]))
    put(f"{p_}.steer_dedent_head_dlq", pf2(N["steer_dedent_head_dlq"])); put(f"{p_}.steer_dedent_ctrl_dlq", pf2(N["steer_dedent_ctrl_dlq"]))
    put(f"{p_}.others_lo", pf3(min(N["ko_inner_others_ded"].values()))); put(f"{p_}.others_hi", pf3(max(N["ko_inner_others_ded"].values())))
    for k in ("share_blk_kw_return", "share_blk_kw_raise", "share_blk_kw_break", "share_blk_kw_continue", "share_blk_kw_pass", "share_layer_return_max",
              "l0h2_line_dqcur_term", "l0h2_all_dqcur_term", "steer_term_head_top", "steer_term_ctrl_top"):
        put(f"{p_}.{k}", fk(k, G2[k]))
    put(f"{p_}.steer_term_head_dlq", pf2(G2["steer_term_head_dlq"])); put(f"{p_}.steer_term_ctrl_dlq", pf2(G2["steer_term_ctrl_dlq"]))
put("cal.q.n.chunks", ni(MC["qwen3-06b-bf16/L23H2/natural"]["n"]["chunks"])); put("cal.q.n.repos", ni(MC["qwen3-06b-bf16/L23H2/natural"]["n"]["repos"]))
put("cal.qg.n.programs", ni(MC["qwen3-06b-bf16/L23H2/gen2"]["n"]["programs"]))
put("cal.q.qcur_base_term", f3(MC["qwen3-06b-bf16/L23H2/gen2"]["qcur_base_term"])); put("cal.q.qcur_neutral_term", f3(MC["qwen3-06b-bf16/L23H2/gen2"]["qcur_neutral_term"]))
put("cal.g.n.chunks", "300")  # r2c03 --calib 300 (round-1 code exploration part); also in the run log name
GC = MC["olmo1b/DEDO/group"]
for k in ("d_lq_dedent", "d_lq_term", "d_lq_same", "rand_d_lq_dedent_min", "rand_d_lq_dedent_max", "d_psame_dedent", "p_open_term_ablated"):
    put(f"cal.g.{k}", fk(k, GC[k]))
# question 5: OLMo-2 checkpoints (r2e08, x1code 200 chunks)
CK = {json.loads(l)["model"]: json.loads(l) for l in open(O2 / "r2e08_200.jsonl")}
put("ck.n", ni(next(iter(CK.values()))["chunks"]))
CKT = {"1B": "olmo1b-1B", "21B": "olmo1b-21B", "84B": "olmo1b-84B", "420B": "olmo1b-420B", "1007B": "olmo1b-1007B",
       "4001B": "olmo1b-4001B", "iax3B": "olmo1b-s2i1-3B", "iax51B": "olmo1b-s2i1-51B", "ibx51B": "olmo1b-s2i2-51B",
       "icx51B": "olmo1b-s2i3-51B", "final": "olmo1b"}
for tag, mname in CKT.items():
    if mname not in CK: continue
    c = CK[mname]; b = c["behaviour"]; p_ = f"ck.{tag}"; z = np.load(O2 / f"r2e08_{mname}_200.npz")
    put(f"{p_}.ded", f3(b["dedent"]["top1"])); put(f"{p_}.term", f3(b["term"]["top1"])); put(f"{p_}.same", f3(b["same"]["top1"]))
    for (l, h) in ((11, 9), (13, 1), (15, 9)):
        put(f"{p_}.L{l}H{h}.att", f2(z["a_ded"][l, h])); put(f"{p_}.L{l}H{h}.pick", f2(z["pick"][l, h]))
    put(f"{p_}.best", c["best_ded_minus_same"])
# weight distances between checkpoints (r2e10b, float64)
DIST = json.loads((O2 / "r2e10b_dist.json").read_text())
for pk, nm in (("final~s1_4001B", "fin_s1end"), ("s1_4001B~s2i1_51B", "s1end_ia51"), ("s2i1_51B~s2i2_51B", "ia51_ib51"), ("s1_1007B~s1_4001B", "s1mid_s1end")):
    put(f"ck.dist.{nm}.rel", f3(DIST[pk]["rel_dist"])); put(f"ck.dist.{nm}.emb", f4(DIST[pk]["emb_cosine"]) if DIST[pk]["emb_cosine"] > 0.99 else f3(DIST[pk]["emb_cosine"]))
# X1 (cross-study test), its frozen verdict values
X1 = J(D3.parent / "X1-shared-parts" / "outputs" / "conf" / "verdicts.json")["tests"]
put("x1.L17H6.width", f3(X1["A1"]["values"]["kept"])); put("x1.L17H6.term", f3(X1["A3"]["values"]["change"]))
put("x1.L17H6.termr", f3(X1["A3"]["values"]["rand"][0]))
put("x1.carry.width", f3(X1["A4"]["values"]["CARRY"])); put("x1.carrynoL17H6.width", f3(X1["A4"]["values"]["CARRY-L17H6"]))
put("x1.term.width", f3(X1["A4"]["values"]["TERM"])); put("x1.termnoL17H6.width", f3(X1["A4"]["values"]["TERM-L17H6"]))
put("x1.npass", str(sum(v["passed"] for v in X1.values()))); put("x1.ntests", str(len(X1)))
put("x1.L11H9.ind", f3(max(abs(x) for x in X1["D3"]["values"].values())))
put("x1.ded.lbw", f3(X1["B4"]["values"]["LBw"])); put("x1.ded.lbs", f3(X1["B4"]["values"]["LBs"]))
# confirmation (held out)
MM = J(O2 / ("measures_calib.json" if os.environ.get("D3_R2_MEASURES") == "calib" else "measures_conf.json"))
for mk, key in (("q", "qwen3-06b-bf16/L23H2"), ("o", "olmo1b/L13H1")):
    N, G2 = MM[key + "/natural"], MM[key + "/gen2"]; p_ = f"c2.{mk}"
    for k in ("att_inner_ded", "att_inner_same", "att_inner_term", "ko_inner_ded", "ko_inner_same", "ko_inner_term", "ko_closed_ded", "ko_inner_ded_closedsub",
              "ko_true_ded", "pick_not_inner", "pick_inner_attended_not_inner", "pick_all_multi", "innermost_rule_all_multi", "steer_dedent_head_top",
              "steer_dedent_ctrl_top", "steer_term_head_top", "steer_term_ctrl_top", "base_lq_ded"):
        put(f"{p_}.{k}", fk(k, N[k]))
    for k in ("steer_dedent_head_dlq", "steer_dedent_ctrl_dlq", "steer_term_head_dlq", "steer_term_ctrl_dlq"): put(f"{p_}.{k}", fk(k, N[k]))
    put(f"{p_}.others_lo", pf3(min(N["ko_inner_others_ded"].values()))); put(f"{p_}.others_hi", pf3(max(N["ko_inner_others_ded"].values())))
    put(f"{p_}.others_maxabs", f3(N["ko_inner_others_maxabs"]))
    for k, v in N["n"].items(): put(f"{p_}.n.{k}", ni(v))
    for k in ("ko_inner_ded", "att_inner_ded", "pick_not_inner"): put(f"{p_}.ci.{k}.lo", f3(N["ci"][k][0])); put(f"{p_}.ci.{k}.hi", f3(N["ci"][k][1]))
    pg = f"c2.{mk}g"
    for k in ("share_blk_kw_return", "share_blk_kw_after_return", "share_blk_kw_raise", "share_blk_kw_break", "share_blk_kw_continue", "share_blk_kw_pass",
              "share_blk_kw_all", "share_layer_return_max", "l0h2_line_dqcur_term", "l0h2_all_dqcur_term", "qcur_base_term", "qcur_neutral_term",
              "steer_term_head_top", "steer_term_ctrl_top", "steer_plain_head_top", "steer_plain_ctrl_top"):
        put(f"{pg}.{k}", fk(k, G2[k]))
    for k in ("steer_term_head_dlq", "steer_term_ctrl_dlq", "steer_plain_head_dlq", "steer_plain_ctrl_dlq"): put(f"{pg}.{k}", fk(k, G2[k]))
    sl = G2["share_layer_return"]; put(f"{pg}.layer_return_arg", str(int(np.nanargmax([x if x is not None else np.nan for x in sl]))))
    for k, v in G2["n"].items(): put(f"{pg}.n.{k}", ni(v))
    put(f"{pg}.ci.share.lo", f3(G2["ci"]["share_blk_kw_return"][0])); put(f"{pg}.ci.share.hi", f3(G2["ci"]["share_blk_kw_return"][1]))
    put(f"{pg}.ci.l0h2.lo", f3(G2["ci"]["l0h2_line_dqcur_term"][0])); put(f"{pg}.ci.l0h2.hi", f3(G2["ci"]["l0h2_line_dqcur_term"][1]))
GG = MM["olmo1b/DEDO/group"]
for k in ("d_lq_dedent", "d_lq_term", "d_lq_same", "d_lq_header", "rand_d_lq_dedent_min", "rand_d_lq_dedent_max", "rand_d_lq_term_min", "d_psame_dedent",
          "p_open_term_ablated", "base_lq_dedent", "base_lq_term"):
    put(f"c2.g.{k}", fk(k, GG[k]))
put("c2.g.ratio", f3(abs(GG["d_lq_term"]) / abs(GG["d_lq_dedent"])))
for k, v in GG["n"].items(): put(f"c2.g.n.{k}", ni(v))
put("c2.g.ci.lo", f3(GG["ci"]["d_lq_dedent"][0])); put("c2.g.ci.hi", f3(GG["ci"]["d_lq_dedent"][1]))
if os.environ.get("D3_R2_MEASURES") != "calib":
    V2 = J(O2 / "conf" / "verdicts.json")
    put("c2.npass", str(sum(V2["predictions"].values()))); put("c2.ntests", str(len(V2["predictions"])))
    for k, v in V2["predictions"].items(): put(f"c2.v.{k}", "pass" if v else "fail")

(PAPER / "numbers.tex").write_text("% Generated by scripts/make_numbers.py from saved outputs. Do not edit.\n% Use in the text as \\nv{key}; an unknown key stops the build.\n"
                                  + "".join(f"\\expandafter\\def\\csname nv@{k}\\endcsname{{{v}}}\n" for k, v in sorted(V.items())))
(PAPER / "checks").mkdir(exist_ok=True)
(PAPER / "checks" / "numbers.json").write_text(json.dumps(V, indent=1, ensure_ascii=False))
print(len(V), "numbers")
