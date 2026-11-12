"""D3 round 3: build the result sections of REPORT-3.md from saved outputs only. No model is loaded.
New (round 3): round3/outputs/conf/verdicts.json (c03), round3/outputs/r2/measures_conf.json and r2/conf/verdicts.json
(r2c04), round3/outputs/conf/verdicts_XE.json and the raw run files it reads (r3_verdicts), round3/outputs/gate_summary.json.
Original (for comparison): D3-indentation/outputs/conf/verdicts.json, outputs/r2/measures_conf.json, X1-shared-parts/
outputs/conf/c01_qwen3-06b-bf16_IND.json, outputs/gate_summary_postfix_final.json.
The hand-written reading (round3/REPORT-3-reading.md) is appended after the generated sections.
Usage: python r3_build_report.py [--draft]   (--draft: layout test with the original outputs standing in; REPORT-3.draft.md)"""
import json, re, sys
from pathlib import Path

R3 = Path(__file__).resolve().parent.parent; D3 = R3.parent; X1 = D3.parent / "X1-shared-parts"
DRAFT = "--draft" in sys.argv
ld = lambda p: json.loads(Path(p).read_text())
MINUS = "\u2212"
s_ = lambda x: MINUS + x[1:] if x.startswith("-") else x
f3 = lambda x: "–" if x is None else s_(f"{(0.0 if round(x, 3) == 0 else x):.3f}")
pf3 = lambda x: "–" if x is None else ("+" if x >= 0 else "") + f3(x)
pf2 = lambda x: ("+" if x >= 0 else "") + s_(f"{x:.2f}")
ni = lambda x: f"{int(x):,}"
L = []; w = L.append

# ---------------- load
O1 = ld(D3 / "outputs" / "conf" / "verdicts.json")
N1 = O1 if DRAFT else ld(R3 / "outputs" / "conf" / "verdicts.json")
OM2 = ld(D3 / "outputs" / "r2" / "measures_conf.json"); OV2 = ld(D3 / "outputs" / "r2" / "conf" / "verdicts.json")
NM2 = OM2 if DRAFT else ld(R3 / "outputs" / "r2" / "measures_conf.json")
NV2 = OV2 if DRAFT else ld(R3 / "outputs" / "r2" / "conf" / "verdicts.json")
XE = ld(R3 / "outputs" / ("calib/verdicts_XE_calib.json" if DRAFT else "conf/verdicts_XE.json"))
XO = ld(X1 / "outputs" / "conf" / "c01_qwen3-06b-bf16_IND.json")
GO = ld(D3 / "outputs" / "gate_summary_postfix_final.json")
GN = GO if DRAFT else (ld(R3 / "outputs" / "gate_summary.json") if (R3 / "outputs" / "gate_summary.json").exists() else {})
BS = ld(R3 / "data" / "code" / "build_summary.json"); FS = ld(R3 / "data" / "code" / "fetch_summary.json")
# ---------------- provenance guard: every round-3 value must come from a file under round3/ written after the
# PROTOCOL-3.md freeze (its mtime), never from an original (pre-protocol) output.
import os
NEWFILES = [R3 / "outputs" / "conf" / "verdicts.json", R3 / "outputs" / "r2" / "measures_conf.json",
            R3 / "outputs" / "r2" / "conf" / "verdicts.json", R3 / "outputs" / "conf" / "verdicts_XE.json",
            R3 / "outputs" / "gate_summary.json", R3 / "x1" / "outputs" / "conf" / "c01_qwen3-06b-bf16_IND.json",
            R3 / "x1" / "outputs" / "conf" / "r3x_l11h9_IND.json", R3 / "outputs" / "conf" / "r3_pairs.json",
            R3 / "outputs" / "conf" / "c01_newcode.json", R3 / "outputs" / "conf" / "c01_gen.json", R3 / "outputs" / "conf" / "c02_gen.json"]
T_FREEZE = os.stat(D3 / "PROTOCOL-3.md").st_mtime
MISSING = []
if not DRAFT:
    for f in NEWFILES:
        if not f.exists(): MISSING.append(f.relative_to(R3).as_posix()); continue
        assert f.resolve().is_relative_to(R3.resolve()), f"round-3 value read from outside round3/: {f}"
        assert f.stat().st_mtime > T_FREEZE, f"round-3 file older than the PROTOCOL-3.md freeze: {f}"
    for a, b in [(N1, O1), (NM2, OM2)]:
        assert a != b, "a round-3 result equals the original output exactly"
fz = (D3 / "FROZEN.md").read_text()
tch = re.search(r"\| CHARTER-3\.md \| [0-9a-f]{64} \| 2026-11-10 (\d\d:\d\d)", fz).group(1)
tpr = re.search(r"\| PROTOCOL-3\.md \| [0-9a-f]{64} \| 2026-11-10 (\d\d:\d\d)", fz).group(1)

# ---------------- X1 original values on the indentation side (same arithmetic as r3_verdicts)
K = {"cU": "code|dedent|lq", "cT": "code|term|lq", "gU": "gen|dedent|lq", "gT": "gen|term|lq"}
xo = lambda g, k: XO[g][K[k]] - XO["base"][K[k]]
xr = lambda g, k: (min(XO[x][K[k]] - XO["base"][K[k]] for x in XO if x.startswith(g + "~r")),
                   max(XO[x][K[k]] - XO["base"][K[k]] for x in XO if x.startswith(g + "~r")))
T = XE["tests"]
vd = lambda p: "–" if p is None else ("pass" if p else "**fail**")

# ---------------- counts
r1n = [x for x in N1["verdicts"]]; r1p = sum(x["verdict"] == "pass" for x in r1n)
r2p = sum(bool(v) for v in NV2["predictions"].values()); r2n = len(NV2["predictions"])
xk = ["XA3", "XA5", "XB1", "XB2", "XB3", "XB4", "XC3", "XC6", "XD3"]; ek = ["E1", "E2", "E3", "E4", "E5", "E6"]
xp = sum(bool(T[k]["passed"]) for k in xk); ep = sum(bool(T[k]["passed"]) for k in ek)
r1_conf = all(x["verdict"] == "pass" for x in r1n if x["id"] in ("H1", "H2", "H3", "H5", "H6"))
r1_part = all(x["verdict"] == "pass" for x in r1n if x["id"] in ("H1", "H2", "H3"))
r1_verdict = "confirmed" if r1_conf else "partly confirmed" if r1_part else "not confirmed"

w("# D3 round 3 report: the code results of paper B on shareable material, and L11H9's second role")
w("")
w(f"10–11 November 2026. **Confirmation** under `PROTOCOL-3.md` (frozen 10 Nov, {tpr} IST), run once on parts sealed before any model read them, under `CHARTER-3.md` (frozen {tch} IST). Plan and reasons: `ROUND-3-PLAN.md`. The result tables below are generated from saved outputs by `round3/code/r3_build_report.py`; the reading at the end is written by hand from the same outputs. An independent script checks every number (`round3/CHECK-3.md`).")
w("")
w("## Result")
w("")
w(f"- **D3 round 1 (H1–H9): {r1p} of {len(r1n)} tests pass**; by the frozen rule the two-route account is **{r1_verdict}** on the new material.")
w(f"- **D3 round 2 (Q1–Q11, O1–O10, G1–G4): {r2p} of {r2n} tests pass**. " + "; ".join(f"{q[0].upper() + q[1:]}: **{v}**" for q, v in NV2["questions"].items()) + ".")
nr = lambda ks: any(T[k]["passed"] is None for k in ks)
w(f"- **X1, indentation side: {xp} of {len(xk)} tests pass.**" if not nr(xk) else "- **X1, indentation side: not run (or incomplete).**")
w("- **L11H9's second role (part E): not run (or incomplete).**" if XE['verdicts']['E: L11H9 second role'] is None or nr(ek) else f"- **L11H9's second role (part E): {XE['verdicts']['E: L11H9 second role']}** (E1 {vd(T['E1']['passed'])}, E3 {vd(T['E3']['passed'])}; qualifiers: E2 {vd(T['E2']['passed'])}, E4 {vd(T['E4']['passed'])}, E5 {vd(T['E5']['passed'])}, E6 {vd(T['E6']['passed'])}).")
w("")
w("## Material")
w("")
lic = ", ".join(f"{k} {ni(v)}" for k, v in sorted(BS["licences"].items(), key=lambda kv: -kv[1]))
w(f"- **Natural code:** {ni(FS['files'])} Python files ({ni(FS['bytes'])} bytes) fetched on 10 November 2026 from {ni(FS['repos_used'])} GitHub repositories created on or after 1 September 2026 whose licence is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0 or Unlicense ({ni(FS['repos_listed'])} listed, {ni(FS['repos_after_exclusion'])} after removing every repository an earlier part used). Each file is pinned to a commit; licence texts and notices are kept. {ni(BS['normalised'])} files survived normalisation and gave {ni(BS['held_chunks'])} chunks from {ni(BS['held_repos'])} repositories ({ni(BS['dup_old'])} chunks that repeat an earlier part and {ni(BS['dup'])} repeats within the build dropped). Chunks by licence: {lic}.")
w("- **Generated programs:** 2,000 from round 1's generator (seeds 300000–301999) and 1,500 from round 2's targeted generator (seeds 10000–11499).")
w("- All parts were sealed before any model read them. Everything here was run in Qwen3-0.6B-Base and, for round 2's questions 4a and 4b, OLMo-2-0425-1B, both in bfloat16, by byte-identical copies of the frozen scripts.")
w("")

# ---------------- R1
w("## Part R1: D3 round 1 (`PROTOCOL.md`)")
w("")
o1 = {x["id"]: x for x in O1["verdicts"]}
w("| id | prediction | original (8 Nov) | round 3 | round 3, context | verdict |")
w("|---|---|---|---|---|---|")
for x in r1n:
    o = o1[x["id"]]
    fm = f3 if x['id'] == 'H3' else pf3
    w(f"| {x['id']} | {x['claim']} | {fm(o['value'])} | {fm(x['value'])} | {s_(x.get('context') or '') or '–'} | {vd(x['verdict'] == 'pass')} |")
w("")
for k in ("n_points", "base"):
    pass
w("")

# ---------------- R2
def r2rows(N, G, P):
    return [
        (f"{P}1", "attention to innermost open header, unmarked dedents; same-level lines", lambda N, G: f"{f3(N['att_inner_ded'])}; {f3(N['att_inner_same'])}"),
        (f"{P}2", "knock-out of that token, unmarked dedents; range over the other heads of the layer", lambda N, G: f"{pf3(N['ko_inner_ded'])}; {pf3(min(N['ko_inner_others_ded'].values()))} to {pf3(max(N['ko_inner_others_ded'].values()))}"),
        (f"{P}3", "same knock-out, same-level lines; terminal dedents", lambda N, G: f"{pf3(N['ko_inner_same'])}; {pf3(N['ko_inner_term'])}"),
        (f"{P}4", "closed-header knock-out against open-header knock-out", lambda N, G: f"{pf3(N['ko_closed_ded'])} against {pf3(N['ko_inner_ded_closedsub'])}"),
        (f"{P}5", "picking where target is not innermost; all multi-header dedents against innermost rule", lambda N, G: f"{f3(N['pick_not_inner'])}; {f3(N['pick_all_multi'])} against {f3(N['innermost_rule_all_multi'])}"),
        (f"{P}6", "steering, natural unmarked dedents: rise (control); becomes top (control)", lambda N, G: f"{pf2(N['steer_dedent_head_dlq'])} ({pf2(N['steer_dedent_ctrl_dlq'])}); {f3(N['steer_dedent_head_top'])} ({f3(N['steer_dedent_ctrl_top'])})"),
        (f"{P}7", "knock-out of the true-level header, unmarked dedents", lambda N, G: pf3(N["ko_true_ded"])),
        (f"{P}8", "steering, generated terminal targets: rise (control); becomes top (control)", lambda N, G: f"{pf2(G['steer_term_head_dlq'])} ({pf2(G['steer_term_ctrl_dlq'])}); {f3(G['steer_term_head_top'])} ({f3(G['steer_term_ctrl_top'])})"),
        (f"{P}9", "`return`: share of the gap closed by removing the keyword edge in all layers", lambda N, G: f3(G["share_blk_kw_return"])),
        (f"{P}10", "largest single-layer share for `return`", lambda N, G: f3(G["share_layer_return_max"])),
    ]
w("## Part R2: D3 round 2 (`PROTOCOL-2.md`)")
w("")
QN, QG, ON, OG, GR = (NM2[k] for k in ("qwen3-06b-bf16/L23H2/natural", "qwen3-06b-bf16/L23H2/gen2", "olmo1b/L13H1/natural", "olmo1b/L13H1/gen2", "olmo1b/DEDO/group"))
qn, qg, on_, og, gr = (OM2[k] for k in ("qwen3-06b-bf16/L23H2/natural", "qwen3-06b-bf16/L23H2/gen2", "olmo1b/L13H1/natural", "olmo1b/L13H1/gen2", "olmo1b/DEDO/group"))
w(f"Natural code: {ni(QN['n']['chunks'])} chunks from {ni(QN['n']['repos'])} repositories ({ni(QN['n']['ded'])} unmarked dedents, {ni(QN['n']['same'])} same-level lines and {ni(QN['n']['term'])} terminal dedents with at least one open header, Qwen3). Generated: {ni(QG['n']['programs'])} targeted programs ({ni(QG['n']['return'])} ending in `return`). Thresholds are those of `PROTOCOL-2.md`; the verdict column is round 3's.")
w("")
w("| id | measure | original (9 Nov) | round 3 | verdict |")
w("|---|---|---|---|---|")
vd2 = lambda k: vd(NV2["predictions"].get(k)) if k in NV2["predictions"] else "–"
for (k, m_, f), (_, _, _) in zip(r2rows(QN, QG, "Q"), r2rows(qn, qg, "Q")): w(f"| {k} | {m_} | {f(qn, qg)} | {f(QN, QG)} | {vd2(k)} |")
w(f"| Q11 | L0H2 ablated on the terminal line and decision token: change in q(current level) | {pf3(qg['l0h2_line_dqcur_term'])} | {pf3(QG['l0h2_line_dqcur_term'])} | {vd2('Q11')} |")
for (k, m_, f) in r2rows(ON, OG, "O"): w(f"| {k} | {m_} (OLMo-2, L13H1) | {f(on_, og)} | {f(ON, OG)} | {vd2(k)} |")
w(f"| G1 | DED-O ablated, unmarked dedents; range of 20 random sets | {pf3(gr['d_lq_dedent'])}; {pf3(gr['rand_d_lq_dedent_min'])} to {pf3(gr['rand_d_lq_dedent_max'])} | {pf3(GR['d_lq_dedent'])}; {pf3(GR['rand_d_lq_dedent_min'])} to {pf3(GR['rand_d_lq_dedent_max'])} | {vd2('G1')} |")
w(f"| G2 | terminal dedents: change; shallower-open mass under ablation | {pf3(gr['d_lq_term'])}; {f3(gr['p_open_term_ablated'])} | {pf3(GR['d_lq_term'])}; {f3(GR['p_open_term_ablated'])} | {vd2('G2')} |")
w(f"| G3 | same-level lines | {pf3(gr['d_lq_same'])} | {pf3(GR['d_lq_same'])} | {vd2('G3')} |")
w(f"| G4 | unmarked dedents: change in mass on the current level | {pf3(gr['d_psame_dedent'])} | {pf3(GR['d_psame_dedent'])} | {vd2('G4')} |")
w("")

# ---------------- X
w("## Part X: the indentation side of X1's tests")
w("")
w("Each X1 test re-scored on its indentation part only, with X1's thresholds and X1's frozen exploration means, on the first 400 new natural-code chunks and the first 400 new generated programs. Values are changes in mean log q(true); brackets give the range of the 20 random draws. The original column is X1's held-out run (8 Nov).")
w("")
w("| id | test | original | round 3 | verdict |")
w("|---|---|---|---|---|")
def rng(v): return f"({f3(v[0])} to {f3(v[1])})"
xv = lambda t: T[t]["values"]
w(f"| XA3 | L17H6, generated terminal: ≤ −0.010, below every random draw | {pf3(xo('L17H6', 'gT'))} {rng(xr('L17H6', 'gT'))} | {pf3(xv('XA3')['change'])} {rng(xv('XA3')['rand'])} | {vd(T['XA3']['passed'])} |")
w(f"| XA5 | L17H6, code unmarked (≤ 0.03); generated unmarked (≤ 0.10) | {pf3(xo('L17H6', 'cU'))}; {pf3(xo('L17H6', 'gU'))} | {pf3(xv('XA5')['cU'])}; {pf3(xv('XA5')['gU'])} | {vd(T['XA5']['passed'])} |")
for t, g in (("XB1", "WRITE"), ("XB2", "COUNT"), ("XB3", "READ-L17H6"), ("XC3", "L0H2"), ("XC6", "L0H3"), ("XD3", "L11H9")):
    lab = "|change| ≤ 0.05 on all four" if t == "XD3" else "indentation quiet"
    o = "; ".join(pf3(xo(g, k)) for k in ("cU", "cT", "gU", "gT")); n = "; ".join(pf3(xv(t)[k]) for k in ("cU", "cT", "gU", "gT"))
    w(f"| {t} | {g.replace('-', '−')}: {lab} (code unmarked; code terminal; generated unmarked; generated terminal) | {o} | {n} | {vd(T[t]['passed'])} |")
w(f"| XB4 | DED, code unmarked: ≤ −0.40, below every random draw | {pf3(xo('DED', 'cU'))} {rng(xr('DED', 'cU'))} | {pf3(xv('XB4')['change'])} {rng(xv('XB4')['rand'])} | {vd(T['XB4']['passed'])} |")
w("")
n = XE["n_points"]
w(f"Points (round 3): code unmarked {ni(n['cU'])}, code terminal {ni(n['cT'])}, generated unmarked {ni(n['gU'])}, generated terminal {ni(n['gT'])}. Unablated mean log q(true): " + ", ".join(f"{k} {f3(v)}" for k, v in XE["base"].items()) + ".")
w("")

# ---------------- E
w("## Part E: L11H9's second role (new, prospective)")
w("")
e = {t: T[t]["values"] for t in ek}
ci = XE["intervals"]
cis = lambda k: "" if not ci.get(k) else f" [{f3(ci[k][0])}, {f3(ci[k][1])}]"
w("| id | prediction | result | verdict |")
w("|---|---|---|---|")
w(f"| E1 | L11H9 alone, generated terminal dedents: ≤ −0.010 and below every random draw (X1's A3 criterion) | {pf3(e['E1']['change'])}{cis('L11H9|gT')}; random {rng(e['E1']['rand'])} | {vd(T['E1']['passed'])} |")
w(f"| E2 | CARRY and TERM, the same criterion | CARRY {pf3(e['E2']['CARRY']['change'])}{cis('CARRY|gT')}, random {rng(e['E2']['CARRY']['rand'])}; TERM {pf3(e['E2']['TERM']['change'])}{cis('TERM|gT')}, random {rng(e['E2']['TERM']['rand'])} | {vd(T['E2']['passed'])} |")
w(f"| E3 | removing L11H9 weakens each set by ≥ +0.010 | CARRY {pf3(e['E3']['CARRY']['without'])} against {pf3(e['E3']['CARRY']['full'])} (difference {pf3(e['E3']['CARRY']['diff'])}); TERM {pf3(e['E3']['TERM']['without'])} against {pf3(e['E3']['TERM']['full'])} ({pf3(e['E3']['TERM']['diff'])}) | {vd(T['E3']['passed'])} |")
w(f"| E4 | L11H9: code unmarked ≤ 0.03, generated unmarked ≤ 0.10 in absolute value | {pf3(e['E4']['cU'])}; {pf3(e['E4']['gU'])} | {vd(T['E4']['passed'])} |")
w(f"| E5 | L11H9, natural-code terminal dedents: below every random draw | {pf3(e['E5']['change'])}{cis('L11H9|cT')}; random {rng(e['E5']['rand'])} | {vd(T['E5']['passed'])} |")
p6 = e["E6"]
if p6:
    def pr(h):
        x = p6.get(h)
        return "–" if not x else f"necessity {f3(x['nec'])} (random at most {f3(x['rand_nec_max'])}), sufficiency {f3(x['suf'])} (random at most {f3(x['rand_suf_max'])})"
    w(f"| E6 | pair patching, L11H9 alone: necessity ≥ 0.05 and above every random draw | {pr('L11H9')}; L17H6, reported: {pr('L17H6')} | {vd(T['E6']['passed'])} |")
w("")
w(f"Verdict rule (frozen): confirmed if E1 and E3 pass; a single-head effect only if E1 passes and E3 fails; not confirmed if E1 fails. **Verdict: {XE['verdicts']['E: L11H9 second role']}.**")
w("")

# ---------------- gate
w("## Part G: the gate on the new material (descriptive)")
w("")
w("Round 1's gate measures; the thresholds are round 1's (header top-1 ≥ 0.90; terminal top-1 ≥ 0.5 and shallower-open mass ≥ 0.7; top-1 minus the copy baseline ≥ 0.10; never-opened mass at dedents ≤ 0.10 on 2-space, 3-space and mixed generated code). Original values (round 1, corrected measure) in brackets; OLMo-2-0425-1B was gated in round 2 on other material, so it has none here.")
w("")
w("| model | points | header top-1 | terminal top-1 | terminal shallower-open | top-1 − copy | never-opened mass, worst width |")
w("|---|---|---|---|---|---|---|")
def gl(G, m):
    r = G.get(m)
    if not r or not r.get("newcode"): return None
    s = r["newcode"]; ol = [r[f"gen_w{x}"]["dedent_shnotopen"] for x in ("2", "3", "mixed") if r.get(f"gen_w{x}")]
    return dict(n=s["n"], h=s["header_top1"], t=s["terminal_top1"], o=s["terminal_shopen"], d=s["top1"] - s["copy"], ol=max(ol) if len(ol) == 3 else None)
names = {"qwen3-06b": "Qwen3-0.6B-Base", "olmo1b": "OLMo-2-0425-1B", "pythia-410m": "Pythia-410M", "gemma4-e2b": "Gemma-4-E2B", "gpt2": "GPT-2"}
for m in names:
    a, b = gl(GN, m), gl(GO, m)
    if a is None: w(f"| {names[m]} | not run | | | | | |"); continue
    br = lambda k: "" if b is None or b[k] is None else f" ({f3(b[k])})"
    w(f"| {names[m]} | {ni(a['n'])} | {f3(a['h'])}{br('h')} | {f3(a['t'])}{br('t')} | {f3(a['o'])}{br('o')} | {f3(a['d'])}{br('d')} | {f3(a['ol'])}{br('ol')} |")
w("")
if MISSING: w("**Not run or missing at build time:** " + ", ".join(f"`{m}`" for m in MISSING) + "."); w("")
gen = "\n".join(L) + "\n"
rd = R3 / "REPORT-3-reading.md"
if rd.exists() and not DRAFT: gen += "\n" + rd.read_text()
(D3 / ("REPORT-3.draft.md" if DRAFT else "REPORT-3.md")).write_text(gen)
print("written", "draft" if DRAFT else "REPORT-3.md")
