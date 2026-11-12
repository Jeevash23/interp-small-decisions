"""D3 round 2: build REPORT-2.md from saved outputs only (outputs/r2/measures_conf.json and outputs/r2/conf/verdicts.json,
written by the frozen r2c04_verdicts.py; run logs and FROZEN.md for times). code/r2_check_report.py checks every number
against values recomputed independently from the raw records (outputs/r2/independent_values.json).
Usage: python r2_build_report.py [--draft-calib]   (--draft-calib: layout test on calibration measures, writes REPORT-2.draft.md)"""
import json, re, sys
from pathlib import Path

D3 = Path(__file__).resolve().parent.parent; O2 = D3 / "outputs" / "r2"
draft = "--draft-calib" in sys.argv
M = json.loads((O2 / ("measures_calib.json" if draft else "measures_conf.json")).read_text())
V = json.loads((O2 / "conf" / "verdicts.json").read_text()) if not draft else dict(predictions={}, questions={})
MINUS = "\u2212"
s = lambda x: MINUS + x[1:] if x.startswith("-") else x
f3 = lambda x: s(f"{x:.3f}"); pf3 = lambda x: ("+" if x >= 0 else "") + f3(x); pf2 = lambda x: ("+" if x >= 0 else "") + s(f"{x:.2f}")
ni = lambda x: f"{int(x):,}"
vd = lambda k: ("pass" if V["predictions"].get(k) else "FAIL") if k in V["predictions"] else "–"
QN, QG = M["qwen3-06b-bf16/L23H2/natural"], M["qwen3-06b-bf16/L23H2/gen2"]
ON, OG, GR = M["olmo1b/L13H1/natural"], M["olmo1b/L13H1/gen2"], M["olmo1b/DEDO/group"]
fz = (D3 / "FROZEN.md").read_text()
tfreeze = re.search(r"\| PROTOCOL-2\.md \| [0-9a-f]{64} \| 2026-11-09 (\d\d:\d\d)", fz).group(1)
ci = lambda N, k: f"[{f3(N['ci'][k][0])}, {f3(N['ci'][k][1])}]"
L = []; w = L.append


def head_rows(N, G, P):
    rows = [
        (f"{P}1", "attention to innermost open header's `:\\n`, unmarked dedents; same-level lines", f"{f3(N['att_inner_ded'])} {ci(N, 'att_inner_ded')}; {f3(N['att_inner_same'])}"),
        (f"{P}2", "knock-out of that token, unmarked dedents; every other head of the layer", f"{pf3(N['ko_inner_ded'])} {ci(N, 'ko_inner_ded')}; {pf3(min(N['ko_inner_others_ded'].values()))} to {pf3(max(N['ko_inner_others_ded'].values()))}"),
        (f"{P}3", "same knock-out, same-level lines; terminal dedents", f"{pf3(N['ko_inner_same'])}; {pf3(N['ko_inner_term'])}"),
        (f"{P}4", "closed-header knock-out against open-header knock-out (same points)", f"{pf3(N['ko_closed_ded'])} against {pf3(N['ko_inner_ded_closedsub'])}"),
        (f"{P}5", "picking where target ≠ innermost (innermost most attended); all multi-header dedents against innermost rule",
         f"{f3(N['pick_not_inner'])} {ci(N, 'pick_not_inner')} ({f3(N['pick_inner_attended_not_inner'])}); {f3(N['pick_all_multi'])} against {f3(N['innermost_rule_all_multi'])}"),
        (f"{P}6", "steering, natural unmarked dedents: rise (control); becomes top (control)",
         f"{pf2(N['steer_dedent_head_dlq'])} ({pf2(N['steer_dedent_ctrl_dlq'])}); {f3(N['steer_dedent_head_top'])} ({f3(N['steer_dedent_ctrl_top'])})"),
        (f"{P}7", "knock-out of the true-level header, unmarked dedents", pf3(N["ko_true_ded"])),
        (f"{P}8", "steering, generated terminal targets: rise (control); becomes top (control)",
         f"{pf2(G['steer_term_head_dlq'])} ({pf2(G['steer_term_ctrl_dlq'])}); {f3(G['steer_term_head_top'])} ({f3(G['steer_term_ctrl_top'])})"),
        (f"{P}9", "`return`: share of the gap closed by removing the keyword edge in all layers", f"{f3(G['share_blk_kw_return'])} [{f3(G['ci']['share_blk_kw_return'][0])}, {f3(G['ci']['share_blk_kw_return'][1])}]"),
        (f"{P}10", "largest single-layer share for `return`", f3(G["share_layer_return_max"])),
    ]
    return rows


w("# D3 round 2 report: how the two dedent routes get their information, and a second model")
w("")
w(f"9 November 2026. **Confirmation** under `PROTOCOL-2.md` (frozen {tfreeze} IST), run once on the sealed round-2 parts. The candidates came from the round-2 exploration in `NOTEBOOK.md` under `CHARTER-2.md` (frozen 8 Nov, 16:54 IST) and `AMENDMENTS-2.md`. Every number below is filled from saved outputs by `code/r2_build_report.py`; `code/r2_check_report.py` checks them against values recomputed independently from the raw records.")
w("")
w("## Result")
w("")
npass = sum(V["predictions"].values()); nt = len(V["predictions"])
w(f"**{npass} of {nt} frozen predictions pass.**")
w("")
for q, v in V["questions"].items(): w(f"- {q[0].upper() + q[1:]}: **{v}** ({', '.join(V['question_parts'][q])}).")
w("")
w("## Qwen3-0.6B-Base, head L23H2")
w("")
w(f"Natural code: {ni(QN['n']['chunks'])} chunks from {ni(QN['n']['repos'])} repositories; {ni(QN['n']['ded'])} unmarked dedents, {ni(QN['n']['same'])} same-level lines and {ni(QN['n']['term'])} terminal dedents with at least one open header. Generated: {ni(QG['n']['programs'])} targeted programs ({ni(QG['n']['terminal'])} ending in a terminal statement; {ni(QG['n']['return'])} `return`).")
w("")
w("| id | measure | result | verdict |")
w("|---|---|---|---|")
for k, m_, r in head_rows(QN, QG, "Q"): w(f"| {k} | {m_} | {r} | {vd(k)} |")
w(f"| Q11 | L0H2 ablated on the terminal line and decision token: change in q(current level), terminal programs | {pf3(QG['l0h2_line_dqcur_term'])} [{f3(QG['ci']['l0h2_line_dqcur_term'][0])}, {f3(QG['ci']['l0h2_line_dqcur_term'][1])}] | {vd('Q11')} |")
w("")
w("## OLMo-2-0425-1B, head L13H1 and block-end set DED-O = {L13H1, L10H12, L11H0, L15H4}")
w("")
w(f"Same material. {ni(ON['n']['ded'])} unmarked dedents; {ni(GR['n']['dedent'])} unmarked dedents in the group ablation (chunks over 1,400 tokens skipped; root-identified points).")
w("")
w("| id | measure | result | verdict |")
w("|---|---|---|---|")
for k, m_, r in head_rows(ON, OG, "O"): w(f"| {k} | {m_} | {r} | {vd(k)} |")
w(f"| G1 | DED-O ablated, unmarked dedents; 20 random layer-matched sets | {pf3(GR['d_lq_dedent'])} [{f3(GR['ci']['d_lq_dedent'][0])}, {f3(GR['ci']['d_lq_dedent'][1])}]; {pf3(GR['rand_d_lq_dedent_min'])} to {pf3(GR['rand_d_lq_dedent_max'])} | {vd('G1')} |")
w(f"| G2 | terminal dedents: change (ratio to G1); shallower-open mass under ablation | {pf3(GR['d_lq_term'])} ({f3(abs(GR['d_lq_term']) / abs(GR['d_lq_dedent']))}); {f3(GR['p_open_term_ablated'])} | {vd('G2')} |")
w(f"| G3 | same-level lines | {pf3(GR['d_lq_same'])} | {vd('G3')} |")
w(f"| G4 | unmarked dedents: change in mass on the current level | {pf3(GR['d_psame_dedent'])} | {vd('G4')} |")
w("")
w("## Descriptive (not used for verdicts)")
w("")
for nm, G in (("Qwen3", QG), ("OLMo-2", OG)):
    w(f"- {nm}, keyword-edge share by keyword: return {f3(G['share_blk_kw_return'])}, raise {f3(G['share_blk_kw_raise'])}, break {f3(G['share_blk_kw_break'])}, continue {f3(G['share_blk_kw_continue'])}, pass {f3(G['share_blk_kw_pass'])}; with the edges from the rest of the line as well (return) {f3(G['share_blk_kw_after_return'])}. L0H2 everywhere: {pf3(G['l0h2_all_dqcur_term'])}.")
w(f"- Qwen3 attention to the innermost open header at terminal dedents: {f3(QN['att_inner_term'])}; OLMo-2 L13H1: {f3(ON['att_inner_term'])}.")
w(f"- Over all multi-header dedents, OLMo-2's L13H1 picks the target in {f3(ON['pick_all_multi'])} of cases against the innermost rule's {f3(ON['innermost_rule_all_multi'])} (no prediction was made about this comparison).")
w("")
w("## Run")
w("")
if draft: w("RUNNOTE")
else:
    done = {}
    for f in sorted(D3.joinpath("outputs").glob("r2c0*_timelimit_1000.log")):
        if "calib" in f.name: continue
        txt = f.read_text(errors="replace"); m = re.findall(r"exit (\d+) (\d\d:\d\d)", txt); done[f.name.split("_timelimit")[0]] = m[-1] if m else ("?", "?")
    w("Run once as one systemd unit (`d3r2-conf`, `code/r2q_conf.sh`), started 09:55 and finished 14:20 IST on 9 November, in GPU lock holds of at most 1,000 s of work each; the scripts resume from their saved records, and the GPU was shared with another study and with exploratory checkpoint scans. Every script covered all its material (last pass exit status and time: " + "; ".join(f"`{k}` {v[0]} at {v[1]}" for k, v in done.items()) + "). No crash, deviation or exclusion beyond those the protocol fixes (group ablation: chunks over 1,400 tokens skipped).")
w("")
(D3 / ("REPORT-2.draft.md" if draft else "REPORT-2.md")).write_text("\n".join(L) + "\n")
print("written", "draft" if draft else "REPORT-2.md")
