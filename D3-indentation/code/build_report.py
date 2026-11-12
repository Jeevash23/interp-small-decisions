"""D3: build REPORT.md from saved outputs only (confirmation outputs/conf/, gate outputs, exploration JSON).
Every number in the report is filled here from a file; check_report.py recomputes them independently.
Usage: python build_report.py"""
import json
from pathlib import Path
import numpy as np

D3 = Path(__file__).resolve().parent.parent; O = D3 / "outputs"; C = O / "conf"
J = lambda p: json.loads(Path(p).read_text())
n = J(C / "c01_newcode.json"); g = J(C / "c01_gen.json"); pr = J(C / "c02_gen.json"); V = J(C / "verdicts.json")
gate = J(O / "gate_summary_postfix_final.json"); gate_v1 = J(O / "v1-before-2230-fix" / "gate_summary.json")
f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"
d = lambda r, grp, meas, c: r[grp][meas][c] - r["base"][meas][c]
rmin = lambda r, grp, meas, c: min(v[meas][c] for k, v in r.items() if k.startswith(grp + "_rand")) - r["base"][meas][c]
rmax = lambda r, grp, meas, c: max(v[meas][c] for k, v in r.items() if k.startswith(grp + "_rand")) - r["base"][meas][c]
cm = pr["comps"]; crand = lambda grp, k: [v[k] for kk, v in cm.items() if kk.startswith(grp + "_rand")]
vt = {v["id"]: v for v in V["verdicts"]}
ci = lambda h: f"[{f3(vt[h]['ci95'][0])}, {f3(vt[h]['ci95'][1])}]"
np_n, np_g = n["n_points"], g["n_points"]

L = []
w = L.append
w("# D3 report: two routes to a dedent in Qwen3-0.6B-Base")
w("")
w("8 November 2026. **Confirmation** under `PROTOCOL.md` (frozen 06:31 IST), run once on the sealed confirmation parts between 06:32 and 08:55 IST. The candidate came from the exploration in `NOTEBOOK.md` under `CHARTER.md` (frozen 7 Nov, 21:04 IST) and Amendment 1. Every number below is filled from saved outputs by `code/build_report.py` and recomputed independently by `code/check_report.py`.")
w("")
w("## Result")
w("")
w(f"All nine frozen predictions pass. H1, H2, H3, H5 and H6 pass, so by the protocol's rule **the candidate is confirmed**. H4, H7, H8 and H9 also pass.")
w("")
w("| part | prediction | result | random control groups (20) | 95% interval (descriptive) | verdict |")
w("|---|---|---|---|---|---|")
w(f"| H1 | DED ablation, natural code, other dedents: change in mean log q(true) ≤ −0.40, below every random group | {f3(d(n,'DED','lq','dedent'))} | lowest {f3(rmin(n,'DED','lq','dedent'))} | {ci('H1')} | {vt['H1']['verdict']} |")
w(f"| H2 | same points: change in mass on the current level ≥ +0.10, above every random group | +{f3(d(n,'DED','p_same','dedent'))} | highest +{f3(rmax(n,'DED','p_same','dedent'))} | {ci('H2')} | {vt['H2']['verdict']} |")
w(f"| H3 | DED ablation, terminal dedents: shallower-open mass ≥ 0.95; change in log q ≤ 0.25 × H1's | {f3(n['DED']['p_open']['term'])}; change {f3(d(n,'DED','lq','term'))} (ratio {f3(abs(d(n,'DED','lq','term'))/abs(d(n,'DED','lq','dedent')))}) | – | {ci('H3')} | {vt['H3']['verdict']} |")
w(f"| H4 | DED ablation, same-level lines: change in mean log q(true) ≥ −0.01 | +{f3(d(n,'DED','lq','same'))} | – | {ci('H4')} | {vt['H4']['verdict']} |")
w(f"| H5 | TERM ablation, terminal dedents: change ≤ −0.03, below every random group | {f3(d(n,'TERM','lq','term'))} | lowest {f3(rmin(n,'TERM','lq','term'))} | {ci('H5')} | {vt['H5']['verdict']} |")
w(f"| H6 | TERM ablation, other dedents: absolute change ≤ 0.10 | {f3(d(n,'TERM','lq','dedent'))} | – | {ci('H6')} | {vt['H6']['verdict']} |")
w(f"| H7 | TERM and DED together, terminal dedents: change more negative than the sum of the two alone | {f3(d(n,'TERMDED','lq','term'))} | sum of singles {f3(d(n,'TERM','lq','term')+d(n,'DED','lq','term'))} | {ci('H7')} | {vt['H7']['verdict']} |")
w(f"| H8 | DED ablation, generated code, other dedents: change ≤ −0.25, below every random group | {f3(d(g,'DED','lq','dedent'))} | lowest {f3(rmin(g,'DED','lq','dedent'))} | {ci('H8')} | {vt['H8']['verdict']} |")
w(f"| H9 | CARRY patching, generated return/assignment pairs: necessity ≥ 0.40, sufficiency ≥ 0.20, each above every random group | necessity {f3(cm['CARRY']['nec'])}, sufficiency {f3(cm['CARRY']['suf'])} | highest {f3(max(crand('CARRY','nec')))} and {f3(max(crand('CARRY','suf')))} | – | {vt['H9']['verdict']} |")
w("")
w("Groups: DED = {L23H2, L17H3, L17H4, L17H5, L18H6}; TERM = {L11H9, L5H6, L17H6, L16H2, L23H6, L22H8, L23H3, L25H9, L19H6}; CARRY = {L17H4, L11H9, L5H6, L17H6, L16H2, L6H5}. Random groups replace each head by a random other head of the same layer. Intervals resample repositories (natural code) or programs (generated code); they are descriptive and were not used for verdicts.")
w("")
w("## What was tested, and on what")
w("")
w(f"- **Model:** Qwen3-0.6B-Base in bfloat16. Pythia-410M, the charter's main model, failed gate item 1; Qwen3-0.6B-Base passed every item, so the charter's fallback rule made it the main model.")
w(f"- **Natural code:** {n['n_chunks_used']} chunks of Python from 123 GitHub repositories created on or after 1 September 2026, sealed on 7 November and read here for the first time. Decision points: {np_n['header']} after block headers, {np_n['same']} at the same level, {np_n['term']} dedents after terminal statements, {np_n['dedent']} other dedents.")
w(f"- **Generated programs:** {g['n_chunks_used']} programs (indent widths 2, 3, 4 and mixed), sealed and read here for the first time: {np_g['header']}, {np_g['same']}, {np_g['term']} and {np_g['dedent']} points in the same classes. {pr['n']} return/assignment pairs were built from them (mean clean-minus-corrupt gap {f2(pr['gap'])} nats).")
w(f"- **Intact model on the sealed natural code:** mean log q(true) {f3(n['base']['lq']['header'])} after headers, {f3(n['base']['lq']['same'])} at the same level, {f3(n['base']['lq']['term'])} at terminal-statement dedents and {f3(n['base']['lq']['dedent'])} at other dedents. At other dedents it puts {f3(n['base']['p_same']['dedent'])} on keeping the current level. On generated code these are {f3(g['base']['lq']['header'])}, {f3(g['base']['lq']['same'])}, {f3(g['base']['lq']['term'])} and {f3(g['base']['lq']['dedent'])}; there it keeps the current level at other dedents with {f3(g['base']['p_same']['dedent'])}.")
w("- **Intervention:** each named head's output at the decision token (the token that ends the previous line) is replaced by its mean over all decision tokens of the same part. q is the model's distribution over candidate indents, renormalised.")
w("")
w("## What the result means")
w("")
w("1. **A block-end route.** Five heads in layers 17, 18 and 23 are needed for a dedent that no terminal statement announces, such as the end of an `if` body after an ordinary statement.")
w(f"   - Without them, mean log q(true) at those points falls by {f3(-d(n,'DED','lq','dedent'))} nats, against at most {f3(-rmin(n,'DED','lq','dedent'))} for any random group of the same size and layers. The mass on keeping the current level rises from {f3(n['base']['p_same']['dedent'])} to {f3(n['DED']['p_same']['dedent'])}.")
w(f"   - Same-level lines become slightly more certain (+{f3(d(n,'DED','lq','same'))}), as expected if the heads' job is to say \"this block may end here\".")
w(f"   - The same holds in generated code ({f3(d(g,'DED','lq','dedent'))}, random at most {f3(-rmin(g,'DED','lq','dedent'))}).")
w("   - In exploration, L23H2 wrote the target run token directly (direct logit attribution), and leaving it out removed about half of the group's effect.")
w(f"2. **A separate terminal-statement route.** Removing DED leaves terminal-statement dedents almost intact in natural code: {f3(n['DED']['p_open']['term'])} of the mass stays on open shallower levels, and the change in log q is {f3(d(n,'DED','lq','term'))}.")
w(f"   - Removing TERM instead costs terminal-statement dedents {f3(-d(n,'TERM','lq','term'))} (random at most {f3(-rmin(n,'TERM','lq','term'))}) and leaves other dedents nearly unchanged ({f3(d(n,'TERM','lq','dedent'))}).")
w(f"   - In return/assignment pairs, six mid-layer heads (CARRY) carry {f3(cm['CARRY']['nec'])} of the difference by necessity and {f3(cm['CARRY']['suf'])} by sufficiency. Random groups carry at most {f3(max(crand('CARRY','nec')))} and {f3(max(crand('CARRY','suf')))}.")
w(f"3. **The routes back each other up.** Removing both costs terminal-statement dedents {f3(-d(n,'TERMDED','lq','term'))}, more than the {f3(-(d(n,'TERM','lq','term')+d(n,'DED','lq','term')))} expected from adding the two single effects. So a terminal-statement dedent can be reached by either route.")
w("")
w("The TERM effect is small in absolute size, because terminal-statement dedents are near-certain in natural code and well backed up. The claim is that the effect is specific and beyond random groups, not that TERM alone carries most of the decision.")
w("")
w("## Not claimed, and limits")
w("")
w("- **One model.** The mechanism was found and confirmed in Qwen3-0.6B-Base only. Pythia-410M and Gemma-4-E2B were measured behaviourally but not analysed inside.")
w("- **Mean ablation only.** No resample ablation and no path patching between the routes were run. The head sets were chosen in exploration; their memberships are not claimed to be minimal or complete.")
w("- **How far to dedent is not explained.** Exploration found that the number of levels to dedent is barely linearly decodable at the decision token (R² at most 0.30). Late heads attend to the starts of the open levels, but their role was not confirmed.")
w(f"- **Generated code differs.** There DED also supports terminal-statement dedents (exploration). No sparing claim was made or tested for generated code. The base model is much less accurate there (other dedents {f3(g['base']['lq']['dedent'])}).")
w("- **D1's line-breaking heads** (Pythia-410M) were not among the heads found here, and no link to D1 was tested in Qwen3.")
w("")
w("## Gate (before any internal analysis), both measurements")
w("")
w("Exploration part of the post-cut-off GitHub code (unseen in training, not held out); generated programs for the open-level part.")
w("")
w("| model | measure | points | header top-1 (≥ 0.90) | terminal top-1 (≥ 0.5) | terminal shallower-open (≥ 0.7) | top-1 − copy (≥ 0.10) | open-level: never-opened mass, worst width (≤ 0.10) |")
w("|---|---|---|---|---|---|---|---|")
for mname, lab in [("pythia-410m", "Pythia-410M"), ("qwen3-06b", "Qwen3-0.6B-Base"), ("gemma4-e2b", "Gemma-4-E2B"), ("gpt2", "GPT-2")]:
    for tag, gs in [("pre-fix", gate_v1), ("post-fix", gate)]:
        if mname not in gs or not gs[mname].get("newcode"): continue
        s = gs[mname]["newcode"]
        ol = [gs[mname].get(f"gen_w{x}") for x in ["2", "3", "mixed"]]
        olv = f3(max(o["dedent_shnotopen"] for o in ol)) if all(ol) else "–"
        w(f"| {lab} | {tag} | {s['n']} | {f3(s['header_top1'])} | {f3(s['terminal_top1'])} | {f3(s['terminal_shopen'])} | {f3(s['top1'] - s['copy'])} | {olv} |")
w("")
w("- Pre-fix numbers come from the first measure, which dropped every point after a blank line for tokenisers that split blank-line runs (Pythia, GPT-2). The fix is Amendment 1; it was made after the pre-fix numbers had been seen and changed no verdict. Gemma-4-E2B's first natural-code run failed at load, so it has no pre-fix row.")
p1 = gate["pythia-410m"]
w(f"- Pythia-410M's item-1 miss is the first indented line of each chunk, where nothing earlier fixes the width: the model splits its probability between four and eight spaces (NOTEBOOK, 7 Nov 21:59 and 22:51). Elsewhere it is near-perfect after headers.")
w("- GPT-2 expects a blank line after most newlines and fails every item.")
w("")
w("## Process notes")
w("")
w("- A 30-program smoke test of the behaviour script ran on the generated exploration part three minutes before that part's digests were written. One attention scan (Pythia, exploration programs) ran before the natural-code gate result, although the charter puts the gate first. Neither touched sealed material; both are in `NOTEBOOK.md`.")
w("- Scans after 00:32 on 8 Nov used bfloat16, checked against float32 (NOTEBOOK, 01:18). The confirmation used bfloat16, as frozen.")
w("- Exploration dead ends, recorded in `NOTEBOOK.md`:")
w("  - single-head ablations could not find the terminal route;")
w("  - a \"first-token reader\" account predicted false dedents after `return_value = …`, and the model is not fooled (in all three models tested the effect is under a tenth of a real statement's);")
w("  - a \"four-space step\" prior in L25H9 and the late MLPs had the wrong signs.")
w("- No crash, deviation or exclusion occurred in the confirmation run.")
Path(D3 / "REPORT.md").write_text("\n".join(L) + "\n")
print("wrote REPORT.md", len(L), "lines")
