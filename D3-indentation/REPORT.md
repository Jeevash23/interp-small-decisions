# D3 report: two routes to a dedent in Qwen3-0.6B-Base

8 November 2026. **Confirmation** under `PROTOCOL.md` (frozen 06:31 IST), run once on the sealed confirmation parts between 06:32 and 08:55 IST. The candidate came from the exploration in `NOTEBOOK.md` under `CHARTER.md` (frozen 7 Nov, 21:04 IST) and Amendment 1. Every number below is filled from saved outputs by `code/build_report.py` and recomputed independently by `code/check_report.py`.

## Result

All nine frozen predictions pass. H1, H2, H3, H5 and H6 pass, so by the protocol's rule **the candidate is confirmed**. H4, H7, H8 and H9 also pass.

| part | prediction | result | random control groups (20) | 95% interval (descriptive) | verdict |
|---|---|---|---|---|---|
| H1 | DED ablation, natural code, other dedents: change in mean log q(true) ≤ −0.40, below every random group | -1.091 | lowest -0.029 | [-1.168, -1.015] | pass |
| H2 | same points: change in mass on the current level ≥ +0.10, above every random group | +0.306 | highest +0.016 | [0.286, 0.323] | pass |
| H3 | DED ablation, terminal dedents: shallower-open mass ≥ 0.95; change in log q ≤ 0.25 × H1's | 0.991; change -0.059 (ratio 0.054) | – | [-0.077, -0.042] | pass |
| H4 | DED ablation, same-level lines: change in mean log q(true) ≥ −0.01 | +0.025 | – | [0.022, 0.028] | pass |
| H5 | TERM ablation, terminal dedents: change ≤ −0.03, below every random group | -0.044 | lowest -0.009 | [-0.054, -0.035] | pass |
| H6 | TERM ablation, other dedents: absolute change ≤ 0.10 | -0.027 | – | [-0.044, -0.012] | pass |
| H7 | TERM and DED together, terminal dedents: change more negative than the sum of the two alone | -0.188 | sum of singles -0.103 | [-0.216, -0.164] | pass |
| H8 | DED ablation, generated code, other dedents: change ≤ −0.25, below every random group | -0.631 | lowest -0.096 | [-0.646, -0.615] | pass |
| H9 | CARRY patching, generated return/assignment pairs: necessity ≥ 0.40, sufficiency ≥ 0.20, each above every random group | necessity 0.650, sufficiency 0.373 | highest 0.075 and 0.032 | – | pass |

Groups: DED = {L23H2, L17H3, L17H4, L17H5, L18H6}; TERM = {L11H9, L5H6, L17H6, L16H2, L23H6, L22H8, L23H3, L25H9, L19H6}; CARRY = {L17H4, L11H9, L5H6, L17H6, L16H2, L6H5}. Random groups replace each head by a random other head of the same layer. Intervals resample repositories (natural code) or programs (generated code); they are descriptive and were not used for verdicts.

## What was tested, and on what

- **Model:** Qwen3-0.6B-Base in bfloat16. Pythia-410M, the charter's main model, failed gate item 1; Qwen3-0.6B-Base passed every item, so the charter's fallback rule made it the main model.
- **Natural code:** 1606 chunks of Python from 123 GitHub repositories created on or after 1 September 2026, sealed on 7 November and read here for the first time. Decision points: 6804 after block headers, 14825 at the same level, 2030 dedents after terminal statements, 2097 other dedents.
- **Generated programs:** 2008 programs (indent widths 2, 3, 4 and mixed), sealed and read here for the first time: 20701, 25179, 6702 and 8025 points in the same classes. 300 return/assignment pairs were built from them (mean clean-minus-corrupt gap 3.35 nats).
- **Intact model on the sealed natural code:** mean log q(true) -0.024 after headers, -0.033 at the same level, -0.053 at terminal-statement dedents and -0.366 at other dedents. At other dedents it puts 0.168 on keeping the current level. On generated code these are -0.675, -0.298, -0.577 and -1.845; there it keeps the current level at other dedents with 0.605.
- **Intervention:** each named head's output at the decision token (the token that ends the previous line) is replaced by its mean over all decision tokens of the same part. q is the model's distribution over candidate indents, renormalised.

## What the result means

1. **A block-end route.** Five heads in layers 17, 18 and 23 are needed for a dedent that no terminal statement announces, such as the end of an `if` body after an ordinary statement.
   - Without them, mean log q(true) at those points falls by 1.091 nats, against at most 0.029 for any random group of the same size and layers. The mass on keeping the current level rises from 0.168 to 0.473.
   - Same-level lines become slightly more certain (+0.025), as expected if the heads' job is to say "this block may end here".
   - The same holds in generated code (-0.631, random at most 0.096).
   - In exploration, L23H2 wrote the target run token directly (direct logit attribution), and leaving it out removed about half of the group's effect.
2. **A separate terminal-statement route.** Removing DED leaves terminal-statement dedents almost intact in natural code: 0.991 of the mass stays on open shallower levels, and the change in log q is -0.059.
   - Removing TERM instead costs terminal-statement dedents 0.044 (random at most 0.009) and leaves other dedents nearly unchanged (-0.027).
   - In return/assignment pairs, six mid-layer heads (CARRY) carry 0.650 of the difference by necessity and 0.373 by sufficiency. Random groups carry at most 0.075 and 0.032.
3. **The routes back each other up.** Removing both costs terminal-statement dedents 0.188, more than the 0.103 expected from adding the two single effects. So a terminal-statement dedent can be reached by either route.

The TERM effect is small in absolute size, because terminal-statement dedents are near-certain in natural code and well backed up. The claim is that the effect is specific and beyond random groups, not that TERM alone carries most of the decision.

## Not claimed, and limits

- **One model.** The mechanism was found and confirmed in Qwen3-0.6B-Base only. Pythia-410M and Gemma-4-E2B were measured behaviourally but not analysed inside.
- **Mean ablation only.** No resample ablation and no path patching between the routes were run. The head sets were chosen in exploration; their memberships are not claimed to be minimal or complete.
- **How far to dedent is not explained.** Exploration found that the number of levels to dedent is barely linearly decodable at the decision token (R² at most 0.30). Late heads attend to the starts of the open levels, but their role was not confirmed.
- **Generated code differs.** There DED also supports terminal-statement dedents (exploration). No sparing claim was made or tested for generated code. The base model is much less accurate there (other dedents -1.845).
- **D1's line-breaking heads** (Pythia-410M) were not among the heads found here, and no link to D1 was tested in Qwen3.

## Gate (before any internal analysis), both measurements

Exploration part of the post-cut-off GitHub code (unseen in training, not held out); generated programs for the open-level part.

| model | measure | points | header top-1 (≥ 0.90) | terminal top-1 (≥ 0.5) | terminal shallower-open (≥ 0.7) | top-1 − copy (≥ 0.10) | open-level: never-opened mass, worst width (≤ 0.10) |
|---|---|---|---|---|---|---|---|
| Pythia-410M | pre-fix | 20249 | 0.870 | 0.965 | 0.991 | 0.340 | 0.085 |
| Pythia-410M | post-fix | 22854 | 0.870 | 0.973 | 0.994 | 0.358 | 0.073 |
| Qwen3-0.6B-Base | pre-fix | 22854 | 1.000 | 0.989 | 0.997 | 0.406 | 0.065 |
| Qwen3-0.6B-Base | post-fix | 22854 | 1.000 | 0.989 | 0.997 | 0.406 | 0.065 |
| Gemma-4-E2B | post-fix | 22854 | 0.995 | 0.986 | 0.996 | 0.406 | 0.061 |
| GPT-2 | pre-fix | 21129 | 0.078 | 0.382 | 0.639 | -0.247 | 0.257 |
| GPT-2 | post-fix | 22854 | 0.078 | 0.337 | 0.696 | -0.243 | 0.208 |

- Pre-fix numbers come from the first measure, which dropped every point after a blank line for tokenisers that split blank-line runs (Pythia, GPT-2). The fix is Amendment 1; it was made after the pre-fix numbers had been seen and changed no verdict. Gemma-4-E2B's first natural-code run failed at load, so it has no pre-fix row.
- Pythia-410M's item-1 miss is the first indented line of each chunk, where nothing earlier fixes the width: the model splits its probability between four and eight spaces (NOTEBOOK, 7 Nov 21:59 and 22:51). Elsewhere it is near-perfect after headers.
- GPT-2 expects a blank line after most newlines and fails every item.

## Process notes

- A 30-program smoke test of the behaviour script ran on the generated exploration part three minutes before that part's digests were written. One attention scan (Pythia, exploration programs) ran before the natural-code gate result, although the charter puts the gate first. Neither touched sealed material; both are in `NOTEBOOK.md`.
- Scans after 00:32 on 8 Nov used bfloat16, checked against float32 (NOTEBOOK, 01:18). The confirmation used bfloat16, as frozen.
- Exploration dead ends, recorded in `NOTEBOOK.md`:
  - single-head ablations could not find the terminal route;
  - a "first-token reader" account predicted false dedents after `return_value = …`, and the model is not fooled (in all three models tested the effect is under a tenth of a real statement's);
  - a "four-space step" prior in L25H9 and the late MLPs had the wrong signs.
- No crash, deviation or exclusion occurred in the confirmation run.
