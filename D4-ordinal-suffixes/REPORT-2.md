# D4 round 2 report: the path from digits to suffix

<!-- nocheck -->9 November 2026. **Exploration** under `CHARTER-2.md` (frozen 8 Nov, 16:54 IST) and its Amendment 1 (`AMENDMENTS-2.md`) used exploration frames and numbers only (`NOTEBOOK.md`, round-2 entries). **Confirmation** under `PROTOCOL-2.md` (frozen 9 Nov, 09:34 IST; digests in `FROZEN.md`) ran once, 09:34–10:46 IST, on held-out material sealed on 8 Nov at 17:01 before any model read it. Every confirmation number comes from `outputs/r2conf/EVALUATION-2.json` or the raw files beside it, written by the frozen code. `code/check_report2.py`, written after the run, recomputes every test from the raw files without the evaluator's code and checks every number in this file against the saved outputs.

## Question

Round 1 (`REPORT.md`) showed that the "teen check" for ordinal suffixes (112th, not 112nd) needs an early attention step that brings the tens digit to the number's last token: head L0H2 in Qwen3-0.6B-Base, and three layer-0 heads in OLMo-2-1B where its tokeniser splits the tens digit off. Round 2 asks what happens after that step. Where in Qwen3 is "teen, so th" decided, and how is it combined with the last digit's own suffix? What carries the teen check when the early heads are removed? Is OLMo-2's failure across the token boundary a missing route or a weak one? And does the early step appear in other one-token-per-digit models?

## Summary

39 of the 41 prospective tests passed. Three of the four accounts were supported in full; the fourth was partly supported.

1. **Qwen3-0.6B-Base: the full path (12 of 12 tests).** L0H2's teen signal is read at once by MLP0: 0.913 (4 digits) and 0.913 (6 digits) of L0H2's teen loss passes through MLP0's input alone, and no other receiver carries more than 0.069. No other layer-0 head's route through MLP0 comes close. The signal is then gathered at the input of MLP24, the largest mediator of both L0H2's loss (0.588, 0.560) and the loss caused by removing L11H9, the suffix head found in the cross-study test X1 (0.611, 0.641). The logit lens shows the teen decision appearing at layer 24 (a rise of 2.889 and 2.384 logits, the largest of any layer). MLP24 writes 0.781 and 0.916 of the teen effect directly. It raises "th" against the other three suffixes by 1.873–2.893 logits for teens, whatever the last digit, while MLP18 writes the last digit's own suffix for teens and non-teens alike. The two signals meet only in the logits.
2. **Qwen3-0.6B-Base: the backup route (7 of 7).** With L0H2 and L1H6 removed, two later previous-token heads, L2H4 and L5H2, carry part of the tens digit across. Removing them as well lowers the teen effect kept from 0.701 to 0.482 at 2 digits, below every control pair, and by less at 3 and 4 digits (a drop of 0.219, against 0.095 and 0.098). On their own they matter little (kept 0.931–0.986). Even with all four heads removed, 0.482 of the two-digit teen effect remains: a further route is still unexplained.
3. **OLMo-2-1B: a weak route, not a missing one (14 of 14).** Across seven lengths, the teen check is strong where the tens and last digits share a token (AUROC 0.887–0.999 at 3, 5, 8 and 9 digits) and weak where they do not (0.640–0.730 at 4, 7 and 10). Where they do not, the three round-1 heads carry most of what arrives (kept 0.209–0.242), MLP0 is its first reader (0.700–0.861 of their loss), and the teen status is linearly readable at the last token early on (probe 0.979–1.000 at layer 2) but less so by layer 15 (0.734–0.854). The signal is about half the size of the in-token one or smaller (norm ratio at most 0.503). The late layers still read a full-sized signal: adding the 3-digit teen direction to non-teen numbers at 4, 7 and 10 digits lowers the suffix margin by 1.986–2.962 logits, while random directions of the same size move it by at most 0.401.
4. **Other one-token-per-digit models (6 of 8; partly supported).** In Qwen3-1.7B-Base, SmolLM2-1.7B and Gemma-4-E2B, early heads chosen only by their attention to the previous token carry part of the teen check (kept 0.377–0.801, against at least 0.991 for random heads), and in Qwen3-1.7B two current-token heads carry the last-digit pull. The miss: in Qwen3-1.7B the previous-token pair also removes much of the pull (kept 0.548 and 0.233, against the 0.55 required), so there the division between the two head types is less clean than in the 0.6B model.

## Material and measures

- **Frames.** 24 held-out frames written for round 2 (rankings, positions and anniversaries are used in the tests; dates are kept unread), split from 24 exploration frames by hash and sealed before any model read them (`data2/MANIFEST.md`).
- **Numbers.** Two-digit numbers are shared by every part, so two-digit items differ from exploration by frame only. Three- to five-digit numbers come from round 1's confirmation and reserve parts: read before, but in new frames here. Six- and seven-digit numbers come from round 1's reserve and had never been read. Eight- to ten-digit numbers were split for round 2 by hash, and the held part had never been read. Each length has 96 teen numbers (ending in 11–13) and 96 non-teen numbers ending in 1–3.
- **m** = logit of the suffix the last digit alone calls for minus logit of "th", at the number's last token. **Teen effect** = mean m (non-teen) − mean m (teen). **Pull** = mean m (non-teen). **AUROC** = probability that a non-teen item has a higher m than a teen item. **Kept** = the value after an intervention ÷ the value before.
- **Interventions**, all at the last token: mean ablation of a head's output; **path ablation**, in which a head's ablation change reaches one receiver only (an attention layer, an MLP or the unembedding); **mediation**, in which the change a head's full ablation causes at a receiver's input is given to that receiver alone. **DLA** projects a component's output through the final norm, linearised at each item's scale, onto the unembedding difference. **Steering** adds a direction to the layer-8 output at the last token.
- Decisions used point estimates, as the protocol fixed.

## Confirmation results

### Qwen3-0.6B-Base: the path (R1–R6)

Base teen effect 2.482 (AUROC 0.877) at 4 digits and 1.950 (0.874) at 6 digits.

| Test | Prediction | 4 digits | 6 digits | Result |
|---|---|---|---|---|
| R1 | L0H2's teen loss through MLP0 ≥ 0.60; every other receiver ≤ 0.15 | 0.913; next 0.049 | 0.913; next 0.069 | pass, pass |
| R2 | L0H2's loss through MLP0 above every other layer-0 head's | 1.138 against at most 0.004 | 0.759 against at most 0.001 | pass, pass |
| R3 | MLP24 the largest mediator in MLP12–27, ≥ 0.40, for L0H2 and for L11H9 | L0H2 0.588 (next 0.504); L11H9 0.611 (next 0.553) | 0.560 (0.489); 0.641 (0.459) | pass, pass |
| R4 | Largest lens rise at layer 24, ≥ 1.0 | 2.889 (next 0.817) | 2.384 (next 1.337) | pass, pass |
| R5 | MLP24 DLA ≥ 0.50 of the teen effect; every other MLP ≤ 0.35 | 0.781; next 0.201 | 0.916; next 0.130 | pass, pass |
| R6 | MLP24 teen profile: th − mean ≥ 1.5, own-suffix ≤ 0.6; MLP18 own-suffix ≥ 0.4 in both classes | th − mean 1.908–2.893, own ≤ 0.250; MLP18 0.907–1.619 | 1.873–2.800, own ≤ 0.278; MLP18 0.793–1.579 | pass, pass |

Removing L11H9 lowers the teen effect by 0.734 and 0.402 logits. In R6 the MLP18 own-suffix values are similar for teens (0.923, 0.907, 1.619 for last digits 1, 2, 3 at 4 digits) and non-teens (0.969, 1.045, 1.438): MLP18 writes the last digit's suffix without regard to teen status.

### Qwen3-0.6B-Base: the backup route (S1–S3)

A = L0H2 + L1H6; B = L2H4 + L5H2; controls are A plus 20 random pairs from layers 2–6.

| Digits | Base teen effect | A kept | A + B kept | A + random pair kept | B alone kept |
|---|---|---|---|---|---|
| 2 | 9.874 | 0.701 | **0.482** | 0.685–0.706 | 0.986 |
| 3 | 3.293 | 0.415 | **0.321** | 0.390–0.425 | 0.951 |
| 4 | 2.624 | 0.341 | **0.243** | 0.334–0.372 | 0.931 |

S1 (A + B below every control) and S2 (B alone ≥ 0.85) pass at each length. S3 passes: B removes 0.219 beyond A at 2 digits, against 0.095 at 3 digits and 0.098 at 4, and A + B still keeps 0.482 at 2 digits.

### OLMo-2-1B: the weak route (O3–O8)

T = L0H5 + L0H0 + L0H9; controls are 20 random triples from layers 0–2. The last-token pieces show where the tens digit sits: inside the last token at 3, 5, 8 and 9 digits, in the previous token at 4, 7 and 10.

| Digits | Teen effect | AUROC | T kept | Random triples, minimum kept | Probe, layer 2 / 15 |
|---|---|---|---|---|---|
| 3 | 4.670 | 0.999 | 1.004 | 0.987 | 0.989 / 1.000 |
| 4 | 1.186 | **0.730** | **0.234** | 0.937 | 1.000 / **0.854** |
| 5 | 3.156 | 0.971 | 0.996 | 0.985 | 1.000 / 0.995 |
| 7 | 0.625 | **0.685** | **0.209** | 0.930 | 0.979 / **0.844** |
| 8 | 2.452 | 0.912 | 0.998 | 0.984 | 1.000 / 0.995 |
| 9 | 1.539 | 0.887 | 0.994 | 0.982 | 1.000 / 0.984 |
| 10 | 0.561 | **0.640** | **0.242** | 0.924 | 0.989 / **0.734** |

- **O3, O4, O5** pass as the table shows.
- **O6** passes: the teen direction's norm, relative to the 3-digit one, is at most 0.503, 0.484 and 0.440 at 4, 7 and 10 digits, and at least 0.674 at 5 digits.
- **O7** passes: steering with the 3-digit teen direction lowers the non-teen mean m by 2.962, 2.359 and 1.986 logits at 4, 7 and 10 digits; random directions of the same norm move it by at most 0.290, 0.401 and 0.242.
- **O8** passes: T's teen loss goes through MLP0's input for 0.700, 0.759 and 0.861 of it at 4, 7 and 10 digits; the next receiver carries at most 0.128.

### Other one-token-per-digit models (X)

PREV and CUR sets were chosen in exploration by attention alone (≥ 0.5 of the last token's attention on the previous or the current token, layers 0–1).

| Model | Set | Digits | Teen kept | Pull kept | Random sets, minimum teen / pull kept | Result |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | PREV L0H2 + L1H6 | 3 | 0.484 | 0.548 | 0.994 / — | **fail** (pull) |
| | | 4 | 0.377 | 0.233 | 0.998 / — | **fail** (pull) |
| | CUR L0H3 + L0H7 | 3 | 1.136 | −0.567 | — / 0.985 | pass |
| | | 4 | 1.024 | −2.754 | — / 0.991 | pass |
| SmolLM2-1.7B | PREV L0H6 + L0H8 | 3 | 0.729 | 0.801 | 0.995 / — | pass |
| | | 4 | 0.661 | 0.588 | 1.000 / — | pass |
| Gemma-4-E2B | PREV L0H4 | 3 | 0.801 | 0.906 | 0.991 / — | pass |
| | | 4 | 0.653 | 0.821 | 0.994 / — | pass |

The previous-token sets pass on the teen side in every model and length. In Qwen3-1.7B they also take away much of the pull, which round-2 exploration had not shown (0.70 and 0.63 kept there). SmolLM2 and Gemma-4 have no early current-token head by the attention rule, so the CUR test applies to Qwen3-1.7B only.

## Exploratory and descriptive results (not tested)

- **When OLMo-2 learns the teen check** (E06; six stage-1 training checkpoints, exploration items). The in-token check forms early: AUROC at 3 digits is 0.789 after 21B tokens and 0.982 after 84B. The cross-token check stays near chance far longer: AUROC at 4 digits is 0.540 at 84B tokens, 0.710 at 1.0T and 0.817 at the end of stage 1 (4.0T). Round 1's triple L0H5 + L0H0 + L0H9 does nothing at any stage-1 checkpoint where the effect is readable (kept 0.997–1.003 from 420B tokens on), so the route it forms in the released model is set up in the final training stage. The stage-2 checkpoints were not on disk and were not fetched.
  - <!-- nocheck -->**Correction, 9 Nov 2026, 14:32 IST (post hoc).** That last inference does not hold. Study D3 found, and D4's own weight comparison confirms (`code/r2post_olmo_lineage.py`, `code/r2post_olmo_basis.py`, `outputs/r2post/`), that the released OLMo-2-0425-1B weights do not descend from these checkpoints. The released embeddings have cosine 0.006 with the end of stage 1, and the layer weights about 0, while all four published stage-2 continuations stay at 0.9996 or above. Comparisons that do not depend on the basis also rule out a reordered or rotated copy. The checkpoints are therefore a published training run of the same recipe. They say nothing about when the released model's route formed, and head positions need not correspond across the two runs. The E06 numbers above stand as a description of that run. Paper v3 (`paper/v3-2026-11-09/`) carries the correction.
- <!-- nocheck -->**The unexplained part of the two-digit check** (exploration values, `NOTEBOOK.md`; not machine-checked). Exploration found the tens position's residue still large at layers 19–21 when A is removed, which suggests a late second move, or a stored pattern for frequent two-digit ordinals. Not tested.

## Relation to X1

The cross-study test X1 (`../X1-shared-parts/REPORT.md`) found that L0H2 is a general previous-token head and L0H3 a general current-token head, both used by several tasks, and that L11H9 is a suffix head. Round 2 fits that picture. The teen check does not have a dedicated early head; it uses a general previous-token mover, and the task-specific work happens in MLP0 (reading) and MLP24 (deciding). L11H9 feeds the same decision point.

## Deviations and limits

- <!-- nocheck -->Overnight stall: nothing ran from 21:40 on 8 Nov to 09:09 on 9 Nov because of a waiting-loop fault. Amendment 1 moved the protocol freeze from 03:00 to 09:34 and let the Gemma-4 exploration run and the E06 checkpoints finish after the exploration window; no held-out material was read before the freeze (`NOTEBOOK.md`, 9 Nov 09:20).
- <!-- nocheck -->Before the freeze, every task except the other-models task was run once at full size on exploration items (31 of 33 draft tests passed); the other-models task had only a small smoke run. Three thresholds were reset from exploration values: S2 (0.90 → 0.85) and the 5-digit side of O6 (0.75 → 0.65) from that run, and X-PREV (0.80 → 0.85) from the full Gemma-4-E2B exploration run. (`PROTOCOL-2.md` calls the dry run "the whole pipeline"; it did not include the other-models task at full size.)
- Three- to five-digit numbers had been read in round 1 (in other frames). Two-digit numbers are shared by all parts.
- One model family carries the path analysis. The interventions are mean ablations at the last token, and DLA linearises the final norm.
- Qwen3-0.6B's pull at 6 digits is close to zero (−0.027), so pull ratios at that length would be unstable; no test uses them.

## Files

<!-- nocheck -->`PROTOCOL-2.md`, `AMENDMENTS-2.md`, `FROZEN.md`; raw values in `outputs/r2conf/` (`qwen.json`, `qwen2.json`, `olmo.json`, `other_*.json`); tests and verdicts in `outputs/r2conf/EVALUATION-2.json`; exploration in `NOTEBOOK.md` and `outputs/r2/`; the number check `code/check_report2.py`.
