# D3 round 2 confirmation protocol: how the two dedent routes get their information, and a second model

Written 9 November 2026, about 10:00 IST, after the round-2 exploration recorded in `NOTEBOOK.md` (8 Nov 16:54 to 9 Nov 09:53) under `CHARTER-2.md` and `AMENDMENTS-2.md`. Frozen (digests in `FROZEN.md`, file made read-only) before any script reads a sealed round-2 part. It is run **once**. Everything below is **prospective and confirmatory**. Anything else computed on the sealed parts is labelled post hoc. Question 5 (training dynamics) is not tested here and stays exploratory (R2-1).

## Models

- **Qwen3-0.6B-Base** (main model, round 1), bfloat16.
- **OLMo-2-0425-1B** (final weights), bfloat16: the second model by the charter's fixed rule. It passed round 1's frozen gate on round-1 exploration material (NOTEBOOK, 8 Nov 17:13–17:25).
- Eager attention for the head-level scripts (r2c01, r2c02), which need attention weights; SDPA for the group ablation (r2c03), as in round 1.

## The candidate account (from exploration)

1. **What the block-end head reads (question 1).** At an unmarked dedent, Qwen3's block-end head L23H2 attends from the decision token to the `:\n` token that ends the header of the block being closed. From there it copies that header's indent: its output from an open header's `:\n` favours that header's indent. Removing that one source position costs the model about as much as removing the whole header line. The other layer-23 heads do not matter there, and a header that is already closed does not matter either. The "block has ended" signal in its query is spread over MLPs 18–20 and L17H5 (exploratory; not tested here).
2. **How far to dedent (question 2).** When several shallower levels are open, the open header that L23H2 attends to most is the target level, also when that is not the innermost one. Setting its attention all-on a chosen open header moves the prediction to that header's indent, in natural and in generated code.
3. **The terminal route (question 3).** After `return`, the decision newline reads the keyword token directly. Removing that one attention edge in every layer closes most of the gap between a terminal and a neutral line. The read is spread over many layers (4–19), with no single layer carrying much of it. X1's lead, previous-token head L0H2, does not act through positions on the terminal line.
4. **Second model (question 4).** OLMo-2 has the same two routes. A block-end group DED-O = {L13H1, L10H12, L11H0, L15H4} is needed for unmarked dedents and spares terminal dedents. Its head L13H1 reads open headers' `:\n` tokens and sets the level as L23H2 does, though less sharply. The terminal keyword is read directly there too.

## Material (sealed; digests in `FROZEN.md`)

- **Natural code:** `data/r2code/part-held.jsonl`, 2,712 chunks of Python from 225 GitHub repositories created on or after 1 Sep 2026 with 5–9 stars, none used by round 1 or X1. Never read by a model. Used by r2c01 (both models) and r2c03.
- **Targeted generated programs:** `data/r2gen2/part-held.jsonl`, 1,510 programs from `code/gen2.py` (held-out half by seed hash). Never read by a model. Used by r2c02 (both models).
- `data/r2gen/part-held.jsonl` (4,000 generator programs) is not used and stays sealed as a reserve.
- Decision points, candidate indents, classes and the indent distribution are computed exactly as in round 1 (`code/pyindent.py`, `code/common.py`, `RootMasses`; points whose candidates are all identified at the first token). Open headers come from `r2common.open_headers` (Python's tokenizer; the `:\n` token of each enclosing header, innermost last). Chunks longer than 1,400 tokens are skipped by r2c01 and r2c03.

## Measures and interventions (as in the exploration scripts; full definitions in the script headers)

- **Attention** of the named head from the decision token to the innermost open header's `:\n`: at unmarked dedents and at same-level lines (points with at least one open header).
- **Knock-out (ko):** the head's output at the decision token is recomputed from its clean attention with one source position removed and the rest renormalised. **ko_inner** removes the innermost open header's `:\n`, **ko_true** the true-level open header's `:\n`, and **ko_closed** the most recent earlier header `:\n` that is no longer open (control). Layer-matched control: ko_inner for **every other head of the same layer**. Measure: change in mean log q(true).
- **Picking:** at dedents (unmarked and terminal) with at least two open headers whose true level is one of them, whether the head's most-attended open header is the true one. Reported for points where the true level is not the innermost, and for all such points against the innermost rule (true level = innermost).
- **Steering:** at dedents with at least two open headers, the head's attention is set all-on open header x's `:\n` (output = its value vector there), for each x. Measure, for x that is not already the top candidate: mean change in log q(x), and the share where x becomes the top candidate. Control: the same for one other head of the same layer, drawn with fixed seeds (`random.Random(4000 + point counter)` natural, `random.Random(5000 + seed)` generated).
- **Terminal route (r2c02, terminal programs):** q(current level) at the target point; the attention edge from the decision token to the keyword token removed in all layers (`blk_kw`), or in one layer at a time; and the neutral twin, where the terminal line is `value = value`. Share of the gap = (ablated − base) / (neutral − base), computed on means over programs (`return` programs for the predictions). L0H2 mean-ablated (mean over all positions of the part's first 40 programs) at the terminal line's positions and the decision token: change in mean q(current level).
- **Group ablation (r2c03):** DED-O's outputs at every decision token replaced by their mean over all decision tokens of the part; 20 layer-matched random groups (`random.Random(3000 + k)`).

## Predictions (each could fail)

Thresholds were set from the calibration runs, which ran the same scripts on exploration material (x1code 250 chunks, gen2 400 programs, round-1 code 300 chunks; NOTEBOOK, 9 Nov 09:53), with margins. Calibration values are in brackets. "Every other head" means all 15 other heads of the layer.

### Qwen3-0.6B-Base, head L23H2

| id | prediction | calibration |
|---|---|---|
| **Q1** | Attention to the innermost open header's `:\n` at unmarked dedents ≥ 0.30, and ≥ 2.5 × the same at same-level lines. | 0.507 and 0.102 |
| **Q2** | ko_inner at unmarked dedents ≤ −0.05, more negative than ko_inner of every other layer-23 head, and at least 2 × the largest absolute change of any of them. | −0.135; others −0.006 to +0.041 |
| **Q3** | ko_inner at same-level lines ≥ −0.02; at terminal dedents, absolute change ≤ 0.25 × the absolute Q2 change. | +0.013; −0.004 |
| **Q4** | On unmarked dedents with a closed earlier header, the absolute ko_closed change is ≤ 0.5 × the absolute ko_inner change on the same points. | −0.006 against −0.087 |
| **Q5** | Picking: where the true level is not the innermost, the most-attended open header is the true one in ≥ 0.60 of points; over all multi-header dedents, picking beats the innermost rule by ≥ 0.02. | 0.875; 0.922 against 0.852 |
| **Q6** | Steering, natural unmarked dedents: mean change in log q(x) ≥ +1.0 and ≥ 5 × the control's (control floored at 0.02); x becomes top in a share ≥ the control's + 0.05. | +2.69 against +0.09; 0.163 against 0.008 |
| **Q7** | ko_true at unmarked dedents whose true level is an open header ≤ −0.10. | −0.210 |
| **Q8** | Steering, generated terminal target points: mean change ≥ +1.0 and ≥ 5 × the control's (floored at 0.02); x becomes top in ≥ 0.20, and ≥ the control's + 0.10. | +1.43 against −0.03; 0.376 against 0.015 |
| **Q9** | `return` programs: removing the decision-token → keyword edge in all layers closes ≥ 0.40 of the gap to the neutral twin. | 0.690 |
| **Q10** | The largest single-layer share for `return` is ≤ 0.35 × the Q9 share. | 0.097 (0.14 × 0.690) |
| **Q11** | L0H2 ablated on the terminal line and the decision token changes mean q(current level) at terminal target points by at most 0.05 in absolute value. | +0.025 |

### OLMo-2-0425-1B, head L13H1 (layer 13 of 16) and group DED-O

| id | prediction | calibration |
|---|---|---|
| **O1** | As Q1, with thresholds ≥ 0.20 and ≥ 1.4 ×. | 0.318 and 0.172 |
| **O2** | As Q2 (every other layer-13 head). | −0.118; others ≤ 0.002 in absolute value |
| **O3** | As Q3. | +0.013; −0.007 |
| **O4** | As Q4. | −0.006 against −0.096 |
| **O5** | Where the true level is not the innermost, the most-attended open header is the true one in ≥ 0.45 of points, and that share exceeds the share where the innermost is most attended by ≥ 0.15. | 0.648 against 0.250 |
| **O6** | As Q6, with mean change ≥ +0.8 and ≥ 4 × the control's; top share ≥ control + 0.05. | +1.57 against +0.16; 0.186 against 0.015 |
| **O7** | ko_true at unmarked dedents ≤ −0.08. | −0.161 |
| **O8** | As Q8, with mean change ≥ +0.8 and ≥ 4 × control; top share ≥ 0.15 and ≥ control + 0.10. | +1.49 against +0.09; 0.346 against 0.017 |
| **O9** | As Q9 (≥ 0.40). | 0.880 |
| **O10** | As Q10 (≤ 0.35 × the O9 share). | 0.162 |
| **G1** | DED-O group ablation, unmarked dedents: change in mean log q(true) ≤ −0.30, and below every one of the 20 random groups. | −0.630; random −0.052 to +0.075 |
| **G2** | DED-O, terminal dedents: absolute change ≤ 0.25 × the absolute G1 change, and shallower-open mass under ablation ≥ 0.90. | −0.029; 0.975 |
| **G3** | DED-O, same-level lines: change ≥ −0.02. | +0.027 |
| **G4** | DED-O, unmarked dedents: mass on the current level rises by ≥ 0.08. | +0.186 |

OLMo-2's picking is weaker than Qwen3's: in calibration, over all multi-header dedents it did not beat the innermost rule (0.759 against 0.852). No prediction is made about that comparison for OLMo-2.

## Verdict rules

Each prediction is reported separately, pass or fail. A question is **confirmed** if all its named predictions pass, otherwise **not confirmed**; the remaining predictions qualify the account whichever way they go.

- **Question 1** (L23H2 reads the `:\n` of the header being closed): Q1, Q2, Q3. Qualifier: Q4 (the header must be open, not just recent).
- **Question 2** (the attended open header sets how far to dedent): Q5, Q6, Q8. Qualifier: Q7.
- **Question 3** (the terminal keyword is read directly by the decision token, spread over layers, not through L0H2 on the terminal line): Q9, Q10, Q11.
- **Question 4a** (both routes exist in OLMo-2): G1, G2, O9. Qualifiers: G3, G4, O10.
- **Question 4b** (L13H1 works like L23H2): O1, O2, O3, O6, O8. Qualifiers: O4, O5, O7.

## Analysis

- `code/r2c04_verdicts.py conf` computes every measure from the saved records and applies the rules above. It writes `outputs/r2/measures_conf.json` and `outputs/r2/conf/verdicts.json`.
- As description only, not used for verdicts, it adds 95% intervals from a bootstrap: 2,000 resamples, seed 0, resampling repositories for natural code and programs for generated code.
- No point, chunk, program or repository is excluded after the run. If a script crashes, it is fixed only for the crash (no change to the computation) and resumed. The fix is logged in `REPORT-2.md`. A prediction that cannot be evaluated is reported as not evaluable, not as a pass.
- `REPORT-2.md` is written with every number taken from the saved outputs, and an independent script checks them.

## Run plan

1. Freeze this file and the scripts (digests in `FROZEN.md`; read-only).
2. Run `code/r2q_conf.sh` as a `systemd-run --user` unit. It runs r2c01 and r2c02 for (qwen3-06b-bf16, L23H2) and (olmo1b, L13H1), and r2c03 for olmo1b DED-O. Each call holds the shared GPU lock for at most 1,000 s of work, and the scripts resume from their saved records.
3. Run `code/r2c04_verdicts.py conf`, then write `REPORT-2.md`.

## Scripts frozen with this protocol

`code/r2c01_header.py`, `code/r2c02_gen2.py`, `code/r2c03_group.py`, `code/r2c04_verdicts.py`, `code/r2common.py`, `code/common.py`, `code/pyindent.py`, `code/gen2.py`, `code/run.sh`, `code/r2q_conf.sh`. The smoke tests (`outputs/r2/conf_smoke/`) and calibration runs (`outputs/r2/conf_calib_*/`) used exploration material only. They are not results.
