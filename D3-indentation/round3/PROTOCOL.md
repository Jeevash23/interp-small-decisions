# D3 confirmation protocol: two routes to a dedent in Qwen3-0.6B-Base

Written 8 November 2026, about 06:30 IST, after the exploration recorded in `NOTEBOOK.md` (7 Nov 21:04 to 8 Nov 06:15) under `CHARTER.md` and Amendment 1. It is frozen (digests in `FROZEN.md`, file made read-only) before any script reads a sealed confirmation part. It is run **once**. Everything below is **prospective and confirmatory**. Anything else computed on the confirmation parts is labelled post hoc.

## Model and why it is the main model

Qwen3-0.6B-Base (Hugging Face `Qwen/Qwen3-0.6B-Base`, on disk), loaded in bfloat16 with SDPA attention. Pythia-410M failed gate item 1 and Qwen3-0.6B-Base passed every item, so the charter's fallback rule makes it the main model (NOTEBOOK, 7 Nov 23:05). bfloat16 was checked against float32 on the exploration scan: base values agree to 0.002, and component effects correlate at r ≥ 0.993 (NOTEBOOK, 8 Nov 01:18).

## The candidate mechanism (from exploration)

At the decision token (the token that ends the previous line, such as `:\n` or `\n`):

1. **Block-end route.** A set of attention heads, DED = {L23H2, L17H3, L17H4, L17H5, L18H6}, is needed for dedents that are not preceded by a terminal statement ("other dedents": the end of a block such as an `if` body after an ordinary statement). Without DED the model keeps the current level at those points, and same-level lines become more certain. L23H2 writes the target run token directly (direct logit attribution).
2. **Terminal-statement route.** A different set of heads carries "the previous line was `return`/`pass`/`raise`/`break`/`continue`" to the decision token: CARRY = {L17H4, L11H9, L5H6, L17H6, L16H2, L6H5} (found by patching `return …` against `value = value`). Mean-ablating a nine-head set, TERM = {L11H9, L5H6, L17H6, L16H2, L23H6, L22H8, L23H3, L25H9, L19H6}, hurts terminal-statement dedents in natural code but not other dedents.
3. **Back-up.** In natural code, terminal-statement dedents survive the loss of either route but suffer much more when both are removed.

Exploration also found what this account does **not** cover, and those findings are not tested here: the model is not misled by a keyword at the start of an identifier (`return_value = …`); how far to dedent is barely linearly decodable at the decision token; and in generated code DED also supports terminal dedents.

## Material (sealed; digests in `FROZEN.md`)

- **Natural code:** `data/newcode/part-conf.jsonl`, 1,606 chunks of Python from 123 GitHub repositories created on or after 1 Sep 2026 (after the model's release). It was split from the exploration part by repository hash and has never been read by a model.
- **Generated programs:** `data/gen/part-conf.jsonl`, 2,008 programs (widths 2, 3, 4 and mixed), split by seed hash; never read by a model.
- All chunks of each part are used. Decision points, candidate indents and the indent distribution are computed exactly as in exploration: `code/pyindent.py` and `code/common.py`, including `prepare` as amended (Amendment 1) and `RootMasses`. Only points whose candidates are all identified at the first token are used, which for Qwen3 is every point the measure keeps.
- Point classes: **header** (the line opens a block), **same**, **term** (dedent after a terminal statement), **dedent** (any other dedent).

## Interventions and controls

- **Mean ablation (c01):** the named heads' outputs at every decision token are replaced by their mean over all decision tokens of the same part (per head, at the input of the attention output projection). Measures, per class: mean log q(true); mean mass on the current level; mean mass on shallower open levels. q is the renormalised candidate distribution.
- **Random control groups (c01, c02):** for every head of a named group, a head drawn uniformly from the other heads of the same layer. Seeds fixed in the scripts (`random.Random(1000 + k)` for c01 and `random.Random(2000 + k)` for c02, k = 0–19). 20 draws per group (DED, TERM, CARRY, and DED in c02). TERMDED has no random draws.
- **Pair patching (c02):** pairs are built from the generated confirmation programs exactly as in exploration (e04, e08). The line before a terminal-statement dedent is replaced by `value = value` at the same indent, and the score is log q(true) − log q(current level). Up to 300 pairs, the first eligible point per program, in a fixed shuffle. A group's outputs at the decision token are patched from the clean run into the corrupt run (sufficiency) and from the corrupt run into the clean run (necessity). Each is reported as a share of the mean clean-minus-corrupt gap.

## Predictions (each could fail)

Thresholds were set from the exploration values (given in brackets), with margins. Every comparison with random groups is against **all 20 draws**: "below every random group" means below the minimum of the 20.

| part | prediction | exploration values |
|---|---|---|
| **H1** | DED, natural, other dedents: the change in mean log q(true) is ≤ −0.40 and below every random group. | −0.99 and −1.01; random at most −0.04 from base |
| **H2** | DED, natural, other dedents: the change in mass on the current level is ≥ +0.10 and above every random group. | +0.29 and +0.28; random at most +0.02 |
| **H3** | DED, natural, terminal dedents: shallower-open mass is ≥ 0.95 under ablation, and the absolute change in mean log q(true) is ≤ 0.25 × the absolute H1 change. | 0.987 and 0.989; ratios 0.025 and 0.023 |
| **H4** | DED, natural, same-level lines: the change in mean log q(true) is ≥ −0.01 (not hurt). | +0.031 and +0.030 |
| **H5** | TERM, natural, terminal dedents: the change in mean log q(true) is ≤ −0.03 and below every random group. | −0.062; random min −0.004 from base |
| **H6** | TERM, natural, other dedents: the absolute change in mean log q(true) is ≤ 0.10. | −0.028 |
| **H7** | TERM + DED together, natural, terminal dedents: the change in mean log q(true) is more negative than the sum of the TERM-alone and DED-alone changes (more than additive). | −0.256 against −0.087 |
| **H8** | DED, generated, other dedents: the change in mean log q(true) is ≤ −0.25 and below every random group. | −0.53; random at most −0.05 |
| **H9** | CARRY, generated pairs: necessity ≥ 0.40 and sufficiency ≥ 0.20, each above every random group. | 0.640 and 0.362; random max 0.047 and 0.039 |

How the parts read together:
- H1–H4 test the block-end route;
- H5 and H6 with H1 and H3 test the two-route dissociation in natural code;
- H7 tests the back-up;
- H8 tests the block-end route in unusual code;
- H9 tests the terminal-signal carriers.

Each part is reported separately, pass or fail. The **candidate is called confirmed** if H1, H2, H3, H5 and H6 all pass. It is **partly confirmed** if H1–H3 pass but H5 or H6 fails (the block-end route holds, the dissociation does not). Otherwise it is **not confirmed**. H4, H7, H8 and H9 qualify the account whichever way they go.

## Analysis

- `code/c03_verdicts.py` applies the rules above to the saved outputs and writes `outputs/conf/verdicts.json`.
- As description only, not used for verdicts, it adds 95% intervals for each named-group change from a bootstrap that resamples repositories (natural code) or programs (generated code), 2,000 resamples, seed 0.
- No point, chunk or repository is excluded after the run. If a script crashes, it is fixed only for the crash (no change to the computation) and resumed. The fix and its reason are logged in `REPORT.md`.
- If a prediction cannot be evaluated (for example, too few points in a class), it is reported as not evaluable, not as a pass.

## Run plan

1. Freeze this file and the scripts (digests in `FROZEN.md`); make them read-only.
2. Run `c01_ablate.py newcode`, `c01_ablate.py gen` and `c02_pairs.py` under the shared GPU lock, in chunks of at most about 18 minutes (the scripts resume from their saved runs).
3. Run `c03_verdicts.py`, then write `REPORT.md`, with every number taken from the saved outputs and checked by a script.

## Scripts frozen with this protocol

`code/c01_ablate.py`, `code/c02_pairs.py`, `code/c03_verdicts.py`, `code/common.py`, `code/pyindent.py`, `code/run.sh`. Pipeline smoke tests ran on exploration material only (`outputs/conf_smoke/`); they are not results.
