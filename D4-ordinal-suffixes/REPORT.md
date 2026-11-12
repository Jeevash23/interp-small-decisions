# D4 report: how small models choose an ordinal suffix

<!-- nocheck -->8 November 2026. **Exploration** under `CHARTER.md` (frozen 7 Nov, 21:01 IST) and Amendment 1 used exploration frames and numbers only (`NOTEBOOK.md`). **Confirmation** under `PROTOCOL.md` (frozen 8 Nov, 04:16 IST; digests in `FROZEN.md`) ran once on the confirmation part, read for the first time, from 04:16 to 06:26 IST. The reserve part is unread. Every number in the confirmation section comes from `outputs/conf/EVALUATION.json`, written by the frozen `code/conf_evaluate.py`; `code/check_report.py` checks every number in this file against the saved outputs; the few exploration numbers outside its reach (lines marked as not machine-checked in the source) are checked by `code/check_report_extra.py`, written after the run.

## Question

How does a small language model choose between "st", "nd", "rd" and "th" after a number, including the exception for numbers ending in 11, 12 and 13 ("112th", not "112nd")? This "teen check" needs the tens digit as well as the last digit, and tokenisers place those two digits differently: one token per digit (Qwen3, SmolLM2, Gemma-4), or multi-digit chunks (OLMo-2 cuts plain numbers into threes from the left, so `1112` becomes `111` + `2`; GPT-2 and Pythia use variable chunks).

## Summary

All prospective predictions held on held-out frames and numbers.

1. **Token-boundary account (behaviour).** OLMo-2-1B's teen check works when the tens and last digits share a token and fails when they do not. In plain form the bias-free teen AUROC was 0.775 at 4 digits and 0.729 at 7 digits (the tens digit in the previous token), against 0.974–1.000 at 3, 5 and 6 digits (one token holds both). Writing the same numbers with a thousands separator, which puts the last three digits in one token, raised AUROC to 0.989 at 4 digits and 0.953 at 7. Plain four-digit teens were given the right suffix in 0.220 of cases, against 1.000 for three-digit ones. GPT-2 and Pythia-1B show the same failure on the few numbers their tokenisers split this way (teen accuracy 0.113 and 0.175, against 0.950 and 0.965 in controls).
2. **Early teen copy in Qwen3-0.6B-Base (mechanism).** Layer-0 head L0H2 attends from the last digit to the tens digit and carries it to the last position. Patching its output alone moved 0.599 (3 digits) and 0.612 (4 digits) of the suffix logit difference between a teen and a non-teen number, more than any other head, and 0.000 when only the last digit changed. Mean-ablating L0H2 and L1H6 at the last position left 0.430 and 0.459 of the teen effect at 3 and 4 digits, and dropped teen accuracy by 0.458 and 0.385 while non-teen accuracy fell by 0.000 and 0.021; random head pairs from the same layers left at least 0.994.
3. **Cross-token move in OLMo-2-1B (mechanism).** Three layer-0 heads (L0H5, L0H0, L0H9) matter only where the tens digit is in the previous token. Mean-ablating them at the last position left 0.110 of the teen effect at 4 digits and 0.076 at 7 digits, but 0.995 at 3 digits and 0.993 at 5 digits, where the tens digit is inside the last token. Random triples left at least 0.921 and 0.856 at 4 and 7 digits.

Together: in both a one-token-per-digit model and a chunked-token model, the teen check depends on an early attention step that brings the tens digit to the last token. Where the tokeniser already puts both digits in one token, no such step is needed, and OLMo-2's late layers read the teen status from the token itself. Where the step is needed in OLMo-2, it is weak, and that is where the model makes its teen errors.

## Material and measures

- Frames from five families (dates, centuries, rankings, anniversaries, positions), each ending just before the suffix ("In the world rankings, the player has dropped to `563`"). Frames and numbers were split by hash into exploration, confirmation and reserve before any model read them (`data/MANIFEST.md`; Amendment 1 extends the number split to seven digits). Confirmation items use confirmation frames and confirmation numbers only; numbers 1–99 belong to every part.
- **m** = logit of the suffix the last digit alone calls for minus logit of "th", at the number's last token. **Teen AUROC** = probability that a non-teen number ending in 1–3 has higher m than a teen (x11–x13); it does not depend on a cell's overall pull towards "th". **Teen effect** = mean m (non-teen) − mean m (teen). **Effect kept** = teen effect with heads mean-ablated at the last position ÷ unablated teen effect. **Patch fraction** = share of the source-minus-target logit difference moved by patching one head's output at the last position, over pairs answered correctly in both runs.
- 95% intervals resample distinct numbers (`2,000` resamples). Decisions used point estimates, as the protocol fixed.

## Gate (exploration)

<!-- nocheck -->Six of seven models met both frozen gate conditions (`outputs/gate_table.md`): `Qwen3-0.6B-Base`, `Pythia-1B` (replacing `Pythia-410M`, which missed condition 2 by 0.006, under the charter's fallback rule), `OLMo-2-1B`, `SmolLM2-1.7B`, `Gemma-4-E2B` and `GPT-2`.

## Confirmation results

### Token-boundary account (B1–B5)

OLMo-2-1B, 200 teen and 200 non-teen items per cell:

| Digits | Tokens (plain) | Plain AUROC [95% CI] | Comma AUROC [95% CI] | Plain teen acc. |
|---|---|---|---|---|
| 3 | one | 1.000 [0.999, 1.000] | — | 1.000 |
| 4 | 3+1 | **0.775** [0.722, 0.825] | 0.989 [0.979, 0.996] | **0.220** |
| 5 | 3+2 | 0.999 [0.997, 1.000] | 0.974 [0.959, 0.987] | 1.000 |
| 6 | 3+3 | 0.974 [0.960, 0.987] | 0.918 [0.890, 0.943] | 0.980 |
| 7 | 3+3+1 | **0.729** [0.679, 0.776] | 0.953 [0.934, 0.970] | 0.535 |

| Test | Prediction | Result | |
|---|---|---|---|
| B1 | plain 4- and 7-digit AUROC ≤ 0.80 | 0.775, 0.729 | pass |
| B2 | the seven one-token cells ≥ 0.88 | lowest 0.918 (6 digits, comma) | pass |
| B3 | comma − plain ≥ 0.10 at 4 and 7 digits | +0.213, +0.224 | pass |
| B4 | AUROC 5 − 4 ≥ 0.10 and 6 − 7 ≥ 0.10; teen acc. 4 ≤ 0.65, 3 ≥ 0.85 | +0.223, +0.245; 0.220, 1.000 | pass |
| B5 | GPT-2 and Pythia-1B: boundary teen acc. ≤ 0.50, control ≥ 0.85, AUROC gap ≥ 0.10 | GPT-2 0.113 vs 0.950, gap 0.574; Pythia-1B 0.175 vs 0.965, gap 0.140 | pass (both) |

Verdict as fixed in the protocol: **supported**.

Notes:
- The three-digit cell rests on 7 distinct teen numbers (there are few three-digit numbers ending in 11–13, and the confirmation part holds about 30% of them); the 4-digit cell on 71, and the 5- to 7-digit cells on 169–200.
- B5 rests on 10 distinct teen numbers per model, each in the confirmation rankings and positions frames (80 teen items; 2,920 and 1,048 non-teen items). Most are numbers such as `1011` or `91013`, whose zero hundreds digit makes the tokeniser cut before the tens digit. GPT-2's boundary AUROC was below chance (0.369 [0.279, 0.465]): its teens look less like teens than its non-teens. Pythia-1B's stayed well above chance (0.831 [0.763, 0.888]) while its teen accuracy collapsed, so in Pythia-1B the teen signal across the boundary is weakened, not lost.

### Early teen copy in Qwen3-0.6B-Base (Q1–Q6)

Ablation, 96 teen and 96 non-teen numbers per digit count; 20 random head pairs from layers 0–2:

| Digits | Base effect (AUROC) | L0H2+L1H6 kept | L0H2 alone | L1H6 alone | Random pairs kept | Teen acc. drop | Non-teen acc. drop |
|---|---|---|---|---|---|---|---|
| 2 | 11.691 (1.000) | 0.709 | 0.873 | 0.984 | 0.996–1.058 | 0.021 | 0.083 |
| 3 | 5.012 (0.979) | **0.430** | 0.583 | 0.942 | 0.996–1.217 | **0.458** | 0.000 |
| 4 | 4.252 (0.986) | **0.459** | 0.586 | 0.946 | 0.994–1.116 | **0.385** | 0.021 |

Patching, 96 pairs per scan, head outputs at the last position:

| Scan | Valid pairs | L0H2 | Top head |
|---|---|---|---|
| 3 digits, into a teen | 84 | **0.599** | L0H2 |
| 4 digits, into a teen | 75 | **0.612** | L0H2 |
| 3 digits, last digit only | 95 | −0.000 | L0H3, 0.598 |
| 4 digits, last digit only | 94 | −0.000 | L0H3, 0.558 |

| Test | Prediction | Result | |
|---|---|---|---|
| Q1 | both heads ablated keep ≤ 0.55 at 3 and 4 digits; every random pair ≥ 0.85 | 0.430, 0.459; random ≥ 0.994 | pass |
| Q2 | teen acc. drops ≥ 0.20, non-teen ≤ 0.10, at 3 and 4 digits | 0.458, 0.385; 0.000, 0.021 | pass |
| Q3 | 2 digits: kept ≥ 0.55, teen acc. after ablation ≥ 0.95 | 0.709; 0.979 | pass |
| Q4 | L0H2 alone ≤ 0.70, L1H6 alone ≥ 0.80, at 3 and 4 digits | 0.583, 0.586; 0.942, 0.946 | pass |
| Q5 | L0H2 the top head and ≥ 0.40 in teen scans at 3 and 4 digits | 0.599, 0.612, top in both | pass |
| Q6 | L0H2 within ±0.05 and top head ≥ 0.30 in last-digit scans | −0.000, −0.000; 0.598, 0.558 | pass |

Verdict: **supported**. Q3 uses only the numbers 11, 12 and 13, which also appear in exploration.

### Cross-token move in OLMo-2-1B (O1–O2)

Ablation of L0H5+L0H0+L0H9 at the last position, 96 teen and 96 non-teen numbers per length, plain form; 20 random triples from layers 0–2:

| Digits | Tens digit | Base effect (AUROC) | Triple kept | L0H5 / L0H0 / L0H9 alone | Random triples kept |
|---|---|---|---|---|---|
| 3 | in the last token | 6.243 (0.998) | 0.995 | 0.997 / 0.999 / 0.997 | 0.989–1.007 |
| 4 | in the previous token | 1.210 (0.768) | **0.110** | 0.608 / 0.781 / 0.724 | 0.921–1.047 |
| 5 | in the last token | 3.711 (0.995) | 0.993 | 0.994 / 1.004 / 1.001 | 0.994–1.020 |
| 7 | in the previous token | 0.620 (0.651) | **0.076** | 0.629 / 0.715 / 0.745 | 0.856–1.031 |

| Test | Prediction | Result | |
|---|---|---|---|
| O1 | (read: base effect ≥ 0.30 at 4 and 7 digits, 1.210 and 0.620) kept ≤ 0.65 at 4 and 7 digits; every random triple ≥ 0.85 | 0.110, 0.076; random ≥ 0.921 and ≥ 0.856 | pass |
| O2 | kept ≥ 0.90 at 3 and 5 digits | 0.995, 0.993 | pass |

Verdict: **supported**. The lowest random triple at 7 digits (0.856) was close to the 0.85 bar; at 7 digits the base teen effect is small (0.620 logits), so ratios there are less stable than at 4 digits.

## What this does and does not show

- **Shown on held-out material:** whether a small model gets "112th" right depends on whether its tokeniser puts the tens and last digits in the same token; the thousands separator repairs the four- and seven-digit failures in OLMo-2; in Qwen3-0.6B-Base one layer-0 head carries the tens digit for the teen check and does nothing for the last digit; in OLMo-2-1B three layer-0 heads are needed only across the token boundary.
- **Partial mechanisms.** About 0.43–0.46 of Qwen3's teen effect survives ablating both heads, so other routes exist (exploration points to L5H2 and later heads). Two-digit teens mostly survive the ablation (0.709 kept), so they use a further route that was not identified. The MLPs that turn the copied digit into the suffix (Qwen3 MLP0, 16, 18, 24, 25; OLMo-2 MLP10, 11, 13 in exploration) were not tested on held-out material.
<!-- nocheck -->- **Not explained.** Qwen3's teen accuracy differs by frame family (exploration: 0.755 in rankings against 0.973 in anniversaries) although L0H2's attention does not (exp06, a dead end). The errors arise downstream of the copy. GPT-2's and Pythia-1B's internals were not studied, and their boundary results rest on ten numbers each.
- **Scope.** Base models with under two billion parameters; English frames; numbers up to seven digits; one-shot suffix choice among four tokens.

## Records

`CHARTER.md`, `AMENDMENTS.md`, `PROTOCOL.md` and `FROZEN.md` (frozen files and digests); `NOTEBOOK.md` (exploration, dead ends included); `outputs/conf/` (confirmation outputs and `EVALUATION.json`); `outputs/` and `outputs/explore/` (exploration outputs).
