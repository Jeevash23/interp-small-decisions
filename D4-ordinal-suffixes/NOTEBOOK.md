# D4 notebook (exploratory)

Everything here is **exploratory**, under `CHARTER.md` (frozen 7 Nov 2026, 21:01 IST). Exploration items only; the conf and reserve parts are unread. Numbers come from saved outputs in `outputs/`; gate tables are printed by `code/gate_summary.py` from `outputs/gate01_<model>.jsonl` and saved as `outputs/gate_summary_<model>.json`.

## 7 Nov 2026

### 21:00 Tokenisers (before the charter froze)
- Checked every charter tokeniser on " the {N}{suffix}". Corrections to the GAPS-2 draft written into the charter: Pythia splits four-digit numbers 2+2, 1+3, 3+1 or keeps them whole, and only 16 four-digit numbers ending in 10–19 split the tens digit from the last digit; OLMo-2 splits every four-digit number 3+1 from the left, so all 900 such teen numbers do. Pythia also writes 356 three-digit numbers as 1+2 (` 5` + `58`).

### 21:02 Material sealed
- `code/build_material.py`: 68 frames in five families, numbers 1–9,999; explore 88,480 items, conf 26,287, reserve 7,145 (`data/MANIFEST.md`). Digests in `FROZEN.md`.

### 21:05–21:30 Gate (`code/gate01_behaviour.py`, `code/gate_summary.py`)
- Sample per model: 10,886 exploration items (every N ≤ 99; every N ≥ 100 ending in 11–13; 3,000 random three-digit and 3,000 random four-digit others). Measure: top-1 among st/nd/rd/th at the number's last token. Pipeline smoke run (200 items, Pythia-410M) first, labelled `_smoke`, not a result.
- Model loading: float32 for Pythia-410M, GPT-2 and Qwen3-0.6B-Base; bfloat16 for the 1B+ models; Gemma-4-E2B with D2's text-only loader.

| Model | G1 dates 1–31 (≥ 0.90) | G2 32–999 ending 1/2/3 (≥ 0.80) | Qualifies |
|---|---|---|---|
| Qwen3-0.6B-Base | 1.000 (248) | 0.977 (1,216) | yes |
| Pythia-410M | 1.000 (248) | 0.794 (1,216) | **no** (narrow miss) |
| Pythia-1B (fallback for Pythia-410M) | 1.000 | 0.938 | yes |
| OLMo-2-1B | 1.000 | 0.965 | yes |

- **Pythia-410M fails condition 2** by 0.006. The thresholds are the charter's and are not moved. Pattern: perfect on 1–99 and on every "th" case, but for larger numbers ending in 1, 2 or 3 it drifts to "th": three-digit 0.70–0.72, four-digit 0.23–0.29. Teens look right (0.99) only because "th" is the default. This is the "knows only the frequent cases, defaults to th" outcome the charter foresaw. Under the charter its fallback, Pythia-1B, was gated and qualifies; Pythia-1B takes Pythia-410M's place as the multi-digit main model.
- **Qwen3-0.6B-Base** (one token per digit): the 11–13 exception is perfect for two-digit numbers (102/102) but 0.834 for three-digit (320) and 0.857 for four-digit (2,212). The errors pick the last digit's suffix ("111st", "1112nd"): of the four-digit teen errors, st 152, nd 44, rd 120. "th" cases 1.000 throughout.
- **OLMo-2-1B, the boundary prediction, behaviour only.** Every four-digit number is two tokens, 3+1 (` 111` + `2`), so the tens digit sits in the token before the last digit. Four-digit teens: **0.463** correct (2,212 items), with st 408, nd 341, rd 438 errors, against 0.944 for three-digit teens (one token) and 0.965 for four-digit numbers with other endings. Exactly the error the tokeniser account predicts.
- **Pythia-1B by split** (four-digit teens): 2+2 (tens and last digit in one token, e.g. ` 11` + `12`) 1.000 (1,806); 1+3 (` 1` + `112`) 0.680 (378); 3+1 (` 101` + `1`) 0.071 (14). Three-digit teens written as one token 0.988. So when the tens digit is in the same token as the last digit Pythia is right, when it is in the previous token it is almost always wrong, and the in-between case (both in a no-space three-digit token) is partial.
- Not yet interpreted: why Pythia-1B's 1+3 teens (both digits in the last token) are worse than whole three-digit tokens. Possibly the no-space `112` token is rarer as a number ending than ` 112`.

### 21:20 Exp01 pilot, Qwen3-0.6B-Base (`code/exp01_patch.py`, n = 24 pairs per type; pilot)
- Residual patching, source into target, by layer and position from the end; fraction of the source–target logit difference moved.
- Last-digit pairs (341 → 342): the whole effect sits at the last position at every layer (1.00), none at the tens position. The last digit is read where the suffix is predicted, as expected.
- Tens pairs (322 ↔ 312): already at layer 0 the effect is split between the last position (0.35–0.61) and the tens position (0.40–0.66). By layer 2 the last position carries 0.55–0.86; the rest stays at the tens position until about layer 19, and moves to the last position at layers 19–20 (last position 0.93–0.98 at layer 20). So the tens digit reaches the suffix position in two steps: early (layers 0–1) and late (around layer 19–20). Three-digit numbers leave more at the tens position until layer 19 (0.20–0.36) than two-digit (0.10–0.22).
- Next: the full run (160 pairs), heads at layers 0–1 and 19–20 that attend from the last digit to the tens digit, and whether the late step explains the three- and four-digit teen errors.

### 21:40 Gate complete, all seven models (`outputs/gate_table.md`, printed by `code/gate_table.py`)

| Model | G1 | G2 | Qualifies | Teens 2-digit | Teens 3-digit | Teens 4-digit | 4-digit ending 1/2/3 |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 1.000 | 0.977 | yes | 1.000 | 0.834 | 0.857 | 0.825 |
| Pythia-410M | 1.000 | 0.794 | no | 1.000 | 0.994 | 0.988 | 0.265 |
| Pythia-1B | 1.000 | 0.938 | yes | 1.000 | 0.991 | 0.939 | 0.659 |
| OLMo-2-1B | 1.000 | 0.965 | yes | 1.000 | 0.944 | 0.463 | 0.879 |
| SmolLM2-1.7B | 1.000 | 0.986 | yes | 1.000 | 0.838 | 0.785 | 0.788 |
| Gemma-4-E2B | 1.000 | 0.999 | yes | 1.000 | 0.953 | 0.874 | 0.903 |
| GPT-2 Small | 1.000 | 0.842 | yes | 1.000 | 0.963 | 0.907 | 0.612 |

- Gate passed: the study goes on. Qwen3-0.6B-Base (one token per digit) and Pythia-1B (in place of Pythia-410M) are the main pair; OLMo-2-1B carries the boundary question. The Qwen3 fallbacks were not needed.
- GPT-2 Small by split, four-digit teens: 2+2 1.000 (1,820), 1+2 1.000 (120), 1+3 0.586 (266), 3+1 0.143 (112). The same ordering as Pythia-1B (2+2 1.000, 1+3 0.680, 3+1 0.071).
- Single-digit models (Qwen3, SmolLM2, Gemma-4) are all perfect on two-digit teens and lose accuracy on three- and four-digit teens, with last-digit errors ("1112nd").
- Teen errors by frame (`code/errmap.py`, exploratory): Qwen3-0.6B-Base three-digit teens are 0.979 right in anniversary frames, 0.884 in positions and 0.661 in rankings (four-digit: positions 0.947, rankings 0.768), so its errors depend on context. OLMo-2's four-digit teen accuracy is flat across frames (0.445–0.482) and hundreds digits (0.383–0.549): a tokenisation effect, not a context one.

### 21:50 Thousands separator and longer numbers (tokenisers only), Amendment 1
- With a thousands separator, OLMo-2, Pythia and GPT-2 put the last three digits in one token (`1`, `,`, `112`), so the tens digit is back in the last token. Qwen3 still writes one token per digit. OLMo-2 cuts longer numbers into threes from the left (` 121` + `12`; ` 112` + `112`; ` 111` + `211` + `2`), so a token account predicts teen accuracy that rises and falls with length: high at three digits, low at four, high at five and six, low at seven.
- Numbers above 9,999 are not in the frozen material, so `AMENDMENTS.md` Amendment 1 extends the charter's hash split to 10,000–9,999,999 (same rule) before any model read such a number. `is_explore_number()` in `code/common.py` applies it; checked to agree with the frozen `data/numbers.json` on all 9,999 numbers.
- `code/exp03_length_sep.py` (constructed inputs, exploration frames and numbers): accuracy by length 2–7 digits, plain against comma form, by whether the last token holds the tens digit. Queued for all six qualifying models.

### 21:55 Exp01 full run stopped (dead end in the tooling, not in the science)
- The full exp01 run (160 pairs, 2–4 digits) one forward pass per patch would have run past the 20-minute lock limit before saving. Stopped after two pair types; its log is kept as `outputs/exp01_patch_qwen3-06b_n160_STOPPED.log` and is not used. The pilot outputs were renamed `_pilot_n24` and `_pilot_n16`.
- Replaced by `code/exp04_scan.py`: the same pairs and metric, but each forward pass carries many copies of the batch, each with a different patch (residual by layer and position, single head at the last position, single MLP at the last position). One job per digit count and pair type, each saving its own file.

### Time correction (21:52)
- Section times above were my estimates and run ahead of the clock. Real times from file timestamps: gate table written 21:27 (heading "21:40"), Amendment 1 frozen 21:29 ("21:50"), exp01 stop about 21:35 ("21:55"). From here on, times come from `date` or file timestamps.

### 21:54 Exp03, OLMo-2-1B: length and separator sweep (`outputs/exp03_olmo2-1b.jsonl`, `outputs/teen_effect_olmo2-1b.json`)
- 120 teen and 120 non-teen items ending in 1–3 (plus 60 "th") per length and form; exploration frames (rankings, positions) and exploration numbers.
- Accuracy on teens, plain form: 2 digits 1.000, 3 digits 0.950, 4 digits **0.433**, 5 digits 0.992, 6 digits 0.942, 7 digits **0.542**. The rise and fall the token account predicts. With a comma (`1`, `,`, `112`): four-digit teens 0.900, seven-digit 0.917.
- Caveat found: plain five- and six-digit numbers ending in 1–3 (not teen) drop to 0.550 and 0.642, so raw teen accuracy is helped by a pull towards "th" when the last token is a no-space two- or three-digit chunk. Raw accuracy mixes the teen check with this bias.
- **Bias-free measure** (`code/teen_effect.py`): m = logit(st/nd/rd for the last digit) − logit(th); teen effect = mean m (non-teen) − mean m (teen), and the AUROC of m between the two. A pure "th" bias moves both equally.

| Digits, form | Tens digit in last token? | Effect | AUROC |
|---|---|---|---|
| 2 plain | yes | +12.06 | 1.000 |
| 3 plain | yes | +5.23 | 0.994 |
| 4 plain | **no** | +0.99 | **0.699** |
| 4 comma | yes | +3.47 | 0.937 |
| 5 plain | yes | +3.35 | 0.965 |
| 5 comma | yes | +3.18 | 0.932 |
| 6 plain | yes | +2.52 | 0.956 |
| 6 comma | yes | +3.06 | 0.930 |
| 7 plain | **no** | +0.89 | **0.694** |
| 7 comma | yes | +3.05 | 0.941 |

- Reading: when the tens digit is in the last token, OLMo-2 separates teens from other numbers almost perfectly (AUROC 0.93–1.00); when it is in the previous token, separation falls to about 0.70 and the effect to about a third or less. So OLMo-2 reads the teen status mainly from the last token's own identity, with a weak cross-token route. Candidate prediction for the protocol: the AUROC split by "tens digit in last token" on held-out numbers and frames.

### 22:04 Exp03, Pythia-1B (`outputs/exp03_pythia-1b.jsonl`, `outputs/teen_effect_pythia-1b.json`)
- Random numbers of 4–7 digits almost never put Pythia's tens digit in the previous token (only 3 of 600 teen items at 5–7 digits; none at 4 or 7), so this sweep cannot test the boundary in Pythia. The gate's 14 split-3+1 teens (1/14 right) remain the only Pythia boundary cases; GPT-2 has 112 in the gate (0.143).
- Where the tens digit is in the last token: teen AUROC 1.000 (2 digits), 0.990 (3), 0.937 (4), 0.955 (5), 0.952 (6), 0.963 (7), plain form.
- Comma forms are slightly worse: AUROC 0.82–0.87 with the last token a no-space three-digit chunk (`112`), and the "th" pull is stronger (mean m on non-teen items below zero from five digits). Same direction as the gate's split-1+3 teens (0.680): no-space three-digit tokens carry the teen status less cleanly than two-digit ones.
- Decision: Pythia-1B stays the multi-digit main model for the whole-token question (how the suffix is stored in number tokens), but the boundary prediction is tested in OLMo-2-1B (all four-digit numbers) and GPT-2 (gate split 3+1), where enough items exist.

### 22:08 Exp04 pilot and Exp05 ablation, Qwen3-0.6B-Base
- Exp04 pilot (24 three-digit pairs, 322 → 312; `outputs/exp04_scan_qwen3-06b_digits_3_kind_intoteen_n_24.log`, 43 s): heads at the last position L0H2 +0.66 (attention to the tens digit 0.88), L11H9 +0.41 (0.03 to the tens digit, so it reads something already at the last position or elsewhere), L1H6 +0.21 (0.97), L19H2 +0.11, L5H2 +0.11 (0.82). MLPs: 0 +0.64, 24 +0.66, 18 +0.38, 16 +0.35, 25 +0.27, 27 −0.18. Residual: tens position holds 0.65 at layer 0, 0.44 at layer 2, 0.08 at layer 19, 0.04 at layer 20.
- Exp05 (`code/exp05_ablate.py`, `outputs/exp05_qwen3-06b_0.2-1.6.json`): mean ablation of L0H2 and L1H6 at the last position; 96 teen and 96 non-teen items ending in 1–3 per digit count; 20 random pairs of heads from layers 0–2 as controls.

| Digits | Base effect (AUROC) | Both ablated: effect kept (AUROC) | L0H2 alone | L1H6 alone | Random pairs, kept | Teen acc. base → ablated | Non-teen acc. base → ablated |
|---|---|---|---|---|---|---|---|
| 2 | +11.44 (1.000) | 0.70 (0.993) | 0.89 | 0.99 | 1.00–1.05 | 1.000 → 1.000 | 1.000 → 0.906 |
| 3 | +4.06 (0.988) | **0.37** (0.785) | 0.53 | 0.93 | 0.99–1.01 | 0.875 → **0.302** | 0.979 → 0.958 |
| 4 | +2.74 (0.895) | **0.36** (0.716) | 0.50 | 0.93 | 0.99–1.04 | 0.833 → **0.531** | 0.844 → 0.812 |

- Reading: in three- and four-digit numbers the teen check depends largely on L0H2 (with L1H6 adding to it), heads in the first two layers that attend from the last digit to the tens digit. Ablating them turns "112th" into "112nd" in most cases while barely touching non-teen numbers; no random pair moves anything. Two-digit teens survive the ablation: there is another route for them (perhaps a stored pattern for frequent two-digit ordinals). This matches the behaviour: two-digit teens are perfect, longer ones are not.
- Candidate mechanism (Qwen3-0.6B-Base): **early teen copy**. L0H2 and L1H6 copy the tens digit into the last digit's position in layers 0–1; MLP0 and the late MLPs (16, 18, 24, 25) turn it into the suffix choice. The last digit's own suffix is read at its own position (exp01: last-digit patches act only there).

### 22:26 Exp03 GPT-2 and Qwen3; Exp06 (dead end)
- Bias-free teen measure (`outputs/teen_effect_<model>.json`):
  - **Qwen3-0.6B-Base** (tens digit always the previous token): AUROC 1.000 (2 digits), 0.968 (3), 0.921 (4), 0.915 (5), 0.897 (6), 0.903 (7); comma forms similar (0.903–0.945). A smooth decline, no rise and fall, as the token account expects when every digit is a token.
  - **GPT-2**, tens digit in the last token: AUROC 0.931–1.000 in plain form; comma forms 0.762–0.816 (last token a no-space three-digit chunk, with a strong pull to "th"). Tens digit in the previous token (random numbers rarely do this): four-digit plain, 10 teen and 20 non-teen items, AUROC **0.230** (teens look *less* like teens than non-teens do); other cells have 0–3 teen items and are not read.
- **Exp06** (`code/exp06_attn_errors.py`, `outputs/exp06_qwen3-06b.jsonl`; 2,000 gate teen items with three or four digits): L0H2 attends 0.882 to the tens digit when the answer is right and 0.883 when wrong (AUROC 0.414); L1H6 0.952 against 0.949 (0.513). Attention is the same in every frame (0.881–0.882), while accuracy is 0.973 in anniversaries, 0.939 in positions and 0.755 in rankings. **Dead end for explaining errors:** the copy step runs the same everywhere; the frame-dependent teen errors arise later, where the copied tens digit and the context-driven suffix preference are combined.

### 22:27 Exp04 metric fixed (tooling)
- The three-digit Qwen3 outteen scan (513 → 523 direction) gave fractions in the tens (L9H9 +13.86): 27% of its teen sources are themselves answered wrongly, and dividing by near-zero per-pair gaps blew the mean up. From 22:26 `exp04_scan.py` reports the moved logit difference summed over pairs whose source and target are both answered correctly, divided by the summed gap over the same pairs (`n_valid` saved). The three three-digit outputs made under the old metric are renamed `_oldmetric` and not used; the intoteen and outteen runs are queued again (`chain_exp04b.sh`). The intoteen pilot numbers above (old metric, 91–99% of pairs valid) agree with the new runs to be checked.

### 22:46 Exp03, SmolLM2-1.7B and Gemma-4-E2B (one token per digit)
- Teen AUROC, plain form, 2–7 digits: SmolLM2 1.000, 0.972, 0.879, 0.867, 0.911, 0.878; Gemma-4 1.000, 0.998, 0.945, 0.954, 0.979, 0.937. Comma forms similar (SmolLM2 0.802–0.865, Gemma 0.963–0.992). Like Qwen3: high at two digits, lower and roughly flat beyond, no rise and fall. So the rise and fall is specific to OLMo-2's chunked tokeniser, as the token account predicts; the one-token-per-digit models all read the tens digit across tokens, with some loss for longer numbers.

### 02:46 (8 Nov) Conf01 pipeline check on exploration material; Exp04 scans for Qwen3 (2 and 4 digits) and OLMo-2 (3 and 4 digits)
- **Conf01 pipeline check** (`code/conf01_boundary.py --part explore`, `outputs/explore/conf01_<model>_summary.json`; finished 23:04). The script runs end to end for all three models. Exploratory numbers, the same material family as exp03 but a fresh draw:

| Model | Cell | Teen / non-teen items | AUROC | Teen acc. | Non-teen acc. |
|---|---|---|---|---|---|
| OLMo-2-1B | 3 digits plain (tens in last token) | 200 / 200 | 0.990 | 0.950 | 0.945 |
| | 4 plain (tens before) | 200 / 200 | **0.724** | **0.435** | 0.875 |
| | 5 plain (in) | 200 / 200 | 0.982 | 0.995 | 0.550 |
| | 6 plain (in) | 200 / 200 | 0.939 | 0.910 | 0.555 |
| | 7 plain (before) | 200 / 200 | **0.708** | 0.590 | 0.680 |
| | 4–7 comma (in) | 200 / 200 each | 0.934–0.964 | 0.865–0.915 | 0.730–0.900 |
| GPT-2 | boundary set (tens before) | 168 / 10,136 | **0.450** | **0.167** | 0.747 |
| | control (tens in last token, 4–5 digits) | 200 / 200 | 0.947 | 0.955 | 0.590 |
| Pythia-1B | boundary set | 140 / 3,780 | 0.803 | **0.321** | 0.924 |
| | control | 200 / 200 | 0.948 | 0.955 | 0.610 |

  - OLMo-2 repeats exp03 closely: the two cells where the tens digit sits in the previous token (4 and 7 digits, plain) are the only low ones, and the thousands separator, which puts the last three digits in one token, restores them. 5- and 6-digit plain non-teens are pulled to "th" (non-teen accuracy 0.55), so raw accuracy is a poor measure there; the bias-free AUROC is the main measure.
  - GPT-2's boundary set is below chance (0.450), as in the gate. Pythia-1B's boundary AUROC is 0.803, much higher than GPT-2's, yet its teen accuracy collapses (0.321 against 0.955 in the control): in Pythia the teen signal across the boundary is weakened, not gone. So a prediction for Pythia on AUROC alone would be fragile; teen accuracy is the clearer measure for these two models.
  - **Caution on n:** the boundary teen items come from only 12 (GPT-2) and 10 (Pythia-1B) distinct teen numbers, each in the 14 exploration frames, mostly of the forms x,x11–x,x13 with a zero hundreds digit (1012, 3011, 81011 …). Items within a number are not independent. On the conf part there will be fewer still, so B5 stays secondary.
- **Exp04, Qwen3-0.6B-Base, new metric** (n = 96 pairs each; valid = pairs with source and target both right):
  - 2 digits (96 valid of 96): intoteen L0H2 +0.35, L11H9 +0.29, L1H6 +0.07; outteen L0H2 +0.47, L11H9 +0.30; last-digit pairs L0H3 +0.55, L0H7 +0.41, L11H9 +0.27.
  - 4 digits: intoteen (63 valid) L0H2 +0.65, L11H9 +0.41, L1H6 +0.15, L5H2 +0.14; outteen (62 valid) L0H2 +0.45, L11H9 +0.26, L19H2 +0.12; last-digit pairs (85 valid) L0H3 +0.62, L0H7 +0.36, L11H9 +0.31.
  - MLPs in every teen scan: MLP0 (+0.25 to +0.62), MLP18 (+0.46 to +0.55), MLP24 (+0.58 to +0.64), MLP25 (+0.30 to +0.36).
  - Reading: L0H2 is the top head for every teen pair type at 2 and 4 digits, and is absent from the last-digit scans, where L0H3 and L0H7 (attending to the last position itself) carry the digit. L11H9 appears in all three pair types with almost no attention to the tens digit, so it is a general suffix head downstream, not part of the teen check proper. At 2 digits L0H2's share is smaller (+0.35 to +0.47), consistent with exp05's finding that two-digit teens survive its ablation.
- **Exp04, OLMo-2-1B** (16 layers × 16 heads; 4-digit numbers are always split three + one):
  - 3 digits, one token (`331`): no head at the last position matters (largest +0.02, sum of all heads +0.05 to +0.09); residual patches at the last position carry 1.00 from layer 0 onwards, and the late MLPs do the work (MLP13 +0.65 to +0.68, MLP11 +0.35 to +0.39, MLP10 +0.21 to +0.22). The teen status is read from the token itself, with nothing moved between positions.
  - 4 digits (`834` + `3`; only 32 and 30 valid pairs because teen accuracy is low): three layer-0 heads carry the tens-digit token to the last position, L0H5 (+0.25 / +0.22, attention to the previous token 0.84), L0H0 (+0.24 / +0.18) and L0H9 (+0.18 / +0.17), with L0H3, L1H8, L3H11 smaller. The residual at the previous token holds 0.36 at layer 0, 0.12 at layer 2 and nothing from layer 3; MLP0 +0.50, MLP1 +0.25 to +0.37, then the same late MLPs (13 +0.75, 11 +0.35).
  - Reading: OLMo-2 uses the same late MLPs whether the tens digit arrives inside the last token or from the previous one, but in the second case it needs an early cross-token move by layer-0 heads, and that move is where the teen signal is lost (4-digit AUROC 0.72 against 0.99 for 3 digits). Candidate for the protocol if an ablation confirms necessity: queued as `chain_exp05olmo.sh` (mean ablation of L0H5+L0H0+L0H9 at the last position, 3/4/5/7 digits, 20 random triples from layers 0–2).
- Qwen3 three-digit reruns with the new metric (`chain_exp04b.sh`) started 02:44.

### 03:50 Qwen3 three-digit patching rerun (final metric); evaluator dry run
- `chain_exp04b.sh` done. Three digits, ratio-of-sums metric: intoteen (86 of 96 pairs valid) L0H2 +0.63 (attention to the tens digit 0.88), L11H9 +0.37, L1H6 +0.17, L5H2 +0.11; MLP0 +0.58, MLP24 +0.61, MLP18 +0.40, MLP16 +0.32. Outteen (67 valid; 27% of teen sources are answered wrongly) L0H2 +0.47, L11H9 +0.18, L19H2 +0.10. The old-metric pilot values (+0.62, +0.39, +0.17) hold.
- `code/conf_evaluate.py --part explore` (a dry run of the protocol's evaluator on exploration outputs; not results) runs end to end and every draft prediction holds on exploration material, as it should, since the thresholds were set from it. Q6 at three digits uses the `last` scan made under the old metric (no `n_valid`; 99–100% of sources and targets right, so the two metrics nearly agree).

### 04:16 Exp05 on OLMo-2-1B: the layer-0 heads matter only across a token boundary
- `chain_exp05olmo.sh` (queued 02:44; it waited for the shared GPU lock until about 04:05), `outputs/exp05_olmo2-1b_0.5-0.0-0.9.json`. Mean ablation of L0H5+L0H0+L0H9 at the last position, 96 teen and 96 non-teen numbers per length, plain form, 20 random triples from layers 0–2:

| Digits (tokens) | Base effect (AUROC) | Triple ablated: kept (AUROC) | L0H5 / L0H0 / L0H9 alone | Random triples kept | Teen acc. base → ablated |
|---|---|---|---|---|---|
| 3 (one token) | +5.41 (0.991) | 1.00 (0.990) | 1.00 / 1.00 / 1.00 | 0.99–1.02 | 0.958 → 0.969 |
| 4 (3+1) | +1.49 (0.751) | **0.41** (0.614) | 0.78 / 0.84 / 0.85 | 0.94–1.02 | 0.531 → 0.344 |
| 5 (3+2) | +3.83 (0.971) | 0.99 (0.971) | 0.99 / 1.00 / 0.99 | 0.98–1.00 | 0.979 → 0.979 |
| 7 (3+3+1) | +0.87 (0.674) | **0.26** (0.552) | 0.69 / 0.75 / 0.81 | 0.94–1.01 | 0.510 → 0.469 |

- Reading: the three heads do nothing when the tens digit is inside the last token (3 and 5 digits) and carry most of what is left of the teen signal when it is in the previous token (4 and 7 digits). No single head does it alone; the three share the move. This is the candidate's sharpest prediction, a component whose role switches on and off with token layout, and it goes into the protocol as section 4.3 (O1, O2).
- Decision (logged under the overnight go-ahead): protocol section 4.3 kept, thresholds O1 ≤ 0.65 kept at 4 and 7 digits with every random triple ≥ 0.85, read only if the base effect is ≥ 0.30 logits; O2 ≥ 0.90 kept at 3 and 5 digits. Reason: margins of about 0.2 from the exploration values on the small-effect side, and a read rule because a ratio over a near-zero base effect is unstable.

### 06:29 Confirmation run done; REPORT.md written
- `PROTOCOL.md` frozen 04:16; `chain_conf.sh` ran once, 04:16–06:26 (it waited for the shared GPU lock between jobs). No job crashed or was restarted. `code/conf_evaluate.py --part conf` → `outputs/conf/EVALUATION.json` (SHA-256 ed045fb4…).
- All 14 predictions passed (B1–B4; B5 for GPT-2 and Pythia-1B; Q1–Q6; O1, O2). All three verdicts "supported". Closest margins: O1's lowest random triple at 7 digits (0.856 against a 0.85 bar), and B1 at 4 digits (0.775 against ≤ 0.80).
- `REPORT.md` written from `EVALUATION.json`. `code/check_report.py` (frozen) found 0 unmatched numbers; three exploration numbers outside its pool (the Pythia-410M gate miss and two exp06 frame accuracies) are checked by `code/check_report_extra.py`, a post hoc helper written after the run, which also passes.

### 06:39 Post hoc note: ties in the exploratory AUROC helper
- While building the write-up's number check, the independent recount (pair counting, ties as one half) disagreed with `outputs/teen_effect_smollm2-17b.json` in the third decimal. `code/teen_effect.py`'s `auroc()` ranks tied scores arbitrarily instead of averaging them; with bf16 logits, ties occur. Over every exp03 cell of the six models the largest difference is 0.0028 (SmolLM2, 6 digits, comma). No conclusion changes.
- Scope: the same helper gives the AUROCs printed by `exp05_ablate.py` and the per-cell summaries of `conf01_boundary.py`. No prospective test used them: `code/conf_evaluate.py` recomputes every AUROC from the per-item records with ties averaged. The frozen files are unchanged. The write-up recomputes the exploration AUROCs it quotes from the per-item records (labelled post hoc there) and drops the exp05 AUROC column.

### 06:40 Write-up draft v1
- `paper/v1-2026-11-08/manuscript.pdf`, 8 pages with references, LaTeX as in A3. All numbers are macros from `scripts/make_numbers.py`; `scripts/check_numbers.py` (independent recount) gives 360 checks, all passed. Twelve references checked against their arXiv abstract pages (`checks/REFERENCE-CHECK.md`). The post hoc tie correction for exploration AUROCs above is stated in the paper. Nothing committed.

## Round 2 (depth), under CHARTER-2.md (frozen 8 Nov 16:54)

### 17:01 Round-2 material sealed
- 48 new frames (rankings 14, positions 14, anniversaries 10, dates 10), split by hash into explore2 and held2 (24 each); `data2/MANIFEST.md`. Number rules in `code/r2common.py`: 3–5-digit held items reuse round-1 conf/reserve-class numbers in new frames; 6–7-digit round-1 reserve-class numbers (never read) and 8–10-digit numbers (hash 6–9) are fresh.
- The local machine restarted at about 16:57, after the build (16:55). Checked on resuming: the digests match, and a rebuild gives identical bytes. No rebuild was needed.

### 18:10 Tooling, and E01 at two digits (Qwen3-0.6B-Base)
- `code/r2lib.py`: right-padded batches, interventions at each row's last token. They are mean ablation of heads and MLPs, residual patching by offset, scaling a head, and **path ablation**. Path ablation adds a sender's ablation change to one receiver's input only: one attention layer, one MLP, one head (other heads of that layer reset to clean), or the unembedding. Also: logit lens, DLA with the final norm linearised per item, and a cross-validated logistic probe. Every script was smoke-tested on CPU with a few items before it went in the GPU queue.
- Dead end in the tooling: the first full E01 job (17:32–17:52) hit the 20-minute cap and saved nothing. Its prints were buffered, and one run of 192 items in float32 takes about 0.8 s, so the full per-head scan did not fit. From 18:06, jobs are split by digit count, flush every line and save after each digit count. The per-head path scan covers only the layers that carry at least 0.03 of the sender's layer-level effect. (An earlier job at 17:08 crashed at once on a device mismatch in the path hook; fixed.)
- **E01, 2 digits** (`outputs/r2/e01_qwen3-06b_2.json`; 96 teen and 96 non-teen; teen effect +10.46, pull +3.98, AUROC 1.000):
  - **Logit lens:** the teen effect is small (−0.4 to +4.1) through layer 23 and jumps at layer 24 (+16.5), then settles to +10.5 by layer 27.
  - **DLA:** MLP24 writes +8.89 of the teen effect and only +0.56 of the pull. MLP25 writes +1.94 of the teen effect and +3.06 of the pull, and MLP26 and MLP27 take some back (−1.72, −1.38). The largest teen head is L26H9 at +0.54.
  - **Reading:** at the last token, "teen, so th" is written mainly by MLP24, and the last digit's own suffix mainly by MLP25 and later parts.
  - **Path ablation of L0H2** (full loss of teen effect 1.43): 0.79 of the loss goes through **MLP0's input**, and no other single receiver carries more than 0.01. L0H3's pull (full loss 1.83) spreads out: the largest single receiver is MLP0 at 0.16. L1H6 matters little at two digits (full loss 0.20).
  - Full ablations at two digits: L0H2 kept 0.86 of the teen effect; L0H2+L1H6 kept 0.67; L11H9 kept 0.84 of the teen effect and 0.74 of the pull; L0H3 kept 0.54 of the pull.
- Queued: E01 at 3 and 4 digits (with per-head scans), E03 OLMo-2, E02 two-digit route, and E05 (each MLP ablated; mediation through every later receiver for L0H2, L0H3 and L11H9; the hop from MLP0 onwards; each component's suffix profile).

### 19:02 E01 at three digits (`outputs/r2/e01_qwen3-06b_3.json`; job 18:58–19:01 after waiting for the lock)
- Base: teen effect +4.10, pull +2.80, AUROC 0.956.
- **Logit lens:** the teen effect stays at +1.2 to +3.6 through layer 23, then jumps at layer 24 (+6.6) and ends at +4.1. **DLA:** MLP24 writes +3.27 of the +4.10 teen effect; MLP18 +0.33 and MLP25 +0.51; MLP26 and MLP27 take back −0.41 and −0.42. The heads are small (the largest is L26H6 at +0.22).
- **Path ablation of L0H2:** of the teen loss (1.83), **0.89 goes through MLP0's input**. The next receiver is MLP3 at 0.04, and every other receiver is at 0.01 or less. No attention layer reaches 0.03, so the per-head scan was empty. The same holds for L0H2's smaller effect on the pull (0.82 through MLP0).
- **L0H3's pull** (full loss 1.70) is spread out: MLP0 0.15, MLP1 0.06, L11H9 0.05 (per head), MLP18 0.05. No single receiver carries much, so the last digit is read by many parts.
- Full ablations at three digits: L0H2 kept 0.55 of the teen effect; L0H2+L1H6 kept 0.40; L11H9 kept 0.82 of the teen effect and 0.78 of the pull; L0H7 kept 0.81 of the pull.
- **Reading so far** (exploratory): the tens digit is copied by L0H2 into the last token, turned into a teen feature by MLP0 at once, and the teen override is written at the end, mostly by MLP24. The next question is what lies between MLP0 and MLP24 (E05's hop and mediation analysis).

### 19:25 E03, OLMo-2-1B at 3, 4 and 7 digits (`outputs/r2/e03_olmo2-1b_3-4-7.json`)

| Digits (tokens) | Teen effect (AUROC) | Teen acc. | Triple ablated, kept | Triple ×2 / ×3, kept | Probe at last token, layers 0 → 15 | Same, triple ablated |
|---|---|---|---|---|---|---|
| 3 (`211`) | +5.38 (0.991) | 0.94 | 1.00 | 1.01 / 1.01 | 1.00 throughout | 1.00 throughout |
| 4 (`541`+`3`) | +1.20 (0.735) | 0.40 | 0.25 | 1.23 / 1.26 | 0.99 → 0.77 | 0.89–0.97 early → 0.69 |
| 7 (`626`+`331`+`1`) | +0.87 (0.696) | 0.38 | 0.27 | 1.18 / 1.20 | 0.99 → 0.71 | 0.86–0.93 early → 0.57 |

- **The tens digit is not missing at the last token.** A logistic probe reads teen status from the last token's residual stream at 0.99 in layers 0–3 at 4 and 7 digits, almost as well as at 3 digits, where the tens digit is inside the token. The signal then fades with depth (0.77 and 0.71 at layer 15), while at 3 digits it stays at 1.00.
- **The late reader barely uses it.** The logit lens and DLA place the teen decision in the late MLPs, as at 3 digits: MLP13 writes +3.56 of the 3-digit effect (MLP11 +0.77) but only +1.06 at 4 digits and +0.45 at 7.
- **More signal does not help much.** Doubling or tripling the three layer-0 movers' outputs raises the teen effect only to 1.23–1.26 of base at 4 digits (AUROC 0.735 → 0.775). Removing them takes the effect to a quarter, yet the probe still reads 0.89–0.97 early on: other heads also move part of the signal.
- **Path ablation:** of the triple's loss, 0.76 (4 digits) and 0.72 (7 digits) passes through MLP0's input, with MLP13 (0.13) and MLP1 (0.10) next. As in Qwen3, the first reader of the moved digit is the layer-0 MLP.
- **Reading** (exploratory): a weak route, not a missing one. The tens digit reaches the last token and stays readable, but the late suffix MLPs respond to it far less than to the same fact inside a three-digit token. E07 tests whether the cross-token signal lies in a different direction from the in-token one.
- The probe at the previous token is at chance at 3 digits, where the previous token is a space. CV accuracy below 0.5 there (0.18–0.22) is the known artefact of a balanced probe on uninformative features.

### 19:52 E01 at four digits; E06 and E07 queued
- **E01, 4 digits** (`outputs/r2/e01_qwen3-06b_4.json`): teen effect +2.99, pull +1.41, AUROC 0.882.
  - Logit lens: the teen effect is at most +1.7 through layer 23, then **+4.6 at layer 24**.
  - DLA: MLP24 writes +2.37, MLP25 +0.60 and MLP27 −0.65.
  - Path ablation of L0H2 (teen loss 1.37): **MLP0 0.92**, MLP3 0.05, nothing else above 0.01.
  - L0H3: kept 0.01 of the pull, so it removes almost all of it at 4 digits; spread over receivers (MLP0 0.14, L11H9 0.05, MLP18 0.05).
  - L11H9: kept 0.75 of the teen effect and 0.54 of the pull.
- The three digit counts agree:

| Digits | L0H2's teen loss through MLP0 | Next receiver | MLP24's share of the teen effect (DLA) | Lens teen effect, layer 23 → 24 |
|---|---|---|---|---|
| 2 | 0.79 | 0.01 | 8.89 of 10.46 | +3.6 → +16.5 |
| 3 | 0.89 | 0.04 | 3.27 of 4.10 | +2.2 → +6.6 |
| 4 | 0.92 | 0.05 | 2.37 of 2.99 | +1.2 → +4.6 |

- **E06 (question 5) needs no download.** D3 downloaded six OLMo-2-0425-1B stage-1 checkpoints (steps 300; 10,000; 40,000; 200,000; 480,000; 1,907,359), about 39 GB in the shared cache. E06 reads them for the teen check by length, the triple ablation and the probe. Queued after chain C.
- E07 (OLMo-2: is the cross-token signal in a different direction from the in-token one, and does the reader respond to the in-token direction?) is queued in chain C.

### 20:09 E02, the two-digit route (Qwen3-0.6B-Base; `outputs/r2/e02_qwen3-06b_2.json`)
- Two digits: teen effect +10.51 (AUROC 1.000). With A = L0H2+L1H6 mean-ablated at the last token it is +7.18 (kept 0.68; AUROC 0.979), as in round 1.
- **Where the tens digit sits, under A** (residual patching from a non-teen source such as 23 into the teen target 13; fraction of the gap moved; last token / tens-digit token):

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|---|---|---|
| Clean | 0.37 / 0.55 | 0.78 / 0.16 | 0.81 / 0.12 | 0.80 / 0.12 | 0.80 / 0.11 | 0.81 / 0.10 | 0.81 / 0.10 | 0.81 / 0.09 |
| Under A | 0.01 / 0.98 | 0.02 / 0.95 | 0.28 / 0.56 | 0.35 / 0.47 | 0.36 / 0.43 | 0.66 / 0.24 | 0.66 / 0.24 | 0.69 / 0.22 |

  - In the clean model the tens digit reaches the last token by layer 1, through L0H2 and L1H6.
  - Under A it stays at its own position until layer 1, then crosses in two steps, at **layer 2** and **layer 5**.
- **Which heads** (further ablation on top of A, kept of A's effect):
  - Whole layers: 2 (0.76), 5 (0.78), 11 (0.80), 26 (0.87).
  - Single heads: **L2H4** 0.85 (attention to the previous token 0.84) and **L5H2** 0.84 (0.18 to the previous token, the rest elsewhere). Then L2H12 0.95 and L3H6 0.96; L11H9 0.77, which is the suffix head, not a mover.
  - MLPs: MLP24 0.38 and MLP18 0.41, the same late teen writers as before.
- **Reading** (exploratory): the two-digit teen check survives A because later heads (L2H4, L5H2) also bring the tens digit across, a backup copy. It then feeds the same late MLPs. E08 tests whether A+L2H4+L5H2 removes it, against random pairs from layers 2–6, at 2, 3 and 4 digits. Why the backup works fully only at 2 digits is open.
- Late in the clean model, a residue of the tens digit (0.09–0.12) still sits at its own position and moves over at layers 19–21 (0.82 → 0.96), as round 1's exp01 pilot saw.

### 20:12 E03, OLMo-2-1B at 5 and 8–10 digits (`outputs/r2/e03_olmo2-1b_5-8-9-10.json`; 8–10 digits are explore2 numbers, never read before)

| Digits (tokens) | Tens digit in last token? | Teen effect (AUROC) | Teen acc. | Triple kept | ×3 kept | Probe, last token, layer 2 → 15 |
|---|---|---|---|---|---|---|
| 5 (`680`+`13`) | yes | +3.31 (0.963) | 0.99 | 0.98 | 1.06 | 1.00 → 0.97 |
| 8 (`614`+`049`+`12`) | yes | +2.48 (0.947) | 1.00 | 0.99 | 1.06 | 1.00 → 0.96 |
| 9 (`751`+`007`+`212`) | yes | +1.76 (0.940) | 1.00 | 0.99 | 0.97 | 0.99 → 0.96 |
| 10 (`208`+`172`+`271`+`3`) | **no** | +0.69 (**0.671**) | 0.77 | **0.25** | 1.32 | 0.99 → **0.65** |

- **The token account carries on to new lengths:** high where the tens digit is inside the last token (5, 8, 9 digits), low where it sits in the previous token (10 digits, as at 4 and 7). The layer-0 triple matters only at 10 digits.
- **At 10 digits the same weak-route signature:** the probe reads teen status at 0.99 early, it fades to 0.65 by layer 15, and tripling the movers gives only 1.32 of base. Of the triple's loss, 0.71 goes through MLP0's input.
- Candidate prospective tests for held2 numbers (fresh 8–10-digit numbers in held2 frames): the length pattern, the triple's role switching with layout, and the probe-versus-behaviour gap.

### 20:22 E05 at 2 and 3 digits: the path from MLP0 to the suffix (`outputs/r2/e05_qwen3-06b_2-3.json`)
- **Mediation** (a sender's full-ablation change at a receiver's input, given to that receiver alone; share of the sender's loss, 2 / 3 digits):
  - L0H2, teen effect: MLP0 0.79 / 0.89; **MLP24 0.57 / 0.62**; MLP18 0.35 / 0.43; MLP25 0.36 / 0.29; attention 11 0.21 / 0.26.
  - L11H9, teen effect: **MLP24 0.56 / 0.70**, MLP18 0.33 / 0.44. L11H9, pull: MLP24 0.54 / 0.46, MLP18 0.29 / 0.46.
  - L0H3, pull: **MLP18 0.50 / 0.51**, MLP20 0.26 / 0.17, MLP24 0.12 / 0.32.
- **Hop 2.** MLP0's change after receiving L0H2's change has no single strong reader: the largest are MLP2 0.12 / 0.16, attention 11 0.15 / 0.13 and MLP18 0.06 / 0.13. The tens signal is carried through the middle layers by many parts and gathered at MLP18 and MLP24.
- **What the late MLPs write.** This is each one's DLA to st/nd/rd/th, centred, by class and last digit:
  - **MLP24** (3 digits): for teens it boosts "th" (+2.0 to +2.5) and lowers st, nd and rd about equally (−0.5 to −1.1), whatever the last digit. For non-teens it is near zero. At 2 digits the same, larger (th +6.0 to +6.7). It is the **teen override**, and it does not need the last digit.
  - **MLP18 and MLP20**: they raise the last digit's own suffix (st for 1, nd for 2, rd for 3) by about +0.3 to +1.1, the same for teens and non-teens. They are **last-digit writers that do not check teen status**.
  - **MLP25**: raises "st" and lowers "th" for every class (largest for numbers ending in 1). MLP26 and MLP27 lower "th" for every class: a general correction.
- **MLP mean ablation** at the last token (teen / pull kept, 3 digits): MLP24 0.39 / 0.50, MLP18 0.67 / 0.44, MLP0 0.64 / 0.74, MLP25 0.74 / 0.79. Mean ablation of a large writer moves both measures, so DLA is the cleaner readout of what each writes.
- **Answer to "where are the two signals combined?"** (exploratory): in no single part. The last digit is written as its own suffix by MLP18/MLP20 (fed by L0H3 and L11H9), the teen status as "th" by MLP24 (fed by L0H2 → MLP0 → the middle layers, and by L11H9). The two add up in the logits. L11H9, the current-token suffix head that X1 found, sits upstream of both writers.

### 20:27 E05 at 4 digits (`outputs/r2/e05_qwen3-06b_4.json`): the same structure
- Mediation of L0H2's teen loss (1.37): MLP0 0.92, **MLP24 0.59**, MLP18 0.49, MLP25 0.30, attention 11 0.29. Of L11H9's teen loss (0.75): **MLP24 0.64**, MLP18 0.52. Of L0H3's pull loss (1.40): **MLP18 0.57**, MLP24 0.25.
- Profiles: MLP24 for teens th +2.0 to +2.9, with st, nd and rd lowered about equally. For non-teens ending in 1 it also gives th +1.2, a leak that fits the round-1 errors running both ways. MLP18 and MLP20 raise the own suffix by +0.3 to +1.1, the same for teens and non-teens.
- The three digit counts agree on every point of the reading at 19:59 above. Candidate prospective tests (Qwen3, held2 frames, 4 and 6 digits):
  1. L0H2's path through MLP0's input carries most of its teen loss, more than any other receiver and more than any other layer-0 head's path through MLP0;
  2. MLP24 is the largest late mediator of L0H2 and of L11H9;
  3. the logit lens jumps at layer 24;
  4. MLP24's DLA is most of the teen effect;
  5. MLP24's teen profile is "th" up and st/nd/rd down without last-digit specificity, while MLP18's is own-suffix up for both classes.

### 20:41 E07, OLMo-2-1B: the cross-token teen signal is small, and the readers respond normally to a full-sized one (`outputs/r2/e07_olmo2-1b.json`)
- **Size.** The teen direction v_d is the mean last-token residual of teens minus that of non-teens. Its norm, relative to the 3-digit one at layers 0, 3, 6, 9, 12, 15: at 4 digits 0.31, 0.50, 0.52, 0.52, 0.47, 0.52; at 7 digits 0.31, 0.38, 0.38, 0.43, 0.34, 0.29. At 5 digits (in-token) 1.19, 1.12, 1.10, 1.05, 0.91, 0.80.
- **Direction.** Cosine with v_3 by layer: 4 digits 0.22 → 0.61; 7 digits 0.20 → 0.46; 5 digits 0.28 → 0.70. A probe trained on 3-digit items, with each length centred on its own mean (no labels), reads 4-digit teens at 0.92 → 0.72 by layer, 7-digit at 0.85 → 0.62, and 5-digit at about 1.00.
- **Steering.** v_3 scaled to its own norm is added at layer l to every non-teen item. The change in mean m (towards "th"):

| Length | Layers 0–14 |
|---|---|
| 3 digits | −4.05 to −4.44 |
| 4 digits | −2.77 to −4.33 |
| 7 digits | −1.90 to −4.21 |
| 5 digits | −1.93 to −4.35 |

  Each length's own direction at the same norm moves m less at 4 and 7 digits (−1.59 to −2.95). Five random directions of that norm move it by −0.37 to +0.28 on average.
- **Reading** (exploratory; refines 19:25):
  - The late readers respond almost as strongly at 4 and 7 digits as at 3, when given a teen signal of 3-digit size in the 3-digit direction.
  - What arrives across the token boundary is about a third to a half as large, and only partly in that direction. So the weak route is weak at the source: the move and its format. It is not a reader that fails to listen.
  - Caveat on the 19:25 gain test: OLMo-2 normalises the whole attention output after each layer (post-norm), so multiplying three heads by 3 is partly cancelled. The small gain there does not show a saturated reader. Steering is the cleaner test.
- Added to the confirmation runner: the norm ratio, the cosine and steering at layer 8 with v_3 against 5 random directions.

### 20:57 E08: L2H4 and L5H2 are a partial backup route (`outputs/r2/e08_qwen3-06b.json`)

| Digits | A kept | A+L2H4 | A+L5H2 | A+B (both) | B alone | A + random pair (layers 2–6), range |
|---|---|---|---|---|---|---|
| 2 | 0.68 | 0.58 | 0.57 | **0.44** | 0.98 | 0.67–0.69 |
| 3 | 0.40 | 0.40 | 0.32 | **0.33** | 0.96 | 0.38–0.40 |
| 4 | 0.41 | 0.41 | 0.30 | **0.29** | 0.92 | 0.40–0.42 |

(Teen accuracy under A+B: 0.93, 0.31, 0.53.)
- B = L2H4+L5H2 matters only once A is removed: alone it keeps 0.92–0.98. A+B is below every random pair at every length. The backup carries most at 2 digits (a drop of 0.24 from A, against 0.07 and 0.12).
- Under A at 3 digits, residual patching shows less moving than at 2 digits: the last token holds 0.07 at layer 2, 0.13 at layer 3 and 0.35 at layer 5, and the tens position still holds 0.61 at layer 9 (0.22 at 2 digits). So L2H4 brings the tens digit across mainly when it is the first digit of the number. That fits a head attending to the previous token after a space.
- **Reading** (exploratory): the two-digit teen check survives A partly through backup movers that act more strongly on two-digit numbers. A further route remains unidentified (A+B still keeps 0.44 at 2 digits): perhaps the late move of the tens-position residue at layers 19–21 seen in E02, or a stored pattern for the frequent two-digit ordinals. Not resolved in this round.

### 21:10 E04: other one-token-per-digit models, heads chosen by attention alone (`outputs/r2/e04_<model>.json`)
- Rule (fixed before any ablation): PREV = heads in layers 0–1 that put ≥ 0.5 of the last token's attention on the previous token (the tens digit); CUR = heads that put ≥ 0.5 on the current token. Each set is mean-ablated at the last token, against 20 random same-size sets from layers 0–1 outside both sets.
- **Check on Qwen3-0.6B-Base** (CPU smoke, 12 items, 20:16): the rule picks PREV = L0H2+L1H6, exactly round 1's pair, and CUR = L0H3+L0H7. The single top PREV head by attention (L1H6) is the weaker one (kept 0.94–0.96), so the set rule is used, not a top-1 rule.
- **Qwen3-1.7B-Base.** Gate: G1 1.000 (403), G2 0.952 (600), qualifies. The rule picks the same heads by index: PREV = L0H2+L1H6, CUR = L0H3+L0H7.

| Digits | Teen effect (AUROC) | PREV: teen kept / pull kept | CUR: teen kept / pull kept | Random sets, teen / pull |
|---|---|---|---|---|
| 3 | +5.85 (0.996) | **0.41** / 0.70 | 1.17 / **0.01** | 1.00–1.02 / 0.99–1.01 |
| 4 | +4.53 (0.964) | **0.41** / 0.63 | 1.01 / **−0.94** | 0.99–1.02 / 0.99–1.05 |

  - The division of labour is the same as in the 0.6B model: the previous-token heads carry the teen check, the current-token heads the last-digit pull.
  - L0H2 alone keeps 0.55 at both lengths, and L0H3 alone keeps 0.52 and −0.04 of the pull.
- E04 for SmolLM2-1.7B crashed at 21:03 (the attention pass ran before the last-token index was set; fixed 21:05; the crashed log is kept as `_CRASHED`). It is queued again after Gemma-4.

### 21:32 E04, SmolLM2-1.7B (rerun after the crash) and Gemma-4-E2B (CPU smoke only so far)
- **SmolLM2-1.7B.** PREV = L0H6 (1.00 of its attention on the previous token) + L0H8. **CUR is empty**: no layer-0/1 head puts ≥ 0.5 on the current token (the top one, L0H29, is below that).
  - PREV: teen kept **0.70** at 3 digits and **0.62** at 4 (random sets 0.99–1.03); pull kept 0.87 and 0.79.
  - L0H6 alone does all of it (0.70, 0.62).
  - The top current-token head, L0H29, changes nothing (1.00).
- **Gemma-4-E2B, CPU smoke** (8 items, so the values are not results): PREV = L0H4 (0.94), teen kept 0.60; no CUR head. The full run is queued after the dry run.
- An extra crash: Gemma-4 mixes layers with different head sizes, and `r2lib.Hooked` assumed one size. Since 21:19 it uses each layer's own size, which changes nothing for the other models. Gemma's first job (21:17) is kept as `_CRASHED`.
- **Reading** (exploratory):
  - The previous-token part of the round-1 account holds in every one-token-per-digit model tried: an early head that copies the previous token carries much of the teen check (kept 0.41–0.70), beyond random heads, and leaves most of the pull.
  - The current-token part is Qwen-specific: SmolLM2 and Gemma-4 have no early current-token head, and the pull is not carried by any single early head there.
- Head sets for PROTOCOL-2 section 4.4 (`code/r2conf_sets.json`): Qwen3-1.7B-Base PREV = L0H2+L1H6 and CUR = L0H3+L0H7; SmolLM2-1.7B PREV = L0H6+L0H8; Gemma-4-E2B PREV from its full exploration run.

### 9 Nov 09:20 Stall overnight; dry run of the confirmation pipeline on exploration items
- **Stall, 21:40 → 09:09.** My waiting loops used `pgrep -f` with a pattern that also matched the loop's own shell, so they never ended, and one of them held up the Gemma-4-E2B run behind it. Nothing ran for about 11.5 hours. A tooling dead end, recorded in `AMENDMENTS-2.md` (Amendment 1). Waits now check a systemd unit or a PID.
- **E06 (checkpoints) ran out of GPU memory** after two of six revisions (the previous model was not freed). Fixed; the other four re-run one per job (`code/r2chain_f.sh`). The crashed logs are kept with `_OOM` in the name.
- **The dry run read only exploration material.** `PROTOCOL-2.md` was not frozen at any point overnight: 20:58 is its last draft edit, and it stayed writable, so `code/r2common.py` would have refused any held2 frame or held-out number. The dry run (`r2conf_run.py --dry`, 21:32–21:39, `outputs/r2dry/`) uses part "explore" and seeds `d4r2-dry-<task>-<digits>`. Rebuilding its item lists from those seeds: 2,304 items (qwen 4/6, qwen2 2/3/4, olmo 3/4/5/7/8/9/10 digits, 96 per class), 2,112 exploration-part numbers plus 192 two-digit items (all two-digit numbers are shared by design), and no held2 frame.
- **Dry-run evaluation** (`r2conf_evaluate.py --dry`, full size, exploration items): 31 of 33 draft tests pass.
  - Qwen path R1–R6 at 4 and 6 digits: all pass. The tightest are R5@6 (largest other MLP 0.32 of the teen effect, threshold 0.35) and R3@6 (L0H2 via MLP24 0.57, next 0.48).
  - Backup route: S2@4 fails narrowly, B alone keeps 0.897 (threshold 0.90). S1 and S3 pass (A+B 0.45/0.30/0.23 against random minima 0.69/0.38/0.36).
  - OLMo-2: O3, O4, O5, O7 and O8 pass. O6 fails on its in-token side: the 5-digit norm ratio dips to 0.72 at one layer (threshold 0.75); the cross-token ratios are ≤ 0.54.
- **Threshold changes before the freeze** (set from exploration only, logged here):
  - S2: B alone keeps ≥ **0.85** (was 0.90). The point of S2 is that B matters little on its own; 0.90 was set from three small exploration values (0.92–0.98).
  - O6: the 5-digit ratio must stay ≥ **0.65** at every layer (was 0.75), so that the same 0.65 line separates the in-token from the cross-token lengths. The cross-token side is unchanged (≤ 0.65).
- **E06, first two checkpoints** (exploratory, descriptive): at step 300 (1B tokens) there is no teen check at any length (AUROC 0.45–0.59). At step 200,000 (420B tokens) the in-token lengths are already strong (AUROC 0.996 at 3 digits, 0.978 at 5) and the cross-token lengths are near chance (0.569 at 4, 0.467 at 7), while the final model has 0.70–0.74. At that checkpoint the triple L0H5+L0H0+L0H9 does nothing at 4 digits (kept 1.00), and the strongest layer-0 previous-token head is L0H7 (0.97). So the weak cross-token route forms late, and probably through different heads. Waiting for the other four checkpoints.

### 9 Nov 09:15 E04, Gemma-4-E2B (full run; the last exploration run)
- Gate passed in round 1. PREV = **L0H4** alone (0.94 of its attention on the previous token); **CUR is empty** (no layer-0/1 head puts ≥ 0.5 on the current token; the top one, L1H0, puts 0.33 on the previous token and changes nothing).

| Digits | Teen effect (AUROC) | PREV L0H4: teen kept / pull kept | Random heads, teen / pull |
|---|---|---|---|
| 3 | +5.67 (1.000) | **0.75** / 0.91 | 1.00–1.01 / 1.00–1.01 |
| 4 | +3.42 (0.949) | **0.62** / 0.84 | 0.99–1.00 / 1.00–1.02 |

- **Reading** (exploratory): the previous-token head again carries part of the teen check and little of the pull, as in Qwen3 and SmolLM2; Gemma-4's head carries less of it at 3 digits (0.75 kept).
- Exploration ends here (`AMENDMENTS-2.md`, Amendment 1). For 4.4 the PREV threshold is set at ≤ 0.85 kept (draft 0.80), because Gemma-4 at 3 digits (0.75) sat within 0.05 of the draft line; random sets stay at 0.99 or above. Head sets are in `code/r2conf_sets.json`. A small smoke run of the `other` task on exploration items checks the code before the freeze.
- 09:34 smoke run of the `other` task (`--dry`, 8 items per class, 2 random sets; `outputs/r2dry/other_*.json`): the code runs for all three models (6–9 s each). Values are too small to read; one test misses at this size (Qwen3-1.7B PREV at 3 digits, pull kept 0.54 against 0.55; the full E04 run gave 0.70), so no threshold is changed. Freezing `PROTOCOL-2.md` next.

### 9 Nov 09:40 E06, OLMo-2-1B stage-1 checkpoints (all six; exploratory and descriptive, after the window by Amendment 1)
E03's exploration items (64 per class). The final released model, from E03, is given for comparison. The checkpoints were downloaded by D3 on 8 Nov; nothing was downloaded here.

| Checkpoint (tokens) | AUROC 3 / 4 / 5 / 7 digits | Teen effect 4 digits | Triple L0H5+L0H0+L0H9 kept, 4 digits | Probe at layer 2 / 14, 4 digits | Top layer-0 previous-token head |
|---|---|---|---|---|---|
| step 300 (1B) | 0.451 / 0.493 / 0.585 / 0.577 | −0.00 | (base too small) | 0.39 / 0.41 | H4 0.14 |
| step 10,000 (21B) | 0.789 / 0.516 / 0.992 / 0.459 | +0.03 | (base too small) | 0.94 / 0.64 | H9 0.10 |
| step 40,000 (84B) | 0.982 / 0.540 / 0.959 / 0.519 | +0.20 | (base too small) | 0.97 / 0.62 | H1 0.11 |
| step 200,000 (420B) | 0.996 / 0.569 / 0.978 / 0.467 | +0.49 | 1.00 | 0.98 / 0.62 | H7 0.97 |
| step 480,000 (1.0T) | 0.998 / 0.710 / 0.982 / 0.584 | +0.98 | 1.00 | 1.00 / 0.74 | H7 0.99 |
| step 1,907,359 (4.0T, end of stage 1) | 0.998 / 0.817 / 1.000 / 0.616 | +1.63 | 1.00 | 0.99 / 0.87 | H7 0.99 |
| released model (E03) | 0.991 / 0.735 / 0.963 / 0.696 | — | 0.25 | 0.99 / 0.77 (layer 15) | — |

- The in-token teen check forms early (3 and 5 digits strong by 84B tokens). The cross-token check (4 and 7 digits) stays near chance until about 1T tokens and is still weak at the end of stage 1.
- The tens digit is linearly readable at the last token from early on (probe at layer 2 ≥ 0.94 from 21B tokens), but much less so later in the network; the late probe rises with training (0.62 → 0.87).
- **The triple does nothing in any stage-1 checkpoint** (kept 1.00 wherever the base effect is readable). In the released model, after stage 2, removing it keeps only 0.25. So the route that round 1 found in L0H5, L0H0 and L0H9 appears only in the final stage of training; in stage 1 the cross-token signal, such as it is, travels another way (the strongest previous-token head there is L0H7). The stage-2 checkpoints are not on disk and were not fetched.
- Not tested in round 2; reported as exploratory and descriptive in REPORT-2 and the paper.

### 9 Nov 10:57 Correction to times written this morning
- Some times in this morning's records were estimated, not read from the clock. File times show the following.
  - Amendment 1 (`AMENDMENTS-2.md`, headed 09:25) was written at 09:11.
  - The notebook entry headed 09:20 (stall and dry run) was written between 09:11 and 09:13. `PROTOCOL-2.md` and Amendment 1 cite it as "09:20"; both are frozen and are left as they are.
  - The two DEPTH-ROUND lines are corrected to 09:13 and 10:49.
- The protocol freeze (09:34) and the confirmation run (09:34:50 to 10:46) were read from the clock and the run files, and are right.
- Nothing else depends on these times. In particular, the order (amendment, then threshold resets, then freeze, then the run) is as recorded.

### 9 Nov 11:02 Paper v2 and its independent check
- `paper/v2-2026-11-09/` (13 pp.; v1 untouched). The second-round section of `scripts/check_numbers.py` was written separately from the raw files and `PROTOCOL-2.md` only. It found one real wording error, now fixed: the text said three thresholds were reset from the dry run, but only S2 and O6 were; the PREV threshold was reset from the Gemma-4-E2B exploration run. Final result: 839 checks, all pass.
- For the record: `PROTOCOL-2.md` (frozen) calls the dry run "the whole pipeline". It covered every task except the other-models task, which had only a small smoke run. `REPORT-2.md` and the paper say this correctly. The protocol is left unchanged.

### 9 Nov 2026, 14:32 IST: E06 checkpoints are a different training run (post hoc)
- The overview passed on D3's finding: the released OLMo-2-0425-1B weights do not descend from the published stage-1 checkpoints.
- I checked it myself on CPU, in the memory cage, with my own code (`code/r2post_olmo_lineage.py`, `code/r2post_olmo_basis.py`; outputs in `outputs/r2post/`). Nothing frozen changed and nothing was downloaded.
- **Released model against the end of stage 1 (4001B):**
  - embedding cosine 0.0059;
  - layer tensors (L0 q and o, L8 MLP down, L15 v) about 0.0000;
  - relative distance 1.47–2.90.
- **The four stage-2 checkpoints against the end of stage 1:** embedding cosine 0.9996–1.0000, layer tensors 0.97–0.995. All stage-2 checkpoints are as far from the released model as stage 1 is (embedding cosine 0.0058).
- **Not a reordered or rotated copy:** a check that does not depend on the basis gives the same answer.
  - Token-norm correlation is 0.933, against 1.000 for stage 2.
  - Gram-matrix correlation over 2000 tokens is 0.754, against 0.999.
  - The singular values of L0 q_proj differ by 1.81 (relative), against 0.04.
  - So the released model is a separate run of the same recipe. Its embedding geometry is similar, as expected, but its weights are not the same.
- My values agree with D3's separately computed `r2e10b_dist.json` (embeddings 0.00585 against 0.0059; ingredient 1 at 51B, 0.99962 against 0.9996).
- **Consequence:** the 09:40 entry above and REPORT-2's E06 item went wrong in concluding that the triple's route "appears only in the final stage of training". The checkpoints describe a published run of the same recipe; they cannot date the released model's route. Head positions need not match across runs either. The E06 numbers stand as a description of that run.
- REPORT-2 has a dated correction under the E06 item. Paper v3 (`paper/v3-2026-11-09/`) rewords the abstract, the subsection, the table caption, the limits and the process paragraph, and adds two lineage macros. v2 is untouched.
- Number check: 847 of 847 passed. It includes eight new checks, which cross-check against D3's file and confirm the old inference is gone. Build: 13 pages.
