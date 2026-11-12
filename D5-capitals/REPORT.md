# D5 report: how two small models keep writing in capitals

**8 November 2026.** Confirmation of the D5 candidate account under the protocol frozen at 05:48 IST on 8 November (`PROTOCOL.md`, digests in `FROZEN.md`). Every prediction below is **prospective**: it was fixed before any model read the held-out `conf` part, which was then run once. Section 4 is **descriptive** (read from the same runs' saved summaries, not a prediction). Exploration results are in `NOTEBOOK.md` and are **exploratory**. The `reserve` part was not read. There were no amendments: all thirteen runs finished at the first attempt.

Every number in this file is filled in by `code/make_report.py` from saved outputs (`outputs/conf_verdicts.json`, written by the frozen `code/conf_verdict.py`; `outputs/explore_verdicts_check.json`; `outputs/conf_exp01_<model>.json`; `outputs/conf_exp08_<model>.json`). The text is `code/report_template.md`.

## Short answer

In both Gemma-4-E2B and Qwen3-1.7B-Base, every prediction passes on new paragraphs (nine in Gemma, ten in Qwen3, whose extra one concerns a head Gemma lacks), and the frozen account is **supported** in both models. Three things hold on held-out text:

- While the model writes an upper-case passage, the logit lens from mid depth puts the coming word's **lower-case** first piece above its actual upper-case piece. The capitals win only in the last three layers (from layer 32 of 35 in Gemma and 25 of 28 in Qwen3).
- A single **capitals direction**, estimated once on exploration text, switches the next word between lower case and capitals in either direction, and mostly keeps the word. Random directions of the same size do nothing. Setting the residual to the middle of this direction leaves the case about half decided.
- The case is set by the last few words. In Qwen3, one layer-0 head reads each token's own case into the residual stream.

The two models agree on every prediction.

## 1. What was tested

- **Models.** Gemma-4-E2B (base; many upper-case words are single tokens) and Qwen3-1.7B-Base (upper-case words split into several pieces). These were the two models that passed the frozen gate during exploration.
- **Material.** The held-out `conf` part: 597 paragraphs from 332 articles, split off by article in the frozen charter. Each run uses a stated range from the start of the shuffled part.
- **Fixed in advance.** Two layers were fixed from exploration: the direction layer ℓ (24 for Gemma, 22 for Qwen3) and the early layer 16. The capitals direction d and its random control r were not re-estimated: they are the exploration means (exploration paragraphs 150–300).
- **Statistics.** Decisions use point estimates. Brackets give 95% intervals from 2,000 bootstrap resamples of paragraphs.

Terms used below. *U* is the true upper-case first piece of the next word, and *Lc* is the same word's lower-case first piece. *Lr* is the lower-case first piece of another word in the paragraph. A *flip* is a change of the top-1 token's case. *Same word* means the new top-1 is the old top-1 word in the other case.

## 2. Results by prediction

Each row gives the conf value with its 95% interval. The bracketed pair after the threshold gives the exploration values for Gemma / Qwen3.

### G1: gate replication (conf paragraphs 0–150)

| Condition (threshold) | Gemma | Qwen3 |
|---|---|---|
| Case kept when continuing an upper-case passage (≥ 0.95) | 0.995 | 0.998 |
| Same word as the lower-case passage (≥ 0.60) | 0.707 | 0.649 |
| Inside-word pieces correct (≥ 0.70) | 0.837 | 0.788 |
| **G1** | **passes** | **passes** |

### L1 and L2: latent lower case (logit lens, conf paragraphs 0–150)

Positions: 17,247 for Gemma and 17,614 for Qwen3 at ℓ, after leaving out positions where U and Lc have equal lens values.

| Measure (threshold) [exploration G / Q] | Gemma | Qwen3 |
|---|---|---|
| Lc > U at ℓ (≥ 0.60) [0.705 / 0.748] | 0.707 [0.696, 0.717] | 0.753 [0.742, 0.764] |
| Lr > U at ℓ (≤ 0.30) [0.164 / 0.130] | 0.162 [0.154, 0.170] | 0.127 [0.121, 0.133] |
| Lc > U at the last layer (≤ 0.05) [0.001 / 0.002] | 0.001 [0.001, 0.002] | 0.002 [0.001, 0.003] |
| First layer from which Lc > U stays below 0.5 (Gemma ≥ 30 of 35, Qwen3 ≥ 23 of 28) [32 / 25] | 32 | 25 |
| **L1** | **passes** | **passes** |
| Lc > Lr at ℓ, upper-case passages (≥ 0.75) [0.852 / 0.871] | 0.852 [0.844, 0.860] | 0.878 [0.872, 0.884] |
| Lc > Lr at ℓ, lower-case passages (≥ 0.75; differs from the line above by ≤ 0.05) [0.853 / 0.885] | 0.852 [0.845, 0.859] | 0.890 [0.884, 0.896] |
| **L2** | **passes** | **passes** |

### D1–D4: the capitals direction (conf paragraphs 0–100, one edit at the prediction position, a = 1)

| Measure (threshold) [exploration G / Q] | Gemma (ℓ = 24) | Qwen3 (ℓ = 22) |
|---|---|---|
| Add d to a lower-case passage: flips to capitals (≥ 0.80) [0.961 / 0.957] | 0.981 [0.964, 0.995] | 0.956 [0.929, 0.978] |
| … same word in capitals (≥ 0.45) [0.775 / 0.600] | 0.799 [0.753, 0.841] | 0.658 [0.601, 0.716] |
| Subtract d from an upper-case passage: flips to lower case (≥ 0.80) [0.955 / 0.971] | 0.961 [0.940, 0.981] | 0.967 [0.948, 0.984] |
| … same word in lower case (≥ 0.45) [0.756 / 0.657] | 0.785 [0.742, 0.824] | 0.668 [0.616, 0.720] |
| Add r: flips (≤ 0.05) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Subtract r: flips (≤ 0.05) | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.008] |
| Positions (add / subtract) | 363 / 363 | 363 / 364 |
| **D1** | **passes** | **passes** |
| Set the component along d to the midpoint: upper-case top-1 left (≤ 0.70) [0.524 / 0.465] | 0.540 [0.483, 0.595] | 0.527 [0.477, 0.584] |
| … still the same word in either case (≥ 0.60) [0.846 / 0.725] | 0.822 [0.781, 0.860] | 0.805 [0.758, 0.849] |
| Same operation along r: upper case left (≥ 0.95) | 0.997 [0.991, 1.000] | 1.000 [1.000, 1.000] |
| **D2** | **passes** | **passes** |
| Ordinary mixed-case prose, add d: capitals (≥ 0.75) [0.952 / 0.959] | 0.957 [0.934, 0.977] | 0.941 [0.913, 0.968] |
| … same word in capitals (≥ 0.50) [0.823 / 0.639] | 0.821 [0.776, 0.861] | 0.678 [0.621, 0.732] |
| … add r: capitals (≤ 0.05) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| **D3** | **passes** | **passes** |
| Layer 16, add d: flips (≤ 0.15) [0.022 / 0.000] | 0.011 [0.003, 0.022] | 0.008 [0.000, 0.017] |
| Layer 16, subtract d: flips (≥ 0.40) [0.756 / 0.500] | 0.736 [0.682, 0.784] | 0.481 [0.433, 0.530] |
| **D4** | **passes** | **passes** |

### R1: the case comes from the last few words (conf paragraphs 0–100)

Each cell is the share of upper-case top-1 tokens among cased top-1 tokens at word starts (word 12 or later). *L+u2* is a lower-case passage whose last two words are in capitals; *U+l2* is an upper-case passage whose last two words are in lower case.

| Passage (threshold) [exploration G / Q] | Gemma | Qwen3 |
|---|---|---|
| L+u1 (no threshold) | 0.297 [0.250, 0.346] | 0.311 [0.262, 0.360] |
| L+u2 (≥ 0.50) [0.689 / 0.687] | 0.708 [0.664, 0.753] | 0.683 [0.631, 0.732] |
| L+u8 (≥ 0.85) [0.933 / 0.933] | 0.932 [0.904, 0.957] | 0.925 [0.897, 0.950] |
| U+l1 (no threshold) | 0.223 [0.176, 0.272] | 0.390 [0.342, 0.440] |
| U+l2 (≤ 0.25) [0.078 / 0.115] | 0.086 [0.057, 0.116] | 0.131 [0.097, 0.171] |
| U+l8 (≤ 0.10) [0.014 / 0.036] | 0.017 [0.005, 0.030] | 0.029 [0.013, 0.049] |
| **R1** | **passes** | **passes** |

### H1: Qwen3's layer-0 case reader, head L0H3 (conf paragraphs 0–50)

| Measure (threshold) [exploration] | Qwen3 |
|---|---|
| Attention on the current token, upper-case passages (≥ 0.90) [0.977] | 0.978 [0.975, 0.981] |
| Attention on the current token, lower-case passages (≥ 0.90) [0.972] | 0.968 [0.965, 0.972] |
| Value from another paragraph in lower case: share of D moved (≥ 0.20) [0.344] | 0.352 [0.323, 0.381] |
| … flips to lower case (≥ 0.25) [0.543] | 0.495 [0.425, 0.565] |
| Value from another paragraph in upper case: share of D moved (≤ 0.08) [0.030] | 0.029 [0.011, 0.049] |
| … flips (≤ 0.05) [0.000] | 0.011 [0.000, 0.027] |
| Control head L0H2 patched from the same word in lower case: share of D moved (≤ 0.08) | 0.011 [0.005, 0.016] |
| Control head L0H0, same: share of D moved (≤ 0.08) | 0.001 [-0.000, 0.002] |
| Reported, no threshold: value from the same word's own lower-case passage, share of D moved / flips | 0.325 / 0.403 |
| Positions (upper-case passages with an upper-case top-1) | 186 |
| **H1** | **passes** |

D is the gap logp(U) − logp(Lc). "Share of D moved" is measured towards the lower-case passage's own D.

### E1: changed words were lower-case runner-ups (gate word starts, conf paragraphs 0–150)

These are positions where the upper-case passage writes a different word from the lower-case passage, and the two words' lower-case first pieces differ. There were 143 such positions for Gemma (in 91 paragraphs) and 170 for Qwen3 (in 109).

| Measure (threshold) [exploration G / Q] | Gemma | Qwen3 |
|---|---|---|
| Changed word's lower-case first piece in the lower-case passage's top 5 (≥ 0.50) [0.762 / 0.694] | 0.713 [0.637, 0.785] | 0.676 [0.615, 0.742] |
| Control word from the same paragraph in the top 5 (≤ 0.10) [0.014 / 0.022] | 0.021 [0.000, 0.046] | 0.029 [0.006, 0.063] |
| Reported, no threshold: median rank of the changed word / of the control word | 3 / 781 | 3 / 689.5 |
| **E1** | **passes** | **passes** |

## 3. Verdict

Under Section 5 of the protocol, the account (the word is chosen in a lower-case-like form and turned into capitals late, under one capitals direction) is **supported** for Gemma-4-E2B and **supported** for Qwen3-1.7B-Base. L1, L2, D1, D2 and D3 pass in both. All the secondary predictions also pass in both models: D4, R1, E1 and G1, plus H1 in Qwen3. The two models agree on every prediction. The conf values are close to the exploration values throughout. Across the 88 measures computed on both, 3 differ by more than 0.05; the largest difference is 0.080 (Qwen3, `rm_dir_same_word_any_case`).

## 4. What the lens shows across depth (descriptive)

This section reads the per-layer summaries of the conf lens runs. It tests nothing. It was written after the predictions had been read, to say where in depth L1 and L2 apply.

At the earliest layers, *any* lower-case piece beats the upper-case one, not only the coming word's. At layer 0 in upper-case passages, Lc > U holds at 0.42 of positions in Gemma and 0.79 in Qwen3. But Lr > U, which uses another word's lower-case piece, holds at 0.64 and 0.75, and Lc > Lr sits near chance (0.31 and 0.48). So early on the lens shows a general lean towards lower-case pieces, not the coming word.

The word itself appears in the lens at mid depth:

| Upper-case passages | Gemma: layers 16 / 20 / 24 | Qwen3: layers 16 / 19 / 22 |
|---|---|---|
| Lc > Lr (the coming word above another word) | 0.47 / 0.67 / 0.85 | 0.69 / 0.83 / 0.88 |
| Lr > U (another word's lower-case piece above U) | 0.64 / 0.43 / 0.16 | 0.61 / 0.15 / 0.13 |
| Lc > U (the coming word in lower case above U) | 0.67 / 0.65 / 0.71 | 0.83 / 0.65 / 0.75 |

In Gemma, the coming word stands out from other words at about layer 20, where Lc > Lr leaves chance and Lr > U starts to fall. In Qwen3, Lc > Lr rises more gradually through the middle layers, and Lr > U drops sharply at layer 19. So the lower-case form of the *specific* word leads only from mid depth. In exploration, adding the capitals direction also began to work around there in Qwen3 (`NOTEBOOK.md`). So "latent lower case" here means: once the model's choice of word shows in the lens, it shows in lower case. The lens says nothing about the form of the word before that depth. The protocol's L1 wording (Lr not above U "from mid depth on") already covered this; this section makes the depth explicit.

## 5. Checks made on the results

- **Gemma's two Lc > Lr shares are identical** (0.852 in both runs). This looked like a possible bug, so it was checked post hoc from the per-position records. The equality is a coincidence of counts: 15,029 of 17,641 positions pass in the upper-case run and 15,029 in the lower-case run, but the two runs agree at only 16,771 positions. So the identity is not an artefact of reading the same run twice. In exploration the two values differed (0.852 and 0.853). The counts are recomputed by `code/make_report.py`; for comparison, Qwen3's counts are 15,491 and 15,697, with agreement at 16,687.
- **E1 filter.** For upper-case words, the changed-word set and the set with different first pieces are the same in conf (identical n), so the frozen filter removed nothing.
- **Runs.** All thirteen confirmation runs ended normally at the first attempt (`outputs/conf_*.log`). `AMENDMENTS.md` was not needed.

## 6. What this does not show

- **It is a description at the level of directions, not a circuit.** The capitals direction is a difference of means, and the experiments show that it is sufficient and largely needed at one layer. They do not show which components write it or read it. Exploration found the writers spread across attention and MLP layers in Gemma (attention 14, MLPs 14–15, 23–24 and 33–34). That component patching was not carried forward, because its donors differed in the word as well as the case.
- **The edit does not always keep the word.** After adding or subtracting d, the new top-1 is the same word in the other case at 0.658–0.668 of positions in Qwen3 and 0.785–0.799 in Gemma, although the case flips at more than 0.95 of them. At the rest it changes the word too. Part of this is expected: a word chosen in lower case can have a different best continuation once in capitals, as E1 shows. But the edit is not a pure case switch.
- **Removal only half-decides the case.** Setting the component along d to the midpoint still leaves 0.540 (Gemma) and 0.527 (Qwen3) of the next words in capitals, not none. That fits a case set by more than one direction, or by the parts of the residual outside this one.
- **The lens is the lens.** The L1 and L2 results read the residual through the model's own final norm and output layer. Section 4 shows that before mid depth the lens reflects a general lean towards lower-case pieces, so it cannot say what form the word takes before that depth.
- **Two models, one text source, one word position per decision.** Both are small base models, tested on the charter's English encyclopaedic paragraphs. SmolLM2-1.7B and Pythia-1B were not tested because they failed the gate. The layer-0 case reader (H1) was tested in Qwen3 only. Exploration found no such head in Gemma, where case enters later (around attention 14).
- **Not tested:** whether the same direction governs other all-capitals settings (headings, acronyms, code constants), and why the initial capital of names and sentence starts does not respond to it. Exploration found subtracting d weak there.

## 7. Files

- Protocol and digests: `PROTOCOL.md`, `FROZEN.md`. Charter: `CHARTER.md`. Exploration log: `NOTEBOOK.md`.
- Frozen confirmation code: `code/conf_*.py`, `code/d5common.py`, `code/run.sh`.
- Outputs: `outputs/conf_gate_<model>.json`, `outputs/conf_exp01_<model>.json` (lens; with residual means not used here), `outputs/conf_exp02_<model>.json`, `outputs/conf_exp04_<model>.json`, `outputs/conf_exp06_qwen3-17b_L0H3.json`, `outputs/conf_exp07_<model>.json`, `outputs/conf_exp08_<model>.json`, and the verdicts `outputs/conf_verdicts.json`.
- This report: `code/report_template.md` filled by `code/make_report.py`.
