# D5 report: how two small models keep writing in capitals

**8 November 2026.** Confirmation of the D5 candidate account under the protocol frozen at 05:48 IST on 8 November (`PROTOCOL.md`, digests in `FROZEN.md`). Every prediction below is **prospective**: it was fixed before any model read the held-out `conf` part, which was then run once. Section 4 is **descriptive** (read from the same runs' saved summaries, not a prediction). Exploration results are in `NOTEBOOK.md` and are **exploratory**. The `reserve` part was not read. There were no amendments: all thirteen runs finished at the first attempt.

Every number in this file is filled in by `code/make_report.py` from saved outputs (`outputs/conf_verdicts.json`, written by the frozen `code/conf_verdict.py`; `outputs/explore_verdicts_check.json`; `outputs/conf_exp01_<model>.json`; `outputs/conf_exp08_<model>.json`). The text is `code/report_template.md`.

## Short answer

In both Gemma-4-E2B and Qwen3-1.7B-Base, every prediction passes on new paragraphs (nine in Gemma, ten in Qwen3, whose extra one concerns a head Gemma lacks), and the frozen account is **supported** in both models. Three things hold on held-out text:

- While the model writes an upper-case passage, the logit lens from mid depth puts the coming word's **lower-case** first piece above its actual upper-case piece. The capitals win only in the last three layers (from layer {{g:first_layer_Lc_below_half_for_good|int}} of 35 in Gemma and {{q:first_layer_Lc_below_half_for_good|int}} of 28 in Qwen3).
- A single **capitals direction**, estimated once on exploration text, switches the next word between lower case and capitals in either direction, and mostly keeps the word. Random directions of the same size do nothing. Setting the residual to the middle of this direction leaves the case about half decided.
- The case is set by the last few words. In Qwen3, one layer-0 head reads each token's own case into the residual stream.

The two models agree on every prediction.

## 1. What was tested

- **Models.** Gemma-4-E2B (base; many upper-case words are single tokens) and Qwen3-1.7B-Base (upper-case words split into several pieces). These were the two models that passed the frozen gate during exploration.
- **Material.** The held-out `conf` part: {{calc:conf_paragraphs|int}} paragraphs from {{calc:conf_articles|int}} articles, split off by article in the frozen charter. Each run uses a stated range from the start of the shuffled part.
- **Fixed in advance.** Two layers were fixed from exploration: the direction layer ℓ (24 for Gemma, 22 for Qwen3) and the early layer 16. The capitals direction d and its random control r were not re-estimated: they are the exploration means (exploration paragraphs 150–300).
- **Statistics.** Decisions use point estimates. Brackets give 95% intervals from 2,000 bootstrap resamples of paragraphs.

Terms used below. *U* is the true upper-case first piece of the next word, and *Lc* is the same word's lower-case first piece. *Lr* is the lower-case first piece of another word in the paragraph. A *flip* is a change of the top-1 token's case. *Same word* means the new top-1 is the old top-1 word in the other case.

## 2. Results by prediction

Each row gives the conf value with its 95% interval. The bracketed pair after the threshold gives the exploration values for Gemma / Qwen3.

### G1: gate replication (conf paragraphs 0–150)

| Condition (threshold) | Gemma | Qwen3 |
|---|---|---|
| Case kept when continuing an upper-case passage (≥ 0.95) | {{g:gate.c1_case_kept}} | {{q:gate.c1_case_kept}} |
| Same word as the lower-case passage (≥ 0.60) | {{g:gate.c2_same_word}} | {{q:gate.c2_same_word}} |
| Inside-word pieces correct (≥ 0.70) | {{g:gate.c3_inside_word}} | {{q:gate.c3_inside_word}} |
| **G1** | **{{g:P.G1|pf}}** | **{{q:P.G1|pf}}** |

### L1 and L2: latent lower case (logit lens, conf paragraphs 0–150)

Positions: {{g:lens_Lc_gt_U_at_l|n}} for Gemma and {{q:lens_Lc_gt_U_at_l|n}} for Qwen3 at ℓ, after leaving out positions where U and Lc have equal lens values.

| Measure (threshold) [exploration G / Q] | Gemma | Qwen3 |
|---|---|---|
| Lc > U at ℓ (≥ 0.60) [{{eg:lens_Lc_gt_U_at_l}} / {{eq:lens_Lc_gt_U_at_l}}] | {{g:lens_Lc_gt_U_at_l|ci}} | {{q:lens_Lc_gt_U_at_l|ci}} |
| Lr > U at ℓ (≤ 0.30) [{{eg:lens_Lr_gt_U_at_l}} / {{eq:lens_Lr_gt_U_at_l}}] | {{g:lens_Lr_gt_U_at_l|ci}} | {{q:lens_Lr_gt_U_at_l|ci}} |
| Lc > U at the last layer (≤ 0.05) [{{eg:lens_Lc_gt_U_last}} / {{eq:lens_Lc_gt_U_last}}] | {{g:lens_Lc_gt_U_last|ci}} | {{q:lens_Lc_gt_U_last|ci}} |
| First layer from which Lc > U stays below 0.5 (Gemma ≥ 30 of 35, Qwen3 ≥ 23 of 28) [{{eg:first_layer_Lc_below_half_for_good|int}} / {{eq:first_layer_Lc_below_half_for_good|int}}] | {{g:first_layer_Lc_below_half_for_good|int}} | {{q:first_layer_Lc_below_half_for_good|int}} |
| **L1** | **{{g:P.L1|pf}}** | **{{q:P.L1|pf}}** |
| Lc > Lr at ℓ, upper-case passages (≥ 0.75) [{{eg:lens_Lc_gt_Lr_upper_at_l}} / {{eq:lens_Lc_gt_Lr_upper_at_l}}] | {{g:lens_Lc_gt_Lr_upper_at_l|ci}} | {{q:lens_Lc_gt_Lr_upper_at_l|ci}} |
| Lc > Lr at ℓ, lower-case passages (≥ 0.75; differs from the line above by ≤ 0.05) [{{eg:lens_Lc_gt_Lr_lower_at_l}} / {{eq:lens_Lc_gt_Lr_lower_at_l}}] | {{g:lens_Lc_gt_Lr_lower_at_l|ci}} | {{q:lens_Lc_gt_Lr_lower_at_l|ci}} |
| **L2** | **{{g:P.L2|pf}}** | **{{q:P.L2|pf}}** |

### D1–D4: the capitals direction (conf paragraphs 0–100, one edit at the prediction position, a = 1)

| Measure (threshold) [exploration G / Q] | Gemma (ℓ = 24) | Qwen3 (ℓ = 22) |
|---|---|---|
| Add d to a lower-case passage: flips to capitals (≥ 0.80) [{{eg:add_dir_L24_flip}} / {{eq:add_dir_L22_flip}}] | {{g:add_dir_L24_flip|ci}} | {{q:add_dir_L22_flip|ci}} |
| … same word in capitals (≥ 0.45) [{{eg:add_dir_L24_same_word}} / {{eq:add_dir_L22_same_word}}] | {{g:add_dir_L24_same_word|ci}} | {{q:add_dir_L22_same_word|ci}} |
| Subtract d from an upper-case passage: flips to lower case (≥ 0.80) [{{eg:sub_dir_L24_flip}} / {{eq:sub_dir_L22_flip}}] | {{g:sub_dir_L24_flip|ci}} | {{q:sub_dir_L22_flip|ci}} |
| … same word in lower case (≥ 0.45) [{{eg:sub_dir_L24_same_word}} / {{eq:sub_dir_L22_same_word}}] | {{g:sub_dir_L24_same_word|ci}} | {{q:sub_dir_L22_same_word|ci}} |
| Add r: flips (≤ 0.05) | {{g:add_rand_L24_flip|ci}} | {{q:add_rand_L22_flip|ci}} |
| Subtract r: flips (≤ 0.05) | {{g:sub_rand_L24_flip|ci}} | {{q:sub_rand_L22_flip|ci}} |
| Positions (add / subtract) | {{g:add_dir_L24_flip|n}} / {{g:sub_dir_L24_flip|n}} | {{q:add_dir_L22_flip|n}} / {{q:sub_dir_L22_flip|n}} |
| **D1** | **{{g:P.D1|pf}}** | **{{q:P.D1|pf}}** |
| Set the component along d to the midpoint: upper-case top-1 left (≤ 0.70) [{{eg:rm_dir_upper_share}} / {{eq:rm_dir_upper_share}}] | {{g:rm_dir_upper_share|ci}} | {{q:rm_dir_upper_share|ci}} |
| … still the same word in either case (≥ 0.60) [{{eg:rm_dir_same_word_any_case}} / {{eq:rm_dir_same_word_any_case}}] | {{g:rm_dir_same_word_any_case|ci}} | {{q:rm_dir_same_word_any_case|ci}} |
| Same operation along r: upper case left (≥ 0.95) | {{g:rm_rand_upper_share|ci}} | {{q:rm_rand_upper_share|ci}} |
| **D2** | **{{g:P.D2|pf}}** | **{{q:P.D2|pf}}** |
| Ordinary mixed-case prose, add d: capitals (≥ 0.75) [{{eg:orig_add_dir_to_upper}} / {{eq:orig_add_dir_to_upper}}] | {{g:orig_add_dir_to_upper|ci}} | {{q:orig_add_dir_to_upper|ci}} |
| … same word in capitals (≥ 0.50) [{{eg:orig_add_dir_to_upper_same_word}} / {{eq:orig_add_dir_to_upper_same_word}}] | {{g:orig_add_dir_to_upper_same_word|ci}} | {{q:orig_add_dir_to_upper_same_word|ci}} |
| … add r: capitals (≤ 0.05) | {{g:orig_add_rand_to_upper|ci}} | {{q:orig_add_rand_to_upper|ci}} |
| **D3** | **{{g:P.D3|pf}}** | **{{q:P.D3|pf}}** |
| Layer 16, add d: flips (≤ 0.15) [{{eg:add_dir_L16_flip}} / {{eq:add_dir_L16_flip}}] | {{g:add_dir_L16_flip|ci}} | {{q:add_dir_L16_flip|ci}} |
| Layer 16, subtract d: flips (≥ 0.40) [{{eg:sub_dir_L16_flip}} / {{eq:sub_dir_L16_flip}}] | {{g:sub_dir_L16_flip|ci}} | {{q:sub_dir_L16_flip|ci}} |
| **D4** | **{{g:P.D4|pf}}** | **{{q:P.D4|pf}}** |

### R1: the case comes from the last few words (conf paragraphs 0–100)

Each cell is the share of upper-case top-1 tokens among cased top-1 tokens at word starts (word 12 or later). *L+u2* is a lower-case passage whose last two words are in capitals; *U+l2* is an upper-case passage whose last two words are in lower case.

| Passage (threshold) [exploration G / Q] | Gemma | Qwen3 |
|---|---|---|
| L+u1 (no threshold) | {{g:upper_top1_L+u1|ci}} | {{q:upper_top1_L+u1|ci}} |
| L+u2 (≥ 0.50) [{{eg:upper_top1_L+u2}} / {{eq:upper_top1_L+u2}}] | {{g:upper_top1_L+u2|ci}} | {{q:upper_top1_L+u2|ci}} |
| L+u8 (≥ 0.85) [{{eg:upper_top1_L+u8}} / {{eq:upper_top1_L+u8}}] | {{g:upper_top1_L+u8|ci}} | {{q:upper_top1_L+u8|ci}} |
| U+l1 (no threshold) | {{g:upper_top1_U+l1|ci}} | {{q:upper_top1_U+l1|ci}} |
| U+l2 (≤ 0.25) [{{eg:upper_top1_U+l2}} / {{eq:upper_top1_U+l2}}] | {{g:upper_top1_U+l2|ci}} | {{q:upper_top1_U+l2|ci}} |
| U+l8 (≤ 0.10) [{{eg:upper_top1_U+l8}} / {{eq:upper_top1_U+l8}}] | {{g:upper_top1_U+l8|ci}} | {{q:upper_top1_U+l8|ci}} |
| **R1** | **{{g:P.R1|pf}}** | **{{q:P.R1|pf}}** |

### H1: Qwen3's layer-0 case reader, head L0H3 (conf paragraphs 0–50)

| Measure (threshold) [exploration] | Qwen3 |
|---|---|
| Attention on the current token, upper-case passages (≥ 0.90) [{{eq:L0H3_self_attention_upper}}] | {{q:L0H3_self_attention_upper|ci}} |
| Attention on the current token, lower-case passages (≥ 0.90) [{{eq:L0H3_self_attention_lower}}] | {{q:L0H3_self_attention_lower|ci}} |
| Value from another paragraph in lower case: share of D moved (≥ 0.20) [{{eq:L0H3_other_lower_moved}}] | {{q:L0H3_other_lower_moved|ci}} |
| … flips to lower case (≥ 0.25) [{{eq:L0H3_other_lower_flip}}] | {{q:L0H3_other_lower_flip|ci}} |
| Value from another paragraph in upper case: share of D moved (≤ 0.08) [{{eq:L0H3_other_upper_moved}}] | {{q:L0H3_other_upper_moved|ci}} |
| … flips (≤ 0.05) [{{eq:L0H3_other_upper_flip}}] | {{q:L0H3_other_upper_flip|ci}} |
| Control head L0H2 patched from the same word in lower case: share of D moved (≤ 0.08) | {{q:L0H3_ctrl_L0H2_same_lower_moved|ci}} |
| Control head L0H0, same: share of D moved (≤ 0.08) | {{q:L0H3_ctrl_L0H0_same_lower_moved|ci}} |
| Reported, no threshold: value from the same word's own lower-case passage, share of D moved / flips | {{q:L0H3_same_lower_moved}} / {{q:L0H3_same_lower_flip}} |
| Positions (upper-case passages with an upper-case top-1) | {{q:L0H3_other_lower_moved|n}} |
| **H1** | **{{q:P.H1|pf}}** |

D is the gap logp(U) − logp(Lc). "Share of D moved" is measured towards the lower-case passage's own D.

### E1: changed words were lower-case runner-ups (gate word starts, conf paragraphs 0–150)

These are positions where the upper-case passage writes a different word from the lower-case passage, and the two words' lower-case first pieces differ. There were {{g:changed_word_top5|n}} such positions for Gemma (in {{g:changed_word_top5|np}} paragraphs) and {{q:changed_word_top5|n}} for Qwen3 (in {{q:changed_word_top5|np}}).

| Measure (threshold) [exploration G / Q] | Gemma | Qwen3 |
|---|---|---|
| Changed word's lower-case first piece in the lower-case passage's top 5 (≥ 0.50) [{{eg:changed_word_top5}} / {{eq:changed_word_top5}}] | {{g:changed_word_top5|ci}} | {{q:changed_word_top5|ci}} |
| Control word from the same paragraph in the top 5 (≤ 0.10) [{{eg:ctrl_word_top5}} / {{eq:ctrl_word_top5}}] | {{g:ctrl_word_top5|ci}} | {{q:ctrl_word_top5|ci}} |
| Reported, no threshold: median rank of the changed word / of the control word | {{x8g:upper_diff_first.word_median_rank|med}} / {{x8g:upper_diff_first.ctrl_median_rank|med}} | {{x8q:upper_diff_first.word_median_rank|med}} / {{x8q:upper_diff_first.ctrl_median_rank|med}} |
| **E1** | **{{g:P.E1|pf}}** | **{{q:P.E1|pf}}** |

## 3. Verdict

Under Section 5 of the protocol, the account (the word is chosen in a lower-case-like form and turned into capitals late, under one capitals direction) is **{{g:P.account|raw}}** for Gemma-4-E2B and **{{q:P.account|raw}}** for Qwen3-1.7B-Base. L1, L2, D1, D2 and D3 pass in both. All the secondary predictions also pass in both models: D4, R1, E1 and G1, plus H1 in Qwen3. The two models agree on every prediction. The conf values are close to the exploration values throughout. Across the {{calc:n_gaps|int}} measures computed on both, {{calc:gap_over_005|int}} differ by more than 0.05; the largest difference is {{calc:maxgap}} ({{calc:maxgap_model|raw}}, `{{calc:maxgap_key|raw}}`).

## 4. What the lens shows across depth (descriptive)

This section reads the per-layer summaries of the conf lens runs. It tests nothing. It was written after the predictions had been read, to say where in depth L1 and L2 apply.

At the earliest layers, *any* lower-case piece beats the upper-case one, not only the coming word's. At layer 0 in upper-case passages, Lc > U holds at {{lg:up.share_Lc_gt_U.0|2}} of positions in Gemma and {{lq:up.share_Lc_gt_U.0|2}} in Qwen3. But Lr > U, which uses another word's lower-case piece, holds at {{lg:up.share_Lr_gt_U.0|2}} and {{lq:up.share_Lr_gt_U.0|2}}, and Lc > Lr sits near chance ({{lg:up.share_Lc_gt_Lr.0|2}} and {{lq:up.share_Lc_gt_Lr.0|2}}). So early on the lens shows a general lean towards lower-case pieces, not the coming word.

The word itself appears in the lens at mid depth:

| Upper-case passages | Gemma: layers 16 / 20 / 24 | Qwen3: layers 16 / 19 / 22 |
|---|---|---|
| Lc > Lr (the coming word above another word) | {{lg:up.share_Lc_gt_Lr.16|2}} / {{lg:up.share_Lc_gt_Lr.20|2}} / {{lg:up.share_Lc_gt_Lr.24|2}} | {{lq:up.share_Lc_gt_Lr.16|2}} / {{lq:up.share_Lc_gt_Lr.19|2}} / {{lq:up.share_Lc_gt_Lr.22|2}} |
| Lr > U (another word's lower-case piece above U) | {{lg:up.share_Lr_gt_U.16|2}} / {{lg:up.share_Lr_gt_U.20|2}} / {{lg:up.share_Lr_gt_U.24|2}} | {{lq:up.share_Lr_gt_U.16|2}} / {{lq:up.share_Lr_gt_U.19|2}} / {{lq:up.share_Lr_gt_U.22|2}} |
| Lc > U (the coming word in lower case above U) | {{lg:up.share_Lc_gt_U.16|2}} / {{lg:up.share_Lc_gt_U.20|2}} / {{lg:up.share_Lc_gt_U.24|2}} | {{lq:up.share_Lc_gt_U.16|2}} / {{lq:up.share_Lc_gt_U.19|2}} / {{lq:up.share_Lc_gt_U.22|2}} |

In Gemma, the coming word stands out from other words at about layer 20, where Lc > Lr leaves chance and Lr > U starts to fall. In Qwen3, Lc > Lr rises more gradually through the middle layers, and Lr > U drops sharply at layer 19. So the lower-case form of the *specific* word leads only from mid depth. In exploration, adding the capitals direction also began to work around there in Qwen3 (`NOTEBOOK.md`). So "latent lower case" here means: once the model's choice of word shows in the lens, it shows in lower case. The lens says nothing about the form of the word before that depth. The protocol's L1 wording (Lr not above U "from mid depth on") already covered this; this section makes the depth explicit.

## 5. Checks made on the results

- **Gemma's two Lc > Lr shares are identical** ({{g:lens_Lc_gt_Lr_upper_at_l}} in both runs). This looked like a possible bug, so it was checked post hoc from the per-position records. The equality is a coincidence of counts: {{calc:lg_pass_upper|int}} of {{calc:lg_n|int}} positions pass in the upper-case run and {{calc:lg_pass_lower|int}} in the lower-case run, but the two runs agree at only {{calc:lg_agree|int}} positions. So the identity is not an artefact of reading the same run twice. In exploration the two values differed ({{eg:lens_Lc_gt_Lr_upper_at_l}} and {{eg:lens_Lc_gt_Lr_lower_at_l}}). The counts are recomputed by `code/make_report.py`; for comparison, Qwen3's counts are {{calc:lq_pass_upper|int}} and {{calc:lq_pass_lower|int}}, with agreement at {{calc:lq_agree|int}}.
- **E1 filter.** For upper-case words, the changed-word set and the set with different first pieces are the same in conf (identical n), so the frozen filter removed nothing.
- **Runs.** All thirteen confirmation runs ended normally at the first attempt (`outputs/conf_*.log`). `AMENDMENTS.md` was not needed.

## 6. What this does not show

- **It is a description at the level of directions, not a circuit.** The capitals direction is a difference of means, and the experiments show that it is sufficient and largely needed at one layer. They do not show which components write it or read it. Exploration found the writers spread across attention and MLP layers in Gemma (attention 14, MLPs 14–15, 23–24 and 33–34). That component patching was not carried forward, because its donors differed in the word as well as the case.
- **The edit does not always keep the word.** After adding or subtracting d, the new top-1 is the same word in the other case at {{q:add_dir_L22_same_word}}–{{q:sub_dir_L22_same_word}} of positions in Qwen3 and {{g:sub_dir_L24_same_word}}–{{g:add_dir_L24_same_word}} in Gemma, although the case flips at more than 0.95 of them. At the rest it changes the word too. Part of this is expected: a word chosen in lower case can have a different best continuation once in capitals, as E1 shows. But the edit is not a pure case switch.
- **Removal only half-decides the case.** Setting the component along d to the midpoint still leaves {{g:rm_dir_upper_share}} (Gemma) and {{q:rm_dir_upper_share}} (Qwen3) of the next words in capitals, not none. That fits a case set by more than one direction, or by the parts of the residual outside this one.
- **The lens is the lens.** The L1 and L2 results read the residual through the model's own final norm and output layer. Section 4 shows that before mid depth the lens reflects a general lean towards lower-case pieces, so it cannot say what form the word takes before that depth.
- **Two models, one text source, one word position per decision.** Both are small base models, tested on the charter's English encyclopaedic paragraphs. SmolLM2-1.7B and Pythia-1B were not tested because they failed the gate. The layer-0 case reader (H1) was tested in Qwen3 only. Exploration found no such head in Gemma, where case enters later (around attention 14).
- **Not tested:** whether the same direction governs other all-capitals settings (headings, acronyms, code constants), and why the initial capital of names and sentence starts does not respond to it. Exploration found subtracting d weak there.

## 7. Files

- Protocol and digests: `PROTOCOL.md`, `FROZEN.md`. Charter: `CHARTER.md`. Exploration log: `NOTEBOOK.md`.
- Frozen confirmation code: `code/conf_*.py`, `code/d5common.py`, `code/run.sh`.
- Outputs: `outputs/conf_gate_<model>.json`, `outputs/conf_exp01_<model>.json` (lens; with residual means not used here), `outputs/conf_exp02_<model>.json`, `outputs/conf_exp04_<model>.json`, `outputs/conf_exp06_qwen3-17b_L0H3.json`, `outputs/conf_exp07_<model>.json`, `outputs/conf_exp08_<model>.json`, and the verdicts `outputs/conf_verdicts.json`.
- This report: `code/report_template.md` filled by `code/make_report.py`.
