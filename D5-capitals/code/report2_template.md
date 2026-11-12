# D5 round 2 report: the parts that write, carry and read the capitals direction

<!-- nocheck -->**8 November 2026.** Exploration under `CHARTER-2.md` (frozen 17:02 IST) is in `NOTEBOOK.md` and is **exploratory**. The confirmation followed `PROTOCOL-2.md` (frozen 20:26 IST, digests in `FROZEN.md`) and ran **once** on held-out material sealed at the charter freeze: the reserve paragraphs (528 paragraphs from 299 articles, unread since round 1 sealed them, and not read by X1) and new heading items from the reserve articles. It ran from 20:26 to 21:28 IST, with no crash and no rerun. Every prediction below is **prospective**. Round 1 (`REPORT.md`) is unchanged.

<!-- nocheck -->Every number comes from `outputs/r2/conf_verdicts2.json`, written by the frozen `code/r2_verdict.py` from the per-position records, and, for exploration values, from the frozen `outputs/r2/explore_verdicts2_check.json`. `code/make_report2.py` fills this text from those files. `code/check_report2.py`, written separately and sharing no code with either, recomputes every value and every pass or fail from the raw records and checks every number here (`outputs/r2/REPORT2-CHECK.md`).

## Short answer

**The capitals direction from round 1 has identifiable parts in both models, and all {{q:V.n_tests|raw}} predictions for Qwen3-1.7B-Base and all {{g:V.n_tests|raw}} for Gemma-4-E2B held on held-out text ({{q:V.n_pass|raw}} of {{q:V.n_tests|raw}} and {{g:V.n_pass|raw}} of {{g:V.n_tests|raw}}).**

- **Writers.** A handful of late components write the direction: in Qwen3 MLPs 19–22 and head L19H0, in Gemma MLPs 22–24 and head L24H6. Moving only their writes along the direction to their lower-case values leaves an upper-case next word at {{q:W_named_upper}} (Qwen3) and {{g:W_named_upper}} (Gemma) of positions, against at least {{q:W_random_upper.min}} and {{g:W_random_upper.min}} for twenty matched random sets.
- **Carriers.** With the current word held in lower case, the case of the earlier words reaches the prediction position through a few heads: Qwen3 L19H0, L17H6, L18H1 and L18H0; Gemma L14H6, L14H7, L14H1, L15H2 and L24H6. Patching them removes {{q:CA_named_proj_removed}} and {{g:CA_named_proj_removed}} of that excess along the direction. Each model has one local carrier that reads the last few words (Qwen3 L17H6, Gemma L14H7) and broad carriers that read the passage further back. Qwen3's L17H6, the format head of the cross-study test X1, carries {{q:L17H6_proj_removed}} of the excess on its own.
- **Readers.** Only MLPs after the direction layer read it: freezing them stops adding the direction from flipping the case ({{q:R_none_flip}} → {{q:R_named_flip}} in Qwen3, {{g:R_none_flip}} → {{g:R_named_flip}} in Gemma), while freezing matched late heads leaves at least {{q:R_random_flip.min}} and {{g:R_random_flip.min}}.
- **Why the edit changes words.** Adding the direction changes the word mostly where writing in capitals changes it anyway: at {{q:E_change_if_up_changes}} of flips where the real upper-case passage also changes the word, against {{q:E_change_if_up_keeps}} where it keeps it (Qwen3; Gemma {{g:E_change_if_up_changes}} against {{g:E_change_if_up_keeps}}), and more often for words whose capitals form needs more tokens.
- **Other capitals.** Removing the direction leaves acronyms ({{q:A_acr_rm_keeps}}, {{g:A_acr_rm_keeps}}) and initial capitals ({{q:A_init_rm_keeps}}, {{g:A_init_rm_keeps}}) in place: they do not use it. Inside capitalised headings it takes capitals away (kept {{q:A_head_rm_keeps}} and {{g:A_head_rm_keeps}}), and the blank line after the heading ends the capitals mode.
- In line with X1, Qwen3's head L0H3 attends to the current token ({{q:attn_L0H3.0}} of its attention) and passes on its case, as one property of the current token. It does not carry the case of earlier words.

## 1. What was tested

- **Models:** Gemma-4-E2B (base) and Qwen3-1.7B-Base, as in round 1. Direction d, its random control r (same norm), the direction layer ℓ (Gemma 24, Qwen3 22) and the round-1 means are reused unchanged; u is the unit vector of d.
- **Runs at word starts:** U = upper-case passage, L = lower-case passage, A = upper-case passage with its last word in lower case (so A and L share the current token and differ only in the earlier words). Edits act at the prediction position only.
- **Writers (W1):** each named component's write along u (its additive term in the residual, after Gemma's post-norms) moved to its value in the paired L run; the rest of its output unchanged.
- **Carriers (C1–C2):** each named head's output replaced by its value in the L run, in A runs and in U runs. **Readers (R1):** each named MLP frozen at its unedited L-run output while d is added after ℓ.
- **Controls:** twenty random sets per named set, matched member by member (a head by another head of the same layer; a writer MLP by another MLP; a reader MLP by a head of the same layer).
- **Material:** sets in reserve paragraphs 0–300 (Qwen3) and 0–250 (Gemma), two word starts each; attention in 0–200, three each; edits in 0–300 (Qwen3) and 0–200 (Gemma), four each; acronyms and initial capitals in all 528 paragraphs; heading items 0–500.
- **Statistics:** decisions use point estimates; brackets give 95% intervals from 2,000 bootstrap resamples of articles.

## 2. Results by prediction

Exploration values are in the last column (Qwen3 / Gemma).

| Test | Prediction | Qwen3 held out | Gemma held out | Exploration |
|---|---|---|---|---|
| W1 | Writers' writes along u moved: upper-case top-1 ≤ 0.30 | {{q:W_named_upper|ci}} | {{g:W_named_upper|ci}} | {{eq:W_named_upper}} / {{eg:W_named_upper}} |
| | … same word in any case ≥ 0.55 | {{q:W_named_same_word|ci}} | {{g:W_named_same_word|ci}} | {{eq:W_named_same_word}} / {{eg:W_named_same_word}} |
| | … every random set ≥ 0.85 (lowest) | {{q:W_random_upper.min}} | {{g:W_random_upper.min}} | {{eq:W_random_upper.min}} / {{eg:W_random_upper.min}} |
| R1 | Adding d flips to capitals ≥ 0.80 | {{q:R_none_flip|ci}} | {{g:R_none_flip|ci}} | {{eq:R_none_flip}} / {{eg:R_none_flip}} |
| | … with the readers frozen ≤ 0.20 | {{q:R_named_flip|ci}} | {{g:R_named_flip|ci}} | {{eq:R_named_flip}} / {{eg:R_named_flip}} |
| | … with random sets frozen ≥ unfrozen − 0.15 (lowest) | {{q:R_random_flip.min}} | {{g:R_random_flip.min}} | {{eq:R_random_flip.min}} / {{eg:R_random_flip.min}} |
| C1 | Carriers patched in A: excess projection removed ≥ 0.60 | {{q:CA_named_proj_removed|ci}} | {{g:CA_named_proj_removed|ci}} | {{eq:CA_named_proj_removed}} / {{eg:CA_named_proj_removed}} |
| | … excess D removed ≥ 0.30 | {{q:CA_named_D_removed|ci}} | {{g:CA_named_D_removed|ci}} | {{eq:CA_named_D_removed}} / {{eg:CA_named_D_removed}} |
| | … random sets ≤ named − 0.25 (highest) | {{q:CA_random_proj_removed.max}} | {{g:CA_random_proj_removed.max}} | {{eq:CA_random_proj_removed.max}} / {{eg:CA_random_proj_removed.max}} |
| C2 | Carriers patched in U: upper-case top-1 ≤ 0.60 | {{q:CU_named_upper|ci}} | {{g:CU_named_upper|ci}} | {{eq:CU_named_upper}} / {{eg:CU_named_upper}} |
| | … every random set ≥ 0.90 (lowest) | {{q:CU_random_upper.min}} | {{g:CU_random_upper.min}} | {{eq:CU_random_upper.min}} / {{eg:CU_random_upper.min}} |
| C3 | Qwen3 L17H6 alone in A: removed ≥ 0.20, above every random layer-17 head | {{q:L17H6_proj_removed|ci}}; random at most {{q:L17H6_random_proj_removed.max}} | n/a | {{eq:L17H6_proj_removed}} |
| C4 | Local carrier: attention on the last eight words ≥ 0.60 | L17H6 {{q:attn_L17H6_recent8|ci}} | L14H7 {{g:attn_L14H7_recent8|ci}} | {{eq:attn_L17H6_recent8}} / {{eg:attn_L14H7_recent8}} |
| | Broad carrier: the same ≤ 0.50 | L19H0 {{q:attn_L19H0_recent8|ci}} | L24H6 {{g:attn_L24H6_recent8|ci}} | {{eq:attn_L19H0_recent8}} / {{eg:attn_L24H6_recent8}} |
| E1 | Edit changes the word where the upper passage does | {{q:E_change_if_up_changes|ci}} | {{g:E_change_if_up_changes|ci}} | {{eq:E_change_if_up_changes}} / {{eg:E_change_if_up_changes}} |
| | … where it does not (difference ≥ 0.30) | {{q:E_change_if_up_keeps|ci}} | {{g:E_change_if_up_keeps|ci}} | {{eq:E_change_if_up_keeps}} / {{eg:E_change_if_up_keeps}} |
| | Changed word in the lower passage's top 5 ≥ 0.40 | {{q:E_changed_in_lower_top5|ci}} | {{g:E_changed_in_lower_top5|ci}} | {{eq:E_changed_in_lower_top5}} / {{eg:E_changed_in_lower_top5}} |
| E2 | Change rate, capitals form needs ≥ 1 extra token | {{q:E_change_extra1p|ci}} | {{g:E_change_extra1p|ci}} | {{eq:E_change_extra1p}} / {{eg:E_change_extra1p}} |
| | … no extra token (difference ≥ 0.10) | {{q:E_change_extra0|ci}} | {{g:E_change_extra0|ci}} | {{eq:E_change_extra0}} / {{eg:E_change_extra0}} |
| A1 | Acronyms: removing d keeps all capitals ≥ 0.85 | {{q:A_acr_rm_keeps|ci}} | {{g:A_acr_rm_keeps|ci}} | {{eq:A_acr_rm_keeps}} / {{eg:A_acr_rm_keeps}} |
| | … removing r ≥ 0.90 | {{q:A_acr_rm_rand_keeps|ci}} | {{g:A_acr_rm_rand_keeps|ci}} | {{eq:A_acr_rm_rand_keeps}} / {{eg:A_acr_rm_rand_keeps}} |
| A2 | Initial capitals: removing d keeps them ≥ 0.90 | {{q:A_init_rm_keeps|ci}} | {{g:A_init_rm_keeps|ci}} | {{eq:A_init_rm_keeps}} / {{eg:A_init_rm_keeps}} |
| | … removing r ≥ 0.95 | {{q:A_init_rm_rand_keeps|ci}} | {{g:A_init_rm_rand_keeps|ci}} | {{eq:A_init_rm_rand_keeps}} / {{eg:A_init_rm_rand_keeps}} |
| A3 | Capitalised headings: removing d keeps all capitals ≤ 0.85 | {{q:A_head_rm_keeps|ci}} | {{g:A_head_rm_keeps|ci}} | {{eq:A_head_rm_keeps}} / {{eg:A_head_rm_keeps}} |
| | … removing r ≥ 0.95 | {{q:A_head_rm_rand_keeps|ci}} | {{g:A_head_rm_rand_keeps|ci}} | {{eq:A_head_rm_rand_keeps}} / {{eg:A_head_rm_rand_keeps}} |
| A4 | Excess along u inside the heading ≥ 0.40 of the gap | {{q:A_head_inside_excess|ci}} | {{g:A_head_inside_excess|ci}} | {{eq:A_head_inside_excess}} / {{eg:A_head_inside_excess}} |
| | … at the first word after it ≤ 0.25 | {{q:A_head_after_excess|ci}} | {{g:A_head_after_excess|ci}} | {{eq:A_head_after_excess}} / {{eg:A_head_after_excess}} |
| | … top-1 all capitals there ≤ 0.10 | {{q:A_head_after_allcaps_top|ci}} | {{g:A_head_after_allcaps_top|ci}} | {{eq:A_head_after_allcaps_top}} / {{eg:A_head_after_allcaps_top}} |

Positions: W1 {{q:W_named_upper|n}} (Qwen3) and {{g:W_named_upper|n}} (Gemma); R1 {{q:R_none_flip|n}} and {{g:R_none_flip|n}}; C1 {{q:CA_named_proj_removed|n}} and {{g:CA_named_proj_removed|n}}; edits {{q:E_change|n}} and {{g:E_change|n}} flips; acronyms {{q:A_acr_rm_keeps|n}} and {{g:A_acr_rm_keeps|n}}; initial capitals {{q:A_init_rm_keeps|n}} and {{g:A_init_rm_keeps|n}}; heading words {{q:A_head_rm_keeps|n}} and {{g:A_head_rm_keeps|n}}.

## 3. Verdicts

As fixed in the protocol:

| | Qwen3-1.7B-Base | Gemma-4-E2B |
|---|---|---|
| Q1, writers and readers | {{q:V.Q1_parts|raw}} | {{g:V.Q1_parts|raw}} |
| Q2, carriers of passage case | {{q:V.Q2_carriers|raw}} | {{g:V.Q2_carriers|raw}} |
| Q3, why the edit changes words | {{q:V.Q3_word_changes|raw}} | {{g:V.Q3_word_changes|raw}} |
| Q4, acronyms and initial capitals separate | {{q:V.Q4_acronyms_initials_separate|raw}} | {{g:V.Q4_acronyms_initials_separate|raw}} |
| Q4, headings use d, the line break ends it | {{q:V.Q4_headings_same_direction_reset|raw}} | {{g:V.Q4_headings_same_direction_reset|raw}} |
| Predictions passed | {{q:V.n_pass|raw}} of {{q:V.n_tests|raw}} | {{g:V.n_pass|raw}} of {{g:V.n_tests|raw}} |

**The parts account (writers, carriers, readers) is supported in both models.** Q5 (training dynamics) was not tested. OLMo-2-0425-1B, the checkpointed candidate, failed the frozen round-1 gate, and the charter's stop rule applied.

## 4. Descriptive detail (held out, not tested beyond the predictions)

- **The writers' edit goes past the lower-case level.** Moving only the writers' writes along u removes {{q:W_named_proj_removed}} (Qwen3) and {{g:W_named_proj_removed}} (Gemma) of the gap between U and L runs along u. These are more than the whole gap: the components between the writers and ℓ do not rebuild it, and the residual ends a little below the lower-case level.
- **Carriers in full capitals.** In U runs, the carriers' patch also changes the current token's context, but not its own case. It leaves an upper-case next word at {{q:CU_named_upper}} and {{g:CU_named_upper}}. The rest of the capitals comes from the current token itself, which in Qwen3 enters through L0H3.
- **Attention by words back** (current word, 1, 2–3, 4–7, 8 or more, first text token, BOS), mean over held-out positions:
  - Qwen3 L17H6: {{q:attn_L17H6.0}}, {{q:attn_L17H6.1}}, {{q:attn_L17H6.2}}, {{q:attn_L17H6.3}}, {{q:attn_L17H6.4}}, {{q:attn_L17H6.5}};
  - Qwen3 L19H0: {{q:attn_L19H0.0}}, {{q:attn_L19H0.1}}, {{q:attn_L19H0.2}}, {{q:attn_L19H0.3}}, {{q:attn_L19H0.4}}, {{q:attn_L19H0.5}};
  - Qwen3 L18H1: {{q:attn_L18H1.4}} on eight or more words back, {{q:attn_L18H1.5}} on the first token;
  - Qwen3 L18H0: {{q:attn_L18H0.4}} and {{q:attn_L18H0.5}};
  - Gemma L14H7: {{g:attn_L14H7.0}}, {{g:attn_L14H7.1}}, {{g:attn_L14H7.2}}, {{g:attn_L14H7.3}}, {{g:attn_L14H7.4}}, {{g:attn_L14H7.5}}, {{g:attn_L14H7.6}};
  - Gemma L24H6: {{g:attn_L24H6.0}}, {{g:attn_L24H6.1}}, {{g:attn_L24H6.2}}, {{g:attn_L24H6.3}}, {{g:attn_L24H6.4}}, {{g:attn_L24H6.5}}, {{g:attn_L24H6.6}};
  - Gemma L14H1 is in between: {{g:attn_L14H1_recent8}} on the last eight words.
  - So passage case comes both from the last few words and from the passage as a whole. In running text the two agree. How the MLPs weigh them to give round 1's recency curve was not tested.
- **Word changes.** The edit changes the word at {{q:E_change}} of flips in Qwen3 and {{g:E_change}} in Gemma. Changes that the real upper-case passage does not make are {{q:E_change_not_by_up}} and {{g:E_change_not_by_up}} of all flips. That is an upper bound on what an impure direction could add. Random directions flipped none of these positions ({{q:E_random_flip}}, {{g:E_random_flip}}).
- **Projection scale:** round-1 means along u are {{q:proj_upper_mean|2}} (upper) and {{q:proj_lower_mean|2}} (lower) in Qwen3, and {{g:proj_upper_mean|2}} and {{g:proj_lower_mean|2}} in Gemma.

## 5. What this does not show

- **These are sets, not single paths.** Writers and readers are tested as sets. Their members overlap with the carriers' downstream MLPs, and we did not trace each path between them. Gemma's random carrier sets share layer 14 with four named heads and removed up to {{g:CA_random_proj_removed.max}}, so layer-14 attention carries case more widely than the named heads alone.
- **Reader freezing is a strong edit.** Freezing late MLPs at their unedited outputs also keeps whatever else they would have written. It shows that the flip needs them, not that they do nothing else.
- **What the carriers read.** The carriers' attention patterns show where they look, not what they read there. Whether L19H0 reads the case of each earlier word or a running summary was not separated.
- **Scope.** Two models, encyclopaedic English, word starts and heading words only. "Acronym" means a word of two or more capitals in mixed-case text, and the test is limited to positions whose top-1 was already all capitals.
- **Held-out text.** The held-out material is held out from these analyses, not from the models' training data.

## 6. Records

`CHARTER-2.md`, `PROTOCOL-2.md` and `FROZEN.md`; `NOTEBOOK.md` (round-2 exploration, dead ends included); `outputs/r2/` (exploration and held-out records, `conf_verdicts2.json`, `explore_verdicts2_check.json`, `REPORT2-CHECK.md`); logs `outputs/r2_*.log`. Nothing is committed.
