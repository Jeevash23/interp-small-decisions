# D5 round 2 report: the parts that write, carry and read the capitals direction

<!-- nocheck -->**8 November 2026.** Exploration under `CHARTER-2.md` (frozen 17:02 IST) is in `NOTEBOOK.md` and is **exploratory**. The confirmation followed `PROTOCOL-2.md` (frozen 20:26 IST, digests in `FROZEN.md`) and ran **once** on held-out material sealed at the charter freeze: the reserve paragraphs (528 paragraphs from 299 articles, unread since round 1 sealed them, and not read by X1) and new heading items from the reserve articles. It ran from 20:26 to 21:28 IST, with no crash and no rerun. Every prediction below is **prospective**. Round 1 (`REPORT.md`) is unchanged.

<!-- nocheck -->Every number comes from `outputs/r2/conf_verdicts2.json`, written by the frozen `code/r2_verdict.py` from the per-position records, and, for exploration values, from the frozen `outputs/r2/explore_verdicts2_check.json`. `code/make_report2.py` fills this text from those files. `code/check_report2.py`, written separately and sharing no code with either, recomputes every value and every pass or fail from the raw records and checks every number here (`outputs/r2/REPORT2-CHECK.md`).

## Short answer

**The capitals direction from round 1 has identifiable parts in both models, and all 12 predictions for Qwen3-1.7B-Base and all 11 for Gemma-4-E2B held on held-out text (12 of 12 and 11 of 11).**

- **Writers.** A handful of late components write the direction: in Qwen3 MLPs 19–22 and head L19H0, in Gemma MLPs 22–24 and head L24H6. Moving only their writes along the direction to their lower-case values leaves an upper-case next word at 0.029 (Qwen3) and 0.009 (Gemma) of positions, against at least 0.985 and 0.949 for twenty matched random sets.
- **Carriers.** With the current word held in lower case, the case of the earlier words reaches the prediction position through a few heads: Qwen3 L19H0, L17H6, L18H1 and L18H0; Gemma L14H6, L14H7, L14H1, L15H2 and L24H6. Patching them removes 0.872 and 0.859 of that excess along the direction. Each model has one local carrier that reads the last few words (Qwen3 L17H6, Gemma L14H7) and broad carriers that read the passage further back. Qwen3's L17H6, the format head of the cross-study test X1, carries 0.363 of the excess on its own.
- **Readers.** Only MLPs after the direction layer read it: freezing them stops adding the direction from flipping the case (0.968 → 0.002 in Qwen3, 0.989 → 0.000 in Gemma), while freezing matched late heads leaves at least 0.966 and 0.985.
- **Why the edit changes words.** Adding the direction changes the word mostly where writing in capitals changes it anyway: at 0.694 of flips where the real upper-case passage also changes the word, against 0.120 where it keeps it (Qwen3; Gemma 0.549 against 0.080), and more often for words whose capitals form needs more tokens.
- **Other capitals.** Removing the direction leaves acronyms (1.000, 1.000) and initial capitals (0.968, 0.982) in place: they do not use it. Inside capitalised headings it takes capitals away (kept 0.669 and 0.768), and the blank line after the heading ends the capitals mode.
- In line with X1, Qwen3's head L0H3 attends to the current token (0.990 of its attention) and passes on its case, as one property of the current token. It does not carry the case of earlier words.

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
| W1 | Writers' writes along u moved: upper-case top-1 ≤ 0.30 | 0.029 [0.016, 0.044] | 0.009 [0.002, 0.019] | 0.021 / 0.000 |
| | … same word in any case ≥ 0.55 | 0.688 [0.651, 0.724] | 0.809 [0.773, 0.845] | 0.691 / 0.836 |
| | … every random set ≥ 0.85 (lowest) | 0.985 | 0.949 | 0.968 / 0.964 |
| R1 | Adding d flips to capitals ≥ 0.80 | 0.968 [0.952, 0.981] | 0.989 [0.979, 0.998] | 0.958 / 0.966 |
| | … with the readers frozen ≤ 0.20 | 0.002 [0.000, 0.006] | 0.000 [0.000, 0.000] | 0.000 / 0.000 |
| | … with random sets frozen ≥ unfrozen − 0.15 (lowest) | 0.966 | 0.985 | 0.937 / 0.948 |
| C1 | Carriers patched in A: excess projection removed ≥ 0.60 | 0.872 [0.860, 0.884] | 0.859 [0.845, 0.874] | 0.880 / 0.839 |
| | … excess D removed ≥ 0.30 | 0.424 [0.414, 0.434] | 0.576 [0.562, 0.588] | 0.459 / 0.584 |
| | … random sets ≤ named − 0.25 (highest) | 0.166 | 0.322 | 0.144 / 0.344 |
| C2 | Carriers patched in U: upper-case top-1 ≤ 0.60 | 0.309 [0.273, 0.345] | 0.255 [0.213, 0.298] | 0.351 / 0.291 |
| | … every random set ≥ 0.90 (lowest) | 0.996 | 0.982 | 0.989 / 1.000 |
| C3 | Qwen3 L17H6 alone in A: removed ≥ 0.20, above every random layer-17 head | 0.363 [0.346, 0.380]; random at most 0.002 | n/a | 0.359 |
| C4 | Local carrier: attention on the last eight words ≥ 0.60 | L17H6 0.929 [0.925, 0.933] | L14H7 0.918 [0.913, 0.923] | 0.929 / 0.914 |
| | Broad carrier: the same ≤ 0.50 | L19H0 0.284 [0.269, 0.299] | L24H6 0.206 [0.195, 0.218] | 0.282 / 0.193 |
| E1 | Edit changes the word where the upper passage does | 0.694 [0.646, 0.744] | 0.549 [0.480, 0.619] | 0.627 / 0.514 |
| | … where it does not (difference ≥ 0.30) | 0.120 [0.096, 0.145] | 0.080 [0.059, 0.101] | 0.105 / 0.058 |
| | Changed word in the lower passage's top 5 ≥ 0.40 | 0.538 [0.483, 0.593] | 0.619 [0.543, 0.695] | 0.683 / 0.520 |
| E2 | Change rate, capitals form needs ≥ 1 extra token | 0.541 [0.482, 0.595] | 0.424 [0.360, 0.486] | 0.432 / 0.406 |
| | … no extra token (difference ≥ 0.10) | 0.205 [0.173, 0.236] | 0.122 [0.097, 0.147] | 0.250 / 0.110 |
| A1 | Acronyms: removing d keeps all capitals ≥ 0.85 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.957 / 1.000 |
| | … removing r ≥ 0.90 | 0.991 [0.971, 1.000] | 0.974 [0.948, 0.994] | 1.000 / 1.000 |
| A2 | Initial capitals: removing d keeps them ≥ 0.90 | 0.968 [0.958, 0.977] | 0.982 [0.974, 0.990] | 0.979 / 1.000 |
| | … removing r ≥ 0.95 | 0.993 [0.988, 0.997] | 0.992 [0.987, 0.997] | 0.995 / 1.000 |
| A3 | Capitalised headings: removing d keeps all capitals ≤ 0.85 | 0.669 [0.609, 0.722] | 0.768 [0.717, 0.816] | 0.554 / 0.694 |
| | … removing r ≥ 0.95 | 0.994 [0.985, 1.000] | 1.000 [1.000, 1.000] | 1.000 / 1.000 |
| A4 | Excess along u inside the heading ≥ 0.40 of the gap | 0.728 [0.697, 0.757] | 0.753 [0.728, 0.776] | 0.713 / 0.749 |
| | … at the first word after it ≤ 0.25 | 0.034 [0.027, 0.041] | 0.116 [0.108, 0.124] | 0.038 / 0.123 |
| | … top-1 all capitals there ≤ 0.10 | 0.018 [0.004, 0.039] | 0.022 [0.006, 0.044] | 0.013 / 0.025 |

Positions: W1 551 (Qwen3) and 451 (Gemma); R1 557 and 454; C1 600 and 500; edits 1,058 and 715 flips; acronyms 115 and 155; initial capitals 1,313 and 1,135; heading words 344 and 375.

## 3. Verdicts

As fixed in the protocol:

| | Qwen3-1.7B-Base | Gemma-4-E2B |
|---|---|---|
| Q1, writers and readers | supported | supported |
| Q2, carriers of passage case | supported | supported |
| Q3, why the edit changes words | supported | supported |
| Q4, acronyms and initial capitals separate | supported | supported |
| Q4, headings use d, the line break ends it | supported | supported |
| Predictions passed | 12 of 12 | 11 of 11 |

**The parts account (writers, carriers, readers) is supported in both models.** Q5 (training dynamics) was not tested. OLMo-2-0425-1B, the checkpointed candidate, failed the frozen round-1 gate, and the charter's stop rule applied.

## 4. Descriptive detail (held out, not tested beyond the predictions)

- **The writers' edit goes past the lower-case level.** Moving only the writers' writes along u removes 1.214 (Qwen3) and 1.279 (Gemma) of the gap between U and L runs along u. These are more than the whole gap: the components between the writers and ℓ do not rebuild it, and the residual ends a little below the lower-case level.
- **Carriers in full capitals.** In U runs, the carriers' patch also changes the current token's context, but not its own case. It leaves an upper-case next word at 0.309 and 0.255. The rest of the capitals comes from the current token itself, which in Qwen3 enters through L0H3.
- **Attention by words back** (current word, 1, 2–3, 4–7, 8 or more, first text token, BOS), mean over held-out positions:
  - Qwen3 L17H6: 0.204, 0.281, 0.212, 0.105, 0.061, 0.138;
  - Qwen3 L19H0: 0.046, 0.037, 0.063, 0.097, 0.615, 0.141;
  - Qwen3 L18H1: 0.617 on eight or more words back, 0.243 on the first token;
  - Qwen3 L18H0: 0.467 and 0.439;
  - Gemma L14H7: 0.278, 0.160, 0.131, 0.060, 0.056, 0.113, 0.201;
  - Gemma L24H6: 0.018, 0.017, 0.035, 0.064, 0.516, 0.102, 0.248;
  - Gemma L14H1 is in between: 0.775 on the last eight words.
  - So passage case comes both from the last few words and from the passage as a whole. In running text the two agree. How the MLPs weigh them to give round 1's recency curve was not tested.
- **Word changes.** The edit changes the word at 0.299 of flips in Qwen3 and 0.194 in Gemma. Changes that the real upper-case passage does not make are 0.082 and 0.060 of all flips. That is an upper bound on what an impure direction could add. Random directions flipped none of these positions (0.000, 0.000).
- **Projection scale:** round-1 means along u are 47.76 (upper) and -541.02 (lower) in Qwen3, and 12.09 and -16.29 in Gemma.

## 5. What this does not show

- **These are sets, not single paths.** Writers and readers are tested as sets. Their members overlap with the carriers' downstream MLPs, and we did not trace each path between them. Gemma's random carrier sets share layer 14 with four named heads and removed up to 0.322, so layer-14 attention carries case more widely than the named heads alone.
- **Reader freezing is a strong edit.** Freezing late MLPs at their unedited outputs also keeps whatever else they would have written. It shows that the flip needs them, not that they do nothing else.
- **What the carriers read.** The carriers' attention patterns show where they look, not what they read there. Whether L19H0 reads the case of each earlier word or a running summary was not separated.
- **Scope.** Two models, encyclopaedic English, word starts and heading words only. "Acronym" means a word of two or more capitals in mixed-case text, and the test is limited to positions whose top-1 was already all capitals.
- **Held-out text.** The held-out material is held out from these analyses, not from the models' training data.

## 6. Records

`CHARTER-2.md`, `PROTOCOL-2.md` and `FROZEN.md`; `NOTEBOOK.md` (round-2 exploration, dead ends included); `outputs/r2/` (exploration and held-out records, `conf_verdicts2.json`, `explore_verdicts2_check.json`, `REPORT2-CHECK.md`); logs `outputs/r2_*.log`. Nothing is committed.
