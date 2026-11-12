# D5 round 2 confirmation protocol: the parts that write, carry and read the capitals direction

**Status: FROZEN 8 Nov 2026** (time and digests in `FROZEN.md`). This file and the files under "Frozen code" are read-only. Later changes are dated amendments in `AMENDMENTS-2.md`, never edits.

Everything below is **prospective**. It was written after the round-2 exploration (`NOTEBOOK.md`, 8 Nov, 17:02–20:25, under `CHARTER-2.md`) and before any model or script reads the held-out material: the `reserve` paragraphs (528, from 299 articles) and the heading items from hash-2 articles. Each run below is made once.

## 1. What exploration found (candidates, all exploratory)

Models: Gemma-4-E2B and Qwen3-1.7B-Base. Direction d and its random control r, the direction layer ℓ (Gemma 24, Qwen3 22), the definitions of U, Lc, upper and lower runs and the same-word rule are round 1's. A is the upper-case passage with its last word in lower case (context only; same current token as the lower-case run L). u is the unit vector of d.

1. **Writers.** Direct writes along u, upper minus lower runs, come almost entirely from a few late components: in Qwen3 MLPs 19–22 and head L19H0; in Gemma MLPs 22–24 and head L24H6. Moving only their writes along u to the lower-run values takes the upper-case top-1 away almost everywhere, and the word mostly survives in lower case. Random matched sets do almost nothing.
2. **Carriers.** With the current token fixed (A against L), the excess along u is brought in by a few heads: Qwen3 L19H0, L17H6, L18H1 and L18H0; Gemma L14H6, L14H7, L14H1, L15H2 and L24H6. Patching them from L removes most of A's excess. In each model one carrier looks at the last few words (Qwen3 L17H6, Gemma L14H7) and another mostly further back (Qwen3 L19H0, Gemma L24H6). Qwen3's L17H6, which X1 found to be a format reader, carries passage case on its own (about a third of A's excess). The current token's own case enters separately (Qwen3: L0H3, a current-token head, as X1 showed).
3. **Readers.** After ℓ, only MLPs react to d: freezing Qwen3's MLPs 23–27, or Gemma's MLPs 25–29 and 32–34, at their unedited values stops adding d from flipping anything, while freezing matched late heads leaves the flips. No late head reads d.
4. **Word changes under the edit.** Adding d changes the word mostly where writing in capitals changes it anyway (the real upper-case passage chooses a different word too), and more often for words whose capitals form needs more tokens. The changed word is usually one of the lower-case runner-ups.
5. **Other capitals.** Removing d leaves acronyms and initial capitals (names, sentence starts) in place, so they are not written with d. Capitalised headings are: removing d takes capitals away inside the heading. The capitals mode stops at the blank line after the heading: the first word of the next paragraph sits near the lower-case level along u, unlike running text, where a last word in lower case keeps about half of the excess.

Not carried forward: training checkpoints (Q5). OLMo-2-0425-1B failed the frozen gate (same word 0.510), and Q5 stopped as the charter's stop rule says.

## 2. Material

- **Held out:** `data/part-reserve.jsonl` (528 paragraphs, shuffled with seed 0 by `load_part2`, cut to 1,200 characters) and `data/headings-reserve.jsonl` (1,335 items, shuffled with seed 0 by `load_headings`), both digested at the charter freeze and refused by `code/d5common2.py` until this file is read-only.
- Directions and random controls are **not** re-estimated: round 1's exploration means (`outputs/exp01_<model>_means.pt`, digested in round 1).
- **Named sets** (fixed now, in `code/r2_sets_qwen.json`, `code/r2_sets_gemma.json` and `code/r2_sets_qwen_l17h6.json`):

| | Qwen3-1.7B-Base | Gemma-4-E2B |
|---|---|---|
| Writers | MLP19, MLP20, MLP21, MLP22, L19H0 | MLP22, MLP23, MLP24, L24H6 |
| Carriers | L19H0, L17H6, L18H1, L18H0 | L14H6, L14H7, L14H1, L15H2, L24H6 |
| Readers | MLP23, MLP24, MLP25, MLP26, MLP27 | MLP25, MLP26, MLP27, MLP28, MLP29, MLP32, MLP33, MLP34 |
| Local and broad carrier (C4) | L17H6; L19H0 | L14H7; L24H6 |

- **Random sets:** twenty per named set, matched member by member (`code/r2_exp03_sets.py`, `matched()`, seed `r2-sets`): a head by another head of the same layer; a writer MLP by another MLP; a reader MLP by a head of the same layer (all MLPs after ℓ may be named, and earlier ones cannot react to d).

## 3. Runs (frozen code)

`code/r2_conf.sh` runs everything, in six holds of the shared GPU lock (`code/run2.sh`: RAM gate, memory cage, at most 20 minutes per hold).

| Run | Script and arguments | Held-out material |
|---|---|---|
| Sets, Qwen3 | `r2_exp03_sets.py qwen3-17b reserve 0 300 r2_sets_qwen.json 2 13 20 conf` | reserve 0–300, 2 word starts each |
| L17H6 alone | `r2_exp03_sets.py qwen3-17b reserve 0 300 r2_sets_qwen_l17h6.json 2 13 20 l17h6` | the same word starts |
| Attention, Qwen3 | `r2_exp06_attn.py qwen3-17b reserve 0 200 L19H0,L17H6,L18H1,L18H0,L0H3` | reserve 0–200, 3 each |
| Edits, Qwen3 | `r2_exp05_edit.py qwen3-17b reserve 0 300 4 15` | reserve 0–300, 4 each |
| Other capitals, Qwen3 | `r2_exp04_other.py qwen3-17b reserve 0 264 0 250` and `… reserve 264 264 250 250` | all 528 paragraphs; heading items 0–500 |
| Sets, Gemma | `r2_exp03_sets.py gemma reserve 0 250 r2_sets_gemma.json 2 13 20 conf` | reserve 0–250, 2 each |
| Attention, Gemma | `r2_exp06_attn.py gemma reserve 0 200 L24H6,L14H6,L14H7,L14H1,L15H2` | reserve 0–200, 3 each |
| Edits, Gemma | `r2_exp05_edit.py gemma reserve 0 200 4 15` | reserve 0–200, 4 each |
| Other capitals, Gemma | `r2_exp04_other.py gemma reserve 0 264 0 250` and `… reserve 264 264 250 250` | all 528 paragraphs; heading items 0–500 |
| Verdicts | `r2_verdict.py conf` (no model) | the outputs above |

If a run fails for a technical reason (out of memory, the hold limit, the lock), it may be rerun unchanged, or on fewer items from the start of the same range, as a dated amendment. No prediction is read before all runs are complete.

## 4. Predictions

Each prediction passes or fails as written, separately for each model, and all are reported whichever way they go. Thresholds come from exploration with a margin and are not tuned on held-out data. `code/r2_verdict.py` recomputes every measure from the per-position records. **Decisions use point estimates**; 95% intervals from 2,000 bootstrap resamples of articles are reported beside them. Exploration values (`outputs/r2/explore_verdicts2_check.json`, the same script on the exploration outputs) are in brackets as Qwen3 / Gemma.

### 4.1 Q1: writers and readers

- **W1 (writers needed).** In upper-case passages with an upper-case top-1, moving only the named writers' writes along u to their values in the paired lower-case run leaves an upper-case top-1 at **≤ 0.30** of positions, with the same word in any case at **≥ 0.55**; each of the twenty random sets leaves **≥ 0.85**. [0.021, 0.691, random ≥ 0.968 / 0.000, 0.836, random ≥ 0.964]
- **R1 (readers needed).** In lower-case passages with a lower-case top-1, adding d after ℓ flips the top-1 to upper case at **≥ 0.80**; with the named readers frozen at their unedited outputs, at **≤ 0.20**; with each random set frozen, at no less than the unfrozen rate minus **0.15**. [0.958, 0.000, random ≥ 0.937 / 0.966, 0.000, random ≥ 0.948]

### 4.2 Q2: carriers of passage case

- **C1 (context carriers).** In A runs, patching the named carriers' outputs from L removes **≥ 0.60** of A's excess projection along u (A minus L, ratio of means), and **≥ 0.30** of its excess D = logp(U) − logp(Lc); every random set removes at most the named value minus **0.25** of the projection. [0.880, 0.459, random ≤ 0.144 / 0.839, 0.584, random ≤ 0.344]
- **C2 (they matter in full capitals too).** In upper-case passages with an upper-case top-1, the same patch leaves an upper-case top-1 at **≤ 0.60**; each random set leaves **≥ 0.90**. [0.351, random ≥ 0.989 / 0.291, random ≥ 1.000]
- **C3 (Qwen3 only: L17H6 alone).** In A runs, L17H6 alone removes **≥ 0.20** of the excess projection, more than every one of twenty random layer-17 heads. [0.359, random ≤ 0.003]
- **C4 (one local and one broad carrier).** Of each head's attention from the prediction position in upper-case passages, leaving out the first text token and BOS, the share on the current word and the seven words before it is **≥ 0.60** for the local carrier and **≤ 0.50** for the broad one. [L17H6 0.929, L19H0 0.282 / L14H7 0.914, L24H6 0.193]

### 4.3 Q3: why the edit changes the word

At lower-case positions where adding d at ℓ gives an upper-case top-1:

- **E1.** The edit changes the word (same-word rule) more often where the real upper-case passage changes it too: the difference between the two change rates is **≥ 0.30**. Among changed words, the edited word's lower-case form is in the lower-case passage's top 5 at **≥ 0.40**. [0.627 − 0.105, 0.683 / 0.514 − 0.058, 0.520]
- **E2.** Words whose capitals form takes at least one more token than their lower-case form change at least **0.10** more often than words that take no extra token. [0.432 vs 0.250 / 0.406 vs 0.110]

### 4.4 Q4: other capitals

Removal is round 1's (component along d set to the midpoint of the round-1 means), at ℓ; r is the random control.

- **A1 (acronyms are not written with d).** At acronym word starts whose top-1 is all capitals (two or more letters), removing d keeps it all capitals at **≥ 0.85**; removing r at **≥ 0.90**. [0.957 / 1.000 (n 9)]
- **A2 (initial capitals are not written with d).** At initial-capital word starts whose top-1 has an initial capital, removing d keeps it at **≥ 0.90**; removing r at **≥ 0.95**. [0.979 / 1.000]
- **A3 (capitalised headings are written with d).** Inside a heading written in capitals, at word starts from its second word whose top-1 is all capitals, removing d keeps it all capitals at **≤ 0.85**; removing r at **≥ 0.95**. [0.554 / 0.694]
- **A4 (the line break ends the capitals mode).** The projection along u, capitalised heading minus the same item with its heading in its original case, paired by word, as a share of the gap between the round-1 upper and lower means: inside the heading **≥ 0.40**; at the first word after the heading **≤ 0.25**; and the top-1 at that first word is all capitals at **≤ 0.10**. [0.713, 0.038, 0.013 / 0.749, 0.123, 0.025]

## 5. Verdicts

For each model:
- **Q1** (the direction is written by the named writers and read by the named late MLPs): supported if W1 and R1 pass; partly supported if one passes.
- **Q2** (passage case is carried by the named heads): supported if C1 and C2 pass; partly supported if one passes. C3 and C4 are reported on their own.
- **Q3** (word changes follow how the model writes capitals, mostly where capitals cost more tokens): supported if E1 and E2 pass; partly if one passes.
- **Q4**: "acronyms and initial capitals are separate" if A1 and A2 pass; "capitalised headings use d and the line break ends it" if A3 and A4 pass.
- **Overall:** the parts account (writers, carriers, readers) is supported if Q1 and Q2 are supported in both models.

A failure is described, not explained away. The report states where the two models agree.

## 6. What would count against the account

- Removing the named writers' writes along u leaving most capitals in place (W1), or freezing the late MLPs leaving the flips (R1): d would be written or read elsewhere as well.
- Random sets doing as much as the named ones (W1, C1, C2, R1): the named parts would not be special.
- The edit changing words where the real upper-case passage keeps them (E1): the direction would carry content.
- Removing d taking away acronyms or initial capitals (A1, A2), or leaving capitalised headings untouched (A3).

## Frozen code

`code/d5common2.py`, `code/run2.sh`, `code/r2_conf.sh`, `code/r2_exp03_sets.py`, `code/r2_exp04_other.py`, `code/r2_exp05_edit.py`, `code/r2_exp06_attn.py`, `code/r2_verdict.py`, `code/r2_sets_qwen.json`, `code/r2_sets_gemma.json`, `code/r2_sets_qwen_l17h6.json`; with round 1's frozen `code/d5common.py`, the exploration means and D1-X's loaders (digests in round 1's section of `FROZEN.md`).
