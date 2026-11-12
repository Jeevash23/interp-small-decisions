# D5 confirmation protocol: how small models write the next word in capitals

**Status: FROZEN 8 Nov 2026, 05:48 IST** (drafted from 04:50). This file and the scripts under "Frozen code" are read-only, with their SHA-256 in `FROZEN.md`. Later changes are dated amendments in `AMENDMENTS.md`, never edits.

Everything below is **prospective**. It was written after exploration (`NOTEBOOK.md`, 7–8 Nov) and before any model reads the `conf` part. The `conf` part is run once. The `reserve` part stays unread.

## 1. What exploration found (candidates, all exploratory)

Models: Gemma-4-E2B (base, many whole-word upper-case tokens) and Qwen3-1.7B-Base (upper-case words split into pieces), the two that passed the frozen gate. SmolLM2-1.7B and Pythia-1B failed the gate's same-word condition and are not tested here.

1. **Latent lower case, late capitals (route R2).** In upper-case passages the logit lens shows the coming word's *lower-case* first piece (Lc) above its actual upper-case piece (U) through the middle layers, while a lower-case piece of *another* word (Lr) is not above U from mid depth on. The word is therefore represented in a lower-case-like form, not lower case in general. U overtakes Lc only in the last three or four layers (Gemma from layer 32 of 35, Qwen3 from layer 25 of 28). Word identity (Lc above Lr) is as strong in upper-case passages as in lower-case ones at the same depth, which goes against upper case being handled as a separate "language" (route R4).
2. **One "write in capitals" direction (route R1, in a specific form).** The difference of mean residuals between upper-case and lower-case passages at word-start positions, taken after one mid-to-late layer, acts as a switch:
   - adding it at the prediction position of a lower-case passage turns the next word into capitals and mostly keeps the word;
   - subtracting it from an upper-case passage does the reverse;
   - setting the component along it to the midpoint leaves the case undecided and mostly keeps the word;
   - adding it inside ordinary mixed-case prose turns the next word into capitals.
   Random directions of the same norm do nothing. Removing capitals with it works earlier in depth than adding them. It does not reliably remove the initial capital of names and sentence starts (small sample), so it is a capitals-mode signal, not a capital-letter signal.
3. **The case comes from the last few words.** Turning only the last two words of a lower-case passage into capitals makes most next words upper case, and the last eight almost all; the reverse holds for lower-case words at the end of an upper-case passage.
4. **A token-level case reader in Qwen3.** Head L0H3 attends almost only to the current token and writes its case into the residual: its value from *another* paragraph in lower case moves the next word towards lower case, while its value from another upper-case paragraph does nothing. Gemma has no such layer-0 head (attention 0 moves little there); case enters around attention 14.
5. **Changed words are lower-case runner-ups.** Where the upper-case passage gives a different word from the lower-case one, the new word's lower-case first piece is usually among the lower-case passage's own top five next tokens (median rank 3), while a random word from the same paragraph almost never is. Words that change between the original and the lower-case passage behave the same way, so this shows how any change of case shifts the choice; it is consistent with latent lower case and not with a separate upper-case vocabulary.

Not carried forward, with reasons: component-level patching (exp03, exp05) used donors that differ in the word as well as the case, so its "same word" rates are not clean case edits; it is reported in the notebook only.

## 2. Material

- **Confirmation part only:** `data/part-conf.jsonl` (597 paragraphs from 332 articles, held out by article hash in the frozen charter), shuffled with seed 0 by `load_part`, cut to 1,200 characters as in exploration. Each script states the paragraph range it reads; all ranges start at the first shuffled paragraph.
- The case directions and their random controls are **not** re-estimated: they are read from the exploration means `outputs/exp01_gemma_means.pt` and `outputs/exp01_qwen3-17b_means.pt` (explore paragraphs 150–300), digested in `FROZEN.md`. The random directions are those of exploration (seed 0, norm matched per layer).
- Word starts, versions (upper, lower, original), the token-case rule and the "same word" rule are those of the frozen charter and of the exploration scripts.
- Fixed layers, chosen on exploration: the **direction layer** ℓ = 24 for Gemma and 22 for Qwen3; the **early layer** 16 for both.

## 3. Runs (frozen code)

Each run is a confirmation copy of an exploration script: it is identical except that it reads `conf` and writes `conf_*` outputs. Each runs once under the local machine rules (`code/run.sh`: GPU lock, memory cage, 20-minute limit). If a run fails for a technical reason (out of memory, time limit, lock), it may be rerun unchanged, or with fewer paragraphs taken from the start of the same range, recorded as a dated amendment; no prediction is read before all runs are complete.

| Run | Script and arguments (both models unless stated) | Material |
|---|---|---|
| Gate replication | `conf_gate.py <m> 150` | conf 0–150 |
| Lens | `conf_exp01_lens.py <m> 0 150` | conf 0–150 |
| Direction | `conf_exp02_direction.py <m> 0 100 16,ℓ` | conf 0–100, up to 4 word starts each (seed 2) |
| Recency | `conf_exp04_switch.py <m> 0 100` | conf 0–100, up to 4 word starts each (seed 3) |
| Removal and prose | `conf_exp07_ablate.py <m> ℓ 0 100` | conf 0–100, up to 4 word starts each (seed 4) |
| Changed words | `conf_exp08_errors.py <m>` (after the gate run) | the gate run's word starts |
| L0H3 (Qwen3 only) | `conf_exp06_head.py qwen3-17b 0 3 0 50 0:2,0:0` | conf 0–50, up to 4 word starts each (seed 2) |
| Verdicts | `conf_verdict.py` (no model) | the outputs above |

## 4. Predictions

Each prediction passes or fails as written, separately for each model, and all are reported whichever way they go. Thresholds come from exploration with margin and are not tuned on confirmation data. `conf_verdict.py` recomputes every measure from the per-position records. **Decisions use point estimates**; 95% intervals from 2,000 bootstrap resamples of paragraphs are reported beside them. Exploration values (from `outputs/explore_verdicts_check.json`, the same script run on the exploration outputs) are given in brackets as Gemma / Qwen3.

### 4.1 Latent lower case (lens)

At word starts in upper-case passages, logit lens with each model's final norm and output layer. U = the true upper-case first piece; Lc = the first piece of the same word in lower case; Lr = Lc of another word in the paragraph (list rolled by one). Comparisons with U leave out positions where U and Lc have equal lens values at that layer, as in exploration (about 2% of Gemma's positions).

- **L1 (late capitals).** At layer ℓ, share(Lc > U) **≥ 0.60** and share(Lr > U) **≤ 0.30**. At the last layer, share(Lc > U) **≤ 0.05**. And the first layer from which share(Lc > U) stays below 0.5 to the end is **in the last five layers** (Gemma ≥ 30, Qwen3 ≥ 23). [0.705, 0.164, 0.001, layer 32 / 0.748, 0.130, 0.002, layer 25]
- **L2 (same word identity in both cases).** At layer ℓ, share(Lc > Lr) in upper-case passages and in lower-case passages are each **≥ 0.75**, and differ by **≤ 0.05**. [0.852 vs 0.853 / 0.871 vs 0.885]

### 4.2 The capitals direction

d = exploration mean residual after the layer, upper minus lower, at word-start prediction positions; r = random direction of the same norm. One edit at the prediction position only, a = 1.

- **D1 (switch, both ways, word kept).** At layer ℓ, at word starts where the unedited top-1 is lower case in a lower-case passage: adding d makes the top-1 upper case in **≥ 0.80** of positions and the same word in upper case in **≥ 0.45**. Where it is upper case in an upper-case passage, subtracting d makes it lower case in **≥ 0.80** and the same word in lower case in **≥ 0.45**. Adding or subtracting r flips **≤ 0.05** in both. [add 0.961 / 0.775, subtract 0.955 / 0.756, random 0 / add 0.957 / 0.600, subtract 0.971 / 0.657, random 0]
- **D2 (needed).** At layer ℓ, in upper-case passages with an upper-case top-1, setting the component along d to the midpoint of the two exploration means leaves an upper-case top-1 in **≤ 0.70** of positions, while the top-1 is still the same word in any case in **≥ 0.60**. The same operation along r leaves **≥ 0.95** upper case. [0.524, 0.846, random 1.000 / 0.465, 0.725, random 1.000]
- **D3 (works in ordinary prose).** At layer ℓ, in the *original* mixed-case passage, at word starts with a lower-case top-1: adding d gives an upper-case top-1 in **≥ 0.75** and the same word in capitals in **≥ 0.50**; adding r gives an upper-case top-1 in **≤ 0.05**. [0.952, 0.823, 0 / 0.959, 0.639, 0]
- **D4 (capitals are removed earlier than they can be added).** At layer 16 with a = 1: adding d to lower-case passages flips **≤ 0.15**, while subtracting d from upper-case passages flips **≥ 0.40**. [0.022 vs 0.756 / 0.000 vs 0.500]

### 4.3 Where the case comes from

- **R1 (the last few words).** Share of upper-case top-1 among cased top-1 tokens at word starts (word 12 or later): lower-case passage with its last 2 words in capitals **≥ 0.50**, last 8 words **≥ 0.85**; upper-case passage with its last 2 words in lower case **≤ 0.25**, last 8 words **≤ 0.10**. [0.689, 0.933, 0.078, 0.014 / 0.687, 0.933, 0.115, 0.036]

### 4.4 Qwen3's token-level case reader

- **H1 (L0H3 reads the token's own case).** From the prediction position, L0H3 puts **≥ 0.90** of its attention on the current token in both upper- and lower-case passages. In upper-case passages with an upper-case top-1, replacing L0H3's value with its value at a word start of *another paragraph in lower case* moves **≥ 0.20** of the gap D = logp(U) − logp(Lc) towards the lower-case passage's own D and flips **≥ 0.25** of top-1s to lower case; its value from *another paragraph in upper case* moves **≤ 0.08** and flips **≤ 0.05**; heads L0H2 and L0H0, patched from the same word's lower-case passage, each move **≤ 0.08**. [0.977 and 0.972; 0.344, 0.543; 0.030, 0.000; 0.021, −0.001]

### 4.5 Changed words

At the gate replication's word starts (four per paragraph, conf 0–150), among positions where the greedy upper-case word is alphabetic and differs from the greedy lower-case word in capitals, and where the two words' lower-case first pieces differ: the rank of the upper-case word's lower-case first piece (" " + word in lower case, first token) in the lower-case passage's next-token distribution at the same word start. Control: another word of the same paragraph, 20 or more words away (seed 5).

- **E1.** The changed word is in the lower-case passage's top 5 at **≥ 0.50** of these positions, and the control word at **≤ 0.10**. [0.761, 0.014 / 0.694, 0.022]

### 4.6 Gate replication

- **G1.** Each model passes all three frozen gate conditions on conf paragraphs 0–150. [both pass]

## 5. Verdicts

For each model, the account in Section 1 (points 1 and 2: the word is chosen in a lower-case-like form and turned into capitals late, under one capitals direction) is:
- **supported** if L1, L2, D1, D2 and D3 all pass;
- **partly supported** if D1 and D3 pass but L1, L2 or D2 fails;
- **not supported** otherwise.

D4, R1, H1, E1 and G1 are secondary: each is reported as passing or failing on its own and does not change the verdict above. A failure is described, not explained away. The report states whether the two models agree.

## 6. What would count against the account

- Lc no longer above U at mid depth on new paragraphs (L1), or the gap closing only because lower-case pieces in general rank above U (Lr > U), would undercut "latent lower case".
- A direction that flips the case but changes the word (D1, D3 same-word rates below threshold) would make it a case-and-content edit rather than a case switch.
- Removal leaving the capitals in place (D2) would mean the direction is sufficient but not used.

## Frozen code

`code/d5common.py`, `code/run.sh`, `code/conf_gate.py`, `code/conf_exp01_lens.py`, `code/conf_exp02_direction.py`, `code/conf_exp04_switch.py`, `code/conf_exp06_head.py`, `code/conf_exp07_ablate.py`, `code/conf_exp08_errors.py`, `code/conf_verdict.py`; the exploration means `outputs/exp01_gemma_means.pt` and `outputs/exp01_qwen3-17b_means.pt`; and D1-X's loader `../D1-line-breaking/xmodel/code/xm3_common.py` and `xm_common.py` (read only, not changed by D5).
