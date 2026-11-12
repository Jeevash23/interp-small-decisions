# D4 round 2: confirmation protocol

**Status: FROZEN 9 Nov 2026, 09:34 IST.** Read-only; digests in `FROZEN.md`. The freeze was later than the charter's 03:00 because of an overnight stall (`AMENDMENTS-2.md`, Amendment 1); no held-out material was read before it.

Study D4 (ordinal suffixes), round 2, under `CHARTER-2.md` (frozen 8 Nov 2026, 16:54 IST). The exploration is in `NOTEBOOK.md` (round-2 entries from 17:03) and is **exploratory**. This protocol fixes, before any model reads held-out material, what each candidate account predicts on that material, the controls, the analysis and the stop rules. The confirmation runs **once**. Every test below is **prospective**. Changes after the freeze are dated amendments in `AMENDMENTS-2.md`.

## 1. Candidate accounts (from exploration)

1. **Qwen3-0.6B-Base: the path from digits to suffix.**
   - L0H2 copies the tens digit into the number's last token.
   - MLP0 reads it there at once. The signal is then carried by many middle-layer parts and gathered at the input of **MLP24**.
   - MLP24 writes the teen override: for teens it raises "th" and lowers st, nd and rd about equally, whatever the last digit.
   - The last digit's own suffix is written separately, by MLP18 (and MLP20), for teens and non-teens alike. The two meet only in the logits.
   - L11H9, the current-token suffix head that X1 found, sits upstream of MLP24.
   - The teen decision becomes readable by the logit lens at layer 24.
2. **Qwen3-0.6B-Base: the backup route.**
   - When L0H2 and L1H6 are removed, the later previous-token movers L2H4 and L5H2 bring part of the tens digit across. They do this more at two digits than at three or four.
   - Part of the two-digit teen check remains unexplained.
3. **OLMo-2-1B: a weak route, weak at its source.**
   - The token account holds at new lengths: the teen check is strong where the tens digit is inside the last token (3, 5, 8, 9 digits) and weak where it sits in the previous token (4, 7, 10 digits).
   - In the weak case, the layer-0 heads L0H5, L0H0 and L0H9 carry what arrives, and MLP0 is its first reader.
   - Teen status is linearly readable at the last token early on, but the signal is a third to a half the size of the in-token one, and it fades with depth.
   - The late readers respond normally to a full-sized in-token teen direction.
4. **Other one-token-per-digit models** (section 4.4, filled in from E04).

## 2. Held-out material (sealed 8 Nov 17:01, `FROZEN.md`)

- **Frames:** the 24 held2 frames in `data2/frames2.jsonl` (rankings, positions, anniversaries and dates; 7, 7, 5 and 5 frames). No model has read them. The tests use the rankings, positions and anniversaries frames, as in round 1.
- **Numbers**, by the rules in `code/r2common.py`:
  - 2 digits: 11–99, all read before; held-out items differ by frame only.
  - 3–5 digits: round-1 confirmation- and reserve-class numbers, read before, in new frames. Reported as such.
  - 6–7 digits: round-1 reserve-class numbers, never read.
  - 8–10 digits: held2 numbers (SHA-256 of `D4R2-num-<N>` modulo 10 is 6–9), never read.
- **Items:** 96 teen (ending in 11–13) and 96 non-teen numbers ending in 1–3 per length, plain form. Frames are taken in turn, and numbers drawn with the seed `d4r2-conf-<task>-<digits>`.

## 3. Measures (as in round 1 and CHARTER-2.md)

- **Basic measures at the number's last token:**
  - m = logit(suffix of the last digit) − logit("th").
  - **Teen effect** = mean m (non-teen) − mean m (teen).
  - **Pull** = mean m (non-teen).
  - **AUROC** of m between non-teen and teen, ties counted as one half.
  - "Kept" = the value after an intervention ÷ the value before.
- **Mean ablation:** a head's output (before the output projection) at the last token is replaced by its mean over that length's items.
- **Path ablation:** a sender's ablation change at the last token is added to one receiver's input only. The receiver is one attention layer, one MLP, or the unembedding. The share is the teen loss through that route ÷ the sender's full teen loss.
- **Mediation:** the change a sender's full ablation causes at a receiver's input is given to that receiver alone, with everything else clean. The share is computed as above.
- **DLA:** a component's output at the last token is projected through the final norm, linearised at each item's own scale, onto the unembedding difference.
  - The teen effect's DLA is the non-teen mean minus the teen mean.
  - A component's **suffix profile** is its DLA onto st, nd, rd and th, centred over the four, averaged by class and last digit.
  - **Own-suffix specificity** = the last digit's own suffix minus the mean of the other two of st/nd/rd.
- **Logit lens:** each layer's last-token residual stream is put through the final norm and the unembedding.
- **Probe:** an L2 logistic regression (λ = 0.01, standardised features), 5-fold cross-validated, for teen against non-teen on the last-token residual stream.
- **Teen direction v_d at layer l:** the mean last-token residual of teens minus that of non-teens.
- **Steering:** v_3 (the 3-digit teen direction at layer 8) is added to the layer-8 block output at the last token of every non-teen item. The response is the change in mean m.

## 4. Predictions

Exploration values are given in brackets. Before the freeze the whole pipeline was run once on full-size exploration items ("dry run", `outputs/r2dry/`, 8 Nov 21:32–21:39); 31 of the 33 draft tests passed, and the two that did not (S2 at 4 digits, O6's 5-digit side) had their thresholds reset from that run (`NOTEBOOK.md`, 9 Nov 09:20). No threshold was set from held-out material.

### 4.1 Qwen3-0.6B-Base, the path (4 and 6 digits; account 1)

- **R1.** L0H2's teen loss through MLP0's input alone is ≥ 0.60 of its full teen loss, and no other single receiver (any attention layer, any other MLP) carries more than 0.15. Read only if L0H2's full teen loss is ≥ 0.30 logits. [3 and 4 digits: 0.89, 0.92; next 0.04, 0.05]
- **R2.** L0H2's teen loss through MLP0 is larger than that of every other layer-0 head through MLP0. These are the 15 other heads, each fully ablated and given to MLP0 alone. [not measured in exploration; a control]
- **R3.** Among MLP12–MLP27, MLP24 is the largest mediator of L0H2's teen loss and of L11H9's teen loss, each with a share ≥ 0.40. [L0H2: 0.62, 0.59 (next 0.43, 0.49); L11H9: 0.70, 0.64 (next 0.44, 0.52)]
- **R4.** The largest single-layer rise of the logit-lens teen effect is from layer 23 to layer 24, and it is ≥ 1.0 logit. [+4.41, +3.44; next +1.40]
- **R5.** MLP24's DLA is ≥ 0.50 of the teen effect, and every other MLP's |DLA| is ≤ 0.35 of it. [0.80, 0.79; next 0.12, 0.22]
- **R6.** Suffix profiles:
  - MLP24 for teens: "th" minus the mean of st/nd/rd is ≥ 1.5 at each last digit, and |own-suffix specificity| ≤ 0.6. [≥ 2.69; ≤ 0.48]
  - MLP18: own-suffix specificity ≥ 0.4 for both classes at each last digit, with the teen and non-teen means within 0.4 of each other. [≥ 0.81; 0.08 apart at 4 digits]

### 4.2 Qwen3-0.6B-Base, the backup route (2, 3, 4 digits; account 2)

A = L0H2+L1H6 and B = L2H4+L5H2, mean-ablated at the last token. The controls are 20 sets of A plus two random heads from layers 2–6 (not L2H4, L5H2).

- **S1.** At each length, A+B keeps less of the teen effect than every A + random pair. [0.44, 0.33, 0.29 against minima 0.67, 0.38, 0.40]
- **S2.** B alone keeps ≥ 0.85 at each length. [0.98, 0.96, 0.92; dry run 0.98, 0.92, 0.90]
- **S3.** At 2 digits, B removes ≥ 0.15 beyond A, more than at 3 and 4 digits, and A+B still keeps ≥ 0.30. [0.24 against 0.07, 0.12; 0.44]

### 4.3 OLMo-2-1B (3, 4, 5, 7, 8, 9, 10 digits; account 3)

T = L0H5+L0H0+L0H9. The controls are 20 random triples from layers 0–2, excluding T.

- **O3.** AUROC is ≥ 0.88 at 3, 5, 8 and 9 digits. At 4, 7 and 10 digits it is ≤ 0.82 and at least 0.10 below every in-token length. [0.991, 0.963, 0.947, 0.940; 0.735, 0.696, 0.671]
- **O4.**
  - At 4, 7 and 10 digits, T keeps ≤ 0.60 of the teen effect and less than every random triple. Each length is read only if its base teen effect is ≥ 0.30.
  - At 3, 5, 8 and 9 digits, T keeps ≥ 0.90.
  - [0.25, 0.27, 0.25; 0.98–1.00]
- **O5.**
  - At 4, 7 and 10 digits, the probe at layer 2 is ≥ 0.90 and the probe at layer 15 is at least 0.10 lower.
  - At 3, 5, 8 and 9 digits, the probe at layer 15 is ≥ 0.93.
  - [layer 2: 0.99 at all three cross-token lengths; layer 15: 0.77, 0.71, 0.65; in-token 0.96–0.98]
- **O6.** |v_d| / |v_3| is ≤ 0.65 at every layer for 4, 7 and 10 digits, and ≥ 0.65 at every layer for 5 digits, so one line separates the two. [≤ 0.52, ≤ 0.43; 0.80–1.19; dry run ≤ 0.54 and ≥ 0.72]
- **O7.** At 4, 7 and 10 digits, steering with v_3 lowers the non-teen mean m by ≥ 1.5 logits, more than each of 5 random directions of the same norm. Those random directions move it by at most 0.5 in absolute value. [at layer 8: −3.26 and −2.44 at 4 and 7 digits; random ≤ 0.37]
- **O8.** At 4, 7 and 10 digits, T's teen loss through MLP0's input is ≥ 0.50 and the largest of any receiver. Read only if T's loss is ≥ 0.30. [0.76, 0.72, 0.71]

### 4.4 Other one-token-per-digit models (account 4)

Models that passed round 1's behaviour gate and split numbers into one token per digit: Qwen3-1.7B-Base, SmolLM2-1.7B and Gemma-4-E2B. Head sets were chosen in exploration (E04) by attention alone, before any ablation: **PREV** = heads in layers 0–1 that put ≥ 0.5 of the last token's attention on the previous token; **CUR** = those that put ≥ 0.5 on the current token. They are fixed in `code/r2conf_sets.json`:

| Model | PREV | CUR |
|---|---|---|
| Qwen3-1.7B-Base | L0H2 + L1H6 | L0H3 + L0H7 |
| SmolLM2-1.7B | L0H6 + L0H8 | none |
| Gemma-4-E2B | L0H4 | none |

Each set is mean-ablated at the last token at 3 and 4 digits. The controls are 20 random sets of the same size from layers 0–1, excluding the named heads.

- **X-PREV** (each model, each length). The PREV set keeps ≤ 0.85 of the teen effect and less than every random set, and keeps ≥ 0.55 of the pull. [Qwen3-1.7B 0.41, 0.41 (pull 0.70, 0.63); SmolLM2 0.70, 0.62 (0.87, 0.79); Gemma-4 0.75, 0.62 (0.91, 0.84); random sets 0.99–1.03]
- **X-CUR** (Qwen3-1.7B-Base only, each length). The CUR set keeps ≤ 0.30 of the pull and less than every random set, and keeps ≥ 0.90 of the teen effect. [pull 0.01, −0.94; teen 1.17, 1.01]
- Reported, not tested: that SmolLM2 and Gemma-4 have no early current-token head by this rule.

## 5. Analysis and verdicts

- `code/r2conf_run.py` writes raw values to `outputs/r2conf/`, one task per job:
  - `qwen` and `qwen2` for Qwen3-0.6B-Base;
  - `olmo` for OLMo-2-1B;
  - `other <model>` for the models in 4.4.
- `code/r2conf_evaluate.py` computes every test and writes `outputs/r2conf/EVALUATION-2.json`.
- A test with a read rule that is not met is reported as "not read" and counts neither way.
- **Verdict per account:**
  - "supported" if every read test passes;
  - "partly supported" if at least half pass;
  - otherwise "not supported".
- Descriptive 95% intervals may be added in the report (bootstrap over items, 2,000 resamples). They do not decide any test.
- A tests-per-account summary and the full raw values go into `REPORT-2.md`. Every number in it is checked by an independent script written after the run.

## 6. Runs and stop rules

- `code/r2chain_conf.sh` runs each task once, in the order `qwen`, `qwen2`, `olmo`, then the 4.4 models.
- Each job runs under the shared GPU lock, in a memory cage, and within the 20-minute cap.
- A job that crashes is reported as crashed; it is not re-run with changes. If it crashed for a reason outside the code (for example the local machine restarting), it may be re-run unchanged once, and that is reported.
- If the queue means a job cannot start by 9 Nov 18:00 IST, it is reported as not run.

## 7. What does not change

- The frozen code, the items and seeds, and the thresholds above.
- No held2 frame or held-out number is read before this file is frozen and read-only. `code/r2common.py` enforces that.

## 8. Frozen code (digests in `FROZEN.md`)

`code/r2common.py` (already frozen), `code/r2lib.py`, `code/r2conf_run.py`, `code/r2conf_evaluate.py`, `code/r2chain_conf.sh`, `code/run.sh` (round 1, unchanged), and `code/r2conf_sets.json` (the 4.4 head sets).
