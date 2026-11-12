# X1 confirmation protocol: where the parts of four discovery studies overlap

Written 8 November 2026, early afternoon IST, after the exploration in `NOTEBOOK.md` (8 Nov, 09:46 onwards) under `CHARTER.md` (frozen 09:45 IST). It is frozen (digests in `FROZEN.md`, file made read-only) before any script reads a sealed held part, and it is run **once**. Everything below is **prospective and confirmatory**; anything else computed on the held parts is labelled post hoc.

## The account to test (from exploration)

Head groups and single heads named by D1-X (line-breaking), D3 (indentation), D4 (ordinal suffixes) and D5 (capitals) were switched off one at a time, at each task's decision positions, in Qwen3-0.6B-Base (capitals in Qwen3-1.7B-Base, heads by index, because 0.6B failed D5's capitals gate). Exploration suggested:

1. **One shared head between line-breaking and indentation, used unevenly.** L17H6, one of D1-X's readers and a member of D3's CARRY and TERM groups, is a strong line-breaking reader and a weak part of the terminal-statement dedent route. It is the only place where D3's groups touch line-breaking. In Qwen3-1.7B it also trims the capitals gap slightly without changing word choice.
2. **Otherwise separate parts.** D1-X's counters, newline writers and other readers do nothing to indentation, and D3's block-end heads (DED) do nothing to line-breaking. The newline writers do not carry what D3's dedent decision needs (lead 2 of the charter: not supported).
3. **The early heads are general movers, not task parts.** D4's L0H2 attends to the previous token and is used by ordinals (the tens digit) and, more weakly, by line-breaking. L0H3 attends to the current token in every task: it carries the last digit for ordinals, is needed for line-breaking decisions, and in Qwen3-1.7B is needed far more for choosing the word than for keeping capitals, so its case role (D5) is one use of reading the current token.
4. **L11H9 is a suffix head.** It is in D3's CARRY and TERM groups and in D4's notebook as a downstream suffix head. It attends to the current token, matters for the ordinal suffix (teen and non-teen alike), and does not matter for line-breaking or (beyond a trace) indentation.

## Models, material and measures

- **Qwen3-0.6B-Base** (bfloat16, SDPA) for line-breaking (LB), indentation (IND) and ordinals (ORD); **Qwen3-1.7B-Base** (bfloat16) for capitals (CAPS). Weights on disk.
- **Held parts, sealed before any model read X1 material** (digests in `FROZEN.md`; never read before this run):
  - prose: `data/prose/part-held.jsonl` (314 paragraphs, 171 new Wikipedia articles of 2025–26 that no earlier study read);
  - natural code: `data/code/part-held.jsonl` (863 chunks, 66 repositories created on or after 1 Oct 2026, none used by D3);
  - generated programs: `data/gen/part-held.jsonl` (2,059 programs from D3's generator, seeds D3 never used);
  - ordinals: `data/ord/part-held.json` (D4's unread reserve frames and the held half of D4's reserve numbers).
- **Items used (fixed in `code/c01_confirm.py`).** LB: 150 width items and 60 natural paragraphs, wrapped at width 60. IND: the first 400 natural-code chunks and the first 400 generated programs. ORD: 4- and 5-digit numbers, up to 96 (number, frame) items per class. CAPS: the first 100 paragraphs.
- **Measures (`code/x1common.py`, as in exploration).**
  - LB: *width effect* = mean change in log P(newline) at the decision point when the previous line is shortened by at least 10 characters (D1-X `width_items`). *Separation* = mean newline log-odds at break points minus at non-break points within 12 characters of the margin (D1-X test G).
  - IND: mean log q(true indent) of D3's renormalised candidate distribution at the decision token, by class: unmarked dedent (`dedent`) and terminal-statement dedent (`term`), in natural code and generated programs.
  - ORD: m = logit(last-digit suffix) − logit("th") at the number's last token. *Teen effect* = mean m (non-teen ending in 1–3) − mean m (teen, x11–x13). *Last-digit pull* = mean m (non-teen ending in 1–3).
  - CAPS: at word starts present in both versions of a paragraph, D_up = log p(U) − log p(Lc) in the upper-case version and D_low = log p(Lc) − log p(U) in the lower-case version (U and Lc are the word's first pieces in each case); word log-prob = log p(Lc) in the lower-case version.
- **Kept** = ablated value ÷ unablated value. **Change** = ablated − unablated (indentation log q; capitals word log-prob).

## Intervention and controls

- **Mean ablation at the decision positions** (scope `dec`): a head's output (input of its attention output projection) is replaced by its mean, at the task's decision positions only. These are, for LB, newline-marker tokens and decision tokens; for IND, the decision token; for ORD, the number's last token; for CAPS, the positions that predict a word start. The means are the **exploration means** saved during exploration (`outputs/means_<model>_<task>_dec.pt`, digests in `FROZEN.md`), not re-estimated.
- **Random controls**: 20 draws per named group. Each head of the group is replaced by a head of the same layer, drawn uniformly from heads not named in any charter group (`x1common.random_sets`, seed `x1-c01-<group>`). "Below every random draw" means below the minimum over all 20 draws.
- **Attention (C8, D4)**: on the first 40 items of each task's held part, the share of each head's attention from the decision positions to the current token (offset 0) and to the previous token (offset 1) (`code/c02_attn.py`, a copy of the exploratory `e02_attn.py` that reads the held parts).

## Predictions (each could fail)

Exploration values (scope dec, exploration parts) are in brackets. "Indentation quiet" means |change| ≤ 0.03 on code unmarked, code terminal and generated terminal dedents, and ≤ 0.10 on generated unmarked dedents. The looser bound on generated unmarked dedents is there because random groups moved it by up to 0.20 in exploration.

**A. L17H6, the shared head**

| | Prediction | Exploration |
|---|---|---|
| A1 | L17H6 alone: LB width effect kept ≤ 0.85 and below every random draw | 0.695; random ≥ 0.997 |
| A2 | L17H6 alone: LB separation kept ≤ 0.95 and below every random draw | 0.853; random ≥ 0.993 |
| A3 | L17H6 alone: generated terminal-dedent change ≤ −0.010 and below every random draw | −0.026; random ≥ −0.013 |
| A4 | CARRY−L17H6 and TERM−L17H6: LB width kept ≥ 0.92; CARRY and TERM with L17H6: ≤ 0.85 | 1.051, 0.991; 0.732, 0.685 |
| A5 | L17H6 alone: \|code unmarked change\| ≤ 0.03, \|generated unmarked change\| ≤ 0.10, and teen effect kept within [0.90, 1.10] at 4 and 5 digits | −0.002, 0.012; 0.997, 1.011 |
| A6 | Qwen3-1.7B, L17H6 alone: D_up kept ≤ 0.97 and below every random draw; \|word log-prob change\| ≤ 0.03 | 0.925 (random ≥ 0.996); −0.004 |

**B. Separate parts for line length and block ends**

| | Prediction | Exploration |
|---|---|---|
| B1 | WRITE: LB width kept ≤ 0.70; indentation quiet | 0.446; largest \|change\| 0.002 |
| B2 | COUNT: LB separation kept ≤ 0.50; indentation quiet | 0.056; largest 0.006 |
| B3 | READ−L17H6: LB width kept ≤ 0.75; indentation quiet | 0.546; largest 0.008 |
| B4 | DED: code unmarked change ≤ −0.40 and below every random draw; LB width and separation kept within [0.95, 1.05] | −1.033; 1.004, 0.995 |

**C. Early heads**

| | Prediction | Exploration |
|---|---|---|
| C1 | L0H2: teen effect kept ≤ 0.75 at 4 and at 5 digits, each below every random draw | 0.444, 0.592; random ≥ 0.998 |
| C2 | L0H2: LB separation kept ≤ 0.93 and below every random draw (L0H2 is not numbers-only) | 0.816; random ≥ 0.972 |
| C3 | L0H2: indentation quiet | largest 0.003 |
| C4 | L0H3: last-digit pull kept ≤ 0.50 at 4 and at 5 digits, each below every random draw; teen effect kept ≥ 0.90 at both | 0.244, 0.124; 1.009, 1.142 |
| C5 | L0H3: LB separation kept ≤ 0.50 and below every random draw | 0.022; random ≥ 0.964 |
| C6 | L0H3: indentation quiet | largest 0.008 |
| C7 | Qwen3-1.7B, L0H3: D_low kept ≤ 0.60 and word log-prob change ≤ −0.50, each below every random draw; and (1 − D_low kept) ≥ 3 × (1 − D_up kept) | 0.345, −1.767; 0.655 against 3 × 0.081 |
| C8 | Attention at decision positions, held parts: L0H3 on the current token ≥ 0.90 in all four tasks; L0H2 on the previous token ≥ 0.50 in LB, ORD and CAPS | L0H3 0.97–0.99; L0H2 0.59, 0.91, 0.70 |

**D. L11H9**

| | Prediction | Exploration |
|---|---|---|
| D1 | L11H9: teen effect kept ≤ 0.85 (4 digits) and ≤ 0.90 (5 digits); last-digit pull kept ≤ 0.85 at both; each below every random draw | 0.662, 0.810; 0.712, 0.665 |
| D2 | L11H9: LB width and separation kept ≥ 0.93 | 1.045, 1.025 |
| D3 | L11H9: \|change\| ≤ 0.05 on all four indentation measures | largest 0.023 |
| D4 | Attention at decision positions, held parts: L11H9 on the current token ≥ 0.85 in all four tasks | 0.90–0.96 |

The attention tests C8 and D4 use the 0.6B model, with the same items for every head.

## Verdicts (applied by `code/c03_verdicts.py`)

Each test is reported pass or fail. If a test cannot be evaluated (a missing output or a class with no points), it is reported as not evaluable, not as a pass.

- **Lead 1 (L17H6).** "Shared part confirmed" if A1, A3 and A4 pass. "Line-breaking reader only" if A1 and A4 pass but A3 fails. Otherwise "not confirmed". A2, A5 and A6 qualify the reading.
- **Lead 2 (separate parts).** Confirmed if B1–B4 all pass.
- **Lead 3 (early heads).** L0H2 as a general previous-token mover is confirmed if C1, C2 and C3 pass. L0H3 as a current-token reader that is neither task- nor case-specific is confirmed if C4, C5 and C7 pass. C6 and C8 qualify.
- **Lead 4 (L11H9).** Confirmed as a suffix head not shared with line-breaking or indentation if D1, D2 and D3 pass. D4 qualifies.
- **Whole account.** Supported if the ten core tests A1, A4, B1, B4, C1, C2, C4, C5, D1 and D2 all pass. Partly supported if some fail, with each part reported. A null (for example, L17H6 no longer standing out) is stated plainly.

Decisions use point estimates. For description only, 95% intervals come from 2,000 bootstrap resamples (numpy seed 0) of width items and paragraphs (LB), repositories and programs (IND), numbers (ORD), and paragraphs (CAPS).

## What is not tested here

- The all-position ablations and the general-damage column (GEN) of exploration. They are reported as exploratory only.
- Lead 5 of the charter (content first, format late). It is a hypothesis, and this protocol does not test it.
- Why L17H6 serves several format decisions, and what L0H3's output encodes beyond token identity.

## Run plan

1. Freeze this file, the scripts and the exploration means (digests in `FROZEN.md`), and make them read-only.
2. Run `c01_confirm.py` for (0.6B, LB), (0.6B, IND), (0.6B, ORD) and (1.7B, CAPS), and `c02_attn.py` for 0.6B. Each runs under the shared GPU lock in holds of at most about 18 minutes, resuming from its saved runs.
3. Run `c03_verdicts.py`. Then write `REPORT.md`, with every number generated from the saved outputs and checked by an independent script.
4. If a script crashes, fix it only for the crash and resume it, and log the fix in `REPORT.md`. Nothing is excluded after the run.

## Scripts and files frozen with this protocol

`code/x1common.py`, `code/c01_confirm.py`, `code/c02_attn.py`, `code/c03_verdicts.py`, `code/run.sh`, `code/queue.sh`, and `outputs/means_qwen3-06b-bf16_{LB,IND,ORD}_dec.pt` and `outputs/means_qwen3-17b_CAPS_dec.pt`. Smoke runs on exploration material (`outputs/conf_smoke/`) are pipeline checks, not results.
