# X1 report: do the parts found in four discovery studies overlap inside one model?

<!-- nocheck -->**8 November 2026.** Exploration under `CHARTER.md` (frozen 09:45 IST) is in `NOTEBOOK.md` and is **exploratory**. The confirmation followed `PROTOCOL.md` (frozen 13:44 IST, digests in `FROZEN.md`). It ran **once** on held-out parts that were sealed before any model read X1 material, from 13:44 to 16:04 IST, with no crash and no re-run. As planned, each job ran in lock holds of at most about 18 minutes and resumed from its saved runs. Every prediction below is **prospective**. Section 5 quotes exploratory results for context and labels them so.

<!-- nocheck -->Tables and verdicts come from `outputs/conf/verdicts.json` (written by the frozen `code/c03_verdicts.py`) and from the raw run files `outputs/conf/c01_*.json` and `outputs/conf/c02_attn_qwen3-06b-bf16.json`. `code/build_report.py` fills this text from those files. `code/check_report.py`, written after the run, works independently of both: it recomputes every value and every pass/fail from the raw files and checks every number in this report.

## Short answer

**Mostly separate parts. Line-breaking and indentation meet at a single head, L17H6, and the early heads that the studies named are general movers of the previous or the current token, not parts of one task. Every frozen prediction held on the held-out material.**

Four studies in this area found attention heads with a role in one behaviour each: line-breaking (D1-X), Python indentation (D3), ordinal suffixes (D4) and writing in capitals (D5). Switching each named group off at the decision positions of every task shows the following.

- **Line-breaking and indentation use separate parts, joined at one head.** D1-X's counters, newline writers and other readers leave indentation unchanged, and D3's block-end heads leave line-breaking unchanged. The one link is **L17H6**. It is a strong line-breaking reader on its own (width effect kept 0.706), and it is the only way D3's groups touch line-breaking: CARRY keeps 0.715 of the width effect and 1.015 without L17H6; TERM keeps 0.675, and 0.966 without it. Its role in the dedent after `return` is small (−0.022 in mean log-probability of the right indent on generated programs). That change is small, but it lies below all 20 random heads of layer 17 (lowest −0.011), as predicted.
- **The early heads are general movers, not task parts.** D4's **L0H2** attends mainly to the previous token in prose, numbers and capitals text. It carries the tens digit for the teen check (teen effect kept 0.507 and 0.546) and also matters for line-breaking (separation kept 0.871). **L0H3** attends to the current token in every task (0.97 to 0.99 of its attention). It carries the last digit (last-digit pull kept 0.155 and 0.314), is needed for line-breaking decisions (separation kept 0.000), and in Qwen3-1.7B it matters far more for choosing the next word than for keeping capitals (D_low kept 0.331, D_up kept 0.923). D5's finding that it reads a token's case is one use of reading the current token.
- **L11H9**, which sits in D3's CARRY and TERM groups, is a suffix head that attends to the current token. It holds part of the ordinal suffix choice (teen effect kept 0.726, last-digit pull kept 0.733 at 4 digits) and leaves line-breaking (1.023) and indentation (largest change 0.030) alone.

22 of 22 frozen tests pass. Failed: none. Verdict on the whole account: **supported**.

## 1. What was tested

- **Models.** Qwen3-0.6B-Base for line-breaking (LB), indentation (IND) and ordinals (ORD). Capitals (CAPS) used Qwen3-1.7B-Base, heads by index: Qwen3-0.6B failed condition 2 of D5's frozen capitals gate (same word 0.547 against the 0.60 bar; `outputs/gate_qwen3-06b.json`), and the charter fixed this fallback in advance. Both models were run in bfloat16.
- **Held-out material**, sealed before any model read X1 material:
  - LB and CAPS: a new batch of English Wikipedia articles created in 2025–26, none of which any earlier study read;
  - IND: Python from GitHub repositories created on or after 1 October 2026, none used by D3, and programs from D3's generator with seeds D3 never used;
  - ORD: D4's unread reserve frames and the held half of its reserve numbers.
  - D5's reserve part was not read.
- **Items.** LB: 150 width items and 60 paragraphs. IND: unmarked and terminal dedents, 504 and 532 points in natural code, 1587 and 1371 in generated programs. ORD: 96 teen and 96 non-teen items at 4 digits, and 96 of each at 5 digits. CAPS: 11489 word starts in 100 paragraphs.
- **Intervention.** Mean ablation of a head's output at the task's decision positions only, using means saved during exploration:
  - LB: newline-marker tokens and decision tokens;
  - IND: the decision token;
  - ORD: the number's last token;
  - CAPS: the positions that predict a word start.
- **Controls.** Each named group is compared with 20 random groups, each replacing every head with a head of the same layer that no study named.
- **Measures**, each as defined by its source study:
  - LB: the width effect (how much a shorter previous line raises log P(newline)) and the separation between break and non-break points;
  - IND: the change in mean log-probability of the right indent;
  - ORD: the teen effect and the last-digit pull, from m = logit(last-digit suffix) − logit("th");
  - CAPS: D_up = log p(upper-case piece) − log p(lower-case piece) when predicting the next word in an upper-case passage, D_low the reverse in a lower-case passage, and the lower-case word's log-probability.
- **Reading the numbers.** "Kept" is the ablated value divided by the unablated one; "change" is ablated minus unablated.

Unablated values on the held parts: LB width 1.383, separation 3.284; IND code unmarked −0.352, code terminal −0.065, generated unmarked −1.871, generated terminal −0.549; ORD teen, 4 digits 3.012, teen, 5 digits 2.624, pull, 4 digits 1.886, pull, 5 digits 1.647; CAPS D_up 8.063, D_low 12.499, word log-prob −3.259.

## 2. Results by prediction

The predictions and exploration values below are as frozen in the protocol. Brackets after a held-out value give a descriptive 95% interval from 2,000 bootstrap resamples of items, paragraphs, repositories or programs, or numbers.

| Test | Prediction | Exploration | Held-out result |
|---|---|---|---|
| A1 | L17H6 alone: LB width effect kept ≤ 0.85 and below every random draw | 0.695; random ≥ 0.997 | pass |
| A2 | L17H6 alone: LB separation kept ≤ 0.95 and below every random draw | 0.853; random ≥ 0.993 | pass |
| A3 | L17H6 alone: generated terminal-dedent change ≤ −0.010 and below every random draw | −0.026; random ≥ −0.013 | pass |
| A4 | CARRY−L17H6 and TERM−L17H6: LB width kept ≥ 0.92; CARRY and TERM with L17H6: ≤ 0.85 | 1.051, 0.991; 0.732, 0.685 | pass |
| A5 | L17H6 alone: \|code unmarked change\| ≤ 0.03, \|generated unmarked change\| ≤ 0.10, and teen effect kept within [0.90, 1.10] at 4 and 5 digits | −0.002, 0.012; 0.997, 1.011 | pass |
| A6 | Qwen3-1.7B, L17H6 alone: D_up kept ≤ 0.97 and below every random draw; \|word log-prob change\| ≤ 0.03 | 0.925 (random ≥ 0.996); −0.004 | pass |
| B1 | WRITE: LB width kept ≤ 0.70; indentation quiet | 0.446; largest \|change\| 0.002 | pass |
| B2 | COUNT: LB separation kept ≤ 0.50; indentation quiet | 0.056; largest 0.006 | pass |
| B3 | READ−L17H6: LB width kept ≤ 0.75; indentation quiet | 0.546; largest 0.008 | pass |
| B4 | DED: code unmarked change ≤ −0.40 and below every random draw; LB width and separation kept within [0.95, 1.05] | −1.033; 1.004, 0.995 | pass |
| C1 | L0H2: teen effect kept ≤ 0.75 at 4 and at 5 digits, each below every random draw | 0.444, 0.592; random ≥ 0.998 | pass |
| C2 | L0H2: LB separation kept ≤ 0.93 and below every random draw (L0H2 is not numbers-only) | 0.816; random ≥ 0.972 | pass |
| C3 | L0H2: indentation quiet | largest 0.003 | pass |
| C4 | L0H3: last-digit pull kept ≤ 0.50 at 4 and at 5 digits, each below every random draw; teen effect kept ≥ 0.90 at both | 0.244, 0.124; 1.009, 1.142 | pass |
| C5 | L0H3: LB separation kept ≤ 0.50 and below every random draw | 0.022; random ≥ 0.964 | pass |
| C6 | L0H3: indentation quiet | largest 0.008 | pass |
| C7 | Qwen3-1.7B, L0H3: D_low kept ≤ 0.60 and word log-prob change ≤ −0.50, each below every random draw; and (1 − D_low kept) ≥ 3 × (1 − D_up kept) | 0.345, −1.767; 0.655 against 3 × 0.081 | pass |
| C8 | Attention at decision positions, held parts: L0H3 on the current token ≥ 0.90 in all four tasks; L0H2 on the previous token ≥ 0.50 in LB, ORD and CAPS | L0H3 0.97–0.99; L0H2 0.59, 0.91, 0.70 | pass |
| D4 | Attention at decision positions, held parts: L11H9 on the current token ≥ 0.85 in all four tasks | 0.90–0.96 | pass |
| D1 | L11H9: teen effect kept ≤ 0.85 (4 digits) and ≤ 0.90 (5 digits); last-digit pull kept ≤ 0.85 at both; each below every random draw | 0.662, 0.810; 0.712, 0.665 | pass |
| D2 | L11H9: LB width and separation kept ≥ 0.93 | 1.045, 1.025 | pass |
| D3 | L11H9: \|change\| ≤ 0.05 on all four indentation measures | largest 0.023 | pass |

Key held-out values with their random ranges (the minimum to the maximum of the 20 random groups):

- **A1/A2, L17H6, line-breaking:** width effect kept 0.706 [0.685, 0.726] (random 0.988 to 1.036); separation kept 0.836 [0.824, 0.849] (random 0.994 to 1.010).
- **A3, L17H6, generated terminal dedents:** change −0.022 [−0.027, −0.019] (random −0.011 to 0.002).
- **A4, D3's groups on line-breaking:** CARRY 0.715, CARRY−L17H6 1.015; TERM 0.675, TERM−L17H6 0.966.
- **A5:** L17H6 on code unmarked dedents −0.010, on generated unmarked dedents 0.021; teen effect kept 0.990 and 1.012.
- **A6, Qwen3-1.7B:** L17H6 D_up kept 0.927 [0.924, 0.931] (random 0.995 to 1.017), D_low kept 0.957, word log-prob change −0.002.
- **B1–B3, line-breaking groups.**
  - WRITE: width kept 0.466; largest indentation change 0.001 (generated unmarked 0.001).
  - COUNT: separation kept 0.040; largest indentation change 0.007 (generated unmarked 0.000).
  - READ−L17H6: width kept 0.568; largest indentation change 0.003 (generated unmarked 0.007).
- **B4, DED:** code unmarked dedents −1.112 [−1.251, −0.968] (random −0.027 to 0.014); line-breaking width kept 0.990, separation kept 0.999.
- **C1–C3, L0H2.** Teen effect kept 0.507 [0.404, 0.582] and 0.546 [0.420, 0.634] (random from 0.988 and 0.973). Separation kept 0.871 [0.833, 0.907] (random 0.959 to 1.018). Largest indentation change 0.008 (generated unmarked 0.003).
- **C4–C7, L0H3.**
  - C4: last-digit pull kept 0.155 [−0.119, 0.341] and 0.314 [0.034, 0.500] (random from 0.980 and 0.988); teen effect kept 1.077 and 1.132.
  - C5: separation kept 0.000 [−0.037, 0.037] (random from 0.966); width kept 0.391.
  - C6: largest indentation change 0.009 (generated unmarked 0.010).
  - C7, Qwen3-1.7B: D_low kept 0.331 [0.315, 0.347] (random from 0.994), word log-prob change −1.816 [−1.894, −1.744] (random from −0.027), D_up kept 0.923. The loss on D_low (0.669) is well over three times the loss on D_up (three times is 0.231).
- **C8, attention at decision positions (held parts, 0.6B).**
  - L0H3 on the current token: LB 0.97, IND 0.99, ORD 0.98, CAPS text 0.98.
  - L0H2 on the previous token: LB 0.60, IND 0.32, ORD 0.90, CAPS text 0.70.
- **D1–D4, L11H9.**
  - Teen effect kept 0.726 [0.664, 0.777] and 0.738 [0.659, 0.794]; last-digit pull kept 0.733 [0.651, 0.797] and 0.643 [0.512, 0.726] (random from 0.988, 0.973, 0.981 and 0.973).
  - Line-breaking: width kept 1.023, separation kept 1.027.
  - Largest indentation change: 0.030.
  - Attention on the current token: LB 0.90, IND 0.96, ORD 0.91, CAPS text 0.94.

## 3. Verdicts

As fixed in the protocol:

- **Lead 1, L17H6:** shared part confirmed.
- **Lead 2, separate parts for line length and block ends:** confirmed.
- **Lead 3, early heads:** L0H2, general previous-token mover confirmed; L0H3, current-token reader, not task- or case-specific, confirmed.
- **Lead 4, L11H9:** suffix head confirmed, not shared with line-breaking or indentation.
- **Whole account:** supported (22 of 22 tests pass).

## 4. The full held-out matrices

Each cell gives the named group's value, with the range of its 20 random groups in brackets. Ratios are "kept"; indentation and the capitals word log-prob are "change".

**Line-breaking (Qwen3-0.6B)**

| Group | width kept | separation kept |
|---|---|---|
| L17H6 | 0.706 (0.988 to 1.036) | 0.836 (0.994 to 1.010) |
| CARRY | 0.715 (0.937 to 1.068) | 0.864 (0.966 to 1.037) |
| CARRY-L17H6 | 1.015 (0.924 to 1.065) | 1.033 (0.960 to 1.014) |
| TERM | 0.675 (0.958 to 1.061) | 0.831 (0.968 to 1.021) |
| TERM-L17H6 | 0.966 (0.975 to 1.028) | 0.990 (0.978 to 1.025) |
| READ-L17H6 | 0.568 (0.966 to 1.008) | 0.512 (0.990 to 1.008) |
| WRITE | 0.466 (0.914 to 1.058) | 0.254 (0.973 to 1.037) |
| COUNT | 0.150 (0.892 to 1.071) | 0.040 (0.888 to 1.028) |
| DED | 0.990 (0.898 to 1.028) | 0.999 (0.909 to 1.013) |
| L0H2 | 0.973 (0.909 to 1.066) | 0.871 (0.959 to 1.018) |
| L0H3 | 0.391 (0.969 to 1.044) | 0.000 (0.966 to 1.015) |
| L11H9 | 1.023 (0.985 to 1.006) | 1.027 (0.993 to 1.001) |

**Indentation (Qwen3-0.6B)**

| Group | code unmarked | code terminal | generated unmarked | generated terminal |
|---|---|---|---|---|
| L17H6 | −0.010 (−0.011 to 0.008) | −0.003 (−0.003 to 0.000) | 0.021 (−0.028 to 0.015) | −0.022 (−0.011 to 0.002) |
| READ-L17H6 | −0.003 (−0.029 to 0.012) | −0.001 (−0.003 to 0.002) | −0.007 (−0.117 to 0.073) | −0.003 (−0.046 to 0.015) |
| WRITE | 0.001 (−0.003 to 0.001) | 0.001 (−0.001 to 0.001) | 0.001 (−0.005 to 0.025) | −0.001 (−0.012 to 0.005) |
| COUNT | 0.007 (−0.007 to 0.011) | −0.001 (−0.001 to 0.001) | 0.000 (−0.021 to 0.063) | −0.003 (−0.009 to 0.012) |
| DED | −1.112 (−0.027 to 0.014) | −0.058 (−0.004 to 0.001) | −0.584 (−0.117 to 0.053) | −0.266 (−0.021 to 0.007) |
| L0H2 | −0.008 (−0.059 to 0.002) | −0.001 (−0.014 to −0.000) | −0.003 (−0.055 to 0.111) | 0.000 (−0.043 to 0.001) |
| L0H3 | −0.009 (−0.059 to 0.001) | −0.004 (−0.003 to −0.000) | 0.010 (−0.055 to 0.002) | −0.007 (−0.024 to 0.001) |
| L11H9 | −0.009 (−0.010 to 0.003) | −0.003 (−0.002 to 0.000) | −0.001 (−0.014 to 0.019) | −0.030 (−0.007 to 0.002) |

**Ordinal suffixes (Qwen3-0.6B)**

| Group | teen kept, 4 digits | teen kept, 5 digits | pull kept, 4 digits | pull kept, 5 digits |
|---|---|---|---|---|
| L17H6 | 0.990 (0.973 to 1.014) | 1.012 (0.954 to 1.019) | 1.002 (0.904 to 1.032) | 1.010 (0.899 to 1.025) |
| CARRY | 0.714 (0.912 to 1.069) | 0.757 (0.879 to 1.086) | 0.742 (0.814 to 1.163) | 0.675 (0.832 to 1.126) |
| CARRY-L17H6 | 0.723 (0.953 to 1.076) | 0.742 (0.962 to 1.077) | 0.736 (0.945 to 1.144) | 0.653 (0.945 to 1.118) |
| TERM | 0.704 (0.842 to 1.091) | 0.785 (0.803 to 1.075) | 0.708 (0.754 to 1.127) | 0.662 (0.847 to 1.136) |
| TERM-L17H6 | 0.714 (0.930 to 1.046) | 0.771 (0.887 to 1.146) | 0.697 (0.865 to 1.120) | 0.631 (0.894 to 1.140) |
| READ | 1.047 (0.966 to 1.038) | 1.021 (0.954 to 1.064) | 1.113 (0.901 to 1.096) | 1.055 (0.880 to 1.108) |
| READ-L17H6 | 1.065 (0.984 to 1.040) | 1.015 (0.979 to 1.106) | 1.111 (0.948 to 1.066) | 1.040 (0.968 to 1.064) |
| WRITE | 0.999 (0.915 to 1.034) | 0.999 (0.957 to 1.032) | 0.997 (0.951 to 1.021) | 0.994 (0.992 to 1.039) |
| COUNT | 1.013 (0.929 to 1.045) | 1.009 (0.916 to 1.030) | 1.008 (0.902 to 1.133) | 1.000 (0.932 to 1.028) |
| DED | 0.976 (0.957 to 1.066) | 1.005 (0.948 to 1.081) | 0.932 (0.879 to 1.140) | 0.977 (0.884 to 1.111) |
| L0H2 | 0.507 (0.988 to 1.033) | 0.546 (0.973 to 1.002) | 0.823 (0.895 to 0.994) | 0.901 (0.784 to 1.000) |
| L0H3 | 1.077 (0.988 to 1.003) | 1.132 (0.992 to 1.005) | 0.155 (0.980 to 1.000) | 0.314 (0.988 to 1.001) |
| L11H9 | 0.726 (0.988 to 1.029) | 0.738 (0.973 to 1.021) | 0.733 (0.981 to 1.046) | 0.643 (0.973 to 1.013) |

**Capitals (Qwen3-1.7B, heads by index)**

| Group | D_up kept | D_low kept | word log-prob change |
|---|---|---|---|
| L0H3 | 0.923 (0.997 to 1.003) | 0.331 (0.994 to 1.000) | −1.816 (−0.027 to 0.001) |
| L0H2 | 1.004 (0.996 to 1.007) | 0.970 (0.992 to 0.999) | −0.038 (−0.027 to −0.000) |
| L11H9 | 0.947 (0.998 to 1.008) | 0.936 (0.996 to 1.007) | −0.033 (−0.007 to 0.002) |
| L17H6 | 0.927 (0.995 to 1.017) | 0.957 (0.996 to 1.019) | −0.002 (−0.014 to 0.000) |
| READ | 0.931 (0.993 to 1.017) | 0.923 (0.950 to 1.021) | −0.013 (−0.039 to −0.000) |
| CARRY | 0.860 (0.993 to 1.018) | 0.880 (0.984 to 1.015) | −0.062 (−0.073 to −0.010) |
| COUNT | 1.009 (0.989 to 1.014) | 1.009 (0.989 to 1.016) | −0.000 (−0.019 to −0.003) |
| DED | 0.996 (0.966 to 1.025) | 0.999 (0.977 to 1.020) | −0.026 (−0.083 to −0.009) |

## 5. Reading, with exploratory context

- **What "separate" means here.** D1-X's groups without L17H6 (COUNT, WRITE, READ−L17H6) each remove much of a line-breaking measure, and on no other task tested do they do more harm than the worst of their random groups. READ−L17H6 even raises the ordinal measures slightly above its random range. DED, likewise, harms only indentation. For COUNT, the held-out separation kept is 0.040, while the largest indentation change is 0.007. In exploration, an all-position ablation added a general-damage control: the mean next-token loss on ordinary prose. Removing COUNT everywhere removed most of the width effect (kept 0.08) yet raised that loss by only 0.007 nats (random groups 0.009 to 0.015). For L17H6 the figures were 0.66 and 0.006 (random 0.002 to 0.010) (exploratory). These are specialised parts.
- **L17H6 as a format reader.** On held-out text L17H6 does three things. It carries the stored line length to the line-break decision (width kept 0.706). It adds a little to the dedent after a terminal statement (−0.022). In Qwen3-1.7B, by index, it trims the case gap (0.927) without changing which word is chosen (−0.002). Each role seems to move a fact about the recent text's format, not its content, to where the format is decided. This fits the charter's lead 5 (content first, format late), but X1 did not test that lead. So L17H6's place in both D1-X's readers and D3's CARRY group is not incidental: the head has a measurable role in both tasks. Its share of the indentation job is small, though. In exploration, CARRY's change on generated terminal dedents was −0.31 with L17H6 and −0.23 without it (exploratory; CARRY was not part of the held-out indentation run).
- **Early heads.** L0H3 attends to the current token in every task. L0H2 attends mainly to the previous token (0.60 to 0.90 in prose, numbers and capitals text). At the newline token in code it puts 0.48 on itself and 0.32 on the previous token. What a task takes from them is decided downstream. Ordinals need the tens digit moved one place (L0H2) and the last digit read where it stands (L0H3); line-breaking needs both. At the newline token that ends a line of code, neither matters to the indent decision. Exploration with all-position ablation (not tested here) found that L0H2 does matter for indentation upstream: removing it everywhere moved generated terminal dedents by −0.36 (random −0.16 to 0.00). Removing it everywhere also raised general loss by 0.16 nats, and L0H3 by 3.56 nats, so neither is a task part.
- **L11H9** attends to the current token. Ablating it shrinks the teen effect and the last-digit pull by similar amounts (0.726 and 0.733 kept at 4 digits). That fits a head that serves the suffix choice in general, not the teen check, as D4's notebook described it. Its place in D3's CARRY and TERM groups adds little to indentation (0.030 at most). In exploration, removing it everywhere raised general loss by 0.048 nats against at most 0.005 for random layer-11 heads. That suggests a broader next-token role, which was not tested here.
- **Lead 2 (the newline as a shared scratchpad) is answered no**, at least for these parts. D1-X's newline writers carry the stored length that line-breaking reads, but D3's indent decision at the same newline token does not use them (largest change 0.001). Indentation reads the previous line through its own heads.

## 6. Limits

- One model family. Line-breaking, indentation and ordinals were tested in Qwen3-0.6B-Base; capitals in Qwen3-1.7B-Base, with heads matched by index (D1-X showed the two models share line-breaking heads by index, but X1 assumed the match for the capitals heads).
- Mean ablation at the decision positions only, with exploration means. Effects that pass through earlier positions (such as L0H2 on the terminal line, seen only in exploration) are outside the confirmatory tests.
- "Separate" means within the range of 20 layer-matched random groups on these measures. Smaller shared effects, or sharing through MLPs and the residual stream, are not excluded.
- Measures are the source studies' own, on new material of the same kinds: Wikipedia prose wrapped at width 60, Python from new repositories and generated programs, English ordinal frames up to five digits, and capitals in Wikipedia paragraphs.
- The held-out material is held out from these analyses, not necessarily from the models' training data.

## Records

`CHARTER.md`, `PROTOCOL.md` and `FROZEN.md` (frozen files and digests), `NOTEBOOK.md` (exploration, dead ends included), `outputs/` (exploration outputs and summaries), `outputs/conf/` (confirmation outputs and `verdicts.json`), `outputs/conf_smoke/` (pipeline checks on exploration material, not results). Nothing is committed.
