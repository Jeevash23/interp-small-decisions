# X1 report: do the parts found in four discovery studies overlap inside one model?

<!-- nocheck -->**8 November 2026.** Exploration under `CHARTER.md` (frozen 09:45 IST) is in `NOTEBOOK.md` and is **exploratory**. The confirmation followed `PROTOCOL.md` (frozen 13:44 IST, digests in `FROZEN.md`). It ran **once** on held-out parts that were sealed before any model read X1 material, from 13:44 to {end_time} IST, with no crash and no re-run. As planned, each job ran in lock holds of at most about 18 minutes and resumed from its saved runs. Every prediction below is **prospective**. Section 5 quotes exploratory results for context and labels them so.

<!-- nocheck -->Tables and verdicts come from `outputs/conf/verdicts.json` (written by the frozen `code/c03_verdicts.py`) and from the raw run files `outputs/conf/c01_*.json` and `outputs/conf/c02_attn_qwen3-06b-bf16.json`. `code/build_report.py` fills this text from those files. `code/check_report.py`, written after the run, works independently of both: it recomputes every value and every pass/fail from the raw files and checks every number in this report.

## Short answer

**{short_answer}**

Four studies in this area found attention heads with a role in one behaviour each: line-breaking (D1-X), Python indentation (D3), ordinal suffixes (D4) and writing in capitals (D5). Switching each named group off at the decision positions of every task shows the following.

- **Line-breaking and indentation use separate parts, joined at one head.** D1-X's counters, newline writers and other readers leave indentation unchanged, and D3's block-end heads leave line-breaking unchanged. The one link is **L17H6**. It is a strong line-breaking reader on its own (width effect kept {a1}), and it is the only way D3's groups touch line-breaking: CARRY keeps {a4c} of the width effect and {a4cm} without L17H6; TERM keeps {a4t}, and {a4tm} without it. Its role in the dedent after `return` is small ({a3} in mean log-probability of the right indent on generated programs). {a3_sentence}
- **The early heads are general movers, not task parts.** D4's **L0H2** attends mainly to the previous token in prose, numbers and capitals text. It carries the tens digit for the teen check (teen effect kept {c1a} and {c1b}) and also matters for line-breaking (separation kept {c2}). **L0H3** attends to the current token in every task ({att_L0H3_LB_0} to {att_L0H3_IND_0} of its attention). It carries the last digit (last-digit pull kept {c4a} and {c4b}), is needed for line-breaking decisions (separation kept {c5}), and in Qwen3-1.7B it matters far more for choosing the next word than for keeping capitals (D_low kept {c7l}, D_up kept {c7u}). D5's finding that it reads a token's case is one use of reading the current token.
- **L11H9**, which sits in D3's CARRY and TERM groups, is a suffix head that attends to the current token. It holds part of the ordinal suffix choice (teen effect kept {d1a}, last-digit pull kept {d1c} at 4 digits) and leaves line-breaking ({d2w}) and indentation (largest change {d3_max}) alone.

{n_passed} of {n_tests} frozen tests pass. Failed: {failed}. Verdict on the whole account: **{account}**.

## 1. What was tested

- **Models.** Qwen3-0.6B-Base for line-breaking (LB), indentation (IND) and ordinals (ORD). Capitals (CAPS) used Qwen3-1.7B-Base, heads by index: Qwen3-0.6B failed condition 2 of D5's frozen capitals gate (same word 0.547 against the 0.60 bar; `outputs/gate_qwen3-06b.json`), and the charter fixed this fallback in advance. Both models were run in bfloat16.
- **Held-out material**, sealed before any model read X1 material:
  - LB and CAPS: a new batch of English Wikipedia articles created in 2025–26, none of which any earlier study read;
  - IND: Python from GitHub repositories created on or after 1 October 2026, none used by D3, and programs from D3's generator with seeds D3 never used;
  - ORD: D4's unread reserve frames and the held half of its reserve numbers.
  - D5's reserve part was not read.
- **Items.** LB: {n_width} width items and {n_nat} paragraphs. IND: unmarked and terminal dedents, {n_code_U} and {n_code_T} points in natural code, {n_gen_U} and {n_gen_T} in generated programs. ORD: {ord_n4} teen and {ord_n4} non-teen items at 4 digits, and {ord_n5} of each at 5 digits. CAPS: {n_caps_pos} word starts in {n_caps_par} paragraphs.
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

Unablated values on the held parts: LB {lb_base}; IND {ind_base}; ORD {ord_base}; CAPS {caps_base}.

## 2. Results by prediction

The predictions and exploration values below are as frozen in the protocol. Brackets after a held-out value give a descriptive 95% interval from 2,000 bootstrap resamples of items, paragraphs, repositories or programs, or numbers.

| Test | Prediction | Exploration | Held-out result |
|---|---|---|---|
{test_table}

Key held-out values with their random ranges (the minimum to the maximum of the 20 random groups):

- **A1/A2, L17H6, line-breaking:** width effect kept {a1} {ci_L17H6_LBw} (random {a1_lo} to {a1_hi}); separation kept {a2} {ci_L17H6_LBs} (random {a2_lo} to {a2_hi}).
- **A3, L17H6, generated terminal dedents:** change {a3} {ci_L17H6_gT} (random {a3_lo} to {a3_hi}).
- **A4, D3's groups on line-breaking:** CARRY {a4c}, CARRY−L17H6 {a4cm}; TERM {a4t}, TERM−L17H6 {a4tm}.
- **A5:** L17H6 on code unmarked dedents {a5cu}, on generated unmarked dedents {a5gu}; teen effect kept {a5t4} and {a5t5}.
- **A6, Qwen3-1.7B:** L17H6 D_up kept {a6} {ci_L17H6_Du} (random {a6_lo} to {a6_hi}), D_low kept {a6l}, word log-prob change {a6w}.
- **B1–B3, line-breaking groups.**
  - WRITE: width kept {b1}; largest indentation change {b1_ind} (generated unmarked {b1_indgu}).
  - COUNT: separation kept {b2}; largest indentation change {b2_ind} (generated unmarked {b2_indgu}).
  - READ−L17H6: width kept {b3}; largest indentation change {b3_ind} (generated unmarked {b3_indgu}).
- **B4, DED:** code unmarked dedents {b4} {ci_DED_cU} (random {b4_lo} to {b4_hi}); line-breaking width kept {b4w}, separation kept {b4s}.
- **C1–C3, L0H2.** Teen effect kept {c1a} {ci_L0H2_t4} and {c1b} {ci_L0H2_t5} (random from {c1a_lo} and {c1b_lo}). Separation kept {c2} {ci_L0H2_LBs} (random {c2_lo} to {c2_hi}). Largest indentation change {c3_ind} (generated unmarked {c3_indgu}).
- **C4–C7, L0H3.**
  - C4: last-digit pull kept {c4a} {ci_L0H3_p4} and {c4b} {ci_L0H3_p5} (random from {c4a_lo} and {c4b_lo}); teen effect kept {c4c} and {c4d}.
  - C5: separation kept {c5} {ci_L0H3_LBs} (random from {c5_lo}); width kept {c5w}.
  - C6: largest indentation change {c6_ind} (generated unmarked {c6_indgu}).
  - C7, Qwen3-1.7B: D_low kept {c7l} {ci_L0H3_Dl} (random from {c7l_lo}), word log-prob change {c7w} {ci_L0H3_wl} (random from {c7w_lo}), D_up kept {c7u}. The loss on D_low ({c7_ratio_l}) is well over three times the loss on D_up (three times is {c7_ratio_u3}).
- **C8, attention at decision positions (held parts, 0.6B).**
  - L0H3 on the current token: LB {att_L0H3_LB_0}, IND {att_L0H3_IND_0}, ORD {att_L0H3_ORD_0}, CAPS text {att_L0H3_CAPS_0}.
  - L0H2 on the previous token: LB {att_L0H2_LB_1}, IND {att_L0H2_IND_1}, ORD {att_L0H2_ORD_1}, CAPS text {att_L0H2_CAPS_1}.
- **D1–D4, L11H9.**
  - Teen effect kept {d1a} {ci_L11H9_t4} and {d1b} {ci_L11H9_t5}; last-digit pull kept {d1c} {ci_L11H9_p4} and {d1d} {ci_L11H9_p5} (random from {d1a_lo}, {d1b_lo}, {d1c_lo} and {d1d_lo}).
  - Line-breaking: width kept {d2w}, separation kept {d2s}.
  - Largest indentation change: {d3_max}.
  - Attention on the current token: LB {att_L11H9_LB_0}, IND {att_L11H9_IND_0}, ORD {att_L11H9_ORD_0}, CAPS text {att_L11H9_CAPS_0}.

## 3. Verdicts

As fixed in the protocol:

- **Lead 1, L17H6:** {lead1_L17H6}.
- **Lead 2, separate parts for line length and block ends:** {lead2_separate}.
- **Lead 3, early heads:** L0H2, {lead3_L0H2}; L0H3, {lead3_L0H3}.
- **Lead 4, L11H9:** {lead4_L11H9}.
- **Whole account:** {account} ({n_passed} of {n_tests} tests pass).

## 4. The full held-out matrices

Each cell gives the named group's value, with the range of its 20 random groups in brackets. Ratios are "kept"; indentation and the capitals word log-prob are "change".

**Line-breaking (Qwen3-0.6B)**

{M_LB}

**Indentation (Qwen3-0.6B)**

{M_IND}

**Ordinal suffixes (Qwen3-0.6B)**

{M_ORD}

**Capitals (Qwen3-1.7B, heads by index)**

{M_CAPS}

## 5. Reading, with exploratory context

{reading}

## 6. Limits

{limits}

## Records

`CHARTER.md`, `PROTOCOL.md` and `FROZEN.md` (frozen files and digests), `NOTEBOOK.md` (exploration, dead ends included), `outputs/` (exploration outputs and summaries), `outputs/conf/` (confirmation outputs and `verdicts.json`), `outputs/conf_smoke/` (pipeline checks on exploration material, not results). Nothing is committed.
