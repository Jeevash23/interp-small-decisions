# D3 round 2 report: how the two dedent routes get their information, and a second model

9 November 2026. **Confirmation** under `PROTOCOL-2.md` (frozen 09:55 IST), run once on the sealed round-2 parts. The candidates came from the round-2 exploration in `NOTEBOOK.md` under `CHARTER-2.md` (frozen 8 Nov, 16:54 IST) and `AMENDMENTS-2.md`. Every number below is filled from saved outputs by `code/r2_build_report.py`; `code/r2_check_report.py` checks them against values recomputed independently from the raw records.

## Result

**25 of 25 frozen predictions pass.**

- Question 1 (what the block-end head reads): **confirmed** (Q1, Q2, Q3).
- Question 2 (how far to dedent): **confirmed** (Q5, Q6, Q8).
- Question 3 (terminal route: keyword read directly, spread, not via L0H2 on the line): **confirmed** (Q9, Q10, Q11).
- Question 4a (two routes in OLMo-2): **confirmed** (G1, G2, O9).
- Question 4b (L13H1 works like L23H2): **confirmed** (O1, O2, O3, O6, O8).

## Qwen3-0.6B-Base, head L23H2

Natural code: 2,712 chunks from 225 repositories; 3,934 unmarked dedents, 18,570 same-level lines and 3,504 terminal dedents with at least one open header. Generated: 1,510 targeted programs (782 ending in a terminal statement; 176 `return`).

| id | measure | result | verdict |
|---|---|---|---|
| Q1 | attention to innermost open header's `:\n`, unmarked dedents; same-level lines | 0.458 [0.439, 0.476]; 0.118 | pass |
| Q2 | knock-out of that token, unmarked dedents; every other head of the layer | −0.148 [−0.162, −0.132]; −0.004 to +0.044 | pass |
| Q3 | same knock-out, same-level lines; terminal dedents | +0.013; −0.005 | pass |
| Q4 | closed-header knock-out against open-header knock-out (same points) | −0.001 against −0.122 | pass |
| Q5 | picking where target ≠ innermost (innermost most attended); all multi-header dedents against innermost rule | 0.893 [0.873, 0.913] (0.060); 0.914 against 0.822 | pass |
| Q6 | steering, natural unmarked dedents: rise (control); becomes top (control) | +2.72 (+0.08); 0.125 (0.004) | pass |
| Q7 | knock-out of the true-level header, unmarked dedents | −0.210 | pass |
| Q8 | steering, generated terminal targets: rise (control); becomes top (control) | +1.41 (−0.01); 0.347 (0.020) | pass |
| Q9 | `return`: share of the gap closed by removing the keyword edge in all layers | 0.716 [0.699, 0.733] | pass |
| Q10 | largest single-layer share for `return` | 0.098 | pass |
| Q11 | L0H2 ablated on the terminal line and decision token: change in q(current level), terminal programs | +0.029 [0.023, 0.034] | pass |

## OLMo-2-0425-1B, head L13H1 and block-end set DED-O = {L13H1, L10H12, L11H0, L15H4}

Same material. 3,934 unmarked dedents; 3,934 unmarked dedents in the group ablation (chunks over 1,400 tokens skipped; root-identified points).

| id | measure | result | verdict |
|---|---|---|---|
| O1 | attention to innermost open header's `:\n`, unmarked dedents; same-level lines | 0.290 [0.273, 0.306]; 0.181 | pass |
| O2 | knock-out of that token, unmarked dedents; every other head of the layer | −0.132 [−0.146, −0.116]; +0.000 to +0.002 | pass |
| O3 | same knock-out, same-level lines; terminal dedents | +0.016; −0.009 | pass |
| O4 | closed-header knock-out against open-header knock-out (same points) | −0.004 against −0.119 | pass |
| O5 | picking where target ≠ innermost (innermost most attended); all multi-header dedents against innermost rule | 0.694 [0.661, 0.722] (0.139); 0.752 against 0.822 | pass |
| O6 | steering, natural unmarked dedents: rise (control); becomes top (control) | +1.58 (+0.15); 0.178 (0.015) | pass |
| O7 | knock-out of the true-level header, unmarked dedents | −0.166 | pass |
| O8 | steering, generated terminal targets: rise (control); becomes top (control) | +1.45 (+0.11); 0.393 (0.024) | pass |
| O9 | `return`: share of the gap closed by removing the keyword edge in all layers | 0.895 [0.870, 0.920] | pass |
| O10 | largest single-layer share for `return` | 0.158 | pass |
| G1 | DED-O ablated, unmarked dedents; 20 random layer-matched sets | −0.775 [−0.831, −0.713]; −0.116 to +0.100 | pass |
| G2 | terminal dedents: change (ratio to G1); shallower-open mass under ablation | −0.045 (0.057); 0.971 | pass |
| G3 | same-level lines | +0.033 | pass |
| G4 | unmarked dedents: change in mass on the current level | +0.194 | pass |

## Descriptive (not used for verdicts)

- Qwen3, keyword-edge share by keyword: return 0.716, raise 0.272, break 0.250, continue 0.106, pass −0.078; with the edges from the rest of the line as well (return) 0.813. L0H2 everywhere: +0.062.
- OLMo-2, keyword-edge share by keyword: return 0.895, raise 0.715, break 0.486, continue 0.405, pass 0.376; with the edges from the rest of the line as well (return) 0.855. L0H2 everywhere: +0.005.
- Qwen3 attention to the innermost open header at terminal dedents: 0.357; OLMo-2 L13H1: 0.255.
- Over all multi-header dedents, OLMo-2's L13H1 picks the target in 0.752 of cases against the innermost rule's 0.822 (no prediction was made about this comparison).

## Run

Run once as one systemd unit (`d3r2-conf`, `code/r2q_conf.sh`), started 09:55 and finished 14:20 IST on 9 November, in GPU lock holds of at most 1,000 s of work each; the scripts resume from their saved records, and the GPU was shared with another study and with exploratory checkpoint scans. Every script covered all its material (last pass exit status and time: `r2c01_header_olmo1b_L13H1` 0 at 12:13; `r2c01_header_qwen306bbf16_L23H2` 0 at 10:44; `r2c02_gen2_olmo1b_L13H1` 0 at 12:34; `r2c02_gen2_qwen306bbf16_L23H2` 0 at 11:10; `r2c03_group_olmo1b_DEDO=L13H1,L10H12,L11H0,L15H4` 0 at 14:20). No crash, deviation or exclusion beyond those the protocol fixes (group ablation: chunks over 1,400 tokens skipped).

