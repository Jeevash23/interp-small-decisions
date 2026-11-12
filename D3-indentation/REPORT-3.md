# D3 round 3 report: the code results of paper B on shareable material, and L11H9's second role

10–11 November 2026. **Confirmation** under `PROTOCOL-3.md` (frozen 10 Nov, 21:37 IST), run once on parts sealed before any model read them, under `CHARTER-3.md` (frozen 20:19 IST). Plan and reasons: `ROUND-3-PLAN.md`. The result tables below are generated from saved outputs by `round3/code/r3_build_report.py`; the reading at the end is written by hand from the same outputs. An independent script checks every number (`round3/CHECK-3.md`).

## Result

- **D3 round 1 (H1–H9): 9 of 9 tests pass**; by the frozen rule the two-route account is **confirmed** on the new material.
- **D3 round 2 (Q1–Q11, O1–O10, G1–G4): 25 of 25 tests pass**. Question 1 (what the block-end head reads): **confirmed**; Question 2 (how far to dedent): **confirmed**; Question 3 (terminal route: keyword read directly, spread, not via L0H2 on the line): **confirmed**; Question 4a (two routes in OLMo-2): **confirmed**; Question 4b (L13H1 works like L23H2): **confirmed**.
- **X1, indentation side: 9 of 9 tests pass.**
- **L11H9's second role (part E): confirmed** (E1 pass, E3 pass; qualifiers: E2 pass, E4 pass, E5 pass, E6 pass).

## Material

- **Natural code:** 1,000 Python files (9,460,326 bytes) fetched on 10 November 2026 from 274 GitHub repositories created on or after 1 September 2026 whose licence is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0 or Unlicense (1,404 listed, 1,068 after removing every repository an earlier part used). Each file is pinned to a commit; licence texts and notices are kept. 652 files survived normalisation and gave 2,764 chunks from 225 repositories (9 chunks that repeat an earlier part and 83 repeats within the build dropped). Chunks by licence: Apache-2.0 1,192, MIT 573, BSD-3-Clause 471, CC0-1.0 203, Unlicense 169, BSD-2-Clause 156.
- **Generated programs:** 2,000 from round 1's generator (seeds 300000–301999) and 1,500 from round 2's targeted generator (seeds 10000–11499).
- All parts were sealed before any model read them. Everything here was run in Qwen3-0.6B-Base and, for round 2's questions 4a and 4b, OLMo-2-0425-1B, both in bfloat16, by byte-identical copies of the frozen scripts.

## Part R1: D3 round 1 (`PROTOCOL.md`)

| id | prediction | original (8 Nov) | round 3 | round 3, context | verdict |
|---|---|---|---|---|---|
| H1 | DED, natural, other dedents: change in mean log q(true) <= -0.40 and below every random group | −1.091 | −0.994 | random min -0.028 | pass |
| H2 | DED, natural, other dedents: change in mass on the current level >= +0.10 and above every random group | +0.306 | +0.283 | random max 0.015 | pass |
| H3 | DED, natural, terminal dedents: shallower-open mass >= 0.95 under ablation and |change in log q| <= 0.25 x |H1 change| | 0.991 | 0.994 | terminal change -0.049; ratio 0.049 | pass |
| H4 | DED, natural, same-level lines: change in mean log q(true) >= -0.01 | +0.025 | +0.024 | – | pass |
| H5 | TERM, natural, terminal dedents: change in mean log q(true) <= -0.03 and below every random group | −0.044 | −0.032 | random min -0.009 | pass |
| H6 | TERM, natural, other dedents: |change in mean log q(true)| <= 0.10 | −0.027 | −0.017 | – | pass |
| H7 | TERM+DED, natural, terminal dedents: change in mean log q(true) more negative than the sum of the two single changes | −0.188 | −0.134 | sum of singles -0.081 | pass |
| H8 | DED, generated, other dedents: change in mean log q(true) <= -0.25 and below every random group | −0.631 | −0.652 | random min -0.100 | pass |
| H9 | CARRY, generated pairs: necessity >= 0.40 and sufficiency >= 0.20, each above every random group | +0.650 | +0.652 | sufficiency 0.363; random max nec 0.070, suf 0.039 | pass |


## Part R2: D3 round 2 (`PROTOCOL-2.md`)

Natural code: 2,764 chunks from 225 repositories (3,780 unmarked dedents, 19,383 same-level lines and 3,855 terminal dedents with at least one open header, Qwen3). Generated: 1,500 targeted programs (149 ending in `return`). Thresholds are those of `PROTOCOL-2.md`; the verdict column is round 3's.

| id | measure | original (9 Nov) | round 3 | verdict |
|---|---|---|---|---|
| Q1 | attention to innermost open header, unmarked dedents; same-level lines | 0.458; 0.118 | 0.457; 0.107 | pass |
| Q2 | knock-out of that token, unmarked dedents; range over the other heads of the layer | −0.148; −0.004 to +0.044 | −0.117; −0.004 to +0.035 | pass |
| Q3 | same knock-out, same-level lines; terminal dedents | +0.013; −0.005 | +0.012; −0.006 | pass |
| Q4 | closed-header knock-out against open-header knock-out | −0.001 against −0.122 | 0.000 against −0.086 | pass |
| Q5 | picking where target is not innermost; all multi-header dedents against innermost rule | 0.893; 0.914 against 0.822 | 0.896; 0.911 against 0.819 | pass |
| Q6 | steering, natural unmarked dedents: rise (control); becomes top (control) | +2.72 (+0.08); 0.125 (0.004) | +2.74 (+0.08); 0.145 (0.006) | pass |
| Q7 | knock-out of the true-level header, unmarked dedents | −0.210 | −0.195 | pass |
| Q8 | steering, generated terminal targets: rise (control); becomes top (control) | +1.41 (−0.01); 0.347 (0.020) | +1.39 (−0.02); 0.363 (0.015) | pass |
| Q9 | `return`: share of the gap closed by removing the keyword edge in all layers | 0.716 | 0.710 | pass |
| Q10 | largest single-layer share for `return` | 0.098 | 0.102 | pass |
| Q11 | L0H2 ablated on the terminal line and decision token: change in q(current level) | +0.029 | +0.030 | pass |
| O1 | attention to innermost open header, unmarked dedents; same-level lines (OLMo-2, L13H1) | 0.290; 0.181 | 0.283; 0.171 | pass |
| O2 | knock-out of that token, unmarked dedents; range over the other heads of the layer (OLMo-2, L13H1) | −0.132; +0.000 to +0.002 | −0.115; 0.000 to +0.001 | pass |
| O3 | same knock-out, same-level lines; terminal dedents (OLMo-2, L13H1) | +0.016; −0.009 | +0.014; −0.009 | pass |
| O4 | closed-header knock-out against open-header knock-out (OLMo-2, L13H1) | −0.004 against −0.119 | −0.003 against −0.096 | pass |
| O5 | picking where target is not innermost; all multi-header dedents against innermost rule (OLMo-2, L13H1) | 0.694; 0.752 against 0.822 | 0.715; 0.768 against 0.819 | pass |
| O6 | steering, natural unmarked dedents: rise (control); becomes top (control) (OLMo-2, L13H1) | +1.58 (+0.15); 0.178 (0.015) | +1.61 (+0.16); 0.194 (0.011) | pass |
| O7 | knock-out of the true-level header, unmarked dedents (OLMo-2, L13H1) | −0.166 | −0.154 | pass |
| O8 | steering, generated terminal targets: rise (control); becomes top (control) (OLMo-2, L13H1) | +1.45 (+0.11); 0.393 (0.024) | +1.41 (+0.10); 0.407 (0.024) | pass |
| O9 | `return`: share of the gap closed by removing the keyword edge in all layers (OLMo-2, L13H1) | 0.895 | 0.859 | pass |
| O10 | largest single-layer share for `return` (OLMo-2, L13H1) | 0.158 | 0.136 | pass |
| G1 | DED-O ablated, unmarked dedents; range of 20 random sets | −0.775; −0.116 to +0.100 | −0.776; −0.109 to +0.098 | pass |
| G2 | terminal dedents: change; shallower-open mass under ablation | −0.045; 0.971 | −0.044; 0.972 | pass |
| G3 | same-level lines | +0.033 | +0.030 | pass |
| G4 | unmarked dedents: change in mass on the current level | +0.194 | +0.199 | pass |

## Part X: the indentation side of X1's tests

Each X1 test re-scored on its indentation part only, with X1's thresholds and X1's frozen exploration means, on the first 400 new natural-code chunks and the first 400 new generated programs. Values are changes in mean log q(true); brackets give the range of the 20 random draws. The original column is X1's held-out run (8 Nov).

| id | test | original | round 3 | verdict |
|---|---|---|---|---|
| XA3 | L17H6, generated terminal: ≤ −0.010, below every random draw | −0.022 (−0.011 to 0.002) | −0.025 (−0.010 to 0.004) | pass |
| XA5 | L17H6, code unmarked (≤ 0.03); generated unmarked (≤ 0.10) | −0.010; +0.021 | −0.003; +0.013 | pass |
| XB1 | WRITE: indentation quiet (code unmarked; code terminal; generated unmarked; generated terminal) | +0.001; +0.001; +0.001; −0.001 | −0.001; +0.001; −0.001; −0.001 | pass |
| XB2 | COUNT: indentation quiet (code unmarked; code terminal; generated unmarked; generated terminal) | +0.007; −0.001; +0.000; −0.003 | +0.002; 0.000; 0.000; −0.003 | pass |
| XB3 | READ−L17H6: indentation quiet (code unmarked; code terminal; generated unmarked; generated terminal) | −0.003; −0.001; −0.007; −0.003 | −0.002; +0.000; −0.014; −0.004 | pass |
| XC3 | L0H2: indentation quiet (code unmarked; code terminal; generated unmarked; generated terminal) | −0.008; −0.001; −0.003; +0.000 | 0.000; +0.000; −0.003; 0.000 | pass |
| XC6 | L0H3: indentation quiet (code unmarked; code terminal; generated unmarked; generated terminal) | −0.009; −0.004; +0.010; −0.007 | −0.011; −0.001; +0.010; −0.004 | pass |
| XD3 | L11H9: |change| ≤ 0.05 on all four (code unmarked; code terminal; generated unmarked; generated terminal) | −0.009; −0.003; −0.001; −0.030 | −0.010; −0.003; −0.012; −0.027 | pass |
| XB4 | DED, code unmarked: ≤ −0.40, below every random draw | −1.112 (−0.027 to 0.014) | −0.801 (−0.015 to 0.012) | pass |

Points (round 3): code unmarked 548, code terminal 540, generated unmarked 1,547, generated terminal 1,288. Unablated mean log q(true): cU −0.304, cT −0.039, gU −1.846, gT −0.566.

## Part E: L11H9's second role (new, prospective)

| id | prediction | result | verdict |
|---|---|---|---|
| E1 | L11H9 alone, generated terminal dedents: ≤ −0.010 and below every random draw (X1's A3 criterion) | −0.027 [−0.033, −0.021]; random (−0.004 to 0.003) | pass |
| E2 | CARRY and TERM, the same criterion | CARRY −0.303 [−0.322, −0.285], random (−0.015 to 0.003); TERM −0.224 [−0.242, −0.207], random (−0.060 to 0.030) | pass |
| E3 | removing L11H9 weakens each set by ≥ +0.010 | CARRY −0.264 against −0.303 (difference +0.039); TERM −0.172 against −0.224 (+0.052) | pass |
| E4 | L11H9: code unmarked ≤ 0.03, generated unmarked ≤ 0.10 in absolute value | −0.010; −0.012 | pass |
| E5 | L11H9, natural-code terminal dedents: below every random draw | −0.003 [−0.005, −0.001]; random (−0.001 to 0.001) | pass |
| E6 | pair patching, L11H9 alone: necessity ≥ 0.05 and above every random draw | necessity 0.122 (random at most 0.033), sufficiency 0.027 (random at most 0.015); L17H6, reported: necessity 0.069 (random at most 0.261), sufficiency 0.024 (random at most 0.121) | pass |

Verdict rule (frozen): confirmed if E1 and E3 pass; a single-head effect only if E1 passes and E3 fails; not confirmed if E1 fails. **Verdict: confirmed.**

## Part G: the gate on the new material (descriptive)

Round 1's gate measures; the thresholds are round 1's (header top-1 ≥ 0.90; terminal top-1 ≥ 0.5 and shallower-open mass ≥ 0.7; top-1 minus the copy baseline ≥ 0.10; never-opened mass at dedents ≤ 0.10 on 2-space, 3-space and mixed generated code). Original values (round 1, corrected measure) in brackets; OLMo-2-0425-1B was gated in round 2 on other material, so it has none here.

| model | points | header top-1 | terminal top-1 | terminal shallower-open | top-1 − copy | never-opened mass, worst width |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 45,708 | 0.998 (1.000) | 0.981 (0.989) | 0.997 (0.997) | 0.419 (0.406) | 0.062 (0.065) |
| OLMo-2-0425-1B | 45,708 | 0.995 | 0.963 | 0.985 | 0.406 | 0.069 |
| Pythia-410M | 45,708 | 0.874 (0.870) | 0.961 (0.973) | 0.994 (0.994) | 0.369 (0.358) | 0.070 (0.073) |
| Gemma-4-E2B | 45,708 | 0.994 (0.995) | 0.980 (0.986) | 0.996 (0.996) | 0.419 (0.406) | 0.057 (0.061) |
| GPT-2 | 45,708 | 0.083 (0.078) | 0.362 (0.337) | 0.703 (0.696) | −0.234 (−0.243) | 0.205 (0.208) |


## Reading

**The indentation account holds on shareable material.** Every D3 round-1 prediction passes again on 2,764 chunks of permissively licensed code and 2,000 new generated programs. Removing the block-end heads DED costs unmarked dedents −0.994 in mean log q(true) (original −1.091; random groups at most −0.028), and the five heads spare terminal-statement dedents as before. The terminal-statement set TERM again hurts terminal dedents beyond every random group, but only just past its frozen bar: −0.032 against a threshold of −0.03 (original −0.044). As in round 1, that effect is small in absolute size because terminal dedents are near-certain in natural code; it is specific, not large. The two routes again back each other up (H7) and CARRY again carries the terminal signal in return/assignment pairs (H9).

**How each route gets its information, and the second model, hold too.** All 25 round-2 tests pass on the same new code and on 1,500 new targeted programs, with values close to the originals. In Qwen3, knocking out the single source position L23H2 reads, the `:\n` of the innermost open header, costs unmarked dedents −0.117 (original −0.148), while the other 15 heads of layer 23 move it by −0.004 to +0.035; steering its attention onto a chosen open header raises that header's indent by +2.74 nats on average (original +2.72). Removing the decision token's read of a `return` keyword in every layer again closes most of the gap to a neutral line (0.710; original 0.716), with no single layer carrying much. In OLMo-2-0425-1B the block-end set DED-O again costs unmarked dedents −0.776 against random sets from −0.109 to +0.098, and its head L13H1 again reads open headers (knock-out −0.115).

**X1's indentation side holds.** L17H6 again meets X1's criterion for a shared head on generated terminal dedents (−0.025, random −0.010 to 0.004; original −0.022). The line-breaking groups and the early heads leave indentation as quiet as before, and DED again does its own job (code unmarked dedents −0.801 on the first 400 chunks; original −1.112). X1's line-breaking, ordinal and capitals sides were not re-run: they rest on Wikipedia text and on our own number frames, which can be shared.

**L11H9's second role is confirmed prospectively.** On new material, frozen before any model read it, L11H9 alone lowers mean log q(true) at generated terminal-statement dedents by 0.027, below all 20 random layer-11 heads (−0.004 to 0.003). This is X1's A3 criterion unchanged, the one that made L17H6 shared, and L11H9 meets it again with a larger value than L17H6. Inside D3's sets it matters too: leaving L11H9 out weakens CARRY by 0.039 and TERM by 0.052 on the same points (E3). The qualifiers agree: L11H9 spares unmarked dedents (E4), has a small effect on natural-code terminal dedents that still lies below its random range (−0.003, random −0.001 to 0.001; E5), and in return/assignment pairs it carries 0.122 of the clean-minus-corrupt gap by necessity, against at most 0.033 for random layer-11 heads (E6). L17H6, run the same way for comparison and not scored, carries 0.069; it lies above 19 of its 20 random layer-17 heads but not above the 20th, which is L17H4, itself a CARRY head and the strongest terminal carrier (0.261). Round 1's frozen draw rule, used unchanged, excludes only the head under test; the same draw appeared in calibration (0.239). So L11H9, confirmed by X1 as an ordinal suffix head, has a second, small role in the dedent after a terminal statement: clear in generated programs, and much smaller, though still outside its random range, in natural code, where terminal dedents are near-certain anyway. It also stays within X1's looser frozen bound for "not shared" (D3: at most 0.05 in absolute value), which is why X1's frozen verdict and this one do not contradict each other: the two tests asked different questions.

**The gate reads the same.** On the new code, Qwen3-0.6B-Base, OLMo-2-0425-1B and Gemma-4-E2B pass every gate item; Pythia-410M again misses only the header item (0.874 against the 0.90 bar; original 0.870), and GPT-2 again fails every item. The gate is descriptive here; no verdict rests on it.

## Run notes

- Run once as unit `d3-r3-conf` (`round3/code/r3q_conf.sh`), from 21:37 on 10 November, in GPU lock holds of at most 1,000 s of work in the memory cage.
- **One departure, a pass cap, not a crash.** The frozen round-2 queue `r2q_conf.sh` allows 12 passes for r2c03 (OLMo-2 group ablation). Round 2 finished within them; here the 12th pass ended at 11:40 on 11 November with 14 of 22 runs saved. The same command was resumed as unit `d3-r3-r2c03` (`round3/code/r3q_resume_r2c03.sh`), which reads the saved record and continues; nothing in the computation changed. It finished all 22 runs at 17:28 on 11 November, sharing the GPU lock with the gate jobs.
- The gate scripts ran after the confirmation jobs. The queue ended at 18:53 on 11 November; every job covered all its material.
- No point, chunk, program or repository was excluded beyond what the frozen scripts do (chunks over 1,400 tokens skipped in round 2; points whose candidates are not all identified at the first token).
- Calibration for part E ran on old material only (X1's exploration parts, D3 round 1's generated exploration part) before the protocol froze; its values are in `PROTOCOL-3.md`.

## What paper B and its release would need (not done here)

**Paper B, a v5** (v4 untouched):
1. Report the replication beside each original code result: D3 round 1 (Table and Appendix B), D3 round 2, the indentation cells of X1's matrix, and the gate table, each with the round-3 value and verdict. A short paragraph in the methods on the new material: permissive licences only, pinned commits, licence notices kept, sealed before use, run by byte-identical copies of the frozen scripts.
2. Replace the post hoc L11H9 paragraph (Section 6, abstract, introduction, Figure 1's dotted arrow, discussion, limitations) by the prospective result of part E: L11H9 meets the A3 criterion on new material frozen in advance, and its share shows inside CARRY and TERM. The wording can say "after a terminal statement" rather than "after `return`": the class covers `return`, `pass`, `raise`, `break` and `continue`, and the pairs replace any of them.
3. State that the original natural-code material is not distributed, and that the round-3 material is.
4. Number keys for every round-3 value, checked by the paper's own script against `round3/outputs/`, and the REPORT-3 check.

**The release** (`<interp-area>/RELEASES/small-decisions/`, still on paper v3):
1. Remove every chunk cut from the old GitHub files: `D3-indentation/data/newcode/part-*.jsonl`, `D3-indentation/data/r2code/part-held.jsonl`, `X1-shared-parts/data/code/part-*.jsonl`, and any copy inside the attached archives. Keep `fetched.jsonl` (repository, path, licence, SHA-256) and the fetch scripts, so anyone can fetch the same files and check them.
2. Saved outputs that repeat pieces of that code (the previous line at each decision point, per-point records such as `e01_*.jsonl` and `r2c01_*.jsonl`) need the code text stripped, keeping the numbers, or they stay out of the release.
3. Ship round 3: `round3/data/` (chunks, `fetched.jsonl` with commits, the licence texts and NOTICE files under `licences/`), its scripts and outputs, with `THIRD-PARTY.md` rewritten to say the shipped code is permissively licensed and carries its notices.
4. Rebuild, rescan and rerun the scratch-copy check, and sync to paper v5.
