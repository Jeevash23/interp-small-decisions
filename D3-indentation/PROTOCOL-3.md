# D3 round 3 protocol: replication on shareable material, and a prospective test of L11H9's second role

Written 10 November 2026 under `CHARTER-3.md` (frozen 20:19 IST). Frozen (digests in `FROZEN.md`, file read-only) after the round-3 material was sealed and after the calibration runs on old material, and **before any model reads round-3 material**. It is run **once**. Everything below is **prospective and confirmatory**; anything else computed on the round-3 parts is labelled post hoc. Plan and reasons: `ROUND-3-PLAN.md`.

## Material (sealed; digests in `FROZEN.md`)

- **Natural code:** `round3/data/code/part-held.jsonl`, 2,764 chunks from 225 GitHub repositories created on or after 1 Sep 2026 whose licence is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0 or Unlicense, none used by any earlier D3 or X1 part. Each file is pinned to a commit and recorded with repository, path, licence and SHA-256 in `round3/data/code/fetched.jsonl`; licence texts and notices are in `round3/data/code/licences/`.
- **Generated programs:** `round3/data/gen/part-held.jsonl` (2,000 programs, round 1's generator, seeds 300000–301999) and `round3/data/gen2/part-held.jsonl` (1,500 programs, round 2's targeted generator, seeds 10000–11499).
- One natural-code part serves every natural-code test below (round 1, round 2, X1 and E); X1's and E's tests use its first 400 chunks and the first 400 programs of `gen`, as X1 did.

## How the frozen scripts are run

`round3/code/` holds byte-identical copies of the frozen scripts (digests equal to the entries in `FROZEN.md`; checked at freeze). They read data and write outputs relative to their own folder, so they read `round3/data/` and write `round3/outputs/` unchanged. The slots they read are links to the round-3 parts:

| slot read by the frozen scripts | round-3 part |
|---|---|
| `round3/data/newcode/part-conf.jsonl` (c01 newcode), `newcode/part-explore.jsonl` (gate) | `code/part-held.jsonl` |
| `round3/data/gen/part-conf.jsonl` (c01 gen, c02), `gen/part-explore.jsonl` (gate) | `gen/part-held.jsonl` |
| `round3/data/r2code/part-held.jsonl` (r2c01, r2c03) | `code/part-held.jsonl` |
| `round3/data/r2gen2/part-held.jsonl` (r2c02) | `gen2/part-held.jsonl` |
| `round3/x1/data/code/part-held.jsonl`, `round3/x1/data/gen/part-held.jsonl` (X1's c01) | `code/part-held.jsonl`, `gen/part-held.jsonl` |

`round3/PROTOCOL.md` and `round3/PROTOCOL-2.md` are byte-identical read-only copies of the frozen round-1 and round-2 protocols, placed there at this freeze because the scripts check them before reading a sealed part. X1's frozen `c01_confirm.py` is run unchanged through `round3/code/r3x_c01.py`, which points X1's loader at `round3/x1/data` and its output folder at `round3/x1/outputs`, where a byte-identical copy of X1's frozen exploration means for indentation lies.

## Part R1: D3 round 1 (`PROTOCOL.md`), H1–H9

Qwen3-0.6B-Base, bfloat16. `c01_ablate.py newcode`, `c01_ablate.py gen`, `c02_pairs.py` and `c03_verdicts.py`, unchanged. Predictions, thresholds, random draws (seeds 1000 + k and 2000 + k), the means (re-estimated on each part, as frozen), the verdict rule ("confirmed" if H1, H2, H3, H5 and H6 pass) and the bootstrap: exactly as in `PROTOCOL.md`.

## Part R2: D3 round 2 (`PROTOCOL-2.md`), Q1–Q11, O1–O10, G1–G4

Qwen3-0.6B-Base (head L23H2) and OLMo-2-0425-1B (head L13H1, group DED-O), bfloat16. `r2q_conf.sh` (r2c01, r2c02, r2c03) and `r2c04_verdicts.py conf`, unchanged. Predictions, thresholds, control seeds and verdict rules for questions 1, 2, 3, 4a and 4b: exactly as in `PROTOCOL-2.md`.

## Part X: the indentation side of X1 (`X1-shared-parts/PROTOCOL.md`)

`c01_confirm.py qwen3-06b-bf16 IND`, unchanged (groups L17H6, READ−L17H6, WRITE, COUNT, DED, L0H2, L0H3, L11H9; 20 random draws each, seed `x1-c01-<group>`; X1's frozen exploration means). Each X1 test is scored on its indentation part only, with X1's thresholds; the line-breaking, ordinal and capitals parts (Wikipedia text and our own number frames) are not re-run and keep their X1 verdicts.

| id | X1 test | indentation part re-tested |
|---|---|---|
| XA3 | A3 | L17H6: generated terminal change ≤ −0.010 and below every random draw |
| XA5 | A5 | L17H6: \|code unmarked change\| ≤ 0.03, \|generated unmarked change\| ≤ 0.10 |
| XB1, XB2, XB3 | B1–B3 | WRITE, COUNT, READ−L17H6: indentation quiet |
| XB4 | B4 | DED: code unmarked change ≤ −0.40 and below every random draw |
| XC3, XC6 | C3, C6 | L0H2, L0H3: indentation quiet |
| XD3 | D3 | L11H9: \|change\| ≤ 0.05 on all four indentation measures |

"Indentation quiet" is X1's: \|change\| ≤ 0.03 on code unmarked, code terminal and generated terminal dedents, and ≤ 0.10 on generated unmarked dedents. Verdicts: Lead 1's indentation side holds if XA3 passes; Lead 2's indentation side if XB1–XB4 pass; Lead 3's if XC3 and XC6 pass; Lead 4's frozen bound if XD3 passes.

## Part E: L11H9's second role (new, prospective)

Background: X1 confirmed L11H9 as an ordinal suffix head (test D1) and bounded its indentation effect loosely (D3, at most 0.05). Read post hoc by the criterion that made L17H6 shared (A3), L11H9 also adds to terminal-statement dedents in generated programs (X1's held part: −0.030, random −0.007 to 0.002). L11H9 is a member of D3's CARRY and TERM. The question is whether that second role holds prospectively on new material, and inside CARRY and TERM.

Scripts: X1's frozen `c01_confirm.py` (for L11H9 alone, in part X's run) and `round3/code/r3x_l11h9.py` (the same computation for CARRY, CARRY−L11H9, TERM and TERM−L11H9, with 20 random draws each, seed `x1-c01-<group>`, from heads named in no X1 group); `round3/code/r3_pairs.py` (round 1's pair patching, for the single heads L11H9 and L17H6). Verdicts by `round3/code/r3_verdicts.py`.

Calibration on old material only (`round3/x1/outputs/calib/`, `round3/outputs/calib/`; exploratory, decides nothing): X1's exploration code and generated parts (400 + 400) for the groups, round 1's generated exploration part for the pairs. Calibration values are in brackets; the held-out X1 value of L11H9 alone is quoted as the source of E1.

| id | prediction (each could fail) | brackets |
|---|---|---|
| **E1** | L11H9 alone: change in mean log q(true) at terminal-statement dedents in generated programs ≤ −0.010 **and below every one of its 20 random draws** (X1's A3 criterion, unchanged) | X1 held, post hoc: −0.030; random −0.007 to 0.002 |
| **E2** | CARRY and TERM, each as a whole: the same criterion | CARRY −0.300 (random −0.017 to 0.004); TERM −0.209 (random −0.063 to 0.030) |
| **E3** | Removing L11H9 weakens each set: change(set without L11H9) − change(set) ≥ +0.010, for CARRY and for TERM, on generated terminal dedents | CARRY −0.268 against −0.300 (difference +0.033); TERM −0.163 against −0.209 (+0.046) |
| **E4** | L11H9 alone: \|code unmarked change\| ≤ 0.03 and \|generated unmarked change\| ≤ 0.10 (spares unmarked dedents, as A5 for L17H6) | X1 held: −0.009, −0.001 |
| **E5** | L11H9 alone: natural-code terminal change below every random draw (no magnitude bound) | X1 held: −0.003; random −0.002 to 0.000 |
| **E6** | Pair patching (round 1's return/assignment pairs, up to 300), L11H9 alone: necessity ≥ 0.05 and above every one of 20 random layer-11 heads. L17H6 is run the same way and reported, not scored | necessity 0.116, random −0.004 to 0.031 (300 pairs, mean gap 3.52); L17H6's calibration was still running at the freeze and is not used |

**Verdict.** L11H9's second role is **confirmed** if E1 and E3 pass; **a single-head effect only** if E1 passes and E3 fails; **not confirmed** if E1 fails. E2, E4, E5 and E6 qualify the reading whichever way they go. Its first role (the ordinal suffix) was confirmed prospectively by X1 on our own number frames and is not re-run.

## Part G: the gate, descriptive

`e01_behaviour.py` and `gate_summary.py` (round 1's gate scripts, exploratory there, unchanged; digests in `FROZEN.md`) for Qwen3-0.6B-Base (float32), OLMo-2-0425-1B, Pythia-410M, Gemma-4-E2B and GPT-2 on the round-3 natural code and the round-3 generated programs. Reported with round 1's gate thresholds as description; no verdict depends on it.

## Analysis and rules

- Decisions use point estimates; intervals are description only.
- No point, chunk, program or repository is excluded after the run. A crash is fixed only for the crash and resumed; the fix is logged in `REPORT-3.md`. A prediction that cannot be evaluated is reported as not evaluable, not as a pass.
- `REPORT-3.md` reports every test of parts R1, R2, X and E, pass or fail, next to the original result, with every number taken from saved outputs; an independent script checks them.
- The earlier rounds' files, outputs and verdicts are not changed.

## Run plan

1. Freeze this file and the scripts (digests in `FROZEN.md`; read-only); create the slot links and the protocol copies.
2. Run `round3/code/r3q_conf.sh` as a `systemd-run --user` unit: E, X, R1, R2, then the gate, each call one GPU lock hold of at most about 1,000 s of work in the memory cage, resuming from saved records.
3. Run `c03_verdicts.py`, `r2c04_verdicts.py conf`, `r3_verdicts.py` and `gate_summary.py`; write `REPORT-3.md`; run the independent check.

## Scripts frozen with this protocol

Copies (digests equal to their earlier freeze): `common.py`, `pyindent.py`, `gemma_load.py`, `c01_ablate.py`, `c02_pairs.py`, `c03_verdicts.py`, `r2common.py`, `r2c01_header.py`, `r2c02_gen2.py`, `r2c03_group.py`, `r2c04_verdicts.py`, `run.sh`, `r2q_conf.sh`; X1's `c01_confirm.py`, `x1common.py` and exploration means (in place). New: `r3x_c01.py`, `r3x_l11h9.py`, `r3_pairs.py`, `r3_verdicts.py`, `r3q_conf.sh`; round 1's gate scripts `e01_behaviour.py`, `gate_summary.py`.
