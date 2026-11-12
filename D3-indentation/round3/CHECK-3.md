# Independent check of REPORT-3.md

11 November 2026. Script: `round3/code/r3_check_independent.py` (run with `python3 -B round3/code/r3_check_independent.py` from `D3-indentation/`, inside the memory cage). It was checked against REPORT-3.md as rebuilt on 11 November, which prints H3 without a '+' sign and includes L17H6's pair result in the Reading.

The script does not import or run `r3_build_report.py`, `r3_verdicts.py`, `c03_verdicts.py`, `r2c04_verdicts.py` or `gate_summary.py`. It does not read `verdicts.json`, `verdicts_XE.json`, `measures_conf.json` or `gate_summary.json`. Every value comes from the raw run records:

- R1: `round3/outputs/conf/c01_newcode.json`, `c01_gen.json` and `c02_gen.json`. The originals come from `outputs/conf/`.
- R2: `round3/outputs/r2/conf/r2c01_*.jsonl`, `r2c02_*.jsonl` and `r2c03_olmo1b_DEDO.json`. The record readers are copied from the round-2 checker `code/r2_independent.py`, with the paths changed. The originals come from `outputs/r2/conf/`.
- X and E: `round3/x1/outputs/conf/c01_qwen3-06b-bf16_IND.json`, `r3x_l11h9_IND.json` and `round3/outputs/conf/r3_pairs.json`. The originals come from X1's `c01_qwen3-06b-bf16_IND.json`.
- Gate: the per-point records `round3/outputs/e01_<model>_{newcode,gen}.jsonl`. The definitions were reimplemented from reading `gate_summary.py`. The originals come from `outputs/e01_*.jsonl`. For the four models it covers, these originals agree exactly with `outputs/gate_summary_postfix_final.json` (largest difference 0).
- Material: the fetch and build summaries, `fetched.jsonl`, `part-held.jsonl` and the gen and gen2 parts.

The script finds each report value by its own text in the current REPORT-3.md. It then compares the value with its own recomputation, rounded to the number of decimals the report prints. The thresholds and verdict rules are taken from PROTOCOL.md, PROTOCOL-2.md, X1's PROTOCOL.md and PROTOCOL-3.md.

## Result

**498 checks, 0 failed.**

- **Numbers:** 388 numeric tokens were compared, and none failed. 375 of them are report values: results, originals, random ranges, counts and material figures. The other 13 are test ids or round numbers inside the quoted text, such as "H1–H9" or "layer-11".
- **Verdicts:** 57 were recomputed, and all agree with the report. They are the 49 test verdicts (H1–H9, Q1–Q11, O1–O10, G1–G4, XA3–XB4 and E1–E6), the R1 rule ("confirmed"), the five round-2 questions (all "confirmed"), and part E's verdict and summary line ("confirmed").
- **Other claims:** 16 were checked, and none failed:
  - L17H6's strongest random draw is L17H4, at 0.261 (0.239 in calibration);
  - L17H6 beats 19 of its 20 draws;
  - L11H9's effect is larger than L17H6's;
  - the gate pass and fail patterns in the Reading, including "again";
  - the eight material claims: licences, creation dates, pinned commits, contiguous seeds, and that the summaries match the raw files.
- **Provenance:** 37 checks, and none failed.

**Mismatches: none.**

Two early mismatches were bugs in this checker and were fixed before the final run. The first was its licence-count pattern, which captured a trailing comma. The second was how it tokenised "layer-11" in the Reading text. Neither was an error in the report.

### Consistency checks made along the way

- The r2c03 per-point means agree exactly with the run-file means.
- The X1 run and the part-E run have identical base values (the same points).
- The random draws for E6 come from the head's own layer: layer 11 for L11H9 and layer 17 for L17H6.
- The X1 data slots `round3/x1/data/{code,gen}/part-held.jsonl` are identical to the round-3 natural-code and gen parts.

## Provenance

Every round-3 run output read here is under `D3-indentation/round3/`, and each was modified after `PROTOCOL-3.md` (10 Nov 21:37:43). The earliest is `r3x_l11h9_IND.json` at 22:58 on 10 Nov, and the latest is `e01_gpt2_gen.jsonl` at 18:53 on 11 Nov.

Three kinds of round-3 file are older than the protocol or are not run outputs. In each case the timing matches the protocol, so none counts as a failure:

- **The sealed material is older than the freeze, as the protocol requires.** These files are `round3/data/code/*` (20:58), `round3/data/gen/part-held.jsonl` and `gen2/part-held.jsonl` (20:20), and the X1 slots. Their SHA-256 digests match FROZEN.md, and so does the digest of PROTOCOL-3.md.
- **The calibration record `round3/outputs/calib/r3_pairs.json` was written at 22:00 on 10 Nov, after the freeze.** This matches PROTOCOL-3.md, which says L17H6's calibration "was still running at the freeze and is not used". The Reading quotes one value from that file (0.239). No threshold or verdict depends on it.
- **Two r2 files also match the run notes:**
  - `r2c03_olmo1b_DEDO.json` was last written at 17:28 on 11 Nov, which matches the stated resume finish time;
  - `e01_gpt2_gen.jsonl` was last written at 18:53, which matches the stated end of the queue.

## What was not checked, and why

- **The bootstrap intervals** in square brackets in part E were not recomputed. They are E1 [−0.033, −0.021], CARRY and TERM in E2, and E5 [−0.005, −0.001], which makes 8 numbers. The protocol says intervals are descriptive only, so the script strips them before comparing.
- **"1,404 listed, 1,068 after removing…"** is checked only against `fetch_summary.json`, because the raw GitHub listing was not saved. **"652 files survived normalisation"** and **"9 … and 83 repeats dropped"** are checked only against `build_summary.json`, because recomputing them would mean rerunning the build. The 2,764 chunks hold 633 distinct files. That does not contradict 652, since the deduplication drops chunks and can remove every chunk of a file. The other material numbers are recomputed from the raw files: 1,000 files, 9,460,326 bytes, 274 repositories, 2,764 chunks, 225 repositories, the licence counts, and the seeds and counts.
- **R1's pair-patching values** (H9 and E6) are stored in the run files only as necessity and sufficiency shares; there are no per-pair records. They were read at that level, and the gap was not recomputed from individual pairs. R1's class means were also taken from the run files, not rebuilt from the per-point `.npy` arrays.
- **The run-note timings from the logs** were not checked against the logs: the 21:37 start, the 12th pass ending at 11:40 with 14 of 22 runs saved, and the 1,000 s holds. Only the file times above were compared.
- **Byte-identity of the script copies** in `round3/code/` against FROZEN.md was not re-verified. Only the material digests and PROTOCOL-3.md's digest were.
- **The "What paper B and its release would need" section** holds no result numbers and was not checked.
