# D3 round 3 charter: replication on shareable material

Written 10 November 2026, about 20:35 IST. Frozen (digest in `FROZEN.md`, file read-only) before any round-3 material is fetched or generated and before any round-3 run. Background and reasons: `ROUND-3-PLAN.md`. This round adds **no exploration**: it re-runs frozen predictions and adds one prospective test. Calibration runs (below) use only material that earlier rounds have already read; they are labelled exploratory and decide nothing.

## Questions

1. Do the frozen code-based predictions of D3 round 1 (`PROTOCOL.md`, H1–H9), D3 round 2 (`PROTOCOL-2.md`, Q1–Q11, O1–O10, G1–G4) and the indentation side of X1 (`X1-shared-parts/PROTOCOL.md`) hold, with their frozen thresholds and verdict rules, on new material that may be redistributed?
2. Does L11H9 in Qwen3-0.6B-Base have a second role in terminal-statement dedents by the criterion X1 used for L17H6 (test A3), and does that role show inside D3's sets CARRY and TERM?

## Models

Qwen3-0.6B-Base (bfloat16) and OLMo-2-0425-1B (bfloat16), as frozen. For the descriptive gate: Qwen3-0.6B-Base (float32), Pythia-410M, Gemma-4-E2B, GPT-2, as in round 1, plus OLMo-2-0425-1B. All on disk; nothing is downloaded but the code material.

## Material (rules fixed here, before the fetch)

- **Natural code (`round3/data/code/`).** Fetched by `round3/code/r3_fetch.py`:
  - Repository search through the GitHub API, one query per licence: `language:Python created:>=2026-09-01 fork:false archived:false stars:>=3 license:<L>` for L in mit, apache-2.0, bsd-3-clause, bsd-2-clause, cc0-1.0, unlicense; sorted by stars, up to 5 pages of 100 each.
  - Dropped: every repository listed in D3 round 1's, D3 round 2's or X1's fetch; any repository whose licence endpoint (`repos/<r>/license`) does not return one of MIT, Apache-2.0, BSD-3-Clause, BSD-2-Clause, CC0-1.0, Unlicense.
  - Order of repositories: sha256("D3R3-repo-" + name). Per repository: the default branch's head commit is recorded and every file is read at that commit. Candidate files as in round 2 (`.py`, 1,500–40,000 bytes, no path part in round 2's SKIP set), up to 4 per repository by sha256("D3R3-file-" + repo + path).
  - A file is dropped if its text has an `SPDX-License-Identifier` naming anything outside the six licences, or contains "GNU General Public License", "GNU Lesser General Public", "GNU Affero", "Mozilla Public License" or "LGPL"/"AGPL"/"GPL-" as a licence word.
  - The repository's licence text and, if present, a root `NOTICE`, `NOTICE.txt` or `NOTICE.md` are saved under `round3/data/code/licences/`.
  - Stop after 1,000 kept files or 30 MB.
  - Recorded per file: repository, path, commit, licence (SPDX), created date, stars, bytes, SHA-256. Nothing fetched is executed.
- **Chunks.** `round3/code/r3_build.py` normalises and chunks every kept file exactly as round 2 (`build_material.chunk_records`: `pyindent.normalise`, `pyindent.chunks`, at least 4 decision points). A chunk is dropped if its text hash occurs in any earlier D3 or X1 part, or earlier in this build. Order: by file key. All chunks form one held part, `round3/data/code/part-held.jsonl`.
- **Generated programs.** Round 1's generator (`gen_programs.G`, unchanged) with seeds 300000–301999, width [2, 3, 4, "mixed"][seed % 4], first attempt of 400–1,400 characters that parses (as `r2_build.gen`): `round3/data/gen/part-held.jsonl`. Round 2's targeted generator (`gen2.program`, unchanged) with seeds 10000–11499: `round3/data/gen2/part-held.jsonl`. All held.
- Every part is sealed (digests in `FROZEN.md`, read-only) straight after its build, before any model reads it.

## How the frozen scripts are run on the new material

`D3-indentation/round3/` mirrors D3's layout. `round3/code/` holds byte-identical copies of the frozen scripts (digests compared with `FROZEN.md` before the run). Because those scripts find their data and outputs relative to their own folder, they read `round3/data/` and write `round3/outputs/` with no change. The slots they read (`newcode/part-conf.jsonl`, `gen/part-conf.jsonl`, `r2code/part-held.jsonl`, `r2gen2/part-held.jsonl`, and the gate's `newcode`/`gen` explore slots) are symbolic links to the round-3 parts. Copies of the frozen `PROTOCOL.md` and `PROTOCOL-2.md` (byte-identical, read-only), which the scripts check before reading a sealed part, are placed in `round3/` only when `PROTOCOL-3.md` is frozen. X1's frozen `c01_confirm.py` is run unchanged through a wrapper that points X1's loader and output folder at the round-3 parts and the frozen exploration means.

## Calibration (exploratory, old material only)

The new tests' margins that no frozen test fixes (CARRY and TERM without L11H9; single-head pair patching) are calibrated on X1's exploration parts and D3 round 1's generated exploration part. No calibration run reads round-3 material.

## Budget and stopping

- Local machine GPU only, about 12 GPU hours, under the shared lock and memory cage. Downloads: the code material only (expected 10–15 MB).
- Each frozen job runs once. A crash is fixed only for the crash and resumed, and logged in `REPORT-3.md`.
- If the fetch yields fewer than 400 kept files, the round stops before any model run and the shortfall is reported to the author.
- Nothing is published or committed.
