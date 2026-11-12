# D3 round 3: a replication on shareable material (plan for the author)

Written 10 November 2026, about 20:15 IST, at the author's request. This is a plan; the binding rules are in `CHARTER-3.md` and `PROTOCOL-3.md` once frozen.

## Why

Paper B (`DX-combined-paper/paper/v4-2026-11-10/`) is held back for two reasons.

1. **Material we may not share.** Every natural-code result in D3 (rounds 1 and 2) and in X1's indentation tests was measured on chunks cut from public GitHub files. Of the 1,760 source files, 439 carry no licence and 132 are copyleft. The author does not want to publish other people's unlicensed code.
2. **A post hoc reading in the headline.** v4 says that L11H9 in Qwen3-0.6B-Base, confirmed by X1 as an ordinal suffix head, also adds to terminal-statement dedents (−0.030 on generated programs, random −0.007 to 0.002). That meets the criterion that made L17H6 "shared" (X1 test A3), but it was read off after the run. X1 never ran D3's sets CARRY and TERM on held-out indentation material.

## (a) New held-out material that can be redistributed

- **Natural code, permissive licences only.** Python files fetched fresh from public GitHub repositories whose declared licence (GitHub's `license` field, checked again through the repository's licence endpoint) is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0 or Unlicense. Repositories created on or after 1 Sep 2026 (after both models' training data), not forks, not archived, at least 3 stars, and none used by D3 round 1, D3 round 2 or X1. Same file rule as before (`.py`, 1,500–40,000 bytes, no vendored or generated folders, up to 4 files per repository). A file is also dropped if its own header names a licence outside the list (an SPDX line, or GPL, LGPL, AGPL or MPL text). Each file is pinned to a commit and recorded with repository, path, commit, licence and SHA-256; the repository's licence text (and an Apache `NOTICE` file, if present) is saved beside it, so a release can carry the notices the licences require. Stop at 1,000 files.
- **Newly generated programs of our own**, from D3's frozen generators with seeds never used before: 2,000 programs from round 1's generator (`gen_programs.py`, seeds 300000–301999) and 1,500 from round 2's targeted generator (`gen2.py`, seeds 10000–11499).
- All of it is built and sealed (digests in `FROZEN.md`, read-only) before any model reads it, and none of it is used for exploration or calibration. Chunks that repeat any earlier part are dropped.

## (b) What is re-tested, with the same thresholds

Every frozen code-based prediction behind paper B, by the frozen scripts, unchanged (byte-identical copies, digests checked against `FROZEN.md`), pointed at the new material:

- **D3 round 1** (`PROTOCOL.md`): H1–H9 in Qwen3-0.6B-Base. Natural-code tests on the new permissive part, generated tests and pair patching on the new generated part.
- **D3 round 2** (`PROTOCOL-2.md`): Q1–Q11 (Qwen3-0.6B-Base, L23H2), O1–O10 and G1–G4 (OLMo-2-0425-1B, L13H1 and DED-O). Natural code from the same new permissive part; targeted programs from the new `gen2` part.
- **X1** (`PROTOCOL.md`): the indentation side of every test that has one: A3, A5 (indentation parts), B1–B3 ("indentation quiet"), B4 (natural-code part), C3, C6 and D3, with the frozen exploration means, on the first 400 new natural-code chunks and the first 400 new generated programs, as X1 did. The line-breaking, ordinal and capitals parts of X1 rest on Wikipedia text (CC BY-SA) and on our own number frames, which can be shared, so they are not re-run.
- **D3 gate (descriptive):** the behavioural gate on the new natural code and generated programs, for the five models in paper B's gate table, with the frozen gate thresholds reported.

Same thresholds, same verdict rules. A test passes or fails as frozen; nothing is re-tuned.

## (c) A prospective test of L11H9's second role, and of CARRY and TERM

Frozen before any model reads the new material:

- **E1 (the A3 criterion, as for L17H6):** L11H9 alone, mean-ablated at the decision token: change in mean log q(true) at terminal-statement dedents in generated programs ≤ −0.010 and below every one of its 20 layer-matched random heads.
- **E2:** CARRY and TERM, each as a whole: the same criterion. X1 never ran them on indentation.
- **E3:** removing L11H9 from each set weakens it: change(set without L11H9) − change(set) ≥ a margin fixed from calibration, for both CARRY and TERM.
- **Qualifiers:** L11H9 leaves unmarked dedents alone (as A5); natural-code terminal dedents (no magnitude bound); single-head patching in return/assignment pairs, necessity above every random head (L11H9, and L17H6 for comparison).
- **Verdict:** the second role is confirmed if E1 and E3 pass; "a single-head effect only" if E1 passes and E3 fails; not confirmed if E1 fails. Its first role (the ordinal suffix) was already confirmed prospectively in X1 on our own number frames and is not re-run.

Thresholds that do not come from a frozen test (E3's margin, the pair-patching qualifiers) are set from calibration runs on old material that has already been read (X1's and D3's exploration parts), never on the new material.

## (d) Compute, time and the old runs

- **Local machine GPU only** (6 GB NVIDIA GPU, 6 GB), under the shared lock `~/.cache/d1-gpu.lock` in holds of at most about 18 minutes, inside the memory cage, as one `systemd-run --user` queue. No Modal, no new weights (Qwen3-0.6B-Base, OLMo-2-0425-1B, Pythia-410M, Gemma-4-E2B and GPT-2 are on disk).
- **Time:** the earlier runs took about 2.4 h (round 1), 4.4 h (round 2) and 1.3 h (X1 indentation); with the new tests and the gate, about 10–12 GPU hours, overnight. Downloads: about 10–15 MB of code.
- **The old runs stay in the record.** No earlier file, output or verdict is changed. The old natural-code material stays on this local machine; a release lists it by fetch script, repository, path and SHA-256 only, and does not ship it. The old outputs (numbers, not code) stay with the release as the original record. Round 3's material can be shipped with its licence notices.

## What paper B and its release would then need

Listed in `REPORT-3.md` once the run is done: a v5 that reports the replication beside the original results (and replaces the post hoc L11H9 paragraph by the prospective result, whichever way it goes), and a release that ships round 3's permissive chunks and gives the old chunks as fetch scripts and checksums only. Those edits are a later step.
