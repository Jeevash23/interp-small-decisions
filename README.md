# Small decisions, mostly separate parts: code release

This repository holds the code, frozen protocols, held-out material and saved outputs behind the paper *Small
decisions, mostly separate parts: how small language models indent code, choose ordinal suffixes and write in
capitals*. Every number and plotted value in the paper can be regenerated from the saved outputs on a CPU in a few
minutes, without network access, and checked by an independent script (`make check`).

## What the paper finds

Three studies asked how small language models make one small decision each, and a fourth asked whether the
mechanisms share parts. Each study first explored under a frozen charter (exploratory), then froze predictions with
thresholds and layer-matched random controls in a protocol and tested them once on sealed held-out material
(prospective). Because the predictions were written after exploration, these tests show that each explored account
repeats on new material; across the three studies' two rounds each and the cross-study test, 151 of
153 held. A third round of the indentation study re-ran every test that used natural code, on
permissively licensed code and newly generated programs that can be redistributed, and added a frozen test of a
second role of head L11H9: 49 of 49 tests held.

- **Indentation** (Qwen3-0.6B-Base and OLMo-2-0425-1B; folder `D3-indentation`). Dedents reach the model by two
  routes. After `return` and similar statements the keyword is read directly from the decision token through many
  heads. Unmarked dedents depend on a small set of late heads, chief among them one that attends to the line-ending
  token of an open block's header and copies that header's indent; which open header it attends to sets how far to dedent, and steering its attention
  moves the prediction.
- **Ordinal suffixes** (Qwen3-0.6B-Base, OLMo-2-0425-1B and five further models; `D4-ordinal-suffixes`). To write
  "112th", an early previous-token head brings the tens digit to the last digit, the first MLP reads it, and a late
  MLP writes a teen override towards "th", while a different MLP writes the last digit's own suffix. Where a tokeniser
  puts the tens digit in another token the signal arrives too faint, and a thousands separator repairs the errors.
  Two of the second-round predictions failed (both cells of one prediction in Qwen3-1.7B-Base).
- **Capitals** (Qwen3-1.7B-Base and Gemma-4-E2B; `D5-capitals`). The next word is chosen in a lower-case-like form
  and turned into capitals in the last layers along one direction, written by a few late components, fed by heads
  that carry the passage's case, and read only by the MLPs after it.
- **Shared parts** (Qwen3-0.6B-Base; `X1-shared-parts`). A prospective test (22 of 22 predictions held) finds little
  sharing: the early heads the studies named are general movers of the previous or the current token, and only one
  late head, L17H6, takes part in both line-breaking (the companion preprint *Counting by dilution and a width
  stored at the newline: a partial account of line-breaking in Pythia-410M*) and indentation; a second late head,
  L11H9, serves both the ordinal suffix and the dedent after a terminal statement (confirmed in D3's third round).

## Layout

| Path | Contents |
|---|---|
| `paper/` | The paper's LaTeX source (`main.tex`, `refs.bib`, `tables/`), the generated values (`numbers.tex`, `fig-*.dat`), the number build (`scripts/make_numbers.py`), the independent check (`scripts/check_numbers.py`) and its last report (`checks/`) |
| `D3-indentation/` | Indentation: charters and protocols of the three rounds (`CHARTER.md`, `PROTOCOL.md`, `CHARTER-2.md`, `PROTOCOL-2.md`, `CHARTER-3.md`, `PROTOCOL-3.md`, with the plan `ROUND-3-PLAN.md`), the freeze record (`FROZEN.md`), dated amendments, reports, `code/`, `data/`, the single-study paper's number scripts and source (`paper/v2-2026-11-09/`), and after `make unpack` its saved outputs (`outputs/`) |
| `D3-indentation/round3/` | The third round: `code/` (byte-identical copies of the frozen scripts it re-ran, and its own scripts), `data/` (the permissively licensed natural code with its licence texts, and new generated programs), the round's independent check (`CHECK-3.md`) and, after `make unpack`, its outputs |
| `D4-ordinal-suffixes/` | Ordinal suffixes: the same, with `data2/` for round 2, the exploration notebook (`NOTEBOOK.md`, which its paper check reads) and `paper/v3-2026-11-09/` |
| `D5-capitals/` | Capitals: the same, with `paper/v2-2026-11-08/` |
| `X1-shared-parts/` | The cross-study test: charter, protocol, freeze record, report, `code/`, `data/` and outputs |
| `D1-line-breaking/` | Four helper modules of the companion line-breaking study that the capitals and cross-study loaders import (unchanged) |
| `data/saved-outputs/` | The saved outputs the paper's numbers come from, as compressed archives (see its README) |
| `SEALED-MATERIAL.json` | Held-out material that is still unread, kept out of this release (below) |
| `models/models.json` | The model checkpoints by Hugging Face repository and revision, with every file's SHA-256 |
| `tools/` | Unpacking, freeze checking, sealing, number comparison and model download |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the studies' originals (redacted saved outputs included), with the original's SHA-256, and every frozen file not shipped |
| `THIRD-PARTY.md`, `THIRD-PARTY-ROUND3.tsv` | The GitHub code and Wikipedia text the studies used and their licences; every shipped round-3 source file with its repository, commit, licence and SHA-256 |

The studies keep their own folder names and internal layout, because their frozen code finds its material and the
other studies' folders by relative path. The paper's number build runs the three single-study papers' own number
builders on the saved outputs and reads the cross-study test's raw run files; its check runs the three single-study
papers' independent checkers on the paper's values and recomputes every cross-study value from the raw files.

## Reproducing the paper's numbers (CPU, no network)

```sh
make setup        # .venv with Python 3.12 and the pinned packages (uv; see ENVIRONMENT.md)
make check
```

If you cloned the repository, first download the saved-output archives attached to the v1.0 release into
`data/saved-outputs/` (their SHA-256 are in `SHA256SUMS`). `make check` then:

1. unpacks the saved outputs and the compressed inputs, checking every file's SHA-256 (`tools/unpack.py`), creates
   the links round 3 reads its material through (`tools/make_links.py`) and makes the frozen files read-only again,
   as they were in the studies (`tools/seal_frozen.py`);
2. checks every file named in the four freeze records (`tools/check_frozen_tables.py`; the third-party parts that are
   not shipped are listed with their frozen SHA-256 instead);
3. regenerates `paper/numbers.tex` (3,298 values) and the figure data from the saved outputs and compares them with
   the shipped copies, which the manuscript was built with (`tools/compare_numbers.py`);
4. runs the paper's independent check, which shares no code with the number build: `1,276 checks, 0 failed`. It
   includes D3's round-2 checker and round 3's independent check, which recompute their values from the raw run
   records. The paper's own check record (`paper/checks/`) lists 1,277: one more check counts the parameters of
   the Gemma-4-E2B weights and runs only where they are in the Hugging Face cache.

It needs NumPy and PyTorch (to read the saved `.pt` records); under 5 GB of memory and about ten minutes on an
ordinary CPU. Any Python 3.12 environment with `numpy` and `torch` works: `make check PYTHON=/path/to/python`.

`make paper` builds the PDF from `paper/` (LaTeX with latexmk). `make tables` recopies the single-study papers'
table layouts into `paper/tables/`.

## Rerunning the experiments (GPU)

The studies ran on one 6 GB NVIDIA GPU (compute capability 8.9) with the environment in `ENVIRONMENT.md`. Each
study's `code/` folder holds every script of both rounds (D3's third round: `D3-indentation/round3/code/`); the
protocols name the commands of each confirmation run, and the queue scripts (`code/*.sh`) hold the sequences that
were run. `code/run.sh` in each study runs one script with a memory limit and stops it after 20
minutes; single runs took from under a minute to that limit, and long confirmations resumed from saved records.
Set `PYTHON` to your environment's interpreter before using the shell scripts.

Models are loaded by Hugging Face Hub name through the Hugging Face cache:

```sh
python tools/fetch_models.py --list           # the eight checkpoints, their revisions and sizes (about 28 GB in all)
python tools/fetch_models.py qwen3-0.6b-base  # download one at the recorded revision and check every file
```

Gemma-4-E2B is gated on the Hub (accept its terms and log in first). The OLMo-2-0425-1B training checkpoints (about
5.9 GB each) are needed only for the exploratory checkpoint analyses of indentation and ordinal suffixes, whose
outputs are included; their branch names are in `models/models.json`.

**What can be rerun from the release alone, and what needs refetching first:**

- **As shipped:** D3's runs on generated programs (rounds 1 and 2) and on the stdlib part; all of D3's third round
  (its natural code is shipped with its licences; `make unpack` restores the raw files and the links); D4's ordinal
  runs and D5's prose runs (except the parts held back as unread sealed material, below); X1's runs on generated
  code, prose, ordinals and line-breaking text.
- **After refetching:** every run on the natural code of D3's rounds 1 and 2 and of X1 (D3 `newcode` and `r2code`,
  X1 `code`), whose chunks are not shipped (`THIRD-PARTY.md`). `python tools/refetch_old_code.py` downloads the
  listed source files again and keeps those whose SHA-256 still matches; the build scripts named in that tool then
  rebuild the parts, and the frozen loaders check them against `FROZEN.md`. A file changed or deleted upstream since
  November 2026 cannot be recovered this way, and a part that misses one cannot be rebuilt exactly. The saved outputs
  of these runs are shipped, redacted, and the paper's numbers and checks are computed from them.

Outputs are bitwise reproducible only with the same GPU, driver and library builds; on other hardware the values can
differ in the last digits, which the frozen thresholds were not set to depend on.

## What is not included

- **Unread sealed material** (`SEALED-MATERIAL.json`). Four held-out files are left out so they can still serve as
  held-out material for later work: D3's round-2 reserve of 4,000 generated programs, D4's round-1 reserve part, D4's
  round-2 frame file (it holds five date frames no model has read; the 19 frames the tests used appear in the
  round-2 outputs), and D5's round-2 heading items (items 0–499 of 1,335 were read). Each file's SHA-256, size and
  line count are recorded, and its SHA-256 equals the one in its study's `FROZEN.md`, so it can be checked when it
  is released. The paper quotes two counts of the heading file; the number scripts read them from
  `SEALED-MATERIAL.json`. Without these files, D4's round-2 confirmation and D5's round-2 heading runs cannot be
  rerun from scratch; their saved outputs are included.
- **The GitHub code of D3's rounds 1 and 2 and of X1** (chunks and source files): many of the source files have no
  licence that allows redistribution and some are copyleft. Fetch records and chunk digest lists ship in their place,
  the saved outputs are redacted (repository names replaced by labels, words of code by `[redacted]`), and
  `RELEASE-CHANGES.md` lists every such file. See `THIRD-PARTY.md`.
- The Wikipedia articles of the companion line-breaking study from which D5's paragraphs and headings were built
  (D5's built parts are included).
- Saved outputs the paper's numbers and checks do not read (other exploratory runs, smoke tests and calibration runs), logs, the
  exploration notebooks other than D4's, model weights, and two other authors' papers that a reference check reads
  (replaced by the short passages it searches).

Frozen files not shipped are listed with their SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the studies' files

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: local paths
in shell scripts became variables, records had local paths and tool names replaced, the paper's number scripts point
at the release's folders, D3's paper check compares frozen digests through `release-changes.json` (so a frozen file
changed for the release passes only as exactly the recorded change), D5's number scripts read the two counts of
the unread heading file from `SEALED-MATERIAL.json`, and the scripts that counted chunks of the unshipped third-party
parts read the chunk digest lists. Data are unchanged; saved outputs are unchanged except for the redaction of
third-party code described above. Times in the records are
local times (IST), as the studies wrote them.

## Use of language-model assistants

Language-model assistants were used throughout this work, under the author's direction, to help design the studies,
to write and run the code for the exploration, the experiments and the analyses, to draft and revise the text and to
check the references against their online records. All reported results come from the saved study outputs, and every
number in the paper was generated from those outputs and checked by an independent script.

## Licence and citation

MIT (`LICENSE`) for the code and the studies' records and outputs; the third-party material in `THIRD-PARTY.md`
keeps its own terms.

Richard Jeevash. *Small decisions, mostly separate parts: how small language models indent code, choose ordinal
suffixes and write in capitals*. Preprint, version 5 (2026-11-11), 2026. See `CITATION.cff`; the
preprint's link will be added here when it is public.
