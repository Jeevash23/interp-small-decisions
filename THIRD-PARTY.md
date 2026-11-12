# Third-party material in this release

The studies measured models on text and code written by other people. The MIT licence in `LICENSE` covers the code
and records written for these studies, not the material listed here, which keeps its own terms.

## Python code from public GitHub repositories, shipped (D3 round 3)

`D3-indentation/round3/data/code/` holds the natural code of D3's third round: 999 Python files from
274 public repositories created on or after 1 September 2026, each fetched at a pinned commit and each from
a repository whose licence GitHub reports as MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0 or Unlicense
(files: Apache-2.0 395, MIT 269, BSD-3-Clause 135, CC0-1.0 91, BSD-2-Clause 63, Unlicense 46). They ship as follows:

- `raw/<key>.py`: the source files, byte for byte, packed in `data/d3-round3-code-raw.tar.xz` (`make unpack` puts
  them in place);
- `part-held.jsonl`: the 2,764 chunks the round's tests read, cut from those files (each record names its
  repository, path, commit and licence);
- `licences/`: each repository's licence text as fetched (`<owner>__<repository>.LICENSE.txt`), and its `NOTICE`
  file where it has one (47 repositories);
- `fetched.jsonl`, `SHA256SUMS`, `fetch_summary.json`, `build_summary.json`, `dropped.jsonl`: the fetch record.
- `raw-not-shipped.tsv`: the 1 of the 1,000 fetched files left out of this release, with repository,
  path, commit, licence, SHA-256 and the reason; no chunk in `part-held.jsonl` comes from it, so no test read it. Its
  line stays in `fetched.jsonl` and `SHA256SUMS` as part of the frozen fetch record.

**`THIRD-PARTY-ROUND3.tsv`** lists every shipped round-3 source file: its path in this release, repository, path in
the repository, commit, licence, licence text (and notice) in this release, and SHA-256. Each file remains under its
repository's licence and copyright; the licence texts above carry the copyright notices. The round-3 saved outputs
(`D3-indentation/round3/outputs/`) repeat short pieces of this code (the first word of a line and of the line before)
and its repositories' names.

## Python code from public GitHub repositories, not shipped (D3 rounds 1 and 2, X1)

D3's first two rounds and X1 used chunks of Python files from public, non-fork repositories created on or after
1 September 2026 (D3) or 1 October 2026 (X1). Of the 1,760 source files those chunks came from, GitHub
reports 1,189 under permissive licences, 132 under copyleft or other licences (GPL-2.0,
GPL-3.0, AGPL-3.0, LGPL-2.1, EUPL-1.2), and 439 with no licence file or one GitHub could not identify
(`NOASSERTION`), which are not licensed for redistribution. **None of this code is in the release.** In its place:

| Folder | Fetch record | Chunk digest lists (chunks) | Fetch script |
|---|---|---|---|
| `D3-indentation/data/newcode/` (round 1) | `fetched.jsonl`, `fetch_summary.json`, `build_summary.json`, `SHA256SUMS` | `part-explore.chunks.jsonl` (1,385), `part-conf.chunks.jsonl` (1,606) | `D3-indentation/code/fetch_newcode.py`, `build_material.py` |
| `D3-indentation/data/r2code/` (round 2) | `fetched.jsonl`, `fetch_summary.json`, `build_summary.json` | `part-held.chunks.jsonl` (2,712) | `D3-indentation/code/r2_fetch_code.py`, `r2_build.py` |
| `X1-shared-parts/data/code/` | `fetched.jsonl`, `fetch_summary.json`, `build_summary.json`, `SHA256SUMS` | `part-explore.chunks.jsonl` (971), `part-held.chunks.jsonl` (863) | `X1-shared-parts/code/fetch_code.py`, `build_material.py` |

`fetched.jsonl` lists every fetched file with its repository, path, creation date, the licence GitHub reports (SPDX
identifier) and its SHA-256. A chunk digest list has one line per chunk of the part: repository, path, file key,
chunk index, length and the SHA-256 of the chunk's text; the paper's scripts take their chunk and repository counts
from it. `tools/refetch_old_code.py` downloads the listed files again and keeps those whose SHA-256 still matches,
and the build scripts then rebuild the parts, which can be checked against `FROZEN.md` and the digest lists.

The saved outputs of those runs are included with the code taken out of them: the repositories' names are replaced
by labels (`repo-0001` to `repo-0837`, numbered in the sorted order of the names in the three
`fetched.jsonl` files), and the words of code in D3's first-round behaviour records are replaced by `[redacted]`
except Python keywords. `RELEASE-CHANGES.md` lists every redacted file with both SHA-256.

## CPython standard library

`D3-indentation/data/stdlib/part-explore.jsonl` holds chunks of the Python 3.14 standard library (D3's exploration
only), which is distributed under the Python Software Foundation License Version 2 and the other terms in
`D3-indentation/data/stdlib/LICENSE-CPython-3.14.txt` (CPython's own `LICENSE` file). Copyright Python Software
Foundation; all rights reserved.

## English Wikipedia text

`D5-capitals/data/` (paragraphs and section headings) and `X1-shared-parts/data/prose/` hold text from English
Wikipedia articles created in 2025 and 2026, fetched through the Wikipedia API with their page IDs and revision IDs.
Wikipedia text is available under the Creative Commons Attribution-ShareAlike 4.0 licence (CC BY-SA 4.0); the
authors are the articles' contributors, listed in each article's history on Wikipedia (look an article up by the
`pageid` and `lastrevid` in each record). The saved outputs of D5 and X1 repeat single words from it.

## Other authors' papers

`D3-indentation/paper/v2-2026-11-09/sources/html-2607.04167.html` and `html-2511.13653.html` are not those papers:
each holds only the one to three short passages that the D3 paper's reference check searches, with a link to the full
text. The `sources/arxiv-*.json` and `arxiv-api.xml` files are arXiv metadata records (titles, authors, dates and
abstracts) used to check the reference lists. `paper/sources/gemma-4-E2B-model-card.txt` quotes one table row and
one paragraph of the model card of Gemma-4-E2B (Hugging Face `google/gemma-4-E2B`), whose parameter counts the paper's
number scripts read.

## Models

No model weights are included. `models/models.json` lists the checkpoints by Hugging Face Hub repository and revision;
each is distributed by its developers under the licence on its model card.
