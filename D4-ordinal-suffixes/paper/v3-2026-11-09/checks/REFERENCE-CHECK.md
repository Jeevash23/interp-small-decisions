# Reference check: D4 write-up, v2 (9 November 2026)

v2 keeps v1's twelve references, checked as below on 8 November, and adds seven, checked on 9 November 2026 (section at the end).

## v1 references (8 November 2026)

All twelve references exist and every arXiv ID points to the paper named. Each arXiv entry was checked on 8 November 2026 against its arXiv abstract page (title, authors, first-version date, abstract), saved in `sources/arxiv-abs-pages.json`. `scripts/check_numbers.py` repeats the title, first-author and year comparison on every build. Eleven entries are copied unchanged from reference lists that passed earlier checks: ten from the D1 line-breaking paper, v3 of 7 November (`../../../D1-line-breaking/paper/v3-2026-11-07/REFERENCE-CHECK.md`), and one from the A3 note of 7 November. One entry (`singh2024tokenization`) is new.

| Key | Source checked | Result | What the paper uses it for |
|---|---|---|---|
| `biderman2023pythia` | arXiv 2304.01373 (13 authors, 3 Apr 2023); PMLR v202 per D1's check | Verified | Names Pythia |
| `radford2019language` | OpenAI report PDF, checked for D1 on 6 Nov 2026 (not re-downloaded) | Verified (D1) | Names GPT-2 |
| `olmo2024olmo2` | arXiv 2501.00656 ("Team OLMo" + 42 authors, 31 Dec 2024) | Verified | Names OLMo 2. The chunks-of-three tokenisation is checked in this study's own tokeniser outputs, not taken from the paper |
| `qwen2025qwen3` | arXiv 2505.09388 (60 authors, 14 May 2025) | Verified | Names Qwen3 |
| `gemma2026gemma4` | arXiv 2607.02770 (Gemma Team, 2 Jul 2026) | Verified | Names Gemma 4 |
| `benallal2025smollm2` | arXiv 2502.02737 (22 authors, 4 Feb 2025) | Verified | Names SmolLM2 |
| `singh2024tokenization` | arXiv 2402.14903 (Aaditya K. Singh, DJ Strouse; 22 Feb 2024) | Verified; new entry | "the direction in which numbers are chunked changes arithmetic accuracy in large models, and commas, which force right-to-left chunks, help": the abstract compares left-to-right and right-to-left tokenisation for GPT-3.5 and GPT-4 and finds right-to-left, "enforced by comma separating numbers at inference time", largely improves performance |
| `hanna2023greaterthan` | arXiv 2305.00586 (3 authors, 30 Apr 2023); NeurIPS 2023 per D1's check | Verified | "circuits that compare years": the abstract studies GPT-2 small predicting end years greater than a start year, and finds final MLPs boost valid years (also cited for MLP-heavy accounts) |
| `nikankin2025heuristics` | arXiv 2410.21272 (4 authors, 28 Oct 2024) | Verified | "neurons that each respond to a numerical pattern": the abstract finds sparse neurons implementing heuristics that each identify a numerical input pattern and output matching answers |
| `meng2022locating` | arXiv 2202.05262 (4 authors, 10 Feb 2022); NeurIPS 2022 per D1's check | Verified | Activation patching (causal tracing) as a method |
| `heimersheim2024patching` | arXiv 2404.15255 (2 authors, 23 Apr 2024) | Verified | Activation patching practice; that methodological choices matter |
| `zhang2024patching` | arXiv 2309.16042 (2 authors, 27 Sep 2023); ICLR 2024 per D1's check | Verified | "methodological choices in patching and ablation can change which components appear important": the abstract finds that varying metrics and corruption methods "could lead to disparate interpretability results" |

## Notes

- The key `nikankin2025heuristics` carries 2025 in its name (from the A3 list), while the entry's year field is 2024, matching arXiv v1. The printed citation shows 2024. Not changed, because the key is not printed.
- No reference was found not to exist, and no quoted number is taken from a reference.


## References added in v2 (9 November 2026)

Each was checked against its arXiv abstract page, saved in `sources/arxiv-abs-pages.json` (title, all authors, first-version date, abstract). For `olsson2022induction` the claim rests on the paper's body, so the PDF of version 1 (about 10 MB; not kept in the paper folder) was read as well. Venues: `geva2021kv` is EMNLP 2021, as the comments field of its abstract page states. `wang2023ioi` is entered as ICLR 2023; the abstract page does not state a venue, so this was not confirmed against a source and should be checked before submission. The other five are cited as arXiv preprints.

| Key | Source checked | Result | What the paper uses it for |
|---|---|---|---|
| `wang2023ioi` | arXiv 2211.00593 (5 authors, 1 Nov 2022) | Verified (arXiv); venue not confirmed | Path patching as a method (used there to trace the indirect-object circuit) |
| `goldowskydill2023path` | arXiv 2304.05969 (4 authors, 12 Apr 2023) | Verified | Path patching as a method; the title names it |
| `belrose2023tuned` | arXiv 2303.08112 (8 authors, 14 Mar 2023) | Verified | The logit lens: the abstract describes the tuned lens as "a refinement of the earlier 'logit lens' technique" |
| `geva2021kv` | arXiv 2012.14913 (4 authors, 29 Dec 2020) | Verified | "MLPs as stores of input patterns that each promote output tokens": the abstract finds keys that correlate with textual patterns and values that induce distributions over the output vocabulary |
| `alain2016probes` | arXiv 1610.01644 (2 authors, 5 Oct 2016) | Verified | Linear probes as a method |
| `turner2023steering` | arXiv 2308.10248 (7 authors, 20 Aug 2023); the bib title is the current one on the abstract page | Verified | Steering by adding a direction to activations |
| `olsson2022induction` | arXiv 2209.11895 (26 authors, 24 Sep 2022); body of v1 read | Verified | "the kind of early head that induction heads also build on": the body describes induction heads formed by K-composition with an earlier "previous token head", and, in one small model, previous-token heads in layer 0 |
