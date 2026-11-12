# Reference check: D3 write-up, v2 (9 November 2026)

Ten references. The nine of v1 are unchanged; their check is in `../../v1-2026-11-08/checks/REFERENCE-CHECK.md` (all verified, none missing, no number taken from a reference). One entry is new in v2 and was checked on 9 November 2026 against its arXiv abstract page (saved in `sources/abs-2501.00656.html`, record in `sources/arxiv-abs-pages.json`). `scripts/check_numbers.py` repeats the title, first-author and year comparison for every arXiv entry on every run (all pass; `NUMBER-CHECK.md`).

| Key | Source checked | Result | What the paper uses it for |
|---|---|---|---|
| `olmo2025olmo2` | arXiv 2501.00656, "2 OLMo 2 Furious" (Team OLMo, Walsh, Soldaini, Groeneveld, Lo and others; first version 31 Dec 2024) | Verified; new | Names the OLMo 2 family and supports "whose developers release intermediate training checkpoints" (the abstract lists "thousands of intermediate checkpoints" among the released artifacts). The abstract covers the 7B, 13B and 32B models; the paper's statements about OLMo-2-0425-1B's stages, token counts and three stage-2 runs come from the names of its published checkpoints, not from this reference, and the text says so. |

No reference was found not to exist, and no number in the paper is taken from a reference. The paper's finding that the released OLMo-2-0425-1B weights are not a descendant of its published checkpoints is the paper's own measurement (weight distances from the downloaded files) and is not attributed to any source.
