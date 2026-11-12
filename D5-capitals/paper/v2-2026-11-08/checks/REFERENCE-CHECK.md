# Reference check: D5 write-up, v2 (8 November 2026)

v2 uses the same seventeen references as v1, unchanged; the table below is v1's check, which `scripts/check_numbers.py` repeats on every build (title, first author and year against the saved arXiv records). The cross-study test X1 is not a public paper; v2 describes it in the text as an unpublished test in the same series and quotes its held-out values from its saved summary, which the number check reads. The novelty statement is unchanged.

## v1 check

All seventeen references exist, and every arXiv ID points to the paper named. On 8 November 2026 each arXiv entry was checked against its record from the arXiv API (title, authors, first-version date, abstract, comments). The records are saved in `sources/arxiv-api.xml`, about 58 kB, and parsed into `sources/arxiv-abs-pages.json`. The one non-arXiv entry was checked against its web page, recorded in `sources/logit-lens-page.json`. `scripts/check_numbers.py` repeats the title, first-author and year comparison on every build.

Seven entries are copied unchanged from the D4 write-up (`../../../D4-ordinal-suffixes/paper/v1-2026-11-08/refs.bib`), whose reference check passed and which were rechecked here. The other ten are new.

| Key | Source checked | Result | What the paper uses it for |
|---|---|---|---|
| `gemma2026gemma4` | arXiv 2607.02770 (Gemma Team, 2 Jul 2026) | Verified (from D4) | Names Gemma 4 |
| `qwen2025qwen3` | arXiv 2505.09388 (60 authors, 14 May 2025) | Verified (from D4) | Names Qwen3 |
| `benallal2025smollm2` | arXiv 2502.02737 (22 authors, 4 Feb 2025) | Verified (from D4) | Names SmolLM2 |
| `biderman2023pythia` | arXiv 2304.01373 (13 authors, 3 Apr 2023); PMLR v202 per D1's check | Verified (from D4) | Names Pythia |
| `meng2022locating` | arXiv 2202.05262 (4 authors, 10 Feb 2022); comment "NeurIPS 2022" | Verified (from D4) | Activation patching as a method |
| `heimersheim2024patching` | arXiv 2404.15255 (2 authors, 23 Apr 2024) | Verified (from D4) | Activation patching practice |
| `zhang2024patching` | arXiv 2309.16042 (2 authors, 27 Sep 2023); comment "ICLR 2024" | Verified (from D4) | Activation patching practice |
| `nostalgebraist2020logitlens` | LessWrong post "interpreting GPT: the logit lens", by nostalgebraist, 31 Aug 2020 (fetched 8 Nov 2026) | Verified; new | The logit lens: "reads each layer's residual stream through the model's own output layer". The post applies GPT's final output projection to intermediate-layer activations |
| `belrose2023tunedlens` | arXiv 2303.08112 (8 authors, 14 Mar 2023) | Verified; new | "The logit lens is known to be brittle". The abstract calls the logit lens "often brittle" |
| `wendler2024llamas` | arXiv 2402.10588 (4 authors, 16 Feb 2024) | Verified; new. Cited as arXiv only, because the saved record gives no venue | "Llama-2 models answering in another language first favour the English form of the answer in their middle layers". The abstract: middle layers "already allow for decoding a semantically correct next token ... but give higher probability to its version in English" |
| `turner2023steering` | arXiv 2308.10248 (7 authors, 20 Aug 2023) | Verified; new. Cited under its current arXiv title, "Steering Language Models With Activation Engineering" | Steering by contrasting activations on prompt pairs (the abstract's ActAdd) |
| `panickssery2023caa` | arXiv 2312.06681 (6 authors, 9 Dec 2023) | Verified; new | Steering by "averaging the difference in residual stream activations between pairs of positive and negative examples" (abstract) |
| `marks2023geometry` | arXiv 2310.06824 (2 authors, 10 Oct 2023); comment "Conference on Language Modeling, 2024" | Verified; new | "interventions along such directions can change what a model treats as true". The abstract reports interventions "causing it to treat false statements as true" |
| `park2023linear` | arXiv 2311.03658 (3 authors, 7 Nov 2023); journal reference ICML 2024 | Verified; new | "relate such directions to linear probing and steering". The abstract proves that its two formalisations "connect to linear probing and model steering" |
| `belrose2023leace` | arXiv 2306.03819 (6 authors, 6 Jun 2023) | Verified; new | Concept removal as a test of use; full linear erasure as a stronger test. The abstract mentions "removing a concept to observe changes in model behavior" |
| `dillitzer2026casing` | arXiv 2608.03711 (4 authors, 4 Aug 2026); comment "Accepted at ECCV 2026" | Verified; new | "letter case shifts where attention goes". The abstract: "letter casing modulates internal attention allocation" |
| `powalski2020unicase` | arXiv 2010.11936 (2 authors, 22 Oct 2020) | Verified; new | "a masked language model that represents case separately from the word does much better on fully upper-cased text than a standard cased one". The abstract reports the UniCase model "works much better when ... all tokens are uppercased (+5.88 point)". The paper quotes no number from it |

## Notes

- No reference was found not to exist, and the paper takes no quoted number from any reference.
- New entries are cited as arXiv preprints even where a venue is known (Park et al., ICML 2024; Marks and Tegmark, COLM 2024; Dillitzer et al., ECCV 2026). A venue can be added on the author's request.
- The novelty statement ("We know of no account of how a model writes in capitals") rests on two web searches on 8 November 2026. They found the two behavioural studies cited and no mechanistic account. It is a statement of what we found, not a full survey.
