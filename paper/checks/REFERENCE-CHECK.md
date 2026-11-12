# Reference check, combined paper (v1 record of 9 November 2026, with a v4 addendum at the end)

Every entry in `refs.bib` (58) is listed with how it was checked. Machine checks are in `scripts/check_numbers.py` (section 5) and pass: each arXiv entry's identifier, title, first author and year are compared with the saved arXiv record `sources/arxiv-api.xml` (fetched 9 Nov 2026 from the arXiv API, 143 KB, one call covering all 54 identifiers). Each non-arXiv entry is compared with `sources/web-records.json`. Venues that the arXiv record does not state were checked by hand as noted.

Downloads this session: `sources/arxiv-api.xml` 143 KB; `sources/html-2605.08348.html` 651 KB; `sources/html-2607.12279.html` 120 KB; `sources/ar5iv-1506.02078.html` 150 KB. Two LessWrong pages and four venue look-ups were read through the web without saving. Total about 1.1 MB.

## Flags (nothing quietly changed)

1. **Xu 2026 (arXiv:2606.05378): title was wrong in my first draft** of `refs.bib` and was caught by the check. The correct title is "Pattern Selectivity is Not Task-Causal Structure: A Cross-Architecture Mechanistic Study of Composed-Task Circuits in 1B-Class Language Models". Corrected to the arXiv record.
2. **Dillitzer et al. 2026 (arXiv:2608.03711).** The D5 v2 sentence ("models find all-capitals text harder", citing this paper) does not match the paper, as PRIOR-WORK-CHECK.md said. The abstract (saved record) says upper-case or alternating-case target spans concentrate attention, and that accuracy is not improved and can fall in high-entropy contexts such as alternating case. The combined paper says exactly that and adds that it concerns case in the input, not writing in capitals. The arXiv comment says "Accepted at ECCV 2026"; the entry is cited as an arXiv preprint because the proceedings entry could not be confirmed.
3. **Bassi and Tomar 2026**: venue added from the arXiv journal reference, "Mechanistic Interpretability Workshop at the 43rd International Conference on Machine Learning, Seoul, South Korea, 2026". D3 v2's entry lacked it.
4. **Li and Subramani 2026 (arXiv:2605.08348, v3 7 Oct 2026): checked in the full text** (`sources/html-2605.08348.html`): the five models are Gemma 2 2B and 2B IT, Llama 3.2 3B and 3B Instruct, and **Qwen3 4B**, plus OLMo-2-1B pre-training checkpoints. The paper also finds the shared components are mostly MLP blocks, and the few shared heads have general-purpose roles; the combined paper says this.
5. **Dunefsky, Gurnee and Ameisen 2026 (arXiv:2607.12279): checked in the full text** (`sources/html-2607.12279.html`). The model is Llama-3.1-70B-Instruct. Tasks: sentence planning, embedded tweets, DNA sequences of the same length, coin-toss sequences, SHA-256 hashes, haikus, rounding numbers, and **aligned cells in ASCII tables**. The paper says "formatting ASCII tables" in the abstract and has a section on it, so citing it for a formatting task is accurate; the combined paper words it as "aligning cells in ASCII tables".
6. **Janiak et al. 2023**: the PRIOR-WORK-CHECK wording "ablating it drops top-1 from 100% to 4%" is a comparison of a clean and a corrupted run, not a head ablation. The combined paper cites the post only for a head that attends to earlier tokens of similar indentation.
7. **Elhage et al. 2021** ("A Mathematical Framework", skip-trigram predicting `else`/`elif`, suggested in PRIOR-WORK-CHECK.md) was **not added**: I could not confirm the `else`/`elif` example in the source in this session.
8. **Not added** (named as "lesser" in PRIOR-WORK-CHECK.md, no identifiers given): Bayazit et al. (Findings of EMNLP 2026), Hiraoka and Inui (Findings of EMNLP 2025), Land and Meister 2026, Ameisen et al. 2025 ("Circuit Tracing"), the Hugging Face Qwen3-1.7B steering-vector repository and the September 2026 blog post. They can be added once checked.
9. The prior-work check's four arXiv identifiers it did not give (Levy and Geva, Lan et al., Ferrando and Costa-jussà, Mondorf et al.) were found and checked against the arXiv record: 2410.11781, 2311.04131, 2410.06496, 2410.01434.

## Entries

Carried over from D3 v2, D4 v3 and D5 v2 (already checked there against saved abstract pages; checked again here against the arXiv record). Duplicates merged: `olmo2025olmo2` into `olmo2024olmo2`, `belrose2023tunedlens` into `belrose2023tuned`. arXiv identifiers that were only in a `note` (Pythia, Meng et al., Zhang and Nanda, Hanna et al.) were added as `eprint` so the machine check covers them.

| Key | Identifier | How checked |
|---|---|---|
| biderman2023pythia | arXiv:2304.01373; ICML 2023 (PMLR 202) | arXiv record; venue from D3/D4 |
| radford2019language | technical report | D3 v2 / D4 v3 REFERENCE-CHECK (PDF); web-records.json |
| qwen2025qwen3 | arXiv:2505.09388 | arXiv record |
| gemma2026gemma4 | arXiv:2607.02770 | arXiv record |
| meng2022locating | arXiv:2202.05262; NeurIPS 2022 | arXiv record (comment: NeurIPS 2022) |
| heimersheim2024patching | arXiv:2404.15255 | arXiv record |
| zhang2024patching | arXiv:2309.16042; ICLR 2024 | arXiv record (comment: ICLR 2024) |
| bassi2026geometry | arXiv:2607.04167; ICML 2026 MI workshop | arXiv record incl. journal reference |
| gao2025weightsparse | arXiv:2511.13653 | arXiv record |
| olmo2024olmo2 | arXiv:2501.00656 | arXiv record |
| benallal2025smollm2 | arXiv:2502.02737 | arXiv record |
| hanna2023greaterthan | arXiv:2305.00586; NeurIPS 2023 | arXiv record (comment) |
| nikankin2025heuristics | arXiv:2410.21272 | arXiv record |
| singh2024tokenization | arXiv:2402.14903 | arXiv record |
| wang2023ioi | arXiv:2211.00593; ICLR 2023 | arXiv record; venue from D4 |
| goldowskydill2023path | arXiv:2304.05969 | arXiv record |
| belrose2023tuned | arXiv:2303.08112 | arXiv record |
| geva2021kv | arXiv:2012.14913; EMNLP 2021 | arXiv record (comment) |
| alain2016probes | arXiv:1610.01644 | arXiv record |
| turner2023steering | arXiv:2308.10248 | arXiv record |
| olsson2022induction | arXiv:2209.11895 | arXiv record |
| nostalgebraist2020logitlens | LessWrong, 2020 | D5 v2 saved page record; web-records.json |
| wendler2024llamas | arXiv:2402.10588 | arXiv record |
| panickssery2023caa | arXiv:2312.06681 | arXiv record |
| marks2023geometry | arXiv:2310.06824 | arXiv record |
| park2023linear | arXiv:2311.03658 | arXiv record |
| belrose2023leace | arXiv:2306.03819 | arXiv record |
| dillitzer2026casing | arXiv:2608.03711 | arXiv record and abstract (flag 2) |
| powalski2020unicase | arXiv:2010.11936 | arXiv record |

New in this paper (from PRIOR-WORK-CHECK.md):

| Key | Identifier | How checked |
|---|---|---|
| rai2025interference | arXiv:2507.00322; NeurIPS 2025 | arXiv record (comment: accepted for NeurIPS 2025) |
| miller2025codecompletion | arXiv:2502.18499; AAAI 2025 workshop | arXiv record and abstract (comment names the workshop) |
| janiak2023polysemantic | LessWrong, 9 Nov 2023 | page read: title, three authors, date, indentation claim (flag 6) |
| ebrahimi2020dyck | arXiv:2010.04303; Findings of EMNLP 2020 | arXiv record incl. journal reference |
| yao2021bounded | arXiv:2105.11115; ACL 2021 | arXiv record (comment: ACL 2021) |
| friedman2024illusions | arXiv:2312.03656; ICML 2024 | arXiv record (comment: ICML 2024) |
| sharma2026dyck | arXiv:2604.22128 | arXiv record and abstract (masking top-of-stack attention hurts long-distance accuracy) |
| karpathy2015visualizing | arXiv:1506.02078 | arXiv record; full text: cells "with increasing strength as the indentation of a block of code increases" |
| gould2024successor | arXiv:2312.09230; ICLR 2024 | arXiv record; ICLR 2024 from the OpenReview conference PDF and ICLR poster page |
| levy2025digits | arXiv:2410.11781; NAACL 2025 short, pp. 385–395, doi:10.18653/v1/2025.naacl-short.33 | arXiv record; ACL Anthology entry |
| stolfo2023arithmetic | arXiv:2305.15054; EMNLP 2023 | arXiv record (comment: EMNLP 2023) |
| feucht2024erasure | arXiv:2406.20086; EMNLP 2024, pp. 9727–9739, doi:10.18653/v1/2024.emnlp-main.543 | arXiv record; ACL Anthology entry |
| lad2024stages | arXiv:2406.19384 | arXiv record; no venue found, cited as preprint |
| baeumel2025modular | arXiv:2508.02513 | arXiv record |
| lan2024sequence | arXiv:2311.04131; EMNLP 2024, pp. 12576–12601, doi:10.18653/v1/2024.emnlp-main.699 | arXiv record; ACL Anthology entry |
| finlayson2021agreement | arXiv:2106.06087; ACL-IJCNLP 2021 | arXiv record (comment) |
| ferrando2024agreement | arXiv:2410.06496; Findings of EMNLP 2024 | arXiv record (comment) |
| merullo2024word2vec | arXiv:2305.16130; NAACL 2024 | arXiv record and abstract (uppercasing is one task; additive feed-forward update) |
| bloom2024saelogitlens | LessWrong / AI Alignment Forum, 11 Mar 2024 | page read: authors Joseph Bloom and Johnny Lin, date, case features |
| todd2024function | arXiv:2310.15213; ICLR 2024 | arXiv record (comment) |
| hendel2023taskvectors | arXiv:2310.15916; Findings of EMNLP 2023 | arXiv record (comment) |
| saji2025romanlens | arXiv:2502.07424; Findings of ACL 2025, doi:10.18653/v1/2025.findings-acl.1354 | arXiv record incl. journal reference and DOI |
| merullo2024reuse | arXiv:2310.08744; ICLR 2024 | arXiv record and abstract (78% head overlap, GPT-2 Medium) |
| li2026circuits | arXiv:2605.08348 | arXiv record; full text (flag 4) |
| xu2026selectivity | arXiv:2606.05378 | arXiv record and abstract (flag 1) |
| dunefsky2026countdown | arXiv:2607.12279 | arXiv record; full text (flag 5) |
| gurnee2024universal | arXiv:2401.12181 | arXiv record |
| gurnee2026manifolds | arXiv:2601.04480 | arXiv record; entry as in D1 v3 |
| mondorf2025compositions | arXiv:2410.01434; ACL 2025 | arXiv record (comment: ACL 2025 main) |

The cited GPT-2 report's institution field names its publisher; no product or company name appears in the paper's body.

## Addendum for v4 (10 November 2026)

Changes to `refs.bib`, none to the list of 58 entries:
- `belrose2023leace`: venue added, NeurIPS 2023 (confirmed on the NeurIPS 2023 proceedings page and in a public index; the arXiv record states no venue).
- `marks2023geometry`: venue added, Conference on Language Modeling (COLM) 2024, from the arXiv record's comment; year now 2024.
- `park2023linear`: venue added, ICML 2024, from the arXiv record's journal reference; year now 2024.
- `dillitzer2026casing`: note "Accepted at ECCV 2026" added, as the arXiv record's comment says; still cited as an arXiv preprint because the proceedings entry was not confirmed.
- `li2026circuits`: the capital "M" after the question mark is protected.
- arXiv-only entries all use the form "arXiv preprint arXiv:…" (they were split between two forms).

Cited-work claims the v3 reviewer could not confirm from abstracts, checked in the full texts (PDFs from arXiv: 2511.13653 4.0 MB, 2507.00322 2.3 MB, 2502.18499 0.8 MB; read, not saved here):
- **Gao et al. 2025** (2511.13653), tasks "include indentation": supported. The table of hand-made tasks lists "indent for" (the correct indentation following a `for` loop) and a task on indentation after `while True` or `return True`.
- **Rai et al. 2025** (2507.00322), "late heads that attend to the still-open bracket" in "pretrained code models": partly supported; flagged and reworded. Seven models are studied (GPT-2 Small to XL, CodeLlama-7b, Llama2-7b, Pythia-6.9b), so "code models" was too narrow, and the attention finding is reported for CodeLlama-7b's generalising heads: L30H0 attends to the first function-name token and L30H16 to the open parenthesis "that has yet to be closed". The paper now says a late head in a code model.
- **Miller et al. 2025** (2502.18499), "name Python indentation as future work": partly supported; flagged and reworded. The discussion says similar counting mechanisms "might also be needed for managing indentation in Python" and that future work will look at whether a code model reuses the same components across tasks and languages. The paper now says the authors suggest this and propose to test such reuse. The abstract's "attention heads that keep track of the number of already closed parentheses" supports the other half of the sentence.
- **Li and Subramani 2026** (2605.08348), "including Qwen3-4B": supported by the saved full text (`sources/html-2605.08348.html`, model table).
- **Gemma 4 Technical Report** (2607.02770): the arXiv record's abstract gives the family range as 2.3B to 31B parameters; the model card of Gemma-4-E2B gives 2.3B effective and 5.1B with embeddings (`sources/gemma-4-E2B-model-card.txt`). v3's "under two billion parameters" was wrong for this model and is corrected.
