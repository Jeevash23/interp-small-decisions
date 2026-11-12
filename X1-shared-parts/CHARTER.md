# X1 charter: do the parts found in four discovery studies overlap inside one model?

Written 8 November 2026, 09:45–10:05 IST, after reading the reports, protocols and notebooks of D1-X, D3, D4 and D5, and before any X1 material was built or any model was run for X1. Frozen with its SHA-256 in `FROZEN.md`. Everything done under this charter is **exploratory** until a frozen `PROTOCOL.md` is run once on sealed held-out material.

## Question

Four studies in this area each found attention heads with a role in one behaviour of Qwen3 models: line-breaking (D1-X), Python indentation (D3), ordinal suffixes (D4) and writing in capitals (D5). Each study tested its heads only on its own task. **Are these parts task-specific, or does one model reuse the same heads across these behaviours?** Concretely, when a head group named in one study is switched off, does it change the measure of another study's task more than layer-matched random heads do?

A clear "the parts are separate" is a full answer. So is a partial overlap confined to one or two heads.

## Leads (from reading the four studies; none tested before this charter)

1. **L17H6** is one of D1-X's four readers (line-breaking, Qwen3-0.6B) and is in D3's CARRY and TERM groups (dedent after `return`). Both roles move a fact about the previous line to the decision at the end of a line.
2. **The newline as a scratchpad.** D1-X's newline writers (L5H12, L5H13, L6H2) write a stored line length at newline tokens. In D3 the decision token is itself a newline token. Do the writers also carry indent or previous-line facts that D3's measure needs, and do D3's groups matter for line-breaking?
3. **Early heads that move local surface facts.** D4's L0H2 moves the tens digit to the last digit (partner L1H6). D5's L0H3 (Qwen3-1.7B) reads each token's own case; in D4's exploration notebook L0H3 was also the top head when only the last digit changed (Qwen3-0.6B). D1-X's counters (L4H0, L4H1, L5H13, L6H2, L7H15) are early-to-mid heads. Is each early head task-specific, or a general mover of the previous or current token?
4. **L11H9** (added while reading for this charter): in D4's exploration it appeared in all three patching scans as a general suffix head downstream of the teen check, and it is in D3's CARRY and TERM groups.
5. **Hypothesis only, not a lead to confirm here:** content is chosen first and format (case, line break, indent) is applied late. X1 may describe where in depth the format heads sit; it does not test this.

## Models

- **Qwen3-0.6B-Base** (`Qwen/Qwen3-0.6B-Base`, on disk), bfloat16, SDPA attention, the main model. D1-X, D3 and D4 found their heads in it.
- **Qwen3-1.7B-Base** (on disk) comes in only where needed. D1-X showed that the two models use the same line-breaking heads by layer and index, and D5's L0H3 was found in 1.7B. Rule, fixed now: before any capitals measure in 0.6B is used, D5's frozen gate (`D5-capitals/code/gate.py`, D5's exploration part, 150 paragraphs) is run on Qwen3-0.6B-Base, with its output written to X1's folder. If 0.6B passes all three gate conditions, the capitals column uses 0.6B. If it fails any, the capitals column uses Qwen3-1.7B-Base with the head groups taken by index, and the 0.6B capitals values are reported as descriptive only. Other uses of 1.7B (for example D4's L0H2 by index) are allowed in exploration and must be named in the protocol before they are tested.

## Head groups (the rows)

Named exactly as in the source studies (layer L, query head H):

| Group | Heads | Source |
|---|---|---|
| COUNT | L4H0, L4H1, L5H13, L6H2, L7H15 | D1-X counters |
| CORR | L4H1 | D1-X token corrector |
| WRITE | L5H12, L5H13, L6H2 | D1-X newline writers |
| READ | L7H14, L15H3, L17H6, L22H3 | D1-X readers |
| DED | L23H2, L17H3, L17H4, L17H5, L18H6 | D3 block-end heads |
| CARRY | L17H4, L11H9, L5H6, L17H6, L16H2, L6H5 | D3 terminal-signal carriers |
| TERM | L11H9, L5H6, L17H6, L16H2, L23H6, L22H8, L23H3, L25H9, L19H6 | D3 terminal route |
| TEEN | L0H2, L1H6 | D4 teen copy |
| L0H2, L1H6 | singles | D4 |
| L0H3 | single | D5 (1.7B); D4 exploration (0.6B) |
| L17H6, L11H9, L17H4 | singles | heads shared between groups |
| READ−L17H6, CARRY−L17H6, TERM−L17H6 | groups without the shared head | to see whether an overlap rests on L17H6 alone |

Further groups may be added in exploration if a lead calls for them (logged in `NOTEBOOK.md`).

## Tasks and measures (the columns)

Each measure reuses the source study's code and definitions by importing it; no frozen file of another study is copied or changed.

- **LB, line-breaking** (D1-X `xm_common`): paragraphs wrapped greedily at width 60. *Width effect*: the change in log P(newline) at a letter-ending decision point when the previous line is shortened by at least 10 characters (`width_items`), and the share of it that survives an ablation. *Natural separation*: mean newline log-odds at true break points minus at near-end non-break points (as D1-X test G), and the share kept.
- **IND, indentation** (D3 `common.prepare`, `RootMasses`, `pyindent`): the renormalised candidate-indent distribution at the decision token. Change in mean log q(true) at unmarked dedents, at terminal-statement dedents and at same-level lines; and, on generated return/assignment pairs, the dedent score gap (as D3 c02).
- **ORD, ordinal suffixes** (D4 `common`): m = logit(last-digit suffix) − logit("th") at the number's last token. *Teen effect* (non-teen ending 1–3 minus teen x11–x13) and share kept; *last-digit pull* (mean m on non-teen items ending in 1–3) and share kept; accuracy by class.
- **CAPS, capitals** (D5 `d5common`): at word starts, D = log p(U) − log p(Lc) in upper-case passages (teacher-forced), and the reverse in lower-case passages, with share kept; top-1 case kept.
- **GEN, general damage** (control column): mean next-token loss on ordinary new prose. A group that raises general loss as much as it hurts a task is not called task-specific.

## Interventions and controls

- **Mean ablation** of a head's output (the input of the attention output projection), per head, with the mean taken over the task's own exploration material. The main scope is **all positions**; scopes restricted to the task's decision positions (newline/decision token, last digit, word start) may be used to locate an effect. Path or activation patching may be added where an ablation result needs a mechanism.
- **Random controls**: for each named group, groups that replace each head with a random head of the same layer, never a head named in any group above; at least 10 draws per group in exploration and at least 20 for each confirmatory test.
- **Overlap by chance**: for any shared-head claim, report the chance of the overlap under random placement within the layers involved.

## Material

No X1 analysis reads material any earlier study has read for results. Every held-out part is split off by hash and sealed (digest in `FROZEN.md`, file read-only, loader refuses it until `PROTOCOL.md` is frozen) **before** any model reads any X1 material.

- **Prose (LB, CAPS, GEN):** a fourth batch of English Wikipedia articles created 2025–2026, fetched by X1 with D1-X's selection rule and excluding every article in D1's three earlier batches (so none was read by D1, D1-X or D5). Split by article hash into `explore` and `held`. Capitals use only paragraphs that keep their length under case change (D5's rule). Licence CC BY-SA 4.0.
- **Code (IND):** new generated programs from D3's frozen generator (`gen_programs.G`) with seeds D3 never used (from 100,000), split by seed hash; and Python files from public GitHub repositories created on or after 1 October 2026 that are not among D3's 314 repositories, fetched with D3's file rule, split by repository hash. Nothing fetched is executed.
- **Ordinals (ORD):** D4's unread `reserve` part (its 13 reserve frames and reserve numbers by D4's own hash rule), split by number into `explore` and `held`. D4's hub handover named the reserve as possible held-out material for a follow-up.
- **D5's reserve part stays unread** (the author has not decided how to spend it).
- Downloads stay under about 300 MB in total (expected: under 20 MB).

## Candidate findings worth confirming

- A named group (or single head) whose ablation moves another task's measure beyond every layer-matched random group, by a margin that is a sizeable fraction of its effect on its own task (rough guide: at least a third), while GEN moves little.
- Or the reverse, a clean dissociation: a group that matters strongly for its own task and leaves the others within the random range. A matrix in which every group is task-specific is itself a candidate finding ("separate parts"), and is confirmed by the same kind of frozen test.
- For L17H6 and L11H9: an effect of the single head on both tasks, beyond random single heads of its layer.

## Budget and stop rules

- About 12 hours of wall-clock time from this freeze to a frozen protocol; GPU only on the local machine (6 GB NVIDIA GPU, 6 GB), under `flock ~/.cache/d1-gpu.lock` in a memory cage, at most about 20 minutes per lock hold, only with at least 6 GB of RAM available. No Modal, no cloud.
- Stop exploring and write the protocol when the matrix is complete with random controls and each lead has a clear reading (overlap, separation, or not measurable). Stop early, and write up without a confirmation, if a task's measure cannot be made to work in Qwen3-0.6B (and 1.7B where allowed) on X1's exploration material: for example, the capitals gate fails in both models.
- The confirmation runs once on the held-out parts. Crashes are fixed only for the crash and resumed; anything else is a dated amendment.

## Records

`FROZEN.md` (digests), `NOTEBOOK.md` (dated exploration log, dead ends included), `AMENDMENTS.md` if needed, `PROTOCOL.md`, `REPORT.md` with every number checked by an independent script. No commits, nothing published, no owner's name.
