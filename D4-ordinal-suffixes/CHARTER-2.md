# D4 charter, round 2 (depth): the full path from digits to suffix

Area 14 (mechanism discovery), study D4, second round. Written 8 November 2026, about 17:05 IST, and frozen before any model was run on round-2 material (digest in `FROZEN.md`). Requested by the author on 8 Nov at about 16:50 IST ("why not do the depth runs for these too … and try to get the complete picture"; `../DECISIONS.md`, 16:55). Round 1 (`CHARTER.md`, `PROTOCOL.md`, `REPORT.md`) is closed and unchanged.

Everything done under this charter is **exploratory**. A candidate found here becomes a result only after `PROTOCOL-2.md` is frozen and its predictions are tested once on held-out material sealed below.

## What round 1 and X1 established (starting points, not results of this round)

- Qwen3-0.6B-Base (one token per digit): mean-ablating L0H2 and L1H6 at the number's last token removes most of the teen effect at 3 and 4 digits (kept 0.430 and 0.459 on held-out material) but not at 2 digits. L0H3 carries the last digit's own suffix pull.
- X1 (`../X1-shared-parts/REPORT.md`, 22 of 22 held-out tests): L0H2 is a general previous-token head (it also matters for line-breaking), L0H3 a general current-token head, and L11H9 a suffix head that attends to the current token and holds part of both the teen effect and the last-digit pull (0.726 and 0.733 kept at 4 digits).
- OLMo-2-1B (numbers chunked in threes from the left): teen AUROC is low only where the tens digit sits in the token before the last digit (4 and 7 digits), and three layer-0 heads (L0H5, L0H0, L0H9) carry the teen signal only in that layout.

## Questions, in the overview's order of usefulness

1. **The Qwen3 path.** How do the tens-digit signal written by L0H2 (and L1H6) and the last-digit signal written by L0H3 reach the suffix decision? Which later components read them, does L11H9 or another mid or late part combine them, and at which layer does "teen, so th" become readable at the last position?
2. **The second route for two-digit teens** in Qwen3, which survived ablation of L0H2+L1H6 in round 1.
3. **OLMo-2 downstream.** After the layer-0 heads move the tens digit across the token boundary, what reads it? Is the 4- and 7-digit weakness a missing route (the tens digit never becomes available at the last token) or a weak one (it arrives but is read poorly)? Does the token account carry on to 8–10 digits (3+3+2, 3+3+3, 3+3+3+1)?
4. **Other one-token-per-digit models** on disk (SmolLM2-1.7B, Qwen3-1.7B-Base, Gemma-4-E2B): do they also carry the tens digit with an early previous-token head and the last digit with an early current-token head, so that the round-1 account predicts which heads their teen check depends on?
5. **Optional, not planned:** OLMo-2 training checkpoints. None are on disk. A download over about 300 MB goes to the overview first with its size; without a yes, this question is not run.

## Models

Qwen3-0.6B-Base (float32), OLMo-2-1B (bfloat16), SmolLM2-1.7B (bfloat16), Qwen3-1.7B-Base (bfloat16), Gemma-4-E2B (bfloat16, D2's text-only loader). All on disk; no downloads. Qwen3-1.7B-Base was not gated in round 1: it is gated on exploration items with round 1's gate rule (conditions 1 and 2) before any internal analysis, and dropped if it fails.

## Tools

- Logit lens at the last position (each layer's residual stream through the final norm and unembedding): the teen effect, the last-digit pull and the teen AUROC by layer.
- Direct logit attribution of every head and MLP at the last position to m and to the teen effect.
- **Path ablation:** a sender's mean-ablation change (for example L0H2 at the last position) is passed to one receiver only (one attention layer's input, one head's input, one MLP's input, or the unembedding directly), with every other component seeing the clean value. This splits a sender's effect by the routes it travels.
- Activation patching between numbers that differ in one digit, as in round 1, with chosen components already ablated (to find what carries the second route).
- Linear probes (logistic regression, cross-validated) for the tens digit, the last digit and teen status, by layer and position.
- Attention patterns, used only to choose candidate heads in other models before any ablation is run on them.
- Random controls for every intervention: random heads of the same layer or layers, random MLPs of nearby layers, same set size.

## Material

Round 2 needs held-out material that no model has read. Round 1's confirmation part was read in round 1, and X1 read D4's reserve frames and every reserve number of three to five digits. So:

- **New frames.** About 48 new frame sentences, written for this round in the round-1 families (rankings, positions, anniversaries, dates), are added by `code/r2_build_material.py` to `data2/frames2.jsonl`. Within each family they are split by the SHA-256 of `D4R2-frame-<text>`: the first half by hash order is **explore2**, the rest **held2**.
- **Numbers.** The number of digits decides what is fresh:
  - 2 digits: every two-digit number has been read; held-out items with 2 digits differ from earlier ones in their frames only.
  - 3–5 digits: too few unread numbers remain. Held-out items use numbers of round 1's confirmation and reserve classes (never round 1's exploration class) in held2 frames. They are new in their frames, not in their numbers, and are reported as such.
  - 6–7 digits: numbers of round 1's reserve class have never been read by any model (X1 used up to 5 digits). These are held2 numbers.
  - 8–10 digits: no D4 or X1 run has read any. Split by the SHA-256 of `D4R2-num-<N>` modulo 10: 0–5 explore2, 6–9 held2.
- **Exploration** may use round 1's exploration frames and numbers, explore2 frames, and explore2 numbers. It does not read held2 frames or held2 numbers.
- The builder, the frame file and the manifest are digested in `FROZEN.md` before any model reads them. `code/r2common.py` refuses held2 frames and held2 numbers until `PROTOCOL-2.md` exists and is read-only.

## Measures

As in round 1, at the number's last token: m = logit(suffix of the last digit) − logit("th"); **teen effect** = mean m (non-teen ending in 1–3) − mean m (teen, x11–x13); **last-digit pull** = mean m (non-teen ending in 1–3), as X1 defined it; teen AUROC between non-teen and teen m, ties averaged; accuracy. For an intervention, "kept" is the value after it divided by the value before. Path-ablation shares are the fraction of a sender's full ablation effect that one route carries.

## Candidate findings worth confirming

- (1) A small set of receivers (heads or MLPs) through which most of L0H2's (or L0H3's) effect passes, beyond random receivers of the same kind, and a layer range where the teen effect becomes readable by the logit lens; ideally one component that needs both signals (combines them).
- (2) A component or set that carries most of the two-digit teen effect that survives L0H2+L1H6 ablation, beyond random sets.
- (3) For OLMo-2, a probe or patching result that tells "not moved" from "moved but read weakly" at 4 and 7 digits, and the 8–10-digit behaviour the token account predicts.
- (4) For each extra model, heads chosen by attention alone whose ablation removes the teen effect (or the last-digit pull) beyond random heads.

Each must beat its random controls on exploration material and lead to predictions that could fail on held2 material.

## Budget and stop rules

- **Time.** Exploration from the freeze of this charter until 9 Nov 2026, 02:00 IST at the latest. `PROTOCOL-2.md` frozen by about 03:00; the confirmation runs once; `REPORT-2.md` and the write-up v2 follow. Aim: round 2 confirmed within about a day of this charter.
- **Per question.** A question with no candidate beating its random controls by the end of exploration is not tested; it is reported as an exploratory result or a null. Questions 1–3 come before question 4; question 4 is cut first if time runs short.
- **Confirmation stop.** If the shared GPU queue means the confirmation cannot finish by 9 Nov 18:00 IST, the jobs that did not run are reported as not run; nothing is re-run.
- **Compute.** Local machine only (6 GB NVIDIA GPU, 6 GB), $0, no Modal. Shared local machine: every job that loads weights takes `flock ~/.cache/d1-gpu.lock`, holds it for about 20 minutes or less, starts only with at least 6 GB of RAM available, and runs in a memory cage (`systemd-run --user --scope -q -p MemoryMax=<N>G -p MemorySwapMax=1G -p TasksMax=256`); long jobs run as `systemd-run --user` units.

## Records

- `NOTEBOOK.md` continues with dated round-2 entries, dead ends included.
- `outputs/r2/` holds round-2 exploration outputs; `outputs/r2conf/` the confirmation outputs.
- Later changes to this charter are dated entries in `AMENDMENTS-2.md`, never edits to this file.
