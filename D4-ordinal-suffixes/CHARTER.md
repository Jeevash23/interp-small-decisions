# D4 charter: how does a small model choose an ordinal suffix?

Area 14 (mechanism discovery), study D4. Written 7 November 2026 and frozen before any model was run on D4 material (digest in `FROZEN.md`). Chosen by the author on 7 Nov at 21:00 IST, when they asked for all three round-2 targets to run in parallel (`../DECISIONS.md`). Drafted in `../GAPS-2.md`, Target 2; this charter keeps that draft and adds the corrections listed under "Changes from the draft".

Everything done under this charter is **exploratory**. A candidate mechanism found here is not a result. It becomes one only after a separate, frozen `PROTOCOL.md` tests its predictions on held-out material.

## Question

After a number, English writes "st" if the number ends in 1, "nd" if it ends in 2, "rd" if it ends in 3 and "th" otherwise, except that numbers ending in 11, 12 or 13 take "th". How do Qwen3-0.6B-Base (one token per digit) and Pythia-410M (numbers written as one or two multi-digit tokens) choose between the four suffixes, and how do they handle the 11–13 exception?

Any answer counts. Candidate routes at the outset, none favoured:

- whole-token memory: each number token (Pythia) or frequent ordinal stores its suffix;
- a mod-10 feature of the last digit, as successor heads carry (arXiv 2312.09230), read at the suffix position;
- a separate "teen" check on the tens digit that overrides the last-digit route;
- a default towards "th", which is right for about 70% of numbers on its own;
- something not on this list.

## Models

- **Main pair:** Qwen3-0.6B-Base and Pythia-410M (final checkpoint). Both on disk.
- **Contrast models:** OLMo-2-1B (`allenai/OLMo-2-0425-1B`), SmolLM2-1.7B, Gemma-4-E2B and GPT-2 Small. All on disk.
- **Fallbacks** if a main model fails the gate: Qwen3-1.7B-Base and Qwen2.5-1.5B. Both on disk. Pythia-1B (on disk) may stand in for Pythia-410M in the same way.
- No downloads are planned. Any download over about 300 MB needs the author's yes.

## Tokenisation (checked on the tokenisers only, 7 Nov 2026, before this charter was frozen)

- Qwen3, Qwen2.5, SmolLM2 and Gemma-4 write one token per digit, after a separate space token.
- Pythia-410M writes 1–2 digit numbers as one token. Of the three-digit numbers, 544 are one token and 356 are a single digit followed by a two-digit token (for example ` 5` + `58`). Four-digit numbers split 2+2 (4,269), 1+3 (3,887), 3+1 (672) or stay whole (172). Only 16 four-digit numbers ending in 10–19 put the tens digit and the last digit in different tokens.
- GPT-2 Small is similar to Pythia, with some 2+1 and 1+2+1 splits (46 four-digit teen numbers cross a token boundary).
- OLMo-2-1B writes up to three digits as one token, after a space token, and splits every four-digit number 3+1 from the left (` 111` + `2`). All 900 four-digit numbers ending in 10–19 put the tens digit in one token and the last digit in the next.
- The suffix tokens `st`, `nd`, `rd` and `th` are single tokens in every model above, and in every case checked the ordinal is written as the number's tokens followed by one suffix token.

## Changes from the draft in `GAPS-2.md`

- The draft said Pythia splits four-digit numbers like `111` + `2`, putting the teen check across a token boundary. The tokeniser check above shows that this is true of OLMo-2-1B for all four-digit numbers, but of Pythia-410M for only 16. **The four-digit boundary prediction is therefore tested mainly in OLMo-2-1B**, with Pythia's 16 cases and GPT-2's 46 as small extra sets. Pythia-410M stays a main model for the whole-token question.
- Pythia writes 356 three-digit numbers as two tokens. These are recorded as their own group rather than counted as whole-token cases.
- The gate's thresholds and the stop rules are the draft's, unchanged.

## Material

All material is generated: a prefix that ends with a number, after which the next token should be the suffix. No natural text is used.

- **Frame families.** Each frame is a short context sentence followed by a phrase ending in the number:
  - **dates** (numbers 1–31), for example "The festival is held every year on the {N}";
  - **centuries** (1–21), for example "The chapel was built in the {N}";
  - **rankings** (1–9,999), for example "Out of all the runners in the marathon, she finished {N}";
  - **anniversaries and birthdays** (1–999), for example "Last week the company celebrated its {N}";
  - **positions in a sequence** (1–9,999): editions, sessions, rows, entries, floors, attempts.
- **Split, fixed before any model reads the material.**
  - Frames are split within each family by the SHA-256 of their text: the first half by hash order goes to exploration, the next 30% to confirmation and the rest to reserve.
  - Numbers from 100 to 9,999 are split by the SHA-256 of `D4-num-<N>`: 60% exploration, 30% confirmation, 10% reserve.
  - Numbers 1–99 are too few to split and appear in every part; their frames differ by part.
  - An item belongs to a part only if both its frame and (for 100 and above) its number belong to that part. The confirmation part is therefore new frames with new numbers.
- The builder script and the part files are digested in `FROZEN.md` before any model reads them. The confirmation and reserve parts are read only under a frozen `PROTOCOL.md`. Constructed inputs made to test a candidate (for example a thousands separator, "1,112", or the digits in a different frame) are exploratory material and may be made freely, as long as they use exploration frames and exploration numbers.

## Measures

At the last token of the number:

- **suffix choice**: which of the four suffix tokens has the highest logit;
- **suffix accuracy**: how often that choice is correct, overall and split by last digit, tens digit (teen or not), number of digits, token-split pattern and frame family;
- **margin**: the logit of the correct suffix minus the highest of the other three;
- **ordinal expectation**: the probability the model puts on the four suffixes together, so that a choice among four is not read where the model does not expect an ordinal at all.

## Gate (before any internal analysis)

For each model, on exploration items only:

1. the suffix is top-1 among the four in **at least 90%** of items for numbers 1–31 in date frames;
2. **at least 80%** for numbers 32–999 ending in 1, 2 or 3 but not in 11, 12 or 13 (the cases a "th" default gets wrong), across the non-date families;
3. recorded separately, not a pass condition: the exception cases (numbers ending in 11–13, including 11–13, 111–113, 211–213 and four-digit ones), the "th" cases (ending in 0 or 4–9), and four-digit numbers by token-split pattern.

A model **qualifies** if it meets conditions 1 and 2. Internal analysis is run only on qualifying models.

- If a main model fails, its fallback is gated (Qwen3-1.7B-Base or Qwen2.5-1.5B for Qwen3-0.6B-Base; Pythia-1B for Pythia-410M).
- If no model in this charter qualifies, D4 stops and the behavioural map is written up as a null: which models know the rule, which know only the frequent cases, and how tokenisation changes the error pattern. A systematic error such as "111st" in a model that is competent on the controls would itself be described.
- Contrast models are gated too, for the behavioural map; a contrast model that qualifies may be used in exploration.

## Tools

- Behaviour sweeps by last digit, tens digit, number length, token-split pattern, frame family and frequency.
- The logit lens on the four suffix tokens.
- Direct logit attribution of heads and MLPs at the suffix position, and ablation scans.
- Activation patching between numbers that differ in one digit (342 against 341 should move nd to st; 312 against 322 tests the teen check), by layer, position, head and MLP.
- Linear probes for last digit, tens digit and teen status at the number's tokens, by layer.
- Random-component and random-direction controls for every intervention.
- Area 10's digit work (`../../10-LM-Arithmetic`, A2 and A3) as a source of starting points in Qwen3-0.6B.

## Candidate finding worth confirming

Components or directions that carry the last digit and the teen status to the suffix position, meeting both:

- (a) patching or steering them moves the suffix as the digit change predicts, beyond random sets of the same size;
- (b) they predict errors or behaviour nobody would predict without them, for example a teen error when tokenisation (OLMo-2's 3+1 split) or a frame (a thousands separator, "1,112") puts the tens digit in a different token from the last digit.

## Budget and stop rules

- Two to three weeks of exploration. Stop early at the gate, or after ten days if no intervention moves the suffix beyond random controls.
- Local machine only (6 GB NVIDIA GPU, 6 GB), $0. No Modal.
- Shared local machine (`../DECISIONS.md`, 7 Nov 21:00): every job that loads model weights takes `flock ~/.cache/d1-gpu.lock`, runs in a memory cage (`systemd-run --user --scope -q -p MemoryMax=<N>G -p MemorySwapMax=1G -p TasksMax=256`), starts only with at least 6 GB of RAM available, and holds the lock for at most about 20 minutes.

## Records

- `NOTEBOOK.md`: dated log of everything tried, dead ends included.
- `outputs/`: every saved result, written by scripts in `code/`.
- Later changes to this charter are dated amendments in `AMENDMENTS.md`, never edits to this file.
