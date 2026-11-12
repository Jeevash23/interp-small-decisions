# D3 charter: how does a small model decide how far to indent the next line of Python?

Area 14 (mechanism discovery), study D3. Written 7 November 2026, 21:10 IST, from the draft under Target 1 in `../GAPS-2.md`, and frozen before any material was built or any model was run (digest in `FROZEN.md`). Started by the author on 7 Nov at 21:00 ("why can not we do all three then? in parrelel prehaps?"; `../DECISIONS.md`), alongside D4 and D5 on the same local machine.

Everything done under this charter is **exploratory**. A candidate mechanism found here is not a result. It becomes one only after a separate, frozen `PROTOCOL.md` tests its predictions once on sealed held-out material.

## Question

After a newline in Python, a model must choose the next line's indentation before it sees the next keyword:

- one level deeper after a block header (a line ending in `:`);
- the same level inside a block;
- shallower after `return`, `pass`, `raise`, `break` or `continue`, and at the ends of blocks;
- possibly several levels shallower at the end of nested blocks.

How does Pythia-410M make this choice, in particular when and how far to dedent? How do models whose tokenisers write indentation differently do the same?

Any answer counts. Candidate accounts at the outset, none favoured:

- copying the previous line's indent by attending back to the line start (as D1's heads attend to the newline);
- a local "+1 level" signal from the `:` token;
- a "dedent" signal read from the previous line's first word;
- a memory of which levels are open (a stack-like variable), needed to avoid levels that were never opened, for example in two-space, three-space or mixed-width code;
- cues such as blank lines, block length, or the enclosing `def` or `class`;
- something not on this list.

Bassi and Tomar's indentation geometry (arXiv 2607.04167; representation only, models of 2B and up) is one candidate representation among several, not a claim to be tested.

## Models

- **Main: Pythia-410M** (EleutherAI, final checkpoint, on disk). Writes the newline as its own token and each indent run as one whitespace token.
- **Contrasts**, all on disk:
  - **GPT-2 Small**: writes single spaces, so it must count;
  - **Qwen3-0.6B-Base**: merges `:` with the newline and writes a run one space short, followed by a word with a leading space;
  - **Gemma-4-E2B**: one token per run, like Pythia (loaded as in D1-X, text decoder only).
- **Optional, later:** the rest of the D1-X set on disk (Pythia-1B, Qwen2.5-0.5B, OLMo-2-1B, Qwen3-1.7B-Base, SmolLM2-1.7B), and Pythia checkpoints already on disk, for when a found mechanism forms in training.
- **Fallback main model:** if Pythia-410M fails the gate and a code-trained model passes, Qwen3-0.6B-Base (or Qwen2.5-0.5B) becomes the main model, as the GAPS-2 draft allows.

## Material

All material is built and its held-out parts are sealed by digest in `FROZEN.md` before any model reads any of it.

- **Generated Python programs.** A seeded generator in `code/` writes valid programs (checked with `ast.parse`) with controlled nesting depth, indent width (2, 3 and 4 spaces, and mixed widths across levels, so that some multiples of the unit are never opened), block types (`def`, `class`, `if`/`elif`/`else`, `for`, `while`, `try`/`except`/`finally`, `with`), dedent type and depth, blank lines and comments, and distractors such as `return` inside a string or a comment. Programs are split by a hash of their seed into an exploration part and a sealed confirmation part.
- **Natural code written after the models' training cut-offs.** Python files from public GitHub repositories created on or after 1 September 2026, which is after the release of every charter model (the latest, Gemma-4, was described in July 2026). Fetched with the GitHub API, well under 300 MB; forks, vendored and generated files excluded. Split by a hash of the repository name into an exploration part and a sealed confirmation part, so that no repository has files on both sides. The files are read as text only; nothing fetched is executed.
- **Python standard-library code on disk** (`/usr/lib/python3.14`), with its indentation kept: exploration only, because older versions of it are likely in every model's training data.
- Normalisation, the same for all sources: Unix newlines, trailing whitespace removed, files with tabs or non-ASCII characters dropped. Long files are cut into chunks of whole top-level statements so that every chunk starts at indent 0 and fits every model's context.

## Measures

- **Decision points.** The start of every physical line that begins a new logical line (from Python's own tokenizer), is not blank, is not a comment-only line, and follows a line that ended a logical line. Lines inside brackets, strings or backslash continuations are not decision points.
- **Truth and classes.** The true indent of the next line, and its class against the previous logical line: deeper, same, or shallower by n levels. The set of open levels is the indentation stack at that point. Subclasses: after a block header; after a terminal statement (`return`, `pass`, `raise`, `break`, `continue`).
- **The model's indent distribution.** For each candidate indent k = 0 to Kmax spaces, the probability of the tokens that commit the model to an indent of exactly k under the tokeniser's own canonical way of writing it before the next line's first word. For each candidate, factors are taken along its canonical token path up to the token that distinguishes it from every other candidate. For tokenisers that write an indent as one token (Pythia, Gemma-4, SmolLM2, Qwen3 for k of 2 or more) this is that one token's probability at the decision position. Where the run is absorbed into the next word's token (k = 0, or Qwen3's k = 1), it is the summed probability of all tokens that start with that many spaces followed by a non-space character. For GPT-2 it is the product along the chain of single-space tokens. The candidates' masses are reported raw and renormalised; the mass left over (blank lines, trailing spaces and so on) is reported as "other". Paths that are not the tokeniser's canonical ones are ignored; this is a stated approximation.
- From the distribution: top-1 accuracy; probability on deeper, same and shallower levels; probability on shallower open levels and on levels that were never opened; log-probability of the true indent; and, by class, the same against a **copy-previous-indent baseline**.

## Gate (before any internal analysis)

Run on the exploration parts only, never on a sealed part.

On the natural post-cut-off code, for the main model:

1. after a block header, the true deeper indent is top-1 in at least 90% of cases;
2. after a terminal statement, at least 0.7 of the renormalised probability goes to shallower open levels, and the true level is top-1 in at least 50% of cases;
3. over all decision points, top-1 accuracy beats the copy-previous-indent baseline by at least 10 percentage points.

Separately, the open-level part: in generated two-space, three-space and mixed-width code, at most 10% of the renormalised probability at dedent points goes to shallower levels that are not open.

The contrast models are measured the same way. They do not have to pass; a contrast that fails is itself recorded.

**How the gate can fail.**

- If the main model mostly copies the previous indent and dedents no better than the baseline (items 2 or 3 fail), switch to the fallback main model if it passes; if no charter model passes items 1 to 3, stop, and write up the behaviour and what the models do not do.
- If only the open-level part fails, drop that part of the question and keep the rest.

## Tools

D1's kit, all at the decision position and nearby:

- behaviour sweeps over the variables the generator controls;
- the logit lens on the indent tokens;
- single-head and MLP ablation scans (zero and mean ablation);
- activation and path patching between programs that differ in one cue (a `:` removed, `return` replaced by an assignment, a header moved one level, a blank line added, the indent width changed);
- linear probes (ridge or logistic, held-out folds within the exploration parts) for the current level, the previous level, the next level, and the set of open levels;
- attention analysis, including whether any of D1's newline heads take part;
- set tests against random components matched by layer and size.

Optional: program-synthesis descriptions of found heads (arXiv 2606.19317); Logogram for looking around.

## What counts as a candidate finding

A small set of named components (heads, MLPs or neurons) or a low-dimensional variable that meets both conditions on exploration material:

- **It moves the indent choice as predicted.** Patching or ablating it moves the choice in the predicted direction, clearly beyond random component sets of the same size matched by layer: for example, removing a dedent signal makes the model keep the block's level after `return`, or moving a header makes it dedent to the new level.
- **It predicts something nobody would predict without it.** For example, a `return` inside a string or a comment triggers a false dedent; or code with an unusual indent width confuses one component but not another.

Partial findings are recorded as such: a decodable variable with no causal effect, or a component set that matters but whose role is unclear.

## Stop rules

- Stop at four weeks from the freeze.
- Stop early, and write up what was found, if the gate fails as described above, or if after two weeks no intervention moves the indent choice by more than its random controls.
- Stop and park for the author if a download over about 300 MB, Modal, a licence or a new account seems necessary.

## Budget and shared local machine

- Local machine only (6 GB NVIDIA GPU, 6 GB VRAM; about 15 GB RAM), $0. No Modal.
- D4 and D5 run at the same time. Every job that loads model weights, on the GPU or the CPU, takes `flock ~/.cache/d1-gpu.lock`, runs inside `systemd-run --user --scope -q -p MemoryMax=<N>G -p MemorySwapMax=1G -p TasksMax=256 --`, and starts only when `free -g` shows at least 6 GB available. Each lock hold lasts at most about 20 minutes; longer work is split into chunks. GPU use stays under about 5 GB.
- Downloads: the GitHub code fetch (expected a few MB). Anything over about 300 MB is parked for the author.

## If nothing is found

Write up a behavioural map of indentation across tokenisers and sizes: when each model dedents, how far, whether it respects open levels, how it behaves with unusual widths and distractors, and whether the decodable indent level (as in arXiv 2607.04167) is used at all.

## Record keeping

- `NOTEBOOK.md`: dated entries of what was tried and what came out, dead ends included.
- Scripts in `code/`, outputs in `outputs/` named after the script.
- Later changes to this charter are dated amendments in `AMENDMENTS.md`, never edits.
- Routine decisions within this charter are taken and logged in `NOTEBOOK.md` with their reason. Anything the shared rules reserve for the author (commits, publishing or sending, the author's name, downloads over about 300 MB, Modal, licences) is parked.
