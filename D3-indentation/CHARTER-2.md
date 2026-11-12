# D3 charter, round 2: the complete picture of the two dedent routes

Area 14, study D3, second round ("depth round"). Written 8 November 2026, about 17:05 IST, at the author's request of about 16:50 ("why not do the depth runs for these too … and try to get the complete picture"; `../DECISIONS.md`, entry "8 Nov 2026, about 16:55 IST"). Frozen before any round-2 material was built and before any round-2 run (digest in `FROZEN.md`).

Everything done under this charter is **exploratory**. Round 1 (`CHARTER.md`, `PROTOCOL.md`, `REPORT.md`) is closed and its frozen files stay untouched. Round 1 confirmed two routes to a dedent in Qwen3-0.6B-Base:
- a block-end route: DED = {L23H2, L17H3, L17H4, L17H5, L18H6}, needed for dedents that no terminal statement marks;
- a terminal-statement route: TERM, nine heads, needed modestly for dedents after `return`, `pass`, `raise`, `break` or `continue`; CARRY, six heads, carries the terminal signal to the newline;
- the two routes back each other up for terminal dedents.

Round 2 asks how these routes get their information, and how general they are. A round-2 candidate becomes a result only after a separate frozen `PROTOCOL-2.md` tests it once on round-2 held-out material.

## Questions, in order of priority

1. **What the block-end heads read.** At unmarked dedents, which earlier positions and cues tell DED that the block has ended? Find where the DED heads attend from the decision token. Then test which source positions matter by removing their contribution (attention knock-out of a position class for the DED heads, and patching of the residual stream at source positions from a matched program), ideally down to the path level (which earlier heads write what the DED heads read).
2. **How far to dedent.** When two or more blocks could close (two or more shallower levels are open), what chooses among the open shallower levels? Candidates, none favoured: attention by a late head to the start of an earlier line at the target level, with its indent token copied; a learned recency or "one level up" default; cues from the kind of enclosing block (`def`, `class`, a loop) or from block length; something else.
3. **The terminal route, end to end.** How does "the previous line was `return`, `pass`, …" get from the keyword's own token to the decision newline? X1's exploratory lead: ablating L0H2 (a previous-token head) at every position cuts generated terminal dedents by −0.36, though L0H2 is quiet at the decision token. Test that lead and other upstream steps, and find where TERM and CARRY get their input: position-by-layer patching of return/assignment pairs, attention of the carrier heads, and ablation restricted to positions on the terminal line.
4. **A second model.** Do the two routes exist in another model? The second model is chosen by a fixed rule below.
5. **Optional: training dynamics.** If the second model has public training checkpoints (OLMo-2), find when the routes form. Checkpoints are about 6 GB each, so any download goes first to the session that started this study and the author. Points 1–4 do not wait on that answer.

Any answer counts, including "the information is spread too thinly to localise" for any question.

## Models

- **Main: Qwen3-0.6B-Base**, as confirmed in round 1. bfloat16 for scans, as in round 1 (checked against float32: r ≥ 0.993).
- **Second model, fixed rule.** The models on disk are gated in this order on round-1 exploration material with round 1's frozen gate (items 1–3 and the open-level part, `CHARTER.md`): OLMo-2-0425-1B (another family; public checkpoints), then Qwen3-1.7B-Base (the same family at a larger size), then SmolLM2-1.7B. The first that passes items 1–3 is the second model. Gemma-4-E2B passed the gate in round 1, and is the fallback if none of the three passes. It is last because it is tight on the 6 GB GPU.
- No downloads of model weights.

## Material

All round-2 held-out material is built and sealed by digest in `FROZEN.md` before any model reads any of it.

- **Held-out natural code (new).** Python files from public GitHub repositories created on or after 1 September 2026, with 5 to 9 stars at fetch time. Round 1 used 15 or more stars and X1 used 10 or more, but star counts change, so the rule is applied by name as well:
  - Repositories listed in round 1's or X1's fetch are excluded.
  - Any chunk whose text hash occurs in round-1 or X1 material is dropped.
  - Fetched with the same rules as round 1 (`code/fetch_newcode.py`): up to 4 files per repository, 1,500–40,000 bytes each, the same exclusions, at most 1,200 files.
  - Read as text only; nothing fetched is executed.
  - All of it is held-out. None of it is used for exploration.
- **Held-out generated programs (new).** Round 1's frozen generator with seeds 200000–203999, which neither round 1 (0–3999) nor X1 (100000–103999) used. All held-out.
- **Targeted generated sets (new) for questions 1–3.** A new generator (`code/gen2.py`), written before it is first run, produces matched programs:
  - programs with several open levels, where the next line returns to a chosen one of them;
  - pairs that differ in one cue: a terminal statement against an assignment, or a block header at another level.
  - Seeds are split by `sha256("D3R2-gen2-" + seed) % 2` into an exploration part and a held-out part. The held-out part is sealed with the other held-out material.
- **Exploration material:**
  - round 1's exploration parts (generated, post-cut-off GitHub code, standard library);
  - X1's code and generated parts, both halves, since X1 has already read them;
  - the exploration half of `gen2`.
  - Round 1's sealed parts have now been read and are not used in round 2.
- Normalisation and chunking are as in round 1 (`code/pyindent.py`, unchanged).

## Measures

As in round 1: the indent distribution over candidate indents (`code/common.py`, unchanged), point classes (header, same, terminal, other dedent), log q(true), mass on the current level, and mass on open shallower levels. New for question 2:
- **Multi-level points**: dedent points with at least two open shallower levels.
- **Target share**: q(true level) divided by the mass on all open shallower levels, together with its log.
- **Level preference**: the share given to the nearest, middle and outermost open levels.

## Tools

Round 1's kit, plus:
- **attention knock-out**: for named heads at the decision token, attention to a class of source positions is removed and the rest renormalised; the head's new output is patched in;
- **source-position patching**: the residual stream at chosen positions and layers, taken from a matched program;
- **position-restricted ablation**: a head is ablated only at positions on a chosen line;
- **path patching** between an upstream head and the DED or TERM heads, through their keys, queries or values where needed;
- linear probes and direct logit attribution, as in round 1.

All are compared with random heads matched by layer, or random position sets matched in size.

## What counts as a candidate finding

For each question, a named set of components, source positions, or a simple rule. It must move the measure as predicted, clearly beyond random controls matched by layer and size, on exploration material. It must also predict something nobody would predict without it, for example:
- knocking out attention to one earlier line moves the dedent to another level;
- a terminal-line ablation removes the terminal dedent while leaving other dedents alone;
- the second model's analogue of DED spares terminal dedents.

Partial findings are recorded as such.

## Stop rules and budget

- **Budget:** round 2 aims to be confirmed within about a day of this freeze, by about 9 November 2026, 18:00 IST. Write-up v2 follows.
- **Per question:**
  - If a question has no candidate beating its random controls by 9 November, 06:00 IST, it is dropped from the protocol and written up as what was found.
  - If question 4's gate fails for all three models and Gemma-4-E2B cannot be run, question 4 is recorded as a null.
- **Whole round:** stop at 10 November 2026, 12:00 IST, whatever the state, and write up what exists.
- **Park for the author** if a download over about 300 MB (for example, training checkpoints), Modal, a licence or an account seems necessary. The other questions carry on meanwhile.
- **Local machine:** local machine only, $0. D4 and D5 run their depth rounds at the same time. Every job loading weights takes `flock ~/.cache/d1-gpu.lock` for at most about 20 minutes per hold. It starts only with at least 6 GB of RAM available, and runs in a `systemd-run` memory cage. Long jobs run as `systemd-run --user` units watched by a waiter. Check `pgrep -af` before restarting anything.

## Record keeping

- `NOTEBOOK.md` continues with a "Round 2" section: dated entries, dead ends included, labelled exploratory.
- Scripts go in `code/` with an `r2_` prefix (`r2e…` for exploration, `r2c…` for confirmation). Outputs go in `outputs/r2/`.
- Changes to this charter go in `AMENDMENTS.md` as dated round-2 amendments, never as edits.
- Routine and scientific decisions inside this charter are taken under the author's go-ahead and logged with their reason. Commits, publishing, sending, permanent deletion, Modal, licences, accounts and the author's name stay parked.
- Milestones are logged under "## D3" in `../DEPTH-ROUND.md`, and in STATUS.md's D3 line.
