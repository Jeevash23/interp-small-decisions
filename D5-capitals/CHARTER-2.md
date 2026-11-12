# D5 round 2 · From a capitals direction to the parts that use it: charter

Written 8 Nov 2026, 17:05 IST, after the author asked for depth rounds on D3, D4 and D5 (8 Nov, about 16:50; `../DECISIONS.md`). The parent chat's brief set the four depth points below; the reserve part was released for this round by the parent's decision. Frozen before any round-2 measurement; digest in `FROZEN.md`. Everything done under this charter is **exploratory**. Changes are dated amendments in `AMENDMENTS-2.md`, never silent edits. Round 1 (`CHARTER.md`, `PROTOCOL.md`, `REPORT.md`) stays untouched.

## Where round 1 left off

Round 1 confirmed, on held-out text, in Gemma-4-E2B and Qwen3-1.7B-Base: the next word is identified in a lower-case-like form and turned into capitals in the last three or four layers; one mean-difference direction d after a mid-to-late layer (ℓ = 24 in Gemma, 22 in Qwen3) switches the next word's case both ways, mostly keeping the word, and is needed; the case follows the last two to eight words; Qwen3's head L0H3 passes on the current token's case. Its main limit was that this is an account in terms of one direction, not of the parts that write and read it. X1 (`../X1-shared-parts/REPORT.md`) then showed that L0H3 attends to the current token in every task tested, so case is one property of the current token that it passes on, not its job; and that L17H6 trims the capitals gap a little in Qwen3-1.7B without changing the word.

## Questions

- **Q1, from direction to parts.** Which components (attention heads and MLPs) write the capitals direction d at the prediction position, up to layer ℓ, and what do they read? Which components after ℓ read d and turn the lower-case word into upper-case pieces?
- **Q2, how passage case reaches the prediction.** With the current token's case held fixed, which heads carry the case of the earlier words to the prediction position, and how far back do they look?
- **Q3, why the edit changes the word.** Adding d changes the word at roughly a fifth to a third of the positions where it flips the case. Is this how the model writes capitals anyway (the edited word is the one it writes in a real upper-case passage), a matter of tokenisation (words whose capitals form takes many more pieces, or whose capitalised form is a different word such as a proper noun or an acronym), or a sign that d carries content as well as case?
- **Q4, other capitals.** Do capitalised headings, acronyms in ordinary text, and initial capitals (names and sentence starts) use the same direction and parts? Round 1 saw, on 21–26 positions, that subtracting d rarely removes Gemma's initial capitals.
- **Q5, optional: training dynamics.** In a model with public training checkpoints that passes the round-1 gate, when do the lower-case-first pattern and the capitals direction appear?

## Models

Gemma-4-E2B (base) and Qwen3-1.7B-Base, the two that passed the frozen round-1 gate, loaded as in round 1 (D1-X's loader; Gemma in bfloat16 with its per-layer table read from disk).

For Q5 only: a model with public intermediate checkpoints, on disk first; otherwise downloaded (approved by the author for research on 8 Nov, 17:00; at least 50 GB of disk kept free; each download's size logged in `../DEPTH-ROUND.md`). It enters Q5 only if its final checkpoint passes the frozen round-1 gate (`code/gate.py`, explore paragraphs 0–150, bars unchanged) at the first try. If no candidate passes, Q5 stops and the gate result is reported. Candidates in order: OLMo-2-0425-1B (stage-1 checkpoints public), then any other checkpointed model of 1–2B parameters that fits the 6 GB GPU.

## Material

- **Exploration:** the `explore` part (613 paragraphs) and the `conf` part (597 paragraphs). `conf` was the round-1 held-out part and has been read; in round 2 it is exploration material only. Heading items (below) from articles whose D5 hash is 0 or 1.
- **Held out for round 2:** the `reserve` part (528 paragraphs from 299 articles), sealed by the round-1 charter and read by no model or script since (X1 did not read it); and heading items from articles whose hash is 2. Both are read only by the frozen round-2 protocol, once.
- **Heading items** (new; built now by `code/build_data2.py` from the same D1 batch, `../D1-line-breaking/data/newtext/articles.jsonl`, the D5 article hash unchanged): for each level-2 or level-3 section heading in an article extract whose previous paragraph has at least 30 alphabetic words and whose next paragraph at least 20, one item: the end of the previous paragraph (cut to at most 600 characters at a space), a blank line, the heading, a blank line, and the start of the next paragraph (cut to at most 300 characters at a space). Headings that open reference sections (References, Notes, See also, External links, Further reading, Bibliography, Sources, Citations, Footnotes) are left out, and every character must keep its length under casing. Each item is kept with its heading in capitals and in its original case. The builder prints counts only. The split files, the builder and their digests go into `FROZEN.md` before any script reads them for results.
- **Loader:** `code/d5common2.py` (new) reads `reserve` and the hash-2 heading items only once `PROTOCOL-2.md` exists and is read-only. The frozen round-1 loader is not changed.

## Definitions

Round 1's definitions stand (words, word starts after the first five words, token case, versions, the same-word rule, U, Lc, Lr, the direction d and its random control r of the same norm, ℓ and the early layer 16). New:

- **Component writes along d.** At the prediction position, the part of the residual after layer ℓ contributed by each attention head, each MLP, the embedding and (Gemma) each layer's per-layer-input block, projected on the unit vector of d. In Gemma the post-attention and post-MLP norms are applied with the scale of the whole block output (exact, since each is a vector scale times a fixed weight), and each later layer's scalar multiplies the contributions that pass through it. The sum of all writes equals the projection of the residual.
- **Context-only runs (Q2).** Two prefixes that end at the same lower-case current word: the upper-case passage with its last word in lower case (A), and the lower-case passage (B). Their difference reaches the prediction position only through attention to earlier positions.
- **Acronym position:** a word start in the original passage whose true next word has at least two letters, all upper case, in a passage that is not in capitals. **Initial-capital position:** a word start whose true next word has an upper-case first letter and at least one lower-case letter.

## Tools (exploration)

- Component writes along d, upper minus lower runs, ranked (Q1); the same for the context-only pair (Q2); attention of the leading heads by word distance from the prediction position.
- Directional ablation: moving only the d-component of a set of component outputs to the value it takes in the other run (or its mean there), leaving the rest of each output unchanged.
- Freezing chosen components after ℓ at their unedited outputs while d is added at ℓ (Q1, readers); direct contributions of components after ℓ to logp(U) − logp(Lc).
- Re-reading round-1 records and new edit runs for Q3: whether the edited word equals the upper-case passage's own word, word length and piece counts in each case, proper nouns and acronyms, and the lower-case runner-up rank.
- Removal of d (midpoint, as in round 1) and subtraction of d at acronym, initial-capital and heading positions (Q4); projections on d at those positions against the round-1 upper and lower means.
- **Controls for every intervention:** random component sets of matched size, type and layers (at least 20 sets where a set is ranked against them), and random directions of matched norm.

## What counts as a candidate finding

- **Q1, writers:** a set of at most 12 components whose directional ablation (d-component moved to the lower-case run's value) in upper-case passages leaves an upper-case top-1 in at most 0.70 of positions, while every one of 20 matched random sets leaves at least 0.90. **Readers:** a set of at most 8 components after ℓ whose freezing cuts the flip rate of adding d at ℓ by at least half, while matched random sets cut it by at most 0.15.
- **Q2, carriers:** a set of at most 12 heads whose patching from B into A removes at least half of A's excess projection on d (or of its excess in logp(U) − logp(Lc)), with matched random sets removing at most 0.15.
- **Q3:** an explanation that accounts for a clear majority of the changed words, or a stated mix, with a prediction about which words will change that could fail on new text.
- **Q4:** for each of headings, acronyms and initial capitals, a clear same-or-separate answer: removing d either takes away most of these capitals (same direction) or leaves at least 0.80 of them (separate), against random removal.
- For any point, a null (no set meets the bar) is a result and is written up as such.

Then write `PROTOCOL-2.md` (predictions that could fail, random controls, a size margin on each threshold, analysis plan), freeze it, and run it once on the held-out material.

## Compute, budget and stop rules

- Local machine only, $0; no Modal. Every job that loads weights takes `flock ~/.cache/d1-gpu.lock`, runs in a memory cage, starts only with at least 6 GB of RAM available (more for Gemma), holds the lock for at most about 20 minutes, and shares the GPU with D3's and D4's depth rounds. Long jobs run as `systemd-run --user` units. Check `pgrep -af` before restarting anything.
- **Budget:** round 2 confirmed within about a day. Exploration stops by 9 Nov 2026, 03:00 IST, or after about seven GPU hours, whichever comes first; the protocol is then frozen with the candidates found, and points without a candidate are reported as exploratory nulls. The confirmation run is limited to about five GPU hours; a run that fails for a technical reason may be rerun unchanged or on fewer items from the start of the same range (dated amendment).
- **Q5 stop rules:** stop if no candidate passes the gate at the first try, if its downloads would pass about 25 GB, or if free disk would fall below 50 GB. Q5 is descriptive (checkpoints are not held-out material) and never delays Q1–Q4.
- No commits, nothing published, no Modal, no licences or accounts, no permanent deletion (the Gemma weights stay), no owner's name. Public text says "language-model assistants" only.
