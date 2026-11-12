# D5 · Writing in capitals: charter

Written 7 Nov 2026, 21:05 IST, from the draft in `../GAPS-2.md` (Target 3), on the author's choice at 21:00 to run all three round-2 targets in parallel (`../DECISIONS.md`). Frozen before any model was run on D5 material; digest in `FROZEN.md`. Everything done under this charter is **exploratory**. Changes are dated amendments in `AMENDMENTS.md`, never silent edits.

## Question

In upper-case text, how does a small language model keep writing in capitals, and how does it choose the upper-case pieces of the next word? How does this differ between a model with many whole-word upper-case tokens (Gemma-4-E2B) and models that split upper-case words into several pieces (Qwen3-1.7B-Base, SmolLM2-1.7B, Pythia-1B)?

Possible routes, to be told apart where the data allow:

- **R1, a case-mode direction:** a direction read from recent tokens and added on top of the ordinary next-word prediction, much as a language vector is.
- **R2, latent lower case:** the model predicts the lower-case word first (visible, for example, in the logit lens of middle layers) and maps it to upper-case pieces late.
- **R3, copying:** upper-case forms are copied from earlier in the passage, so words already seen are handled differently from new ones.
- **R4, upper case as a "language":** upper-case text is handled by its own features, so the upper-case and lower-case forms of a word are not represented alike in the middle layers.

## Models

Charter models (all on disk): Gemma-4-E2B (base), Qwen3-1.7B-Base, SmolLM2-1.7B, Pythia-1B. Smaller models on disk (Qwen3-0.6B-Base, Pythia-410M, Pythia-160M) may be added only if they pass the gate below. No downloads are expected; any download over about 300 MB is the author's decision.

## Material

- **Source.** The first D1 new-text batch already on disk (`../D1-line-breaking/data/newtext/articles.jsonl`, English Wikipedia articles created January 2025 to September 2026, fetched 6 Nov 2026; licence and attribution as in its `MANIFEST.md`), using D1's paragraph rule as already applied in its seven `part-*.jsonl` files (1,740 paragraphs). D1's unread second-batch reserve (`newtext2/part-reserve-pythia410m.jsonl`) is not used.
- **Deviation from the GAPS-2 draft, with reason.** The draft asked for a fresh fetch. The study brief (7 Nov, 21:00) asks for the batches already on disk, and the author's data is metered. These paragraphs were read by D1 only for line-breaking measurements and never in capitals; D5's split below is new and is sealed before any D5 measurement. The text may overlap the newer models' training data (as noted for D1); this matters little for a question about case.
- **Eligible paragraphs.** Every character keeps its length under upper- and lower-casing (so the three versions align character by character), and the paragraph has at least 60 alphabetic words.
- **Split (fixed now).** By article: `int(sha256("D5-caps-" + str(pageid)), 16) % 3` gives 0 = `explore`, 1 = `conf` (held out for the confirmation protocol), 2 = `reserve` (held out, unread, for a later round). Built by `code/build_data.py`; the three part files and the script are digested in `FROZEN.md` before any script reads them for results. The loader refuses `conf` until `PROTOCOL.md` exists and is read-only, and refuses `reserve` always.
- **Conditions.** Each paragraph in four versions: upper case (`str.upper()`), lower case (`str.lower()`), title case (each alphabetic word's first letter upper, the rest lower), and the original. Upper case is the condition of interest; lower and title case are the controls the draft names; the original is a reference only.

## Definitions used by the gate

- **Words.** Maximal runs of letters. A **word start** is a character that begins a letter run, is preceded by a space, and is the first non-space character of a token in that version's tokenisation. Only word starts after the first five words of a paragraph count, so the model has seen the passage's case.
- **Token case.** A token string with at least one cased letter is *upper* if it has no lower-case letter, *lower* if it has no upper-case letter, and *mixed* otherwise; a token with no cased letter is *caseless*.
- **Inside-word positions.** The next token starts inside a letter run (not at its start), in the text's own tokenisation.

## Gate (on exploration text; per model)

Run on 150 exploration paragraphs (fixed order: shuffled with seed 0), with greedy decoding for condition 2 at four word starts per paragraph (chosen with seed 1).

1. **Case kept.** At word starts in upper-case passages, among top-1 next tokens that contain a cased letter, at least 95% are upper case.
2. **Same word.** At the same word starts, greedily decode one word (until a token begins with a space or adds a character that is not a letter) from the lower-case passage and from the upper-case passage. Among positions where the lower-case word is alphabetic, the upper-case word equals the lower-case word in capitals at least 60% of the time.
3. **Spelling inside words.** At inside-word positions in upper-case passages, the top-1 next token is a correct continuation of the word (its string is a non-empty prefix of the rest of the word) at least 70% of the time. The strict version (top-1 equals the text's own next token) is reported beside it.

The same measures are reported for the lower-case and title-case controls (condition 1 as "lower kept" / "title kept"; condition 2 as agreement of the original-case word with the lower-case word, a ceiling for how often the chosen word survives a change of case alone). A model enters exploration only if it meets all three conditions. The thresholds are the draft's and are not tuned after seeing results.

**If the gate fails.** If the splitting models fail conditions 2 or 3, fall back to Gemma-4-E2B alone, or restrict the question to words that are a single upper-case token in that model (stated as a dated amendment). If Gemma also fails, stop and write up the behaviour (a map of how well these models write in capitals, by tokeniser, and where they break: word starts against inside words).

## Tools (exploration)

- Comparing the same passage in lower and upper case at word starts and inside words (logit lens with each model's own final norm and output layer; ranks of the lower-case and upper-case forms of the same word by layer).
- Patching the residual stream, head outputs or MLP outputs from an upper-case passage into the lower-case one at the same position, and the reverse; head and MLP scans.
- A single "case direction" (difference of means between upper-case and lower-case runs, per layer), and its effect when added or removed.
- Seen against new words (has the upper-case form already appeared in the passage?), and attention to earlier copies.
- Random-direction controls (matched norm) and random component sets (matched size) for every intervention.

## What counts as a candidate finding

- (a) A direction or a set of components that, when patched or added at word starts, changes the case of the next word (for example upper to lower) without changing which word it is, clearly beyond random controls (as a guide: a case flip in at least half of positions, the word kept in at least half of those, matched random controls flipping under 5%).
- (b) A new prediction that follows from it and could fail: for example, what happens at a mid-sentence switch into capitals, or that upper-case errors follow the lower-case runner-up word rather than spelling noise.

Then write `PROTOCOL.md` (predictions that could fail, controls, analysis plan), freeze it, and run it once on the `conf` part.

## Compute, budget and stop rules

- Local machine only, $0; no Modal. Every job that loads weights takes `flock ~/.cache/d1-gpu.lock`, runs in a memory cage (`systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=1G -p TasksMax=256`), starts only with at least 6 GB of RAM available, and holds the lock for at most about 20 minutes. Gemma-4-E2B runs alone under the lock, in bfloat16, with D1-X's loader (`../D1-line-breaking/xmodel/code/xm3_common.py`). Short contexts (one paragraph).
- Three weeks in all. Stop early at the gate (above), or after ten days of exploration with no case-specific intervention.
- A null is a result: if no mechanism turns up, write up the behaviour and any partial path.
