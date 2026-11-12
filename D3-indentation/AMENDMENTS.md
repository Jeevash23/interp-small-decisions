# D3 amendments to the charter

The charter (`CHARTER.md`, frozen 7 Nov 2026, 21:04 IST) is never edited. Changes are listed here, dated, in order.

## Amendment 1, 7 Nov 2026, 22:30 IST: decision points after blank lines (measurement bug fix)

**What was wrong.** The charter defines a decision point at every line that begins a new logical line and follows a line that ended one, blank lines included ("Measures"). The first implementation of the indent distribution (`code/common.prepare`) kept a point only if every candidate indent left the tokenisation before the original decision token unchanged. Pythia's tokeniser (GPT-2-style pre-tokenisation) writes a blank-line run before a column-0 line as `\n\n` + `\n`, but before an indented line as `\n\n\n` + the run. So the candidate token sequences disagreed before the newline, and every Pythia point after a blank line was silently dropped. Qwen3 and Gemma merge the run the same way in both cases and lost few points. GPT-2 is affected like Pythia.

**Why it is a bug and not a choice.** The dropped points are inside the charter's definition of a decision point. Dropping them was never decided; it followed from an assertion in the code. It also biased the measured gate towards points with no blank line before them, including most "terminal statement, then the next top-level definition" dedents. That bias differed between tokenisers.

**The fix.** The decision position becomes the last token shared by every candidate's tokenisation. Each candidate's probability is taken along its path at least past its last token that contains a newline, so the number of blank lines is never counted as part of the indent choice. Points with no blank line before them are measured exactly as before. The only points added back are those that follow one or more blank lines. On the first 40 natural exploration chunks, Pythia goes from 682 points to 768, 86 of them after blank lines (none before).

**Order of events, stated plainly.** The fix was made after I had seen the pre-fix gate numbers for Pythia-410M (natural and generated code), GPT-2 (both) and Qwen3-0.6B-Base (both). It was prompted by the gap between Qwen3's 1,836 and Pythia's 1,100 terminal-statement points on the same code. Pre-fix, Pythia missed item 1 (0.870 against 0.90) and passed items 2, 3 and the open-level part. Qwen3 passed every item, and GPT-2 failed every item. Pre-fix outputs are kept in `outputs/v1-before-2230-fix/`.

**What does not change.** The bars (0.90, 0.7 and 0.5, 10 points, 0.10), the item definitions, the material, the models and the fallback rule all stay as frozen. In particular, the first indented line of a chunk stays in item 1. Pythia's split between 4 and 8 spaces there is the model's behaviour, not a measurement fault. Any analysis leaving chunk starts out is reported separately as post hoc and does not enter the gate.

**Reporting.** The notebook, and any later report, give both the pre-fix and the post-fix gate tables, whichever way the models land. If the fix changes a model's gate verdict, that is said plainly in the gate entry.
