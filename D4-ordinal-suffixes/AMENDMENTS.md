# D4 amendments to CHARTER.md

## Amendment 1 (7 Nov 2026, about 21:50 IST): numbers above 9,999

**Why.** The gate showed that OLMo-2-1B gets four-digit numbers ending in 11–13 wrong about half the time, and the tokeniser check shows that OLMo-2 cuts digits into chunks of three from the left. A token account then predicts accuracy on teens that rises and falls with length (one token at three digits, 3+1 at four, 3+2 at five, 3+3 at six, 3+3+1 at seven). Testing that needs numbers above 9,999, which the frozen material does not cover. No model has read any number above 9,999 under D4.

**Change.** Numbers from 10,000 to 9,999,999 are split by the charter's rule unchanged: the SHA-256 of `D4-num-<N>` modulo 10, 0–5 exploration, 6–8 confirmation, 9 reserve. The rule is deterministic, so no new file is written; `code/common.py` applies it through `is_explore_number()`. Exploration uses only exploration numbers, in exploration frames. Confirmation and reserve numbers above 9,999 are read only under a frozen `PROTOCOL.md`.

**Not changed.** Questions, models, gate, measures, budget and stop rules. Pythia-1B replaced Pythia-410M as the multi-digit main model under the charter's own fallback rule (Pythia-410M missed gate condition 2), so that needs no amendment; it is recorded in `NOTEBOOK.md`.
