# D4 round 2: amendments to `CHARTER-2.md`

Dated changes after the charter was frozen (8 Nov 2026, 16:54 IST). The charter itself is unchanged.

## Amendment 1 (9 Nov 2026, 09:25 IST): schedule after an 11.5-hour stall

**What happened.** From 21:40 on 8 Nov to 09:09 on 9 Nov nothing ran. The waiting loops used `pgrep -f "<pattern>"` from a shell whose own command line held the same pattern, so each loop found itself and never ended. One of them (waiting for the dry run to finish) also held up `code/r2chain_e.sh`, whose own `pgrep` found that loop, so the full Gemma-4-E2B exploration run (queued 21:32) did not start. The loops were stopped at 09:09; the Gemma run started at once. Waits now use `systemctl --user is-active` on a unit, or a PID.

Separately, the training-checkpoint run E06 (question 5) finished two of six revisions and then ran out of GPU memory at 21:40, because each revision's model was not freed before the next was loaded. That was a code fault: fixed at 09:10, and the four missing revisions are re-run one per job.

**Changes.**
1. Exploration was to end by 9 Nov 02:00 and `PROTOCOL-2.md` to be frozen by about 03:00. Both are moved: exploration ends when the Gemma-4-E2B run (queued before 02:00 and unchanged since) has finished, and the protocol is frozen straight after, by 12:00 at the latest. No other exploration is added, and no threshold is set from anything except exploration material.
2. The four E06 checkpoint revisions run after the window, as exploratory and descriptive results only. No prediction in `PROTOCOL-2.md` depends on them.
3. The confirmation's stop rule (no job starts after 9 Nov 18:00) is unchanged.
4. Held-out material: no held2 frame and no held-out number was read in the delay. `code/r2common.py` refuses held material while `PROTOCOL-2.md` is writable, and the dry run's item lists, rebuilt from their seeds, are all exploration material (`NOTEBOOK.md`, 9 Nov 09:20).
