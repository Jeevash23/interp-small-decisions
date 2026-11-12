# D3 round 2: amendments to `CHARTER-2.md`

`CHARTER-2.md` (frozen 8 Nov 2026, 16:54 IST) is never edited. The charter says round-2 amendments go in `AMENDMENTS.md`, but that file was frozen with round 1's protocol (digest in `FROZEN.md`, read-only). Round-2 amendments are therefore listed here, dated, in order.

## R2-1 (9 Nov 2026, about 09:30 IST): training checkpoints downloaded

`CHARTER-2.md` says "No downloads of model weights" and parks checkpoint downloads for the author. On 8 Nov at about 17:00 the author approved research downloads through the coordinating session ("no need to give permission for research works, the limit is fine"; keep at least 50 GB free; `../DECISIONS.md`, 8 Nov 17:00). Six OLMo-2-0425-1B stage-1 checkpoints (1B, 21B, 84B, 420B, 1,007B and 4,001B tokens; 6 × 5.6 GiB) were downloaded on 8 Nov between 17:22 and 19:12 into the shared Hugging Face cache. Question 5 is therefore run. It stays exploratory: no prediction about training dynamics is frozen in `PROTOCOL-2.md`. No other part of the charter changes.

## R2-2 (9 Nov 2026, about 09:30 IST): lost time, no change to the stop rules

Between 8 Nov 20:00 and 9 Nov 09:10 nothing ran: every exploration job had finished by 19:59, but the waiting loop watched a log name that was never written (`../DEPTH-ROUND.md`, 9 Nov 09:10). The stop rules stay as frozen. "Drop a question with no candidate by 9 Nov 06:00": every question had a candidate beating its controls by 8 Nov 19:59 (`NOTEBOOK.md`), so none is dropped. Confirmation still aims at 9 Nov 18:00, with the hard stop at 10 Nov 12:00.
