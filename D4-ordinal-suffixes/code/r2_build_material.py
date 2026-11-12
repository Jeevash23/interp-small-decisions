"""Round 2 material (CHARTER-2.md): new frames, split explore2/held2 by SHA-256 of 'D4R2-frame-<text>' within
each family (first half by hash order explore2, rest held2). Numbers are not written to a file: their parts are
given by rules in code/r2common.py. Writes data2/frames2.jsonl, data2/MANIFEST.md, data2/SHA256SUMS.
No model is run here."""
import json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data2"; D.mkdir(exist_ok=True)
BIG = 9_999_999_999
FR = {
 "rankings": (1, BIG, [
  "In the regional quiz league, our team finished {N}",
  "Of all the bakeries reviewed in the guide, hers was rated {N}",
  "In the long-distance swim across the lake, he came {N}",
  "On the list of the most visited websites, the site now ranks {N}",
  "In the national lottery draw, her ticket came out {N}",
  "Among the hundreds of schools inspected, ours was placed {N}",
  "In the global index of innovation, the country sits {N}",
  "Out of every song released that year, this one charted {N}",
  "In the cycling time trial, the youngest rider finished {N}",
  "Among the many films in the festival, the jury ranked hers {N}",
  "In the ranking of the busiest airports, this one is now {N}",
  "Out of all the players on the server, he is currently ranked {N}",
  "In the queue for the new phone, she was standing {N}",
  "Among the cars in the endurance race, ours crossed the line {N}"]),
 "positions": (1, BIG, [
  "The robot stopped working on its {N}",
  "The bus was full, so we waited for the {N}",
  "The comet returns on its {N}",
  "He scrolled down the long list and clicked on the {N}",
  "The counter on the turnstile shows that you are the {N}",
  "The printer jammed on the {N}",
  "The ledger is enormous, and the payment is recorded in the {N}",
  "The bell has rung for the {N}",
  "The log file is huge, and the crash happened on the {N}",
  "The building is very tall, and the restaurant is on the {N}",
  "The website greeted me as its {N}",
  "The record says the volcano erupted for the {N}",
  "The train was delayed again, for the {N}",
  "The dataset is large, and the corrupted record is the {N}"]),
 "anniversaries": (1, 999, [
  "The lighthouse keepers are marking the tower's {N}",
  "The football club is preparing for its {N}",
  "The family threw a surprise party for her {N}",
  "The theatre will stage a gala for its {N}",
  "The old bridge has just reached its {N}",
  "The choir sang at the church's {N}",
  "The magazine printed a gold cover for its {N}",
  "My uncle is planning a trip for his {N}",
  "The library is holding an exhibition for its {N}",
  "The brewery released a special beer for its {N}"]),
 "dates": (1, 31, [
  "The exam has been scheduled for the {N}",
  "Our flight home leaves on the {N}",
  "The market is held on the {N}",
  "The report is due on Friday the {N}",
  "Tickets go on sale on the {N}",
  "The garden party is planned for the {N}",
  "The builders will start work on the {N}",
  "The museum is closed on the {N}",
  "His interview is on Tuesday the {N}",
  "The new timetable begins on the {N}"]),
}
old = {json.loads(l)["text"] for l in open(ROOT / "data" / "frames.jsonl")}
rows = []
for fam, (lo, hi, texts) in FR.items():
    assert len(set(texts)) == len(texts) and not (set(texts) & old)
    hs = sorted(texts, key=lambda t: hashlib.sha256(f"D4R2-frame-{t}".encode()).hexdigest())
    for i, t in enumerate(hs):
        fid = f"{fam}-r2-" + hashlib.sha256(t.encode()).hexdigest()[:8]
        rows.append(dict(fid=fid, family=fam, part="explore2" if i < len(hs) // 2 else "held2", lo=lo, hi=hi, text=t))
with open(D / "frames2.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
from collections import Counter
c = Counter((r["family"], r["part"]) for r in rows)
man = ["# D4 round-2 material manifest", "", "Built by `code/r2_build_material.py`. Only counts were taken before freezing.", "",
       "| Family | explore2 frames | held2 frames |", "|---|---|---|"]
man += [f"| {fam} | {c[(fam, 'explore2')]} | {c[(fam, 'held2')]} |" for fam in FR]
man += ["", "Numbers (rules in `code/r2common.py`, CHARTER-2.md): 2 digits all read before; 3-5 digits held2 items use",
        "round-1 conf- and reserve-class numbers (read before, new frames); 6-7 digits round-1 reserve class are held2",
        "(never read); 8-10 digits split by SHA-256 of `D4R2-num-<N>` mod 10, 0-5 explore2, 6-9 held2."]
(D / "MANIFEST.md").write_text("\n".join(man) + "\n")
(D / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((D / n).read_bytes()).hexdigest()}  {n}\n" for n in ("frames2.jsonl", "MANIFEST.md")))
print(c)
