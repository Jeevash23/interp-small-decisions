"""D4 material builder (CHARTER.md, 'Material'). Run once; writes data/ and is digested in FROZEN.md
before any model reads the parts.

Frames are split within each family by SHA-256 of their text (first half by hash order -> explore,
next 30% -> conf, rest -> reserve). Numbers 100-9999 are split by SHA-256 of 'D4-num-<N>' (60/30/10).
Numbers 1-99 appear in every part. An item belongs to a part only if its frame and (for N >= 100) its
number both belong to that part.
"""
import hashlib, json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"
D.mkdir(exist_ok=True)

FAMILIES = {
    "dates": (1, 31, [
        "The festival is held every year on the {N}",
        "Her birthday falls on the {N}",
        "The rent is due on the {N}",
        "The deadline has been moved to the {N}",
        "We are flying out on the {N}",
        "The next board meeting is on the {N}",
        "School starts again on the {N}",
        "The parcel should arrive by the {N}",
        "The concert was postponed until the {N}",
        "Payment must be received before the {N}",
        "The shop reopens on Monday the {N}",
        "The wedding took place on the {N}",
        "Applications close on the {N}",
        "The results will be published on the {N}",
        "Our anniversary dinner is booked for the {N}",
        "The match was played on Saturday the {N}",
    ]),
    "centuries": (1, 21, [
        "The chapel was built in the {N}",
        "The manuscript dates from the {N}",
        "The custom spread across Europe in the {N}",
        "The town was founded in the {N}",
        "These coins were minted in the early {N}",
        "The poem was written in the late {N}",
        "The empire collapsed in the {N}",
        "The bridge was rebuilt during the {N}",
        "Trade with the east grew rapidly during the {N}",
        "The tradition goes back to the {N}",
        "The painting is thought to come from the {N}",
        "Most of the castle walls date from the {N}",
    ]),
    "rankings": (1, 9999, [
        "Out of all the runners in the city marathon, she finished {N}",
        "In the final standings, the team ended up {N}",
        "Among the thousands of entries, his photograph was ranked {N}",
        "In this year's league table, the university is now placed {N}",
        "On the global list of the largest companies, the firm sits {N}",
        "In the national spelling contest, she came {N}",
        "Of the many applicants, he was placed {N}",
        "On the bestseller chart this week, the novel is {N}",
        "In the world rankings, the player has dropped to {N}",
        "Out of every cyclist in the race, he crossed the line {N}",
        "In the annual league table, the school came {N}",
        "Among all the cities surveyed, ours was rated {N}",
        "In the online chess tournament, she finished {N}",
        "In the city's list of busiest streets, this road comes {N}",
    ]),
    "anniversaries": (1, 999, [
        "Last week the company celebrated its {N}",
        "The town is preparing to mark its {N}",
        "Yesterday my grandmother celebrated her {N}",
        "This year the club will celebrate its {N}",
        "The couple are celebrating their {N}",
        "The university marked its {N}",
        "The newspaper is printing a special issue for its {N}",
        "Next month the cathedral celebrates its {N}",
        "The school held a party for its {N}",
        "The village fair is celebrating its {N}",
        "The museum opened a new wing to mark its {N}",
        "Friends gathered at the pub for his {N}",
    ]),
    "positions": (1, 9999, [
        "The spreadsheet has thousands of rows, and the error is in the {N}",
        "The festival is now in its {N}",
        "The assembly opened its {N}",
        "The magazine has just released its {N}",
        "She tried the password again and finally got in on the {N}",
        "The lift stopped at the {N}",
        "The catalogue has many items, and the one you want is the {N}",
        "The novel is now in its {N}",
        "The sequence is long, and the value we need is the {N}",
        "Among the many visitors that day, she was the {N}",
        "The parliament is now in its {N}",
        "The pages are numbered, and the quote appears on the {N}",
        "The tower has many floors, and our office is on the {N}",
        "The orchestra performed the symphony for the {N}",
    ]),
}


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def suffix(n):
    if n % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def number_part(n):
    if n < 100:
        return "all"
    r = int(h(f"D4-num-{n}"), 16) % 10
    return "explore" if r < 6 else "conf" if r < 9 else "reserve"


frames = []
for fam, (lo, hi, fs) in FAMILIES.items():
    order = sorted(fs, key=h)
    k = len(order); ne = round(k * 0.5); nc = round(k * 0.3)
    for i, f in enumerate(order):
        part = "explore" if i < ne else "conf" if i < ne + nc else "reserve"
        frames.append(dict(fid=f"{fam}-{h(f)[:8]}", family=fam, part=part, lo=lo, hi=hi, text=f))

with open(D / "frames.jsonl", "w") as fo:
    for f in frames:
        fo.write(json.dumps(f) + "\n")
nums = {n: number_part(n) for n in range(1, 10000)}
json.dump({str(n): p for n, p in nums.items()}, open(D / "numbers.json", "w"))

counts = {}
for part in ("explore", "conf", "reserve"):
    c = 0
    with open(D / f"part-{part}.jsonl", "w") as fo:
        for f in frames:
            if f["part"] != part:
                continue
            for n in range(f["lo"], f["hi"] + 1):
                if nums[n] not in ("all", part):
                    continue
                fo.write(json.dumps(dict(fid=f["fid"], family=f["family"], n=n, target=suffix(n),
                                         prefix=f["text"].replace("{N}", str(n)))) + "\n")
                c += 1
    counts[part] = c

lines = ["# D4 material manifest", "", "Built by `code/build_material.py`. Only counts were taken before freezing.", "",
         "| Part | Frames | Items |", "|---|---|---|"]
for part in ("explore", "conf", "reserve"):
    lines.append(f"| {part} | {sum(f['part'] == part for f in frames)} | {counts[part]} |")
lines += ["", "Numbers 100-9999 by part: " + ", ".join(
    f"{p} {sum(v == p for v in nums.values())}" for p in ("explore", "conf", "reserve"))]
(D / "MANIFEST.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
