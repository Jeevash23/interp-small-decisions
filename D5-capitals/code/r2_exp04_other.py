"""D5 round 2, exploration 4 (exploratory; CHARTER-2.md Q4): do acronyms, initial capitals and
capitalised headings use the capitals direction?
Usage: r2_exp04_other.py <model> <part> <first> <n> <hfirst> <hn>
Paragraph positions (original-case passage, word starts after the first five words):
  acr  = every word start whose true word has >= 2 letters, all upper case;
  init = up to 4 word starts (seed 21) whose true word starts with an upper-case letter and has a lower-case one;
  low  = up to 2 word starts (seed 22) whose true word is all lower case (reference).
Heading items (heading_text with the heading in capitals, and in its original case):
  hin  = word starts inside the heading from its second word on;
  hbody = the first word of the next paragraph (prediction position: the blank line after the heading).
At each position, at the direction layer l: the projection of the residual on the unit direction u, and
the top-1 next token after (0) nothing, (1) removing d (component along d set to the midpoint of the
round-1 upper and lower means), (2) the same for a random direction r, (3) subtracting d, (4) subtracting
r, (5) adding d, (6) adding r. Writes outputs/r2/exp04_<model>_<part>_<first>.json"""
import re, sys, time
from collections import Counter
from d5common2 import *  # noqa: F401,F403
from d5common2 import LastBatch, load_headings, heading_text

short, part = sys.argv[1], sys.argv[2]; FIRST, N, HF, HN = (int(x) for x in sys.argv[3:7])
LAYER = {"gemma": 24, "qwen3-17b": 22}[short]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; LB = LastBatch(M_)
md = torch.load(OUT / f"exp01_{short}_means.pt")
mu, ml = md["means"]["upper"].float()[LAYER + 1], md["means"]["lower"].float()[LAYER + 1]
D = (mu - ml); g = torch.Generator().manual_seed(0)
Rall = torch.randn((md["means"]["upper"].shape), generator=g)
R = Rall[LAYER + 1]; R = R / R.norm() * D.norm()  # same construction as round 1 (seed 0, row l+1)
mid = (mu + ml) / 2; u = D / D.norm(); ur = R / R.norm()
PU, PL = float(mu @ u), float(ml @ u)


def remove(h, dirn, mid):
    v = dirn / dirn.norm(); return h - ((h - mid) @ v) * v


def measure(ids):
    m.capture = True
    with torch.no_grad(): m.model(torch.tensor([ids], device=DEV), logits_to_keep=1)
    m.capture = False
    h = m.rcap[LAYER][-1].float().cpu()
    LB.clear(); LB.prime(ids)
    vals = [h, remove(h, D, mid), remove(h, R, mid), h - D, h - R, h + D, h + R]
    m.rpatch = {LAYER: [(-1, v.to(DEV), j) for j, v in enumerate(vals)]}
    lg = LB(ids, len(vals)); m.rpatch = {}
    return {"proj": round(float(h @ u), 3), "tops": [vs[int(t)] for t in lg.argmax(-1)]}


t0 = time.time(); out = {"para": [], "head": []}
rows = load_part2(part, seed=0)[FIRST:FIRST + N]; rng = random.Random(21); rng2 = random.Random(22)
for pi, r in enumerate(rows):
    text = cut(r["text"]); ids, first = encode(tok, text); ws, _ = positions(text, first); rs = runs(text)
    kinds = {}
    for k, i in ws.items():
        w = text[rs[i][0]:rs[i][1]]
        if len(w) >= 2 and w.isupper(): kinds[k] = "acr"
        elif w[0].isupper() and any(c.islower() for c in w): kinds.setdefault(k, "init?")
        elif w.islower(): kinds.setdefault(k, "low?")
    init = [k for k, v in kinds.items() if v == "init?"]; low = [k for k, v in kinds.items() if v == "low?"]
    pick = [(k, "acr", ws[k]) for k, v in kinds.items() if v == "acr"]
    pick += [(k, "init", ws[k]) for k in rng.sample(init, min(4, len(init)))] + [(k, "low", ws[k]) for k in rng2.sample(low, min(2, len(low)))]
    for k, kind, i in sorted(pick):
        rec = {"para": FIRST + pi, "pageid": r["pageid"], "kind": kind, "word": text[rs[i][0]:rs[i][1]], **measure(ids[:k])}
        out["para"].append(rec)
    if pi % 20 == 0: print("para", pi, round(time.time() - t0), flush=True)

items = load_headings(part, seed=0)[HF:HF + HN]
for hi, it in enumerate(items):
    for caps in (True, False):
        text, hs, bs = heading_text(it, caps); ids, first = encode(tok, text)
        rs = runs(text); starts = {s: j for j, (s, e) in enumerate(rs)}
        hw = [j for j, (s, e) in enumerate(rs) if hs <= s < bs - 2]
        for k, c in enumerate(first):
            if c < 0 or k == 0 or c not in starts: continue
            j = starts[c]
            if j in hw[1:] and text[c - 1] == " ":
                out["head"].append({"item": HF + hi, "pageid": it["pageid"], "caps": caps, "kind": "hin", "word": text[c:rs[j][1]], **measure(ids[:k])})
            elif c == bs:
                out["head"].append({"item": HF + hi, "pageid": it["pageid"], "caps": caps, "kind": "hbody", "word": text[c:rs[j][1]], **measure(ids[:k])})
    if hi % 20 == 0: print("head", hi, round(time.time() - t0), flush=True)

out.update({"model": short, "part": part, "rows": [FIRST, FIRST + N], "hrows": [HF, HF + HN], "layer": LAYER, "proj_upper_mean": PU,
            "proj_lower_mean": PL, "seconds": round(time.time() - t0)})
save(out, f"r2/exp04_{short}_{part}_{FIRST}.json")


def cls(s):
    s = s.strip(); let = [c for c in s if c.isalpha()]
    if not let: return "none"
    if len(let) >= 2 and all(c.isupper() for c in let): return "ALL"
    if let[0].isupper() and any(c.islower() for c in let): return "Init"
    if len(let) == 1 and let[0].isupper(): return "One"
    if all(c.islower() for c in let): return "low"
    return "mix"


NM = ["base", "rm_d", "rm_r", "sub_d", "sub_r", "add_d", "add_r"]
print("proj means: upper %.2f lower %.2f" % (PU, PL))
for grp, recs in (("para", out["para"]), ("head", out["head"])):
    keys = sorted({(rc["kind"], rc.get("caps")) for rc in recs}, key=str)
    for key in keys:
        g = [rc for rc in recs if (rc["kind"], rc.get("caps")) == key]
        print(key, "n", len(g), "proj %.2f" % np.mean([rc["proj"] for rc in g]),
              "base:", dict(Counter(cls(rc["tops"][0]) for rc in g)))
        for base_cls in ("ALL", "Init", "low"):
            gg = [rc for rc in g if cls(rc["tops"][0]) == base_cls]
            if len(gg) < 5: continue
            print("   base", base_cls, len(gg), {NM[j]: dict(Counter(cls(rc["tops"][j]) for rc in gg).most_common(3)) for j in range(1, 7)})
