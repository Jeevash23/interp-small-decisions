"""D5 round 2, exploration 5 (exploratory; CHARTER-2.md Q3): when adding d changes the word.
Usage: r2_exp05_edit.py <model> <part> <first> <n> [per_para=4] [seed=15]
At up to `per_para` word starts per paragraph (word 12 or later), in the lower-case passage: the top-1,
and the top-1 after adding d (a = 1) or a random direction r (round-1 construction) at the direction layer;
in the upper-case passage: the top-1. Also the lower-case passage's rank of the edited top-1's lower-case
form (" " + edited piece in lower case, first token). Writes outputs/r2/exp05_<model>_<part>_<first>.json"""
import sys, time
from d5common2 import *  # noqa: F401,F403
from d5common2 import LastBatch

short, part = sys.argv[1], sys.argv[2]; FIRST, N = int(sys.argv[3]), int(sys.argv[4])
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 4; SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 15
LAYER = {"gemma": 24, "qwen3-17b": 22}[short]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; LB = LastBatch(M_)
md = torch.load(OUT / f"exp01_{short}_means.pt")
Dall = (md["means"]["upper"] - md["means"]["lower"]).float(); g = torch.Generator().manual_seed(0)
Rall = torch.randn(Dall.shape, generator=g); Rall = Rall / Rall.norm(dim=1, keepdim=True) * Dall.norm(dim=1, keepdim=True)
d, r_ = Dall[LAYER + 1], Rall[LAYER + 1]


def pre(text, rs, i, f):
    p = text[:rs[i][0] - 1]
    return p.upper() if f == "U" else p.lower()


rows = load_part2(part, seed=0)[FIRST:FIRST + N]; rng = random.Random(SEED); t0 = time.time(); out = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); rs = runs(text)
    cand = [i for i, (s, e) in enumerate(rs) if i >= 12 and text[s - 1] == " "]
    for i in sorted(rng.sample(cand, min(PER, len(cand)))):
        ids = {f: tok(pre(text, rs, i, f), add_special_tokens=False)["input_ids"] for f in ("U", "L")}
        m.capture = True
        with torch.no_grad(): lgL = m.model(torch.tensor([ids["L"]], device=DEV), logits_to_keep=1).logits[0, -1].float()
        m.capture = False
        h = m.rcap[LAYER][-1].float().cpu()
        with torch.no_grad(): lgU = m.model(torch.tensor([ids["U"]], device=DEV), logits_to_keep=1).logits[0, -1].float()
        LB.clear(); LB.prime(ids["L"])
        m.rpatch = {LAYER: [(-1, (h + d).to(DEV), 0), (-1, (h + r_).to(DEV), 1)]}
        lg = LB(ids["L"], 2); m.rpatch = {}
        e = vs[int(lg[0].argmax())]
        lw = " " + e.strip().lower() if e.strip() else e
        lt = tok(lw, add_special_tokens=False)["input_ids"][:1] if lw.strip() else []
        rank = int((lgL > lgL[lt[0]]).sum()) + 1 if lt else None
        out.append({"para": FIRST + pi, "pageid": r["pageid"], "run": i, "word": text[rs[i][0]:rs[i][1]],
                    "lo": vs[int(lgL.argmax())], "up": vs[int(lgU.argmax())], "edit": e, "rand": vs[int(lg[1].argmax())],
                    "edit_lower_rank": rank,
                    "pieces_lo": len(tok(" " + vs[int(lgL.argmax())].strip(), add_special_tokens=False)["input_ids"]),
                    "pieces_up": len(tok(" " + vs[int(lgL.argmax())].strip().upper(), add_special_tokens=False)["input_ids"])})
    if pi % 20 == 0: print(pi, round(time.time() - t0), flush=True)
save({"model": short, "part": part, "rows": [FIRST, FIRST + N], "layer": LAYER, "records": out, "seconds": round(time.time() - t0)},
     f"r2/exp05_{short}_{part}_{FIRST}.json")


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


npc = lambda w: len(tok(" " + w, add_special_tokens=False)["input_ids"])
el = [x for x in out if case_of(x["lo"]) == "lower" and case_of(x["edit"]) == "upper"]
ch = [x for x in el if not same_word(x["lo"], x["edit"])]
upch = [x for x in el if not same_word(x["lo"], x["up"])]
print("flips %d, changed %.3f; P(change | upper passage changes) %.3f, P(change | upper keeps) %.3f; changed in lower top 5 %.3f" % (
    len(el), len(ch) / len(el), np.mean([not same_word(x["lo"], x["edit"]) for x in upch]),
    np.mean([not same_word(x["lo"], x["edit"]) for x in el if x not in upch]),
    np.mean([x["edit_lower_rank"] is not None and x["edit_lower_rank"] <= 5 for x in ch])))
for e in range(3):
    gg = [x for x in el if min(npc(x["lo"].strip().upper()) - npc(x["lo"].strip()), 2) == e]
    if gg: print(" extra pieces", e, "n", len(gg), "changed %.3f" % np.mean([not same_word(x["lo"], x["edit"]) for x in gg]))
