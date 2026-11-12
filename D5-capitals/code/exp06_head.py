"""D5 exploration 6 (exploratory): what one attention head carries about case.
Usage: exp06_head.py <model> <layer> <head> [first=300] [n=50] [control heads l:h,l:h]
Same word starts as exp03/exp05 (explore paragraphs [first, first+n), seed 2, up to four per paragraph).
1. Attention of the head from the prediction position (last token of the prefix) in the upper and lower
   runs: share on the token itself, on tokens 1, 2, 3, 4-7, 8-15, 16+ back, and on BOS; and by word:
   share on the last word (word i-1), word i-2, words i-3..i-4, older.
2. Patching the head's o_proj input at the prediction position with donors:
   same_lower: the same word's lower-case run (case and token identity both differ);
   other_upper: the upper run at another picked word in another paragraph (same case, other content);
   other_lower: the lower run at another word in another paragraph (other case, other content);
   mean_lower: the mean over all lower-run prefixes.
   And the reverse into the lower run (same_upper, other_lower, other_upper, mean_upper).
   Control heads (other heads, patched from same_lower / same_upper) are reported alongside.
   Measures: share of D = logp(U) - logp(Lc) moved, top-1 case flips, and flips to the same word.
Writes outputs/exp06_<model>_L<l>H<h>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short, LY, HD = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
FIRST = int(sys.argv[4]) if len(sys.argv) > 4 else 300; N = int(sys.argv[5]) if len(sys.argv) > 5 else 50
CTRL = [tuple(int(a) for a in x.split(":")) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else []
rows = load_part("explore", seed=0)[FIRST:FIRST + N]
M_ = Model(short, attn="eager"); m = M_.m; vs = M_.vs; tok = M_.tok
HEADS = [(LY, HD)] + CTRL


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


def word_of_tokens(text, ids, rs):
    """Run index of each token (by its last non-space character), -1 if it has no letter run."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == list(ids)
    starts = [s for s, e in rs]; out = []
    import bisect
    for s, e in enc["offset_mapping"]:
        j = bisect.bisect_right(starts, e - 1) - 1
        out.append(j if j >= 0 and rs[j][0] <= e - 1 < rs[j][1] + 1 else -1)
    return out


OFFB = [("0", 0, 1), ("1", 1, 2), ("2", 2, 3), ("3", 3, 4), ("4-7", 4, 8), ("8-15", 8, 16), ("16+", 16, 10**6)]
WB = [("w-1", 1, 2), ("w-2", 2, 3), ("w-3..4", 3, 5), ("older", 5, 10**6)]

# pass 1: picks, attention, captured head values and baseline log-probs
rng = random.Random(2); t0 = time.time(); items = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); V = versions(text); rs = runs(text)
    enc = {v: encode(tok, V[v]) for v in ("upper", "lower")}
    wsu, _ = positions(V["upper"], enc["upper"][1]); wsl, _ = positions(V["lower"], enc["lower"][1])
    both = sorted(set(wsu.values()) & set(wsl.values()))
    wt = {v: word_of_tokens(V[v], enc[v][0], rs) for v in ("upper", "lower")}
    for i in sorted(rng.sample(both, min(4, len(both)))):
        kk = {"upper": [k for k, j in wsu.items() if j == i][0], "lower": [k for k, j in wsl.items() if j == i][0]}
        it = {"para": FIRST + pi, "run": i, "U": int(enc["upper"][0][kk["upper"]]), "Lc": int(enc["lower"][0][kk["lower"]])}
        for v in ("upper", "lower"):
            pre = enc[v][0][:kk[v]]; T = len(pre)
            m.capture = True
            with torch.no_grad():
                o = m.model(torch.tensor([pre], device=DEV), output_attentions=True)
                lg = m.model(torch.tensor([pre], device=DEV), logits_to_keep=1).logits[0, -1].float()
            m.capture = False
            att = o.attentions[LY][0, HD, -1].float().cpu().numpy()  # (T,) after BOS strip
            offs = {k: float(att[max(T - b, 0):max(T - a, 0)].sum()) for k, a, b in OFFB}
            offs["bos"] = float(1 - att.sum())
            wd = [i - w if w >= 0 else 10**6 for w in wt[v][:T]]
            byw = {k: float(sum(att[t] for t in range(T) if a <= wd[t] < b)) for k, a, b in WB}
            byw["noword"] = float(sum(att[t] for t in range(T) if wd[t] >= 10**6))
            lp = torch.log_softmax(lg, -1)
            it[v] = {"pre": pre, "att_off": offs, "att_word": byw,
                     "z": {(l, h): m.cap[l][-1][m.hs(l, h)].clone() for l, h in HEADS},
                     "D0": float(lp[it["U"]] - lp[it["Lc"]]), "top0": vs[int(lp.argmax())]}
        items.append(it)
    if pi % 10 == 0: print("p1", pi, round(time.time() - t0), flush=True)

mean_z = {v: {hh: torch.stack([it[v]["z"][hh] for it in items]).mean(0) for hh in HEADS} for v in ("upper", "lower")}

# pass 2: patches. Donor for "other_*" is the item 7 places on (a different paragraph in almost all cases)
nI = len(items); res = []
for n, it in enumerate(items):
    ot = items[(n + 7) % nI]
    rec = {"para": it["para"], "run": it["run"], "patched": {}}
    for v, w in (("upper", "lower"), ("lower", "upper")):
        jobs = [(f"same_{w}", (LY, HD), it[w]["z"][(LY, HD)]), (f"other_{v}", (LY, HD), ot[v]["z"][(LY, HD)]),
                (f"other_{w}", (LY, HD), ot[w]["z"][(LY, HD)]), (f"mean_{w}", (LY, HD), mean_z[w][(LY, HD)])]
        jobs += [(f"ctrl_L{l}H{h}_same_{w}", (l, h), it[w]["z"][(l, h)]) for l, h in CTRL]
        m.patch = {}
        for j, (_, (l, h), val) in enumerate(jobs): m.patch.setdefault(l, []).append((h, -1, val, j))
        x = torch.tensor([it[v]["pre"]] * len(jobs), device=DEV)
        with torch.no_grad(): lg = m.model(x, logits_to_keep=1).logits[:, -1].float()
        m.patch = {}
        lp = torch.log_softmax(lg, -1)
        rec["patched"][v] = {name: [round(float(lp[j][it["U"]] - lp[j][it["Lc"]]), 3), vs[int(lp[j].argmax())]]
                             for j, (name, _, _) in enumerate(jobs)}
    res.append(rec)
    if n % 40 == 0: print("p2", n, round(time.time() - t0), flush=True)

summ = {"attention": {}, "patch": {}}
for v in ("upper", "lower"):
    summ["attention"][v] = {"by_offset": {k: float(np.mean([it[v]["att_off"][k] for it in items])) for k in [b[0] for b in OFFB] + ["bos"]},
                            "by_word": {k: float(np.mean([it[v]["att_word"][k] for it in items])) for k in [b[0] for b in WB] + ["noword"]}}
for v, want in (("upper", "lower"), ("lower", "upper")):
    idx = [n for n, it in enumerate(items) if case_of(it[v]["top0"]) == v]
    D0 = np.array([items[n][v]["D0"] for n in idx]); Dw = np.array([items[n][want]["D0"] for n in idx])
    row = {"n": len(idx)}
    for name in res[0]["patched"][v]:
        Dp = np.array([res[n]["patched"][v][name][0] for n in idx])
        flip = [n for n in idx if case_of(res[n]["patched"][v][name][1]) == want]
        kept = [n for n in flip if same_word(items[n][v]["top0"], res[n]["patched"][v][name][1])]
        row[name] = {"frac_D_moved": float((D0.mean() - Dp.mean()) / (D0.mean() - Dw.mean())),
                     "flip": len(flip) / len(idx), "flip_kept": len(kept) / len(idx)}
    summ["patch"][v] = row
for it in items:
    for v in ("upper", "lower"): it[v].pop("z"); it[v].pop("pre")
save({"model": short, "head": [LY, HD], "controls": CTRL, "rows": [FIRST, FIRST + N], "n_items": nI,
      "summary": summ, "items": items, "records": res, "seconds": round(time.time() - t0)}, f"exp06_{short}_L{LY}H{HD}.json")
print(json.dumps(summ["attention"], indent=None))
for v in summ["patch"]:
    print(v, "n", summ["patch"][v]["n"])
    for k, s in summ["patch"][v].items():
        if k != "n": print(" ", k, {a: round(b, 3) for a, b in s.items()})
