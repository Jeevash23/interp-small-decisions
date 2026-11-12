"""D5 round 2, exploration 6 (exploratory; CHARTER-2.md Q2): how far back do the carrier heads look?
Usage: r2_exp06_attn.py <model> <part> <first> <n> <heads e.g. L19H0,L17H6> [per_para=3] [seed=16]
At up to `per_para` word starts per paragraph (word 12 or later) in the upper-case passage, the attention
of each named head from the prediction position, summed by the word each key token belongs to, counted
back from the current word (0 = the word the prediction position is in, 1 = the word before, ...);
tokens outside any word (spaces, punctuation-only tokens) count with the word before them; the beginning-of-
sequence token (Gemma) is the remainder. Bins: 0, 1, 2-3, 4-7, 8+ (text token 0 excluded), the first
text token on its own (an attention sink in models without a BOS token), and the BOS remainder. Writes outputs/r2/exp06_<model>_<part>_<first>.json"""
import re, sys, time
from d5common2 import *  # noqa: F401,F403

short, part = sys.argv[1], sys.argv[2]; FIRST, N = int(sys.argv[3]), int(sys.argv[4])
HEADS = [tuple(int(x) for x in re.fullmatch(r"L(\d+)H(\d+)", s).groups()) for s in sys.argv[5].split(",")]
PER = int(sys.argv[6]) if len(sys.argv) > 6 else 3; SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 16
M_ = Model(short, attn="eager"); m = M_.m; tok = M_.tok
BINS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 10 ** 6)]
rows = load_part2(part, seed=0)[FIRST:FIRST + N]; rng = random.Random(SEED); t0 = time.time(); out = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); rs = runs(text)
    cand = [i for i, (s, e) in enumerate(rs) if i >= 12 and text[s - 1] == " "]
    for i in sorted(rng.sample(cand, min(PER, len(cand)))):
        pre = text[:rs[i][0] - 1].upper()
        e = tok(pre, return_offsets_mapping=True, add_special_tokens=False); ids = e["input_ids"]
        # word index of each token: the last run starting at or before its first character
        widx = []
        for a, b in e["offset_mapping"]:
            k = a
            while k < b and pre[k].isspace(): k += 1
            widx.append(max([j for j, (s, t) in enumerate(rs) if s <= k] or [0]))
        cur = widx[-1]
        with torch.no_grad(): o = m.model(torch.tensor([ids], device=DEV), output_attentions=True)
        rec = {"para": FIRST + pi, "pageid": r["pageid"], "run": i}
        for (l, h) in HEADS:
            a = o.attentions[l][0, h, -1].float().cpu().numpy()
            dist = np.array([cur - w for w in widx]); dist[0] = -1  # first text token binned separately
            rec[f"L{l}H{h}"] = [float(a[(dist >= lo) & (dist <= hi)].sum()) for lo, hi in BINS] + [float(a[0]), float(1 - a.sum())]
        out.append(rec)
        del o
    if pi % 10 == 0: print(pi, round(time.time() - t0), flush=True)
save({"model": short, "part": part, "rows": [FIRST, FIRST + N], "bins": ["0", "1", "2-3", "4-7", "8+", "first", "bos"], "records": out}, f"r2/exp06_{short}_{part}_{FIRST}.json")
for (l, h) in HEADS:
    k = f"L{l}H{h}"; v = np.mean([rc[k] for rc in out], 0)
    print(k, "mean attention by words back [0, 1, 2-3, 4-7, 8+, first, bos]:", np.round(v, 3).tolist())
