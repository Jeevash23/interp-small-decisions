"""Exp01 (exploratory): residual patching by layer and position between numbers differing in one digit.
Usage: exp01_patch.py <model> [--n 160] [--digits 2,3,4]

Pair types (same frame, same number of digits and same token split; source -> target):
  last   : last digit differs among 1-4, tens digit not 1 (e.g. 341 -> 342): nd -> st etc.
  intoteen: tens digit 2-9 in source, 1 in target, last digit 1-3 (322 -> 312): source nd, target th
  outteen : tens digit 1 in source, 2-9 in target (312 -> 322): source th, target nd
Each type with 2-digit and 3-digit numbers. Metric per (layer, position): fraction of the
source-minus-target logit difference (source suffix minus target suffix) moved by patching the
source residual into the target run. Positions are counted from the end of the prompt (-1 = last token).
"""
import sys, random, json
from common import *
from hooks import Resid, batch_ids

name = sys.argv[1]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 160
DIG = [int(x) for x in sys.argv[sys.argv.index("--digits") + 1].split(",")] if "--digits" in sys.argv else [2, 3]
rng = random.Random("d4-exp01")
frames = [f for f in explore_frames() if f["family"] in ("rankings", "positions", "anniversaries")]
nums = set(explore_numbers())


def pairs(kind, digits):
    out = []
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    cand = [n for n in nums if lo <= n <= hi]
    for _ in range(20000):
        t = rng.choice(cand); s = None
        if kind == "last":
            if (t // 10) % 10 == 1 or t % 10 not in (1, 2, 3, 4): continue
            s = t - t % 10 + rng.choice([d for d in (1, 2, 3, 4) if d != t % 10])
        elif kind == "intoteen":
            if (t // 10) % 10 != 1 or t % 10 not in (1, 2, 3): continue
            s = t + 10 * rng.randint(1, 8)
        elif kind == "outteen":
            if (t // 10) % 10 == 1 or (t // 10) % 10 == 0 or t % 10 not in (1, 2, 3): continue
            s = t - 10 * ((t // 10) % 10 - 1)
        if s not in nums or suffix(s) == suffix(t): continue
        f = rng.choice([f for f in frames if f["hi"] >= max(s, t)])
        out.append((f["text"].replace("{N}", str(s)), f["text"].replace("{N}", str(t)), s, t))
        if len(out) == N: break
    return out


tok, model = load(name)
cols = suffix_ids(tok)
R = Resid(model)
L = len(R.bl)
res = {}


@torch.no_grad()
def run(ids, patch=None, capture=False):
    R.patch = patch or {}; R.capture = capture; R.cap = {}
    h = model.base_model(input_ids=ids).last_hidden_state[:, -1]
    R.patch = {}; R.capture = False
    return unembed(model, h)[:, cols]


for digits in DIG:
    for kind in ("last", "intoteen", "outteen"):
        P = pairs(kind, digits)
        # keep pairs whose source and target tokenise to the same length and split
        P = [p for p in P if len(encode(tok, p[0])) == len(encode(tok, p[1]))
             and [len(x) for x in number_pieces(tok, p[0], p[2])[1]] == [len(x) for x in number_pieces(tok, p[1], p[3])[1]]]
        if not P: continue
        npos = len(number_pieces(tok, P[0][0], P[0][2])[0]) + 1
        si = torch.tensor([SUFFIXES.index(suffix(p[2])) for p in P]); ti = torch.tensor([SUFFIXES.index(suffix(p[3])) for p in P])
        fr = np.zeros((L, npos)); cnt = 0
        base_t = []; base_s = []
        for b in range(0, len(P), 32):
            B = P[b:b + 32]
            # group by length so batches are rectangular
            groups = {}
            for i, p in enumerate(B): groups.setdefault(len(encode(tok, p[0])), []).append(i)
            for _, idx in groups.items():
                src = batch_ids(tok, [B[i][0] for i in idx]); tgt = batch_ids(tok, [B[i][1] for i in idx])
                s_, t_ = si[[b + i for i in idx]], ti[[b + i for i in idx]]
                ls = run(src, capture=True); caps = dict(R.cap)
                lt = run(tgt)
                ar = torch.arange(len(idx))
                ld = lambda lg: (lg[ar, s_] - lg[ar, t_]).cpu().numpy()
                d_s, d_t = ld(ls), ld(lt)
                base_s += list(d_s); base_t += list(d_t)
                T = src.shape[1]
                for l in range(L):
                    for k in range(npos):
                        pos = torch.full((len(idx),), T - 1 - k, device=DEVICE)
                        lp = run(tgt, patch={l: (pos, caps[l][torch.arange(len(idx), device=DEVICE), pos])})
                        fr[l, k] += np.sum((ld(lp) - d_t) / np.maximum(d_s - d_t, 1e-3))
                cnt += len(idx)
        fr /= cnt
        key = f"{kind}_{digits}d"
        res[key] = dict(n=cnt, frac=fr.tolist(), src_correct=float(np.mean(np.array(base_s) > 0)),
                        tgt_correct=float(np.mean(np.array(base_t) < 0)), mean_gap=float(np.mean(np.array(base_s) - np.array(base_t))),
                        pieces=number_pieces(tok, P[0][0], P[0][2])[1])
        print(f"{key}: n={cnt} src_ok={res[key]['src_correct']:.2f} tgt_ok={res[key]['tgt_correct']:.2f}", flush=True)
        for l in range(0, L, max(1, L // 12)):
            print(f"   L{l:2d} " + " ".join(f"pos{-1 - k}:{fr[l, k]:+.2f}" for k in range(npos)), flush=True)
save_json(res, f"exp01_patch_{name}_" + "".join(map(str, DIG)) + f"_n{N}.json")
