"""Exp05 (exploratory): necessity of candidate heads for the teen check.
Usage: exp05_ablate.py <model> --heads 0.2,1.6 --digits 2,3,4 [--per 96] [--random 20] [--pool 0-2] [--part explore|conf]

Items: exploration frames (rankings, positions, anniversaries) and exploration numbers, `per` teen (x11-x13)
and `per` non-teen ending in 1-3 for each digit count. Each listed head's output (before the output projection)
at the LAST position is replaced by its mean over all items of that digit count (mean ablation).
Measures per digit count: teen effect and AUROC of m = logit(last-digit suffix) - logit(th) between non-teen
and teen items (as in code/teen_effect.py), and accuracy on each class. Controls: `random` sets of the same
size drawn from heads in the `pool` layers, excluding the candidates.
Writes outputs/exp05_<model>_<heads>.json.
"""
import sys, random
from common import *
from teen_effect import auroc

name = sys.argv[1]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
HEADS = [tuple(map(int, x.split("."))) for x in arg("--heads", "0.2").split(",")]
DIG = [int(x) for x in arg("--digits", "2,3,4").split(",")]
PER, NR = int(arg("--per", "96")), int(arg("--random", "20"))
lo, hi = map(int, arg("--pool", "0-2").split("-"))
PART = arg("--part", "explore")   # 'conf' only under the frozen PROTOCOL.md (test Q1)
require_frozen_protocol(PART)
rng = random.Random("d4-exp05" if PART == "explore" else f"d4-exp05-{PART}")
frames = [f for f in part_frames(PART) if f["family"] in ("rankings", "positions", "anniversaries")]


def oproj(b):
    for p in ("self_attn.o_proj", "attention.dense", "attn.c_proj"):
        m = b
        try:
            for k in p.split("."): m = getattr(m, k)
            return m
        except AttributeError:
            pass
    raise AttributeError


tok, model = load(name)
cols = suffix_ids(tok)
BL = blocks(model); L = len(BL)
H = model.config.num_attention_heads
dh = getattr(model.config, "head_dim", None) or model.config.hidden_size // H
S = dict(abl={}, cap=None)


def hook(l):
    def f(m, a):
        x = a[0]
        if S["cap"] is not None:
            S["cap"].setdefault(l, []).append(x[:, -1].detach().float().cpu())
        if l in S["abl"]:
            x = x.clone()
            for h, v in S["abl"][l]:
                x[:, -1, h * dh:(h + 1) * dh] = v.to(x.dtype).to(x.device)
            return (x,) + tuple(a[1:])
    return f


for l, b in enumerate(BL): oproj(b).register_forward_pre_hook(hook(l))


def draw(d, cls):
    a, z = 10 ** (d - 1), 10 ** d - 1
    while True:
        n = rng.randint(a, z)
        n = n - n % 100 + rng.choice([11, 12, 13]) if cls == "teen" else n - n % 10 + rng.choice([1, 2, 3])
        if a <= n <= z and in_part(n, PART) and (cls == "teen" or n % 100 not in (11, 12, 13)):
            return n


def measure(items):
    # batches must have equal token lengths, so that position -1 is every row's last real token
    lg = np.zeros((len(items), 4), dtype=np.float32)
    by = {}
    for i, it in enumerate(items): by.setdefault(len(encode(tok, it[0])), []).append(i)
    for _, idx in sorted(by.items()):
        l_, _ = last_logits(model, tok, [items[i][0] for i in idx], cols, bs=64)
        lg[idx] = l_
    m = np.array([l[n % 10 - 1] - l[3] for (_, n, c), l in zip(items, lg)])
    ok = np.array([SUFFIXES[int(np.argmax(l))] == suffix(n) for (_, n, c), l in zip(items, lg)])
    te = np.array([c == "teen" for _, _, c in items])
    return dict(effect=float(m[~te].mean() - m[te].mean()), auroc=auroc(m[~te], m[te]),
                acc_teen=float(ok[te].mean()), acc_end123=float(ok[~te].mean()))


cands = [(l, h) for l in range(lo, hi + 1) for h in range(H) if (l, h) not in HEADS]
rand_sets = [rng.sample(cands, len(HEADS)) for _ in range(NR)]
res = dict(heads=HEADS, per=PER, digits={})
for d in DIG:
    items = []
    for cls in ("teen", "end123"):
        for _ in range(PER):
            n = draw(d, cls); f = rng.choice([f for f in frames if f["hi"] >= n] or frames)
            items.append((f["text"].replace("{N}", str(n)), n, cls))
    # means of every head output at the last position over these items
    S["cap"] = {}; S["abl"] = {}
    base = measure(items)
    means = {l: torch.cat(v).mean(0) for l, v in S["cap"].items()}; S["cap"] = None

    def ablate(hs):
        S["abl"] = {}
        for l, h in hs: S["abl"].setdefault(l, []).append((h, means[l][h * dh:(h + 1) * dh]))
        r = measure(items); S["abl"] = {}
        return r
    cand = ablate(HEADS)
    singles = {f"{l}.{h}": ablate([(l, h)]) for l, h in HEADS} if len(HEADS) > 1 else {}
    rnd = [ablate(s) for s in rand_sets]
    re_ = np.array([r["effect"] for r in rnd])
    res["digits"][d] = dict(base=base, candidate=cand, singles=singles, random=rnd,
                            random_effect_mean=float(re_.mean()), random_effect_min=float(re_.min()),
                            frac_effect_kept=cand["effect"] / base["effect"],
                            random_frac_kept_max=float((re_ / base["effect"]).max()),
                            random_frac_kept_min=float((re_ / base["effect"]).min()))
    print(f"{d} digits: base effect {base['effect']:+.2f} auroc {base['auroc']:.3f} acc teen {base['acc_teen']:.3f} end123 {base['acc_end123']:.3f}")
    print(f"   ablate {HEADS}: effect {cand['effect']:+.2f} ({cand['effect'] / base['effect']:.2f} kept) auroc {cand['auroc']:.3f} "
          f"acc teen {cand['acc_teen']:.3f} end123 {cand['acc_end123']:.3f}")
    for k, v in singles.items():
        print(f"   ablate {k}: effect {v['effect']:+.2f} ({v['effect'] / base['effect']:.2f} kept) acc teen {v['acc_teen']:.3f}")
    print(f"   random sets ({NR}): kept {res['digits'][d]['random_frac_kept_min']:.2f}-{res['digits'][d]['random_frac_kept_max']:.2f}, "
          f"mean {re_.mean() / base['effect']:.2f}")
tag = "-".join(f"{l}.{h}" for l, h in HEADS)
if PART == "explore":
    save_json(res, f"exp05_{name}_{tag}.json")
else:
    (OUT / PART).mkdir(exist_ok=True); save_json(res, f"{PART}/exp05_{name}_{tag}.json")
