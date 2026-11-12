"""D5 exploration 7 (exploratory): is the case direction needed, and does it work on ordinary text?
Usage: exp07_ablate.py <model> <layers a,b,c> [first=400] [n=50]
d_l = exp01 mean difference (upper minus lower, paragraphs 150-300) after layer l; r = random direction
(seed 0, same norm). Word starts in explore paragraphs [first, first+n), up to four per paragraph (seed 4).
At the prediction position only:
 A. Removal: in the upper run and in the lower run, set the residual's component along d_l to the midpoint
    of the two runs' mean components (so the run carries no case on that axis); same for r. Report the
    share of upper-case top-1, and the share whose top-1 is still the unpatched word in any case.
 B. Ordinary text: in the original-case passage, add a*d_l (a = 1, 2) or a*r. At word starts where the
    unpatched top-1 is lower case: share turned upper case, and share turned into the same word in upper
    case (case-insensitive prefix relation). Also subtract a*d_l at word starts whose top-1 is
    capitalised (title or upper): share turned lower case, same word.
Writes outputs/exp07_<model>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short = sys.argv[1]; LAY = [int(x) for x in sys.argv[2].split(",")]
FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else 400; N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
rows = load_part("explore", seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok
md = torch.load(OUT / f"exp01_{short}_means.pt")
mu, ml = md["means"]["upper"].float(), md["means"]["lower"].float()
D = mu - ml
g = torch.Generator().manual_seed(0)
R = torch.randn(D.shape, generator=g); R = R / R.norm(dim=1, keepdim=True) * D.norm(dim=1, keepdim=True)
ALPHAS = [1.0, 2.0]


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


def capcase(s):
    s = s.strip(); return bool(s) and s[0].isalpha() and s[0].isupper()


def remove(h, d, mid):
    u = d / d.norm(); return h - ((h - mid) @ u) * u


rng = random.Random(4); t0 = time.time(); out = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); V = versions(text)
    enc = {v: encode(tok, V[v]) for v in ("upper", "lower", "orig")}
    ws = {v: positions(V[v], enc[v][1])[0] for v in enc}
    both = sorted(set(ws["upper"].values()) & set(ws["lower"].values()) & set(ws["orig"].values()))
    for i in sorted(rng.sample(both, min(4, len(both)))):
        rec = {"para": FIRST + pi, "run": i}
        for v in ("upper", "lower", "orig"):
            k = [k for k, j in ws[v].items() if j == i][0]; ids = enc[v][0][:k]
            m.capture = True
            with torch.no_grad(): lg = m.model(torch.tensor([ids], device=DEV), logits_to_keep=1).logits[0, -1]
            m.capture = False
            Hc = {l: m.rcap[l][-1].float().cpu() for l in LAY}
            rec[v] = {"top0": vs[int(lg.argmax())]}
            jobs = []
            for l in LAY:
                d, rr = D[l + 1], R[l + 1]
                if v in ("upper", "lower"):
                    mid = (mu[l + 1] + ml[l + 1]) / 2
                    jobs += [(f"rm_dir_{l}", l, remove(Hc[l], d, mid)), (f"rm_rand_{l}", l, remove(Hc[l], rr, mid))]
                else:
                    for a in ALPHAS:
                        jobs += [(f"add_dir_a{a:g}_{l}", l, Hc[l] + a * d), (f"add_rand_a{a:g}_{l}", l, Hc[l] + a * rr),
                                 (f"sub_dir_a{a:g}_{l}", l, Hc[l] - a * d), (f"sub_rand_a{a:g}_{l}", l, Hc[l] - a * rr)]
            for s in range(0, len(jobs), 16):
                js = jobs[s:s + 16]; m.rpatch = {}
                for j, (_, l, val) in enumerate(js): m.rpatch.setdefault(l, []).append((-1, val, j))
                with torch.no_grad(): lg = m.model(torch.tensor([ids] * len(js), device=DEV), logits_to_keep=1).logits[:, -1]
                m.rpatch = {}
                for j, (name, _, _) in enumerate(js): rec[v][name] = vs[int(lg[j].argmax())]
        out.append(rec)
    if pi % 10 == 0: print(pi, round(time.time() - t0), flush=True)

summ = {}
for l in LAY:
    row = {}
    for v in ("upper", "lower"):
        el = [rc for rc in out if case_of(rc[v]["top0"]) == v]
        for kind in ("dir", "rand"):
            p = [rc[v][f"rm_{kind}_{l}"] for rc in el]
            row[f"rm_{kind}_{v}"] = {"n": len(el), "upper_share": float(np.mean([case_of(x) == "upper" for x in p])),
                                    "lower_share": float(np.mean([case_of(x) == "lower" for x in p])),
                                    "same_word_any_case": float(np.mean([same_word(rc[v]["top0"], x) for rc, x in zip(el, p)]))}
    elo = [rc for rc in out if case_of(rc["orig"]["top0"]) == "lower"]
    elc = [rc for rc in out if capcase(rc["orig"]["top0"])]
    for a in ALPHAS:
        for kind in ("dir", "rand"):
            p = [rc["orig"][f"add_{kind}_a{a:g}_{l}"] for rc in elo]
            row[f"add_{kind}_a{a:g}"] = {"n": len(elo), "to_upper": float(np.mean([case_of(x) == "upper" for x in p])),
                                         "to_upper_same_word": float(np.mean([case_of(x) == "upper" and same_word(rc["orig"]["top0"], x) for rc, x in zip(elo, p)]))}
            p = [rc["orig"][f"sub_{kind}_a{a:g}_{l}"] for rc in elc]
            row[f"sub_{kind}_a{a:g}"] = {"n": len(elc), "to_lower": float(np.mean([case_of(x) == "lower" for x in p])),
                                         "to_lower_same_word": float(np.mean([case_of(x) == "lower" and same_word(rc["orig"]["top0"], x) for rc, x in zip(elc, p)]))}
    summ[l] = row
save({"model": short, "layers": LAY, "rows": [FIRST, FIRST + N], "summary": summ, "records": out, "seconds": round(time.time() - t0)}, f"exp07_{short}.json")
for l in LAY:
    print(l, json.dumps({k: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in s.items()} for k, s in summ[l].items()}), flush=True)
