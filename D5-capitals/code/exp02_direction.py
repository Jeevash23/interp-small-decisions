"""D5 exploration 2 (exploratory): a single case direction per layer (route R1).
Usage: exp02_direction.py <model> [first=300] [n=50] [layers=all|a,b,c]
d_l = mean residual (after layer l) at word-start prediction positions of upper runs minus that of
lower runs, from exp01 (paragraphs 150-300). Test on explore paragraphs [first, first+n), up to four
word starts each (seed 2). At one position at a time (the prefix ends at the prediction position):
 lower run + a*d_l and + a*r (r random, same norm), a in {1, 2}: does the top-1 turn upper case, and is
   it still the same word (case-insensitive prefix relation with the unpatched top-1)? does it equal
   the upper run's own top-1 at that word?
 upper run - a*d_l and - a*r: the reverse.
Writes outputs/exp02_<model>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short = sys.argv[1]; FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 300; N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
rows = load_part("explore", seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; L = m.L
md = torch.load(OUT / f"exp01_{short}_means.pt")
D = (md["means"]["upper"] - md["means"]["lower"])  # (L+1, d); row l+1 = after layer l
LAY = list(range(L)) if len(sys.argv) <= 4 or sys.argv[4] == "all" else [int(x) for x in sys.argv[4].split(",")]
g = torch.Generator().manual_seed(0)
R = torch.randn(D.shape, generator=g); R = R / R.norm(dim=1, keepdim=True) * D.norm(dim=1, keepdim=True)
ALPHAS = [1.0, 2.0]
rng = random.Random(2)


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


t0 = time.time(); out = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); V = versions(text)
    enc = {v: encode(tok, V[v]) for v in ("upper", "lower")}
    wsu, _ = positions(V["upper"], enc["upper"][1]); wsl, _ = positions(V["lower"], enc["lower"][1])
    runs_both = sorted(set(wsu.values()) & set(wsl.values()))
    pick = sorted(rng.sample(runs_both, min(4, len(runs_both))))
    for i in pick:
        rec = {"para": FIRST + pi, "run": i}
        kk = {v: [k for k, j in (wsu if v == "upper" else wsl).items() if j == i][0] for v in ("upper", "lower")}
        for v, sign in (("lower", 1.0), ("upper", -1.0)):
            ids = enc[v][0][:kk[v]]; x = torch.tensor([ids] * (1 + 2 * len(ALPHAS)), device=DEV)
            rec[v] = {}
            m.capture = True
            with torch.no_grad(): m.model(x[:1], logits_to_keep=1)
            m.capture = False
            H = {l: m.rcap[l][-1].float().cpu() for l in LAY}
            for l in LAY:
                h = H[l]
                vals = []
                for a in ALPHAS: vals += [h + sign * a * D[l + 1], h + sign * a * R[l + 1]]
                m.rpatch = {l: [(-1, vals[j], j + 1) for j in range(len(vals))]}
                with torch.no_grad(): lg = m.model(x, logits_to_keep=1).logits[:, -1].float()
                m.rpatch = {}
                top = lg.argmax(-1).tolist(); s = [vs[t] for t in top]
                rec[v][l] = s
        out.append(rec)
    if pi % 10 == 0: print(pi, round(time.time() - t0), flush=True)

# upper-run top-1 (unpatched) at each word, for the "matches upper run" measure
summary = {}
for l in LAY:
    row = {}
    for v, want in (("lower", "upper"), ("upper", "lower")):
        for ai, a in enumerate(ALPHAS):
            for kind, j in (("dir", 1 + 2 * ai), ("rand", 2 + 2 * ai)):
                base = [rc[v][l][0] for rc in out]; pat = [rc[v][l][j] for rc in out]
                other = [rc[want][l][0] for rc in out]
                elig = [n for n in range(len(out)) if case_of(base[n]) == v]
                flip = [n for n in elig if case_of(pat[n]) == want]
                kept = [n for n in flip if same_word(base[n], pat[n])]
                match = [n for n in elig if pat[n].strip() == other[n].strip()]
                row[f"{v}_{kind}_a{a:g}"] = {"n": len(elig), "flip": len(flip) / max(len(elig), 1),
                                              "flip_kept": len(kept) / max(len(elig), 1), "match_other_run": len(match) / max(len(elig), 1)}
    summary[l] = row
res = {"model": short, "rows": [FIRST, FIRST + N], "layers": LAY, "alphas": ALPHAS, "norms": D.norm(dim=1).tolist(),
       "summary": summary, "records": out, "seconds": round(time.time() - t0)}
save(res, f"exp02_{short}.json")
for l in LAY:
    s = summary[l]
    print(l, " ".join(f"{k}:{s[k]['flip']:.2f}/{s[k]['flip_kept']:.2f}" for k in s if k.endswith("a1") or k.endswith("a2")), flush=True)
