"""D5 exploration 3 (exploratory): which layers write the case of the next word?
Usage: exp03_components.py <model> [first=300] [n=50]
Same word starts as exp02 (explore paragraphs [first, first+n), seed 2, up to four per paragraph).
Prefixes end at the prediction position. For every layer l, the attention output (o_proj output) or
the MLP output at the last position is replaced by its value from the other run (upper <- lower, and
lower <- upper), one component per batch row. Measures, per component:
  D = logp(U) - logp(Lc) at the last position (U: the upper run's own first piece; Lc: the lower run's);
  in the upper run, fraction of D removed; in the lower run, fraction of the gap closed;
  case flip of the top-1 and whether the flipped top-1 is the same word (case-insensitive prefix relation).
Writes outputs/exp03_<model>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short = sys.argv[1]; FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 300; N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
rows = load_part("explore", seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; L = m.L
mods = {("attn", l): m.layers[l].self_attn.o_proj for l in range(L)}
mods.update({("mlp", l): m.layers[l].mlp for l in range(L)})
KEYS = list(mods)
cap, patch, capturing = {}, {}, [False]


def mk(key):
    def f(mod, a, o):
        x = o[0] if isinstance(o, tuple) else o
        if capturing[0]: cap[key] = x[0, -1].detach().clone()
        rowsv = patch.get(key)
        if not rowsv: return None
        x = x.clone()
        for row, val in rowsv: x[row, -1] = val
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    return f
for k, mod in mods.items(): mod.register_forward_hook(mk(k))


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


def run_prefix(ids, cap_on=False):
    capturing[0] = cap_on
    with torch.no_grad(): lg = m.model(torch.tensor([ids], device=DEV), logits_to_keep=1).logits[0, -1].float()
    capturing[0] = False
    return torch.log_softmax(lg, -1), {k: v for k, v in cap.items()}


rng = random.Random(2); t0 = time.time(); out = []; CH = 24
for pi, r in enumerate(rows):
    text = cut(r["text"]); V = versions(text)
    enc = {v: encode(tok, V[v]) for v in ("upper", "lower")}
    wsu, _ = positions(V["upper"], enc["upper"][1]); wsl, _ = positions(V["lower"], enc["lower"][1])
    both = sorted(set(wsu.values()) & set(wsl.values()))
    for i in sorted(rng.sample(both, min(4, len(both)))):
        kk = {"upper": [k for k, j in wsu.items() if j == i][0], "lower": [k for k, j in wsl.items() if j == i][0]}
        pre = {v: enc[v][0][:kk[v]] for v in kk}
        U, Lc = enc["upper"][0][kk["upper"]], enc["lower"][0][kk["lower"]]
        base, caps = {}, {}
        for v in ("upper", "lower"):
            lp, c = run_prefix(pre[v], True); base[v] = lp; caps[v] = c
        rec = {"para": FIRST + pi, "run": i, "U": vs[U], "Lc": vs[Lc]}
        for v, other in (("upper", "lower"), ("lower", "upper")):
            D0 = float(base[v][U] - base[v][Lc]); top0 = vs[int(base[v].argmax())]
            res = {}
            for s in range(0, len(KEYS), CH):
                ks = KEYS[s:s + CH]
                patch.clear()
                for j, k in enumerate(ks): patch[k] = [(j, caps[other][k])]
                x = torch.tensor([pre[v]] * len(ks), device=DEV)
                with torch.no_grad(): lg = m.model(x, logits_to_keep=1).logits[:, -1].float()
                patch.clear()
                lp = torch.log_softmax(lg, -1)
                for j, k in enumerate(ks):
                    t1 = vs[int(lp[j].argmax())]
                    res[f"{k[0]}{k[1]}"] = [round(float(lp[j][U] - lp[j][Lc]), 3), t1]
            rec[v] = {"D0": round(D0, 3), "top0": top0, "patched": res}
        out.append(rec)
    if pi % 10 == 0: print(pi, round(time.time() - t0), flush=True)

summary = {}
for k in KEYS:
    name = f"{k[0]}{k[1]}"; row = {}
    for v, want in (("upper", "lower"), ("lower", "upper")):
        recs = [rc for rc in out if case_of(rc[v]["top0"]) == v]
        Ds = np.array([rc[v]["D0"] for rc in recs]); Dp = np.array([rc[v]["patched"][name][0] for rc in recs])
        flip = [rc for rc in recs if case_of(rc[v]["patched"][name][1]) == want]
        kept = [rc for rc in flip if same_word(rc[v]["top0"], rc[v]["patched"][name][1])]
        row[v] = {"n": len(recs), "mean_D0": float(Ds.mean()), "mean_Dp": float(Dp.mean()),
                  "frac_D_moved": float((Ds.mean() - Dp.mean()) / Ds.mean()) if v == "upper" else float((Dp.mean() - Ds.mean()) / -Ds.mean()),
                  "flip": len(flip) / max(len(recs), 1), "flip_kept": len(kept) / max(len(recs), 1)}
    summary[name] = row
save({"model": short, "rows": [FIRST, FIRST + N], "summary": summary, "records": out, "seconds": round(time.time() - t0)}, f"exp03_{short}.json")
for k in KEYS:
    n = f"{k[0]}{k[1]}"; s = summary[n]
    print(n, "up<-lo moved %.2f flip %.2f kept %.2f | lo<-up moved %.2f flip %.2f kept %.2f" % (
        s["upper"]["frac_D_moved"], s["upper"]["flip"], s["upper"]["flip_kept"], s["lower"]["frac_D_moved"], s["lower"]["flip"], s["lower"]["flip_kept"]), flush=True)
