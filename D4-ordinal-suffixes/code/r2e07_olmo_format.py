"""R2 exploration E07 (CHARTER-2.md question 3), OLMo-2-1B: is the cross-token teen signal in a different form from
the in-token one that the late MLPs read?
Items: 96 teen and 96 non-teen ending 1-3, plain, at 3, 4, 5 and 7 digits (E03's items).
By layer, at the last token:
  - teen direction v_d = mean(teen) - mean(non-teen); cosine between lengths;
  - probe transfer: logistic probe trained on 3-digit items, tested on 4, 5, 7 digits (and on its own length, CV);
  - steering at layer l: add v_3(l) (the 3-digit direction) or v_d(l) (the item length's own direction), each scaled
    to the norm of v_3(l), or a random direction of that norm (5 draws), to every non-teen item of length d; the
    drop in mean m (towards "th") is the response. Reported for d = 4, 7 (cross-token) and 5 (in-token) and 3.
Writes outputs/r2/e07_<model>.json."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
PER = int(arg("--per", "96")); NAME = arg("--model", "olmo2-1b"); LAYERS = arg("--layers", "")
M = Hooked(NAME); L = M.L
res = dict(model=NAME); t0 = time.time()
IT, RS, base = {}, {}, {}
for d in (3, 4, 5, 7):
    IT[d] = make_items(d, PER, f"d4r2-e03-{d}")
    M.clear(); lg = M.run(IT[d], cap=("resid",)); RS[d] = M.c["resid"]; base[d] = summary(lg, IT[d])
y = {d: np.array([it["cls"] == "teen" for it in IT[d]]) for d in IT}
V = {d: {l: RS[d][l][y[d]].mean(0) - RS[d][l][~y[d]].mean(0) for l in range(L)} for d in IT}
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
res["base"] = base
res["norm"] = {d: [float(V[d][l].norm()) for l in range(L)] for d in IT}
res["cos_to_3"] = {d: [cos(V[d][l], V[3][l]) for l in range(L)] for d in (4, 5, 7)}


def transfer(Xtr, ytr, Xte, yte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - Xte.mean(0)) / sd    # each length centred on its own mean (no labels used)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    w = torch.zeros(A.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad()
        with torch.enable_grad():
            loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt) + lam * (w ** 2).sum(); loss.backward()
        return loss
    with torch.enable_grad(): opt.step(cl)
    return float(((((B @ w + b) > 0).numpy()) == yte).mean())


res["probe_transfer_from3"] = {d: [transfer(RS[3][l], y[3], RS[d][l], y[d]) for l in range(L)] for d in (4, 5, 7)}
print(f"probes done {time.time() - t0:.0f}s")
print("  |v| by layer:", {d: " ".join(f"{x:.1f}" for x in res["norm"][d][::3]) for d in IT})
for d in (4, 5, 7):
    print(f"  cos(v{d}, v3):", " ".join(f"{x:+.2f}" for x in res["cos_to_3"][d]))
    print(f"  probe 3->{d}:", " ".join(f"{x:.2f}" for x in res["probe_transfer_from3"][d]))
layers = [int(x) for x in LAYERS.split(",")] if LAYERS else list(range(L))
rng = torch.Generator().manual_seed(0)
res["steer"] = {}
for d in (3, 4, 5, 7):
    non = [it for it in IT[d] if it["cls"] == "end123"]
    m0 = float(mvals(M.run(non), non).mean())
    S = {}
    for l in layers:
        nrm = V[3][l].norm()
        dirs = {"v3": V[3][l], "own": V[d][l] / V[d][l].norm() * nrm}
        for j in range(5):
            r = torch.randn(V[3][l].shape, generator=rng); dirs[f"rand{j}"] = r / r.norm() * nrm
        out = {}
        for k, v in dirs.items():
            M.clear(); M.rpatch = {}
            # add at the block output of layer l (last token): use the next layer's attention input and MLP input
            # via a residual patch of (clean resid + v)
            M.rpatch = {l: (0, RS[d][l][[i for i, it in enumerate(IT[d]) if it["cls"] == "end123"]] + v)}
            out[k] = float(mvals(M.run(non), non).mean()) - m0
        S[str(l)] = out
    M.clear()
    res["steer"][str(d)] = dict(m0=m0, by_layer=S)
    print(f"  steer d={d} (m0 {m0:+.2f}), change in mean m: " + " ".join(
        f"L{l}: v3 {S[str(l)]['v3']:+.2f} own {S[str(l)]['own']:+.2f} rand {np.mean([S[str(l)][f'rand{j}'] for j in range(5)]):+.2f}" for l in layers[::2]))
json.dump(res, open(OUT2 / f"e07_{NAME}.json", "w"), default=float)
print(f"done {time.time() - t0:.0f}s")
