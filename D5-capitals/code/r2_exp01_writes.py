"""D5 round 2, exploration 1 (exploratory; CHARTER-2.md Q1, Q2): who writes the capitals direction?
Usage: r2_exp01_writes.py <model> <part> <first> <n> [per_para=3] [seed=11]
At up to `per_para` word starts per paragraph (word 12 or later, seed), three prefixes ending at the
prediction position:
  U: the upper-case passage;  L: the lower-case passage;
  A: the upper-case passage with its last word in lower case (context only: same current word as L).
For each prefix, every additive term of the residual stream at the prediction position (embedding,
each attention head, each MLP, Gemma's per-layer-input blocks; d5common2.Writes) is
  (a) projected on the unit capitals direction u = d/|d| after layer l (round-1 means, explore 150-300),
      for terms from layers <= l, with Gemma's layer scalars applied; the check value is the
      projection of the captured residual itself;
  (b) for every layer, its direct contribution to logit(U) - logit(Lc) through the final norm
      (linearised at the run's own final-norm scale) and the output layer.
Also records D = logp(U) - logp(Lc) and the top-1 for each prefix. Writes outputs/r2/exp01_<model>_<part>.pt"""
import sys, time
from d5common2 import *  # noqa: F401,F403
from d5common2 import Writes

short, part = sys.argv[1], sys.argv[2]; FIRST, N = int(sys.argv[3]), int(sys.argv[4])
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 3; SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 11
LAYER = {"gemma": 24, "qwen3-17b": 22}[short]
rows = load_part2(part, seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; L, H = m.L, m.H
W = Writes(M_); sc = W.scalars()
md = torch.load(OUT / f"exp01_{short}_means.pt")
d = (md["means"]["upper"] - md["means"]["lower"]).float()[LAYER + 1].to(DEV); u = d / d.norm()
raw = M_.raw; base = getattr(raw.model, "language_model", raw.model)
fn = {}
base.norm.register_forward_hook(lambda mod, a, o: fn.update(x=a[0][0, -1].float(), y=o[0, -1].float()) if W.on else None)
WU = raw.get_output_embeddings().weight.detach()


def build(text, rs, i, form):
    pre = text[:rs[i][0] - 1]
    if form == "U": return pre.upper()
    if form == "L": return pre.lower()
    c0 = rs[i - 1][0]
    return pre[:c0].upper() + pre[c0:].lower()


def cum(j_from, j_to):
    p = 1.0
    for k in range(j_from, j_to + 1): p *= sc[k]
    return p


rng = random.Random(SEED); t0 = time.time(); recs = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); rs = runs(text)
    cand = [i for i, (s, e) in enumerate(rs) if i >= 12 and text[s - 1] == " "]
    for i in sorted(rng.sample(cand, min(PER, len(cand)))):
        word = text[rs[i][0]:rs[i][1]]
        Utok = tok(" " + word.upper(), add_special_tokens=False)["input_ids"][0]
        Ltok = tok(" " + word.lower(), add_special_tokens=False)["input_ids"][0]
        rec = {"para": FIRST + pi, "pageid": r["pageid"], "run": i, "word": word, "U": vs[Utok], "Lc": vs[Ltok]}
        dU = (WU[Utok] - WU[Ltok]).float()
        for f in ("U", "L", "A"):
            ids = tok(build(text, rs, i, f), add_special_tokens=False)["input_ids"]
            W.t = {}; W.on = True; m.capture = True
            with torch.no_grad(): lg = m.model(torch.tensor([ids], device=DEV), logits_to_keep=1).logits[0, -1].float()
            W.on = False; m.capture = False
            lp = torch.log_softmax(lg, -1)
            T = W.terms()
            g = fn["y"] / torch.where(fn["x"].abs() > 1e-6, fn["x"], torch.ones_like(fn["x"]))
            g = torch.where(fn["x"].abs() > 1e-6, g, torch.zeros_like(g))
            proj = {"emb": float(T["emb"] @ u) * cum(0, LAYER)}
            dla = {"emb": float((T["emb"] * cum(0, L - 1)) @ (g * dU))}
            A_p = np.zeros((L, H), np.float32); A_d = np.zeros((L, H), np.float32)
            M_p = np.zeros(L, np.float32); M_d = np.zeros(L, np.float32); P_p = np.zeros(L, np.float32); P_d = np.zeros(L, np.float32)
            for j in range(L):
                cl, cL = (cum(j, LAYER) if j <= LAYER else 0.0), cum(j, L - 1)
                heads = torch.stack([T[("attn", j, h)] for h in range(H)])
                A_p[j] = (heads @ u).cpu().numpy() * cl; A_d[j] = (heads @ (g * dU)).cpu().numpy() * cL
                M_p[j] = float(T[("mlp", j)] @ u) * cl; M_d[j] = float(T[("mlp", j)] @ (g * dU)) * cL
                if ("ple", j) in T:
                    P_p[j] = float(T[("ple", j)] @ u) * cl; P_d[j] = float(T[("ple", j)] @ (g * dU)) * cL
            resid = float(m.rcap[LAYER][-1].float() @ u)
            rec[f] = {"top": vs[int(lg.argmax())], "D": float(lp[Utok] - lp[Ltok]), "last_tok": ids[-1],
                      "resid_proj": resid, "emb_proj": proj["emb"], "emb_dla": dla["emb"],
                      "attn_proj": A_p, "attn_dla": A_d, "mlp_proj": M_p, "mlp_dla": M_d, "ple_proj": P_p, "ple_dla": P_d,
                      "logit_diff": float(lg[Utok] - lg[Ltok])}
        recs.append(rec)
    if pi % 10 == 0: print(pi, round(time.time() - t0), flush=True)

(OUT / "r2").mkdir(exist_ok=True)
torch.save({"model": short, "part": part, "rows": [FIRST, FIRST + N], "layer": LAYER, "scalars": sc, "records": recs,
            "seconds": round(time.time() - t0)}, OUT / "r2" / f"exp01_{short}_{part}_{FIRST}.pt")
# quick check: sum of term projections vs the residual's own projection
err = [abs(rc[f]["emb_proj"] + rc[f]["attn_proj"][:LAYER + 1].sum() + rc[f]["mlp_proj"][:LAYER + 1].sum()
           + rc[f]["ple_proj"][:LAYER + 1].sum() - rc[f]["resid_proj"]) / (abs(rc[f]["resid_proj"]) + 1e-6) for rc in recs for f in ("U", "L", "A")]
print("relative decomposition error: median %.4f, max %.4f" % (np.median(err), np.max(err)))
same_last = np.mean([rc["A"]["last_tok"] == rc["L"]["last_tok"] for rc in recs]); print("A and L share the last token:", same_last)
for f in ("U", "L", "A"):
    print(f, "resid proj mean %.2f" % np.mean([rc[f]["resid_proj"] for rc in recs]), "D mean %.2f" % np.mean([rc[f]["D"] for rc in recs]),
          "upper top-1 %.3f" % np.mean([case_of(rc[f]["top"]) == "upper" for rc in recs]))
dif = np.mean([rc["U"]["attn_proj"] - rc["L"]["attn_proj"] for rc in recs], 0)
dm = np.mean([rc["U"]["mlp_proj"] - rc["L"]["mlp_proj"] for rc in recs], 0)
tot = np.mean([rc["U"]["resid_proj"] - rc["L"]["resid_proj"] for rc in recs])
print("U-L total %.2f; emb %.2f" % (tot, np.mean([rc["U"]["emb_proj"] - rc["L"]["emb_proj"] for rc in recs])))
top = sorted([(float(dif[j, h]), f"L{j}H{h}") for j in range(L) for h in range(H)] + [(float(dm[j]), f"MLP{j}") for j in range(L)], reverse=True)
print("top writers U-L:", [(n, round(v, 2)) for v, n in top[:20]])
difa = np.mean([rc["A"]["attn_proj"] - rc["L"]["attn_proj"] for rc in recs], 0)
dma = np.mean([rc["A"]["mlp_proj"] - rc["L"]["mlp_proj"] for rc in recs], 0)
print("A-L total %.2f" % np.mean([rc["A"]["resid_proj"] - rc["L"]["resid_proj"] for rc in recs]))
top = sorted([(float(difa[j, h]), f"L{j}H{h}") for j in range(L) for h in range(H)] + [(float(dma[j]), f"MLP{j}") for j in range(L)], reverse=True)
print("top writers A-L:", [(n, round(v, 2)) for v, n in top[:20]])
