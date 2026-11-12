"""D3 e05 (exploratory): linear probes on the residual stream at the decision position, by layer.
Targets: true next indent (spaces), previous indent, change (true - prev), class (deeper/same/shallower, ridge
one-vs-rest, accuracy), number of open levels, and at dedent points the number of levels dedented.
Ridge regression, 5 folds grouped by chunk; reports cross-validated R^2 (accuracy for class).
Also a control: the same probes on the residual at the token before the decision position.
Usage: python e05_probe.py <model> <source> <n_points>"""
import sys, json
from common import *

name, source, npts = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = M(name, attn="sdpa")
chunks = load_chunks(source, "explore"); random.Random(5).shuffle(chunks)
X = {l: [] for l in range(m.L + 1)}; Xp = {l: [] for l in range(m.L + 1)}; meta = []
emb = m.model.get_input_embeddings()
for ci, c in enumerate(chunks):
    if len(meta) >= npts: break
    ids, pts = prepare(m, c["text"])
    if not pts: continue
    m.capture = True; m.logprobs(ids, positions=[(0, 0)]); m.capture = False
    ds = torch.tensor([p["d"] for p in pts])
    with torch.no_grad(): e0 = emb(torch.tensor([ids], device=DEV))[0].float()
    X[0].append(e0[ds].cpu()); Xp[0].append(e0[(ds - 1).clamp(min=0)].cpu())
    for l in range(m.L):
        X[l + 1].append(m.rcap[l][ds].float().cpu()); Xp[l + 1].append(m.rcap[l][(ds - 1).clamp(min=0)].float().cpu())
    for p in pts:
        meta.append(dict(chunk=ci, true=p["true"], prev=p["prev"], cls=p["cls"], nopen=len(p["stack"]), nded=p["ndedent"]))
N = len(meta); print("points", N, flush=True)
g = np.array([x["chunk"] for x in meta]); folds = np.array([hash(int(c)) % 5 for c in g])
Y = {"true": np.array([x["true"] for x in meta], float), "prev": np.array([x["prev"] for x in meta], float),
     "delta": np.array([x["true"] - x["prev"] for x in meta], float), "nopen": np.array([x["nopen"] for x in meta], float)}
cls = np.array([["deeper", "same", "shallower"].index(x["cls"]) for x in meta])
ded = cls == 2; Yd = np.array([x["nded"] for x in meta], float)


def ridge_cv(A, y, lam=None, mask=None):
    A = A.numpy().astype(np.float64); idx = np.arange(len(y)) if mask is None else np.where(mask)[0]
    pred = np.zeros(len(idx)); yy = y[idx]
    for f in range(5):
        tr = idx[folds[idx] != f]; te_m = folds[idx] == f
        if te_m.sum() == 0 or len(tr) < 10: continue
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
        At = (A[tr] - mu) / sd; ym = y[tr].mean()
        lamv = lam if lam is not None else 0.1 * At.shape[0]
        w = np.linalg.solve(At.T @ At + lamv * np.eye(At.shape[1]), At.T @ (y[tr] - ym))
        pred[te_m] = ((A[idx[te_m]] - mu) / sd) @ w + ym
    return pred, yy


res = {}
for l in range(m.L + 1):
    A = torch.cat(X[l]); Ap = torch.cat(Xp[l]); r = {}
    for k, y in Y.items():
        pr, yy = ridge_cv(A, y); r[k] = float(1 - ((pr - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    pr, yy = ridge_cv(A, Yd, mask=ded); r["nded"] = float(1 - ((pr - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    sc = np.stack([ridge_cv(A, (cls == c).astype(float))[0] for c in range(3)], 1); r["cls_acc"] = float((sc.argmax(1) == cls).mean())
    prp, yy = ridge_cv(Ap, Y["true"]); r["true_prevtok"] = float(1 - ((prp - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    res[l] = r; print(l, {k: round(v, 3) for k, v in r.items()}, flush=True)
res["n"] = N; res["class_counts"] = np.bincount(cls, minlength=3).tolist()
(OUT / f"e05_{name}_{source}_{N}.json").write_text(json.dumps(res, indent=1))
