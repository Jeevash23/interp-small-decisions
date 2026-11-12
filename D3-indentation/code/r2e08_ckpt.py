"""D3 round 2, r2e08 (exploratory, question 5): the block-end head and the behaviour across OLMo-2-0425-1B training.
For each named model (checkpoints registered in r2common as olmo1b-<tokens>; the final model is olmo1b), on the
same natural-code chunks (x1code shuffled with seed 27, as r2e07, chunks <= 900 tokens):
  behaviour at root-identified points by class (same / term / dedent): mean log q(true) and top-1 accuracy;
  for every head: a_ded, a_same (attention to the innermost open header's ':\\n' at unmarked dedents / same-level
  lines) and pick (at multi-level dedents, share where the head's most-attended open header is the true one).
Usage: python r2e08_ckpt.py <n_chunks> <model> [<model> ...]   Output: outputs/r2/r2e08_<model>_<n>.npz + json line."""
import sys, json, gc
from r2common import *

nch = int(sys.argv[1])
for name in sys.argv[2:]:
    m = M(name, attn="eager"); tap = Tap(m, list(range(m.L)))
    chunks = r2_sources("x1code"); random.Random(27).shuffle(chunks)
    S = {k: np.zeros((m.L, m.H)) for k in ["a_ded", "a_same", "pick", "a_true", "a_inner_pick"]}
    N = dict(ded=0, same=0, pick=0); B = {c: [] for c in ["same", "term", "dedent"]}; used = 0
    for c in chunks:
        if used >= nch: break
        ids, pts = prepare(m, c["text"])
        if not pts or len(ids) > 900: continue
        offs = m.tok(c["text"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        sel = []
        for p in pts:
            if max(p["u"]) or point_class(p) == "header": continue
            p["OH"] = [(ind, ti) for ind, ti in open_headers(c["text"], offs, p) if ti is not None and ti < p["d"]]
            if p["OH"]: sel.append(p)
        if not sel: continue
        used += 1
        tap.rows = [p["d"] for p in sel]; tap.on = True
        qs = RootMasses(m, sel)(m.logprobs(ids, positions=[(0, p["d"]) for p in sel])); tap.on = False
        A = torch.stack([tap.A[l] for l in range(m.L)])
        for qi, p in enumerate(sel):
            k = point_class(p); inner = p["OH"][-1][1]; K = p["K"]; q = qs[qi]
            B[k].append((float(np.log(q[K.index(p["true"])])), int(K[int(np.argmax(q))] == p["true"])))
            if k == "dedent": S["a_ded"] += A[:, :, qi, inner].cpu().numpy(); N["ded"] += 1
            if k == "same": S["a_same"] += A[:, :, qi, inner].cpu().numpy(); N["same"] += 1
            if k in ("dedent", "term") and len(p["OH"]) >= 2 and p["OH"][-1][0] != p["true"] and any(ind == p["true"] for ind, _ in p["OH"]):
                tis = torch.tensor([ti for _, ti in p["OH"]], device=DEV); att = A[:, :, qi, tis]
                tj = [j for j, (ind, _) in enumerate(p["OH"]) if ind == p["true"]][-1]
                S["pick"] += (att.argmax(-1) == tj).float().cpu().numpy(); S["a_true"] += att[:, :, tj].cpu().numpy()
                S["a_inner_pick"] += att[:, :, -1].cpu().numpy(); N["pick"] += 1
    R = dict(a_ded=S["a_ded"] / max(N["ded"], 1), a_same=S["a_same"] / max(N["same"], 1), pick=S["pick"] / max(N["pick"], 1),
             a_true=S["a_true"] / max(N["pick"], 1), a_inner_pick=S["a_inner_pick"] / max(N["pick"], 1))
    np.savez(OUT2 / f"r2e08_{name}_{nch}.npz", **R)
    beh = {k: dict(n=len(v), lq=float(np.mean([a for a, _ in v])), top1=float(np.mean([b for _, b in v]))) for k, v in B.items()}
    sc = R["a_ded"] - R["a_same"]; l, h = np.unravel_index(np.argmax(sc), sc.shape); l2, h2 = np.unravel_index(np.argmax(R["a_true"]), sc.shape)
    out = dict(model=name, chunks=used, N=N, behaviour=beh,
               L13H1=dict(a_ded=float(R["a_ded"][13, 1]), a_same=float(R["a_same"][13, 1]), pick=float(R["pick"][13, 1]), a_true=float(R["a_true"][13, 1])),
               best_ded_minus_same=f"L{l}H{h}", best_ded_minus_same_val=[float(R["a_ded"][l, h]), float(R["a_same"][l, h]), float(R["pick"][l, h])],
               best_a_true=f"L{l2}H{h2}", best_a_true_val=[float(R["a_true"][l2, h2]), float(R["pick"][l2, h2])])
    print(json.dumps(out), flush=True)
    with open(OUT2 / f"r2e08_{nch}.jsonl", "a") as fh: fh.write(json.dumps(out) + "\n")
    del m, tap; gc.collect(); torch.cuda.empty_cache()
