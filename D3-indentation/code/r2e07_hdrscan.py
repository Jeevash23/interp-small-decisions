"""D3 round 2, r2e07 (exploratory, questions 2 and 4): rank every head by how it reads open headers' ':\\n' tokens.
Natural code. For each head, at root-identified points with at least one open header:
  a_ded / a_same: attention from the decision token to the innermost open header's ':\\n' at unmarked dedents /
  same-level lines; pick: at dedents (unmarked and terminal) whose true level is an open header other than the
  innermost, the share of points where the head's most-attended open header is the true one, and its mean
  attention to the true header there.
Usage: python r2e07_hdrscan.py <source> <n_chunks> <model>"""
import sys, json
from r2common import *

src, nch, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = M(name, attn="eager"); tap = Tap(m, list(range(m.L)))
chunks = r2_sources(src); random.Random(27).shuffle(chunks)
S = {k: np.zeros((m.L, m.H)) for k in ["a_ded", "a_same", "pick", "a_true", "a_inner_pick"]}; N = dict(ded=0, same=0, pick=0); used = 0
for c in chunks:
    if used >= nch: break
    ids, pts = prepare(m, c["text"])
    if not pts or len(ids) > 900: continue
    offs = m.tok(c["text"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    sel = []
    for p in pts:
        if max(p["u"]) or point_class(p) == "header": continue
        oh = [(ind, ti) for ind, ti in open_headers(c["text"], offs, p) if ti is not None and ti < p["d"]]
        if oh: p["OH"] = oh; sel.append(p)
    if not sel: continue
    used += 1
    tap.rows = [p["d"] for p in sel]; tap.on = True; m.logprobs(ids, positions=[(0, p["d"]) for p in sel]); tap.on = False
    A = torch.stack([tap.A[l] for l in range(m.L)])  # L, H, nq, T
    for qi, p in enumerate(sel):
        k = point_class(p); inner = p["OH"][-1][1]
        if k == "dedent": S["a_ded"] += A[:, :, qi, inner].cpu().numpy(); N["ded"] += 1
        if k == "same": S["a_same"] += A[:, :, qi, inner].cpu().numpy(); N["same"] += 1
        if k in ("dedent", "term") and len(p["OH"]) >= 2 and p["OH"][-1][0] != p["true"] and any(ind == p["true"] for ind, _ in p["OH"]):
            tis = torch.tensor([ti for _, ti in p["OH"]], device=DEV); att = A[:, :, qi, tis]  # L, H, nh
            tj = [j for j, (ind, _) in enumerate(p["OH"]) if ind == p["true"]][-1]
            S["pick"] += (att.argmax(-1) == tj).float().cpu().numpy(); S["a_true"] += att[:, :, tj].cpu().numpy()
            S["a_inner_pick"] += att[:, :, -1].cpu().numpy(); N["pick"] += 1
R = dict(a_ded=S["a_ded"] / max(N["ded"], 1), a_same=S["a_same"] / max(N["same"], 1), pick=S["pick"] / max(N["pick"], 1),
         a_true=S["a_true"] / max(N["pick"], 1), a_inner_pick=S["a_inner_pick"] / max(N["pick"], 1))
np.savez(OUT2 / f"r2e07_{name}_{src}_{nch}.npz", **R)
print(name, src, "chunks", used, N)
score = R["a_true"] - R["a_inner_pick"]
order = np.dstack(np.unravel_index(np.argsort(-R["a_ded"], axis=None), score.shape))[0]
print("by attention to the innermost header end at unmarked dedents:")
for l, h in order[:10]:
    print(f"  L{l}H{h}: a_ded {R['a_ded'][l,h]:.3f} a_same {R['a_same'][l,h]:.3f} | pick {R['pick'][l,h]:.3f} a_true {R['a_true'][l,h]:.3f} a_inner {R['a_inner_pick'][l,h]:.3f}")
order = np.dstack(np.unravel_index(np.argsort(-R["a_true"], axis=None), score.shape))[0]
print("by attention to the true-level header when it is not the innermost:")
for l, h in order[:10]:
    print(f"  L{l}H{h}: a_true {R['a_true'][l,h]:.3f} a_inner {R['a_inner_pick'][l,h]:.3f} pick {R['pick'][l,h]:.3f} | a_ded {R['a_ded'][l,h]:.3f} a_same {R['a_same'][l,h]:.3f}")
