"""CONFIRMATION COPY (PROTOCOL.md): identical to exp01_lens.py except that it reads the held-out `conf` part
and writes conf_* outputs. The case directions are still read from the exploration means (exp01, explore).
"""
"""D5 exploration 1 (exploratory). Usage: exp01_lens.py <model> [first=150] [n=150]
Explore paragraphs [first, first+n) (seed-0 order; the gate used the first 150).
At every word start (CHARTER definitions), in the upper- and lower-case runs:
 (a) teacher-forced log-prob and top-1 correctness of the true first piece, split by whether the word
     already appeared earlier in the paragraph (route R3, copying);
 (b) logit lens over layers (with controls Lr, Ur: the same pieces of another word in the paragraph) (each model's own final norm and output layer) at the prediction
     position: log-prob of the upper-case first piece U (text's token in the upper run) and of the
     lower-case first piece Lc (first token of ' '+word.lower() as tokenised alone) - route R2;
 (c) residual means at prediction positions per layer for the upper and lower runs (for exp02's case
     direction), saved as a .pt file.
Writes outputs/exp01_<model>.json and outputs/exp01_<model>_means.pt."""
import sys, time, operator
from d5common import *  # noqa: F401,F403

short = sys.argv[1]; FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 150; N = int(sys.argv[3]) if len(sys.argv) > 3 else 150
rows = load_part("conf", seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; raw = M_.raw; vs = M_.vs; tok = M_.tok


def find(obj, names):
    for n in names:
        try: return operator.attrgetter(n)(obj)
        except AttributeError: pass
    raise SystemExit("no final norm")
norm = find(raw, ["gpt_neox.final_layer_norm", "model.language_model.norm", "model.norm"])
head = raw.get_output_embeddings()
cfg = raw.config.get_text_config() if hasattr(raw.config, "get_text_config") else raw.config
cap = getattr(cfg, "final_logit_softcapping", None)


def lens_lp(h, idx):
    """log-softmax over the vocabulary of the logit lens at residual rows h (P, d); returns columns idx (P, k)."""
    out = []
    with torch.no_grad():
        for a in range(0, len(h), 32):  # 32 positions at a time (Gemma's 262k-wide output)
            lg = head(norm(h[a:a + 32].to(head.weight.dtype))).float()
            if cap is not None: lg = torch.tanh(lg / cap) * cap
            out.append((lg.gather(1, idx[a:a + 32]) - torch.logsumexp(lg, -1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


first_piece = {}
def piece(word):
    if word not in first_piece: first_piece[word] = tok(" " + word, add_special_tokens=False)["input_ids"][0]
    return first_piece[word]


L = m.L; t0 = time.time()
recs = []; sums = {v: torch.zeros(L + 1, raw.get_input_embeddings().weight.shape[1], dtype=torch.float64) for v in ("upper", "lower")}
cnt = {"upper": 0, "lower": 0}
for pi, r in enumerate(rows):
    text = cut(r["text"]); V = versions(text); rs = runs(text)
    per = {}
    for v in ("upper", "lower"):
        s = V[v]; ids, first = encode(tok, s); ws, _ = positions(s, first)
        # keep only what is needed per position (true-token log-prob and whether it is the top-1);
        # full log-softmax rows for Gemma's 262k vocabulary do not fit in 6 GB
        m.capture = True; tlp, ttop = [], []
        with torch.no_grad():
            k0 = 0
            for c in M_.chunks(ids):
                nxt = torch.tensor((ids[k0 + 1:k0 + 1 + len(c)] + [0] * len(c))[:len(c)], device=c.device)
                tlp.append((c.gather(1, nxt[:, None])[:, 0] - torch.logsumexp(c, -1)).cpu()); ttop.append((c.argmax(-1) == nxt).cpu()); k0 += len(c)
        m.capture = False
        tlp, ttop = torch.cat(tlp), torch.cat(ttop)
        res = torch.stack([m.rcap[l] for l in range(L)])  # (L, T, d) after each layer
        ks = sorted(ws); P = torch.tensor([k - 1 for k in ks], device=DEV)
        for k in ks:
            i = ws[k]; w = text[rs[i][0]:rs[i][1]]
            per.setdefault(i, {})[v] = {"k": k, "tok": ids[k], "lp": float(tlp[k - 1]), "top1": int(ttop[k - 1])}
        if ks:
            hP = res[:, P]  # (L, P, d)
            sums[v][1:] += hP.double().sum(1).cpu(); cnt[v] += len(ks)
            U = torch.tensor([ids[k] for k in ks], device=DEV)
            Lc = torch.tensor([piece(text[rs[ws[k]][0]:rs[ws[k]][1]].lower()) for k in ks], device=DEV)
            # controls: the same pieces of another word in the paragraph (list rolled by one)
            Lr, Ur = torch.roll(Lc, 1), torch.roll(U, 1)
            idx = torch.stack([U, Lc, Lr, Ur], 1)
            lens = np.stack([lens_lp(hP[l], idx) for l in range(L)])  # (L, P, 2)
            for j, k in enumerate(ks): per[ws[k]][v]["lens"] = lens[:, j].round(3).tolist()
    seen_lower = set()
    for i, (s0, e0) in enumerate(rs):
        w = text[s0:e0].lower()
        if i in per and "upper" in per[i] and "lower" in per[i]:
            recs.append({"para": FIRST + pi, "run": i, "word": text[s0:e0], "seen": w in seen_lower,
                         "upper": per[i]["upper"], "lower": per[i]["lower"]})
        seen_lower.add(w)
    if pi % 25 == 0: print(pi, round(time.time() - t0), flush=True)

means = {v: (sums[v] / max(cnt[v], 1)).float() for v in sums}
torch.save({"means": means, "count": cnt, "rows": [FIRST, FIRST + N]}, OUT / f"conf_exp01_{short}_means.pt")


def summ(sel):
    out = {}
    for v in ("upper", "lower"):
        out[v] = {"n": len(sel), "top1": float(np.mean([x[v]["top1"] for x in sel])) if sel else None,
                  "mean_lp": float(np.mean([x[v]["lp"] for x in sel])) if sel else None}
    return out
res = {"model": short, "rows": [FIRST, FIRST + N], "n": len(recs),
       "all": summ(recs), "seen": summ([x for x in recs if x["seen"]]), "new": summ([x for x in recs if not x["seen"]])}
# logit lens summary: per layer, mean log-prob of U and Lc, and share of positions with Lc above U, in each run
for v in ("upper", "lower"):
    A = np.array([x[v]["lens"] for x in recs])  # (n, L, 2)
    diff = A[:, :, 0] != A[:, :, 1]  # positions where U and Lc are different tokens (always in the upper run)
    res[f"lens_{v}"] = {"mean_lp_U": A[:, :, 0].mean(0).round(3).tolist(), "mean_lp_Lc": A[:, :, 1].mean(0).round(3).tolist(),
                        "share_Lc_gt_U": [float((A[diff[:, l], l, 1] > A[diff[:, l], l, 0]).mean()) if diff[:, l].any() else None for l in range(L)],
                        "mean_lp_Lr": A[:, :, 2].mean(0).round(3).tolist(), "mean_lp_Ur": A[:, :, 3].mean(0).round(3).tolist(),
                        "share_Lc_gt_Lr": [float((A[:, l, 1] > A[:, l, 2]).mean()) for l in range(L)],
                        "share_U_gt_Ur": [float((A[:, l, 0] > A[:, l, 3]).mean()) for l in range(L)],
                        "share_Lr_gt_U": [float((A[diff[:, l], l, 2] > A[diff[:, l], l, 0]).mean()) if diff[:, l].any() else None for l in range(L)]}
res["records"] = recs; res["seconds"] = round(time.time() - t0)
save(res, f"conf_exp01_{short}.json")
print(json.dumps({k: res[k] for k in ("all", "seen", "new")}))
for v in ("upper", "lower"):
    for key in ("share_Lc_gt_U", "share_Lr_gt_U", "share_Lc_gt_Lr", "share_U_gt_Ur"):
        print(v, key, [None if x is None else round(x, 2) for x in res[f"lens_{v}"][key]])
