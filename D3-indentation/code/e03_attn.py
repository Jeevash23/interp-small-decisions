"""D3 e03 (exploratory): where do heads look from the decision position? For each head, mean attention from
the decision position to: prevstart (the first token of the previous logical line: its indent token, or its
first word at indent 0), prevword (the previous line's first word token), last (the token just before the
decision position, e.g. ':'), target (first token of the most recent earlier logical line, other than the
previous one, whose indent equals the true next indent: for dedents, the header line of the closed block),
openstarts (summed attention to first tokens of all earlier logical lines whose indent is an open shallower
level). By class. Single-token-indent models only (Pythia, Gemma).
Usage: python e03_attn.py <model> <source> <n_chunks>"""
import sys, json
from common import *

name, source, nch = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = M(name, attn="eager")
chunks = load_chunks(source, "explore"); random.Random(2).shuffle(chunks)
acc = {}; cnt = {}
used = 0
for c in chunks:
    if used >= nch: break
    text = c["text"]
    enc = m.tok(text, return_offsets_mapping=True, add_special_tokens=False); ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > 1000: continue
    pts = P.decision_points(text)
    if not pts: continue
    lines = text.split("\n")[:-1]; lo = [0]
    for l in lines: lo.append(lo[-1] + len(l) + 1)
    start_tok = {}
    for i, (a, b) in enumerate(offs):
        if b > a:
            for n in range(len(lines)):
                if lo[n] == a or (a < lo[n] < b and False): start_tok.setdefault(n + 1, i)
    tok_at = lambda ch: next((i for i, (a, b) in enumerate(offs) if a <= ch < b), None)
    starts = P._logical_starts(text)
    ind = {n: len(lines[n - 1]) - len(lines[n - 1].lstrip(" ")) for n in starts}
    with torch.no_grad():
        x = torch.tensor([ids], device=DEV)
        if m.off: x = torch.cat([torch.full((1, 1), m.tok.bos_token_id, device=DEV), x], 1)
        att = m.model(x, output_attentions=True).attentions  # L x (1, H, T, T)
        A = torch.stack([a[0, :, m.off:, m.off:] for a in att]).float().cpu().numpy()  # L,H,T,T
    used += 1
    for p in pts:
        s = p["s"]; d = max([i for i, (a, b) in enumerate(offs) if b <= s and b > a], default=None)
        if d is None: continue
        prior = [n for n in sorted(starts) if n < p["line"]]
        if not prior: continue
        pn = prior[-1]
        cats = {}
        cats["prevstart"] = [tok_at(lo[pn - 1])]
        cats["prevword"] = [tok_at(lo[pn - 1] + ind[pn])]
        cats["last"] = [d - 1]
        tg = [n for n in prior[:-1] if ind[n] == p["true"]]
        if tg and p["cls"] != "deeper": cats["target"] = [tok_at(lo[tg[-1] - 1])]
        op = [tok_at(lo[n - 1]) for n in prior[:-1] if ind[n] in p["stack"][:-1] and ind[n] > 0]
        if op: cats["openstarts"] = op
        k = "header" if p["header"] else "same" if p["cls"] == "same" else "term" if p["terminal"] else "dedent"
        for cat, poss in cats.items():
            poss = sorted({q for q in poss if q is not None and q <= d})
            if not poss: continue
            v = A[:, :, d, poss].sum(-1)
            acc[(k, cat)] = acc.get((k, cat), 0) + v; cnt[(k, cat)] = cnt.get((k, cat), 0) + 1
res = {}
for (k, cat), v in acc.items():
    mean = v / cnt[(k, cat)]
    top = np.dstack(np.unravel_index(np.argsort(-mean, axis=None)[:12], mean.shape))[0]
    res[f"{k}|{cat}"] = dict(n=cnt[(k, cat)], top=[(f"L{l}H{h}", round(float(mean[l, h]), 3)) for l, h in top])
    np.save(OUT / f"e03_{name}_{source}_{k}_{cat}.npy", mean)
(OUT / f"e03_{name}_{source}.json").write_text(json.dumps(res, indent=1))
for k, v in sorted(res.items()): print(k, v["n"], v["top"][:6])
