"""D3 e07b (exploratory): as e07, but splits each contribution into the true-token term and the current-token
term (+logit(true) and -logit(current)), records attention to the decision token itself and to position 0,
and groups by indent width for generated code (key suffix _w<width>).
Original e07 docstring: D3 e07 (exploratory): direct logit attribution of every head and MLP at the decision position, onto the
first token of the true indent minus the first token of the current level's indent (the copy alternative).
Points: root-identified, with both true and current indent written as a whitespace token (indent >= 2 in
Qwen3, >= 1 in Pythia/Gemma), true != current. Final norm applied with the run's own scale (RMSNorm or
LayerNorm treated as a fixed linear map at that position). Also attention from the decision position to the
start token of the most recent line at the true indent ("target") and at the current indent ("current").
By class: term (terminal-statement dedent), dedent (other dedents), header (deeper).
Usage: python e07_dla.py <model> <source> <n_chunks>"""
import sys, json
from common import *

name, source, nch = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = M(name, attn="eager")
chunks = load_chunks(source, "explore"); random.Random(7).shuffle(chunks)
base = m.model.base_model
fn = base.norm if hasattr(base, "norm") else (base.final_layer_norm if hasattr(base, "final_layer_norm") else base.ln_f)
WU = m.model.get_output_embeddings().weight
acc = {}; cnt = {}; att = {}; selfacc = {}; used = 0
for c in chunks:
    if used >= nch: break
    text = c["text"]; ids, pts = prepare(m, text)
    if not pts: continue
    pts = [p for p in pts if max(p["u"]) == 0 and p["true"] != p["stack"][-1]]
    sel = []
    for p in pts:
        K = p["K"]; et = p["paths"][K.index(p["true"])][0]; ec = p["paths"][K.index(p["stack"][-1])][0]
        if et[0] == "t" and ec[0] == "t": sel.append((p, et[1], ec[1]))
    if not sel: continue
    used += 1
    enc = m.tok(text, return_offsets_mapping=True, add_special_tokens=False); offs = enc["offset_mapping"]
    lines = text.split("\n")[:-1]; lo = [0]
    for l in lines: lo.append(lo[-1] + len(l) + 1)
    starts = P._logical_starts(text); ind = {n: len(lines[n - 1]) - len(lines[n - 1].lstrip(" ")) for n in starts}
    tok_at = lambda ch: next((i for i, (a, b) in enumerate(offs) if a <= ch < b), None)
    with torch.no_grad():
        x = torch.tensor([ids], device=DEV)
        if m.off: x = torch.cat([torch.full((1, 1), m.tok.bos_token_id, device=DEV), x], 1)
        m.capture = True
        out = m.model.base_model(input_ids=x, output_attentions=True); m.capture = False
        A = [a[0, :, m.off:, m.off:].float() for a in out.attentions]
        hfin = out.last_hidden_state[0, m.off:].float()
        for p, tt, tc in sel:
            d = p["d"]; k = "header" if p["header"] else "term" if p["terminal"] else "dedent"
            if c.get("width") is not None: k = f"{k}_w{c['width']}"
            sv = np.array([[float(A[l][h, d, d]), float(A[l][h, d, 0])] for l in range(m.L) for h in range(m.H)]).reshape(m.L, m.H, 2)
            selfacc[k] = selfacc.get(k, 0) + sv
            r = m.rcap[m.L - 1][d].float()
            # final norm frozen at this position: RMSNorm y = g * x / rms(x); LayerNorm y = g * (x - mean) / sd + b
            ln = "layernorm" in fn.__class__.__name__.lower() and "rms" not in fn.__class__.__name__.lower()
            sd = ((r - r.mean()) if ln else r).pow(2).mean().sqrt()
            g = (fn.weight.float() if getattr(fn, "weight", None) is not None else torch.ones_like(r)) / sd
            vt = WU[tt].float() * g; vc = -WU[tc].float() * g
            if ln: vt = vt - vt.mean(); vc = vc - vc.mean()
            v = np.zeros((m.L, m.H + 1, 2))
            for l in range(m.L):
                w = m.dhl[l]; z = m.zcap[l][d].float()
                Wo = m.oproj[l].weight.float()  # (d_model, H*w) for Linear; GPT-2 Conv1D would be (H*w, d_model)
                if Wo.shape[1] != z.shape[0]: Wo = Wo.T
                for h in range(m.H):
                    o = Wo[:, h * w:(h + 1) * w] @ z[h * w:(h + 1) * w]
                    v[l, h, 0] = float(o @ vt); v[l, h, 1] = float(o @ vc)
                mo = m.mcap[l][d].float(); v[l, m.H, 0] = float(mo @ vt); v[l, m.H, 1] = float(mo @ vc)
            acc[k] = acc.get(k, 0) + v; cnt[k] = cnt.get(k, 0) + 1
            prior = [n for n in sorted(starts) if n < p["line"]]
            tgt = [n for n in prior if ind[n] == p["true"]]; cur = [n for n in prior if ind[n] == p["stack"][-1]]
            if tgt and cur:
                ti = tok_at(lo[tgt[-1] - 1]); ci = tok_at(lo[cur[-1] - 1])
                if ti is not None and ci is not None and ti < d and ci < d:
                    av = np.array([[float(A[l][h, d, ti]), float(A[l][h, d, ci])] for l in range(m.L) for h in range(m.H)]).reshape(m.L, m.H, 2)
                    att[k] = att.get(k, 0) + av; cnt[k + "_att"] = cnt.get(k + "_att", 0) + 1
res = {}
for k in acc:
    mean = acc[k] / cnt[k]; np.save(OUT / f"e07b_{name}_{source}_{k}_dla.npy", mean)
    np.save(OUT / f"e07b_{name}_{source}_{k}_selfsink.npy", selfacc[k] / cnt[k])
    tot = mean.sum(-1)
    flat = [(f"L{l}H{h}" if h < m.H else f"mlp{l}", float(tot[l, h]), float(mean[l, h, 0]), float(mean[l, h, 1])) for l in range(m.L) for h in range(m.H + 1)]
    flat.sort(key=lambda t: -t[1])
    res[k] = dict(n=cnt[k], total=float(tot.sum()), top=[(a, round(b, 3), round(t, 3), round(cc, 3)) for a, b, t, cc in flat[:12]],
                  bottom=[(a, round(b, 3), round(t, 3), round(cc, 3)) for a, b, t, cc in flat[-6:]])
    print(k, json.dumps(res[k]), flush=True)
(OUT / f"e07b_{name}_{source}_{nch}.json").write_text(json.dumps(res, indent=1))
