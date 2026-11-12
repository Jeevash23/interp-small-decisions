"""D3 round 2, r2e01 (exploratory, question 1): where do heads look from the decision token, by token role
(r2common.line_roles), per point class, for every head of selected layers. Also the attention-weighted token
strings for the DED heads at unmarked dedents, and a check that Tap.head_out reproduces the captured head output.
Usage: python r2e01_attn.py <source> <n_chunks> [model]"""
import sys, json, collections
from r2common import *

src, nch = sys.argv[1], int(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else "qwen3-06b-bf16"
m = M(name, attn="eager")
LAY = list(range(m.L)); tap = Tap(m, LAY)
chunks = r2_sources(src); random.Random(21).shuffle(chunks)
ROLES = ["self", "sink", "prev", "prev_start", "encl", "encl_start", "target", "target_start", "other"]
acc = {}; cnt = collections.Counter(); toks = collections.defaultdict(collections.Counter); used = 0; chk = []
DED = ["L23H2", "L17H3", "L17H4", "L17H5", "L18H6"] if name.startswith("qwen3-06b") else []
for c in chunks:
    if used >= nch: break
    text = c["text"]
    ids, pts = prepare(m, text)
    if not pts or len(ids) > 900: continue
    pts = [p for p in pts if max(p["u"]) == 0]
    if not pts: continue
    offs = m.tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    tap.rows = [p["d"] for p in pts]; tap.on = True; m.capture = True
    m.logprobs(ids, positions=[(0, p["d"]) for p in pts]); tap.on = False; m.capture = False
    used += 1
    if len(chk) < 20:
        l, h = min(23, m.L - 1), 2; w = m.dhl[l]
        chk.append(float((tap.head_out(l, h, 0) - m.zcap[l][pts[0]["d"], h * w:(h + 1) * w].float()).abs().max()))
    for qi, p in enumerate(pts):
        R = line_roles(text, offs, p, p["d"]); k = point_class(p)
        if k == "dedent" and p.get("ndedent", 1) >= 2: k2 = "dedent_multi"
        else: k2 = None
        for kk in [k] + ([k2] if k2 else []) + ([f"{k}_blank" if p["blank"] else f"{k}_noblank"]):
            cnt[kk] += 1
            for r in ROLES:
                if not R[r]: continue
                ix = torch.tensor(R[r], device=DEV)
                v = torch.stack([tap.A[l][:, qi, ix].sum(-1) for l in LAY]).cpu().numpy()  # L, H
                acc[(kk, r)] = acc.get((kk, r), 0) + v
        if k == "dedent":
            for hname in DED:
                l, h = hn(hname); a = tap.A[l][h, qi, :p["d"] + 1].cpu().numpy()
                rl = {i: r for r, v in R.items() for i in v}
                for i in np.argsort(-a)[:3]:
                    toks[hname][f"{rl.get(int(i), '?')}:{m.vs[ids[i]]!r}"] += float(a[i])
res = dict(source=src, chunks=used, counts=dict(cnt), recon_maxerr=max(chk) if chk else None, roles={}, top_tokens={})
for (kk, r), v in acc.items():
    mean = v / cnt[kk]; np.save(OUT2 / f"r2e01_{name}_{src}_{kk}_{r}.npy", mean)
    top = np.dstack(np.unravel_index(np.argsort(-mean, axis=None)[:10], mean.shape))[0]
    res["roles"][f"{kk}|{r}"] = dict(top=[(f"L{l}H{h}", round(float(mean[l, h]), 3)) for l, h in top],
                                    ded={hname: round(float(mean[hn(hname)]), 3) for hname in DED + ["L22H8", "L25H9", "L23H6", "L11H9", "L17H6"] if hn(hname)[0] < m.L})
for hname, c in toks.items():
    tot = cnt["dedent"]; res["top_tokens"][hname] = [(k, round(v / tot, 3)) for k, v in c.most_common(12)]
(OUT2 / f"r2e01_{name}_{src}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(dict(chunks=used, counts=dict(cnt), recon=res["recon_maxerr"])))
for kk in ["dedent", "dedent_blank", "dedent_noblank", "term", "same"]:
    if cnt[kk]: print(kk, cnt[kk], {r: res["roles"][f"{kk}|{r}"]["ded"] for r in ROLES if f"{kk}|{r}" in res["roles"]}, sep="\n  ")
for hname, v in res["top_tokens"].items(): print(hname, v[:8])
