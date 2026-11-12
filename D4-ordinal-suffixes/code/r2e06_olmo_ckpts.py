"""R2 exploration E06 (CHARTER-2.md question 5), OLMo-2-1B training checkpoints already on disk (downloaded by
D3 on 8 Nov; no download here): when does the cross-boundary teen check form, and does the 4/7-digit weakness
persist through training?
Usage: r2e06_olmo_ckpts.py rev [rev ...]
For each revision: teen effect, AUROC and teen accuracy at 3, 4, 5, 7 digits (E03's items, 64 per class);
at 4 digits the triple L0H5+L0H0+L0H9 mean-ablated (kept), the most attended-to-previous-token layer-0 heads,
and the teen probe at the last token (layers 2 and 14).
Writes outputs/r2/e06_olmo2-1b_<rev>.json per revision."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
import r2lib
from r2lib import *

REVS = sys.argv[1:]
PER = int(__import__("os").environ.get("E06_PER", "64"))
TRI = [(0, 5), (0, 0), (0, 9)]
t0 = time.time()
for rev in REVS:
    def _load(name, rev=rev):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
        model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", revision=rev, dtype=torch.bfloat16,
                                                     attn_implementation="eager").to(DEVICE).eval()
        return tok, model
    r2lib.load = _load
    M = Hooked("olmo2-1b"); L, Hn = M.L, M.H
    out = dict(rev=rev, digits={})
    for d in (3, 4, 5, 7):
        items = make_items(d, PER, f"d4r2-e03-{d}")
        M.clear(); lg = M.run(items, cap=("oin", "resid")); C = M.c
        R = dict(base=summary(lg, items))
        if d == 4:
            means = {l: v.mean(0) for l, v in C["oin"].items()}
            M.ablate_heads(TRI, means); R["tri"] = summary(M.run(items), items); M.clear()
            y = np.array([it["cls"] == "teen" for it in items])
            R["probe_last"] = {l: probe_cv(C["resid"][l], y) for l in (2, 14)}
            ids, am, last = M.encode_batch([it["prefix"] for it in items[:64]]); M.last = last
            o = M.model.base_model(input_ids=ids, attention_mask=am, output_attentions=True)
            w = o.attentions[0].float(); r = torch.arange(len(last), device=w.device)
            R["att0_prev"] = w[r, :, last, last - 1].mean(0).cpu().tolist()
        out["digits"][str(d)] = R
    json.dump(out, open(OUT2 / f"e06_olmo2-1b_{rev}.json", "w"), default=float)
    b = {d: out["digits"][str(d)]["base"] for d in (3, 4, 5, 7)}
    R4 = out["digits"]["4"]
    print(f"{rev} ({time.time() - t0:.0f}s): AUROC 3/4/5/7 " + " ".join(f"{b[d]['auroc']:.3f}" for d in (3, 4, 5, 7))
          + "; teen effect " + " ".join(f"{b[d]['teen']:+.2f}" for d in (3, 4, 5, 7))
          + f"; 4d triple kept {R4['tri']['teen'] / b[4]['teen']:.2f}; probe L2/L14 {R4['probe_last'][2]:.2f}/{R4['probe_last'][14]:.2f}"
          + "; L0 prev-att top " + " ".join(f"H{h}:{a:.2f}" for a, h in sorted(((a, h) for h, a in enumerate(R4['att0_prev'])), reverse=True)[:4]))
    M.clear(); del M, C, lg; import gc; gc.collect(); torch.cuda.empty_cache()
print(f"done {time.time() - t0:.0f}s")
