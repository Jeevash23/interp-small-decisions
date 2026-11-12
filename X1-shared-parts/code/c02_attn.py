"""X1 confirmation c02 (PROTOCOL.md, tests C8 and D4): CONFIRMATION COPY of e02_attn.py run on the SEALED held parts
(identical computation; only the part read and the output name differ). Original e02 docstring follows.
X1 e02 (exploratory): where the early and shared heads look, task by task, on the exploration parts.
For each head in HEADS: share of attention from a query position to the token at offset 0 (itself), 1, 2, 3, 4-7,
8-15, 16+ back, at (a) every position and (b) the task's decision positions (as e01's dec scope). Also, for (b),
the share on the most recent newline-marker token before the query. Material: the first 40 items of each task.
Usage: e02_attn.py <model: qwen3-06b-bf16|qwen3-17b>  -> outputs/e02_attn_<model>.json"""
import sys
from x1common import *

MODEL = sys.argv[1]; PART = "explore" if "--smoke" in sys.argv else "held"
HEADS = parse("L0H2,L0H3,L1H6,L0H7,L4H0,L4H1,L5H12,L5H13,L6H2,L7H15,L7H14,L15H3,L17H6,L22H3,L11H9,L17H4,L5H6,L6H5,L16H2,L23H2,L17H3,L17H5,L18H6")
BINS = [("0", 0, 1), ("1", 1, 2), ("2", 2, 3), ("3", 3, 4), ("4-7", 4, 8), ("8-15", 8, 16), ("16+", 16, 10 ** 6)]
M = Model(MODEL)
# attention weights need the eager implementation: reload the same weights with eager attention
M.m.model.config._attn_implementation = "eager"
for l in M.m.layers: l.self_attn.config._attn_implementation = "eager"
prose = load_part("prose", PART)
T = dict(LB=LB(M, [r["text"] for r in prose], n_width=40, n_nat=40),
         IND=IND(M, {"code": load_part("code", PART)[:40], "gen": load_part("gen", PART)[:40]}),
         ORD=ORD(M, load_part("ord", PART), per=20), CAPS=CAPS(M, [r for r in prose if r["caps_ok"]][:40]))
res = {}
for t, task in T.items():
    seqs = task.seqs_for_means()[:40]; decs = task.dec_for_means()[:40]
    acc = {nm([h]): {"all": np.zeros(len(BINS)), "dec": np.zeros(len(BINS)), "dec_nl": 0.0, "n_all": 0, "n_dec": 0} for h in HEADS}
    for ids, dp in zip(seqs, decs):
        with torch.no_grad():
            o = M.m.model(torch.tensor([ids], device=DEV), output_attentions=True)
        nlp = np.array(M.nlpos(ids)); Tn = len(ids)
        off = np.arange(Tn)[:, None] - np.arange(Tn)[None, :]
        masks = [((off >= a) & (off < b)) for _, a, b in BINS]
        for (l, h) in HEADS:
            A = o.attentions[l][0, h].float().cpu().numpy()
            a = acc[nm([(l, h)])]
            a["all"] += np.array([(A * mk).sum() for mk in masks]); a["n_all"] += Tn
            for q in dp:
                a["dec"] += np.array([A[q][mk[q]].sum() for mk in masks]); a["n_dec"] += 1
                prev = nlp[nlp < q]
                if len(prev): a["dec_nl"] += float(A[q, prev[-1]])
    res[t] = {k: dict(all=dict(zip([b[0] for b in BINS], (v["all"] / max(v["n_all"], 1)).round(4).tolist())),
                      dec=dict(zip([b[0] for b in BINS], (v["dec"] / max(v["n_dec"], 1)).round(4).tolist())),
                      dec_last_newline=round(v["dec_nl"] / max(v["n_dec"], 1), 4), n_dec=v["n_dec"]) for k, v in acc.items()}
    print(t, "done", flush=True)
(OUT / ("conf_smoke" if PART == "explore" else "conf")).mkdir(exist_ok=True)
save_json(res, OUT / ("conf_smoke" if PART == "explore" else "conf") / f"c02_attn_{MODEL}.json")
for k in [nm([h]) for h in HEADS]:
    print(k, " | ".join(f"{t}: self {res[t][k]['dec']['0']:.2f} prev {res[t][k]['dec']['1']:.2f} nl {res[t][k]['dec_last_newline']:.2f}" for t in res))
