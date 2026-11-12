"""Exp02 (exploratory): which attention heads and MLPs carry the tens digit (teen status) to the suffix.
Usage: exp02_heads.py <model> --digits 3 [--n 160] [--kinds intoteen,outteen,last]

Pairs as in exp01 (same frame, same split). For each attention head (output slice before the output
projection) and each MLP, patch the source value at the LAST position into the target run and record the
fraction of the source-minus-target logit difference moved. Also records the attention weight from the
last position to each earlier number token, per head, in the target run.
"""
import sys, random
from common import *
from hooks import batch_ids
import exp01_patch_pairs as PP

name = sys.argv[1]
DIG = int(sys.argv[sys.argv.index("--digits") + 1])
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 160
KINDS = sys.argv[sys.argv.index("--kinds") + 1].split(",") if "--kinds" in sys.argv else ["intoteen", "outteen", "last"]

tok, model = load(name)
cols = suffix_ids(tok)
BL = blocks(model)
cfg = model.config
H = cfg.num_attention_heads
dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H


def oproj(b):
    for path in ("self_attn.o_proj", "attention.dense", "attn.c_proj"):
        m = b
        try:
            for k in path.split("."): m = getattr(m, k)
            return m
        except AttributeError:
            continue
    raise AttributeError


def mlp(b):
    return b.mlp


OP = [oproj(b) for b in BL]
MLP = [mlp(b) for b in BL]
L = len(BL)
state = dict(mode=None)


def pre_hook(l):
    def f(m, a):
        x = a[0]
        if state["mode"] == "cap":
            state["head"][l] = x[:, -1].detach().clone()
        elif state["mode"] == "patch_head" and state["l"] == l:
            x = x.clone(); h = state["h"]
            x[:, -1, h * dh:(h + 1) * dh] = state["src_head"][l][:, h * dh:(h + 1) * dh]
            return (x,) + tuple(a[1:])
    return f


def mlp_hook(l):
    def f(m, a, o):
        if state["mode"] == "cap":
            state["mlp"][l] = o[:, -1].detach().clone()
        elif state["mode"] == "patch_mlp" and state["l"] == l:
            o = o.clone(); o[:, -1] = state["src_mlp"][l]
            return o
    return f


hs = [OP[l].register_forward_pre_hook(pre_hook(l)) for l in range(L)] + [MLP[l].register_forward_hook(mlp_hook(l)) for l in range(L)]


@torch.no_grad()
def run(ids, att=False):
    o = model.base_model(input_ids=ids, output_attentions=att)
    lg = unembed(model, o.last_hidden_state[:, -1])[:, cols]
    return lg, (o.attentions if att else None)


res = {}
rng = random.Random("d4-exp02")
for kind in KINDS:
    P = PP.pairs(tok, kind, DIG, N, rng)
    if not P: continue
    npos = len(number_pieces(tok, P[0][0], P[0][2])[0])
    head_fr = np.zeros((L, H)); mlp_fr = np.zeros(L); att = np.zeros((L, H, npos + 1)); cnt = 0
    groups = {}
    for p in P: groups.setdefault(len(encode(tok, p[0])), []).append(p)
    for _, G in groups.items():
        for b in range(0, len(G), 32):
            B = G[b:b + 32]
            src = batch_ids(tok, [p[0] for p in B]); tgt = batch_ids(tok, [p[1] for p in B])
            si = torch.tensor([SUFFIXES.index(suffix(p[2])) for p in B]); ti = torch.tensor([SUFFIXES.index(suffix(p[3])) for p in B])
            ar = torch.arange(len(B))
            ld = lambda lg: (lg[ar, si] - lg[ar, ti]).cpu().numpy()
            state.update(mode="cap", head={}, mlp={}); ls, _ = run(src)
            state.update(src_head=state["head"], src_mlp=state["mlp"], mode=None)
            lt, A = run(tgt, att=True)
            d_s, d_t = ld(ls), ld(lt); den = np.maximum(d_s - d_t, 1e-3)
            T = tgt.shape[1]
            for l in range(L):
                a = A[l][:, :, -1].float().cpu().numpy()  # B, H, T
                for k in range(npos + 1):
                    att[l, :, k] += a[:, :, T - 1 - k].sum(0)
                for h in range(H):
                    state.update(mode="patch_head", l=l, h=h); lp, _ = run(tgt)
                    head_fr[l, h] += np.sum((ld(lp) - d_t) / den)
                state.update(mode="patch_mlp", l=l); lp, _ = run(tgt)
                mlp_fr[l] += np.sum((ld(lp) - d_t) / den)
            state["mode"] = None
            cnt += len(B)
    head_fr /= cnt; mlp_fr /= cnt; att /= cnt
    top = sorted(((head_fr[l, h], l, h) for l in range(L) for h in range(H)), reverse=True)[:12]
    res[kind] = dict(n=cnt, head=head_fr.tolist(), mlp=mlp_fr.tolist(), att=att.tolist(),
                     pieces=number_pieces(tok, P[0][0], P[0][2])[1])
    print(f"{kind} {DIG}d n={cnt}", flush=True)
    print("  top heads:", " ".join(f"L{l}H{h}:{v:+.2f}(att->tens {att[l, h, 1]:.2f})" for v, l, h in top), flush=True)
    print("  MLPs:", " ".join(f"{l}:{v:+.2f}" for l, v in enumerate(mlp_fr) if abs(v) > 0.03), flush=True)
save_json(res, f"exp02_heads_{name}_{DIG}d.json")
