"""Exp04 (exploratory): batched patching scans (replaces the slow exp01/exp02 loops; same pairs and metric).
Usage: exp04_scan.py <model> --digits 3 --kind intoteen [--n 96] [--what resid,head,mlp]

Source -> target pairs from exp01_patch_pairs (same frame, same token split). Metric (from 22:26, 7 Nov):
logit difference (source suffix minus target suffix) moved by the patch, summed over pairs whose source and
target the model gets right, divided by the summed source-minus-target gap over the same pairs.
(Before 22:26 it was the mean of per-pair fractions over all pairs, which blew up when a source was wrong.)
  resid: block output at (layer, position from end), positions -1 .. -(number tokens + 1)
  head : one attention head's output (before the output projection) at the last position
  mlp  : one MLP's output at the last position
Each forward pass runs many copies of the target batch, each copy with a different patch.
Also saves each head's attention from the last position to the previous positions (target run).
Writes outputs/exp04_<model>_<digits>d_<kind>.json.
"""
import sys, random, time
from common import *
from hooks import batch_ids, _out, _set
import exp01_patch_pairs as PP

name = sys.argv[1]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
DIG, KIND, N = int(arg("--digits", "3")), arg("--kind", "intoteen"), int(arg("--n", "96"))
WHAT = arg("--what", "resid,head,mlp").split(",")
ROWS = int(arg("--rows", "384"))
PART = arg("--part", "explore")   # 'conf' only under the frozen PROTOCOL.md (test Q4)
require_frozen_protocol(PART)

tok, model = load(name)
cols = suffix_ids(tok)
BL = blocks(model); L = len(BL)
cfg = model.config
H = cfg.num_attention_heads
dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H


def sub(b, path):
    m = b
    for k in path.split("."): m = getattr(m, k)
    return m


def oproj(b):
    for p in ("self_attn.o_proj", "attention.dense", "attn.c_proj"):
        try: return sub(b, p)
        except AttributeError: pass
    raise AttributeError


S = dict(spec=None)   # spec: list of (what, layer, idx) per copy; B rows per copy


def _rows(c, B, dev):
    return torch.arange(c * B, (c + 1) * B, device=dev)


def resid_hook(l):
    def f(m, a, o):
        if S.get("cap"): S["resid"][l] = _out(o).detach().clone()
        if not S["spec"]: return
        h = _out(o); ch = False
        for c, (w, ll, k) in enumerate(S["spec"]):
            if w == "resid" and ll == l:
                if not ch: h = h.clone(); ch = True
                T = h.shape[1]; r = _rows(c, S["B"], h.device)
                h[r, T - 1 - k] = S["resid_src"][l][:, T - 1 - k].to(h.dtype)
        return _set(o, h) if ch else None
    return f


def head_hook(l):
    def f(m, a):
        x = a[0]
        if S.get("cap"): S["head"][l] = x[:, -1].detach().clone()
        if not S["spec"]: return
        ch = False
        for c, (w, ll, k) in enumerate(S["spec"]):
            if w == "head" and ll == l:
                if not ch: x = x.clone(); ch = True
                r = _rows(c, S["B"], x.device)
                x[r, -1, k * dh:(k + 1) * dh] = S["head_src"][l][:, k * dh:(k + 1) * dh]
        return ((x,) + tuple(a[1:])) if ch else None
    return f


def mlp_hook(l):
    def f(m, a, o):
        if S.get("cap"): S["mlp"][l] = o[:, -1].detach().clone()
        if not S["spec"]: return
        ch = False
        for c, (w, ll, k) in enumerate(S["spec"]):
            if w == "mlp" and ll == l:
                if not ch: o = o.clone(); ch = True
                r = _rows(c, S["B"], o.device)
                o[r, -1] = S["mlp_src"][l]
        return o if ch else None
    return f


for l, b in enumerate(BL):
    b.register_forward_hook(resid_hook(l)); oproj(b).register_forward_pre_hook(head_hook(l)); b.mlp.register_forward_hook(mlp_hook(l))


@torch.no_grad()
def run(ids, att=False):
    o = model.base_model(input_ids=ids, output_attentions=att)
    return unembed(model, o.last_hidden_state[:, -1])[:, cols], (o.attentions if att else None)


rng = random.Random(f"d4-exp04-{KIND}-{DIG}" if PART == "explore" else f"d4-exp04-{PART}-{KIND}-{DIG}")
P = PP.pairs(tok, KIND, DIG, N, rng, part=PART)
npos = len(number_pieces(tok, P[0][0], P[0][2])[0]) + 1
specs = []
if "resid" in WHAT: specs += [("resid", l, k) for l in range(L) for k in range(npos)]
if "head" in WHAT: specs += [("head", l, h) for l in range(L) for h in range(H)]
if "mlp" in WHAT: specs += [("mlp", l, 0) for l in range(L)]
acc = {s: 0.0 for s in specs}
att = np.zeros((L, H, npos)); cnt = 0; t0 = time.time()
base = dict(src_ok=0, tgt_ok=0, gap=0.0)
gap_sum, nval = [0.0], [0]
groups = {}
for p in P: groups.setdefault(len(encode(tok, p[0])), []).append(p)
for _, G in groups.items():
    for b0 in range(0, len(G), 32):
        Bp = G[b0:b0 + 32]; B = len(Bp)
        src = batch_ids(tok, [p[0] for p in Bp]); tgt = batch_ids(tok, [p[1] for p in Bp])
        si = torch.tensor([SUFFIXES.index(suffix(p[2])) for p in Bp]); ti = torch.tensor([SUFFIXES.index(suffix(p[3])) for p in Bp])
        S.update(spec=None, cap=True, resid={}, head={}, mlp={}); ls, _ = run(src)
        S.update(cap=False, resid_src=S["resid"], head_src=S["head"], mlp_src=S["mlp"])
        lt, A = run(tgt, att=True)
        ar = torch.arange(B)
        ld = lambda lg, idx=ar: (lg[idx, si] - lg[idx, ti]).cpu().numpy()
        d_s, d_t = ld(ls), ld(lt)
        val = (d_s > 0) & (d_t < 0)   # pairs where the model gets both source and target right
        gap_sum[0] += float((d_s - d_t)[val].sum()); nval[0] += int(val.sum())
        base["src_ok"] += int((d_s > 0).sum()); base["tgt_ok"] += int((d_t < 0).sum()); base["gap"] += float((d_s - d_t).sum())
        T = tgt.shape[1]
        for l in range(L):
            a = A[l][:, :, -1].float().cpu().numpy()
            for k in range(npos): att[l, :, k] += a[:, :, T - 1 - k].sum(0)
        per = max(1, ROWS // B)
        for c0 in range(0, len(specs), per):
            chunk = specs[c0:c0 + per]
            S.update(spec=chunk, B=B)
            lp, _ = run(tgt.repeat(len(chunk), 1))
            S["spec"] = None
            for c, s in enumerate(chunk):
                v = lp[c * B:(c + 1) * B]
                acc[s] += float(np.sum((ld(v) - d_t)[val]))
        cnt += B
norm = gap_sum[0]  # metric: moved logit difference summed over valid pairs / summed source-target gap
res = dict(n=cnt, n_valid=nval[0], metric="ratio of sums over pairs with source and target both correct", kind=KIND, digits=DIG, pieces=number_pieces(tok, P[0][0], P[0][2])[1],
           src_ok=base["src_ok"] / cnt, tgt_ok=base["tgt_ok"] / cnt, mean_gap=base["gap"] / cnt,
           att=(att / cnt).tolist(), seconds=time.time() - t0)
if "resid" in WHAT: res["resid"] = [[acc[("resid", l, k)] / norm for k in range(npos)] for l in range(L)]
if "head" in WHAT: res["head"] = [[acc[("head", l, h)] / norm for h in range(H)] for l in range(L)]
if "mlp" in WHAT: res["mlp"] = [acc[("mlp", l, 0)] / norm for l in range(L)]
if PART != "explore": (OUT / PART).mkdir(exist_ok=True)
save_json(res, (f"{PART}/" if PART != "explore" else "") + f"exp04_{name}_{DIG}d_{KIND}.json")
print(f"{name} {DIG}d {KIND}: n={cnt} valid={nval[0]} src_ok={res['src_ok']:.2f} tgt_ok={res['tgt_ok']:.2f} pieces={res['pieces']} {res['seconds']:.0f}s")
if "resid" in WHAT:
    for l in range(L): print(f"  resid L{l:2d} " + " ".join(f"{-1 - k}:{res['resid'][l][k]:+.2f}" for k in range(npos)))
if "head" in WHAT:
    top = sorted(((res["head"][l][h], l, h) for l in range(L) for h in range(H)), reverse=True)[:12]
    print("  top heads:", " ".join(f"L{l}H{h}:{v:+.2f}(att-2 {res['att'][l][h][1]:.2f})" for v, l, h in top))
    print("  sum of all heads:", f"{sum(map(sum, res['head'])):+.2f}")
if "mlp" in WHAT:
    print("  MLPs:", " ".join(f"{l}:{v:+.2f}" for l, v in enumerate(res["mlp"]) if abs(v) > 0.03))
