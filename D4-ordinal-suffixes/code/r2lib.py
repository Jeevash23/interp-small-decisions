"""Round-2 toolkit (CHARTER-2.md): items, measures and a hooked model for ablation, path ablation, logit lens and
direct logit attribution at the number's last token. Exploratory code until listed in PROTOCOL-2.md.

Batches are right-padded; every intervention acts at each row's last real token (causal models never attend to
the padding that follows).

Interventions (all at the last token):
  abl[l]        list of (head, vector): replace that head's output (o_proj input slice) by the vector
  add[key]      tensor [B, d] added to one receiver's input only: ('attn', l), ('mlp', l) or ('final',)
                (Qwen3/Llama: the input of input_layernorm, post_attention_layernorm or the final norm;
                 OLMo-2, post-norm blocks: the input of self_attn or mlp, or the final norm)
  keep[l]       (head, clean o_proj input [B, H*dh]): every head of layer l except `head` is reset to its clean
                value, so a change made to layer l's input reaches only that head
"""
import random
import numpy as np
import torch
from r2common import *


# ------------------------------------------------------------------ items
def draw_number(rng, d, cls, part="explore"):
    a, z = 10 ** (d - 1), 10 ** d - 1
    for _ in range(100000):
        n = rng.randint(a, z)
        if cls == "teen":
            if d < 2: raise ValueError
            n = n - n % 100 + rng.choice([11, 12, 13]) if d >= 3 else rng.choice([11, 12, 13])
        elif cls == "end123":
            n = n - n % 10 + rng.choice([1, 2, 3])
            if n % 100 in (11, 12, 13): continue
        elif cls == "th":
            n = n - n % 10 + rng.choice([0, 4, 5, 6, 7, 8, 9])
        if a <= n <= z and r2_number_ok(n, part):
            return n
    raise RuntimeError("no number found")


def fmt(n, form):
    return f"{n:,}" if form == "comma" else str(n)


def make_items(d, per, seed, part="explore", form="plain", families=("rankings", "positions", "anniversaries"),
               classes=("teen", "end123"), frames=None):
    """per items of each class with d digits; each item a dict(prefix, n, cls, fid)."""
    rng = random.Random(seed)
    frames = frames or [f for f in r2_frames(part) if f["family"] in families]
    out = []
    for cls in classes:
        for i in range(per):
            f = frames[i % len(frames)] if cls != "teen" or d > 2 else frames[(i // 3) % len(frames)]
            n = draw_number(rng, d, cls, part) if not (cls == "teen" and d == 2) else 11 + i % 3
            out.append(dict(prefix=f["text"].replace("{N}", fmt(n, form)), n=n, cls=cls, fid=f["fid"], d=d, form=form))
    return out


# ------------------------------------------------------------------ measures
def mvals(lg, items):
    """m = logit(suffix of the last digit) - logit(th); lg is [B, 4] in SUFFIXES order."""
    idx = np.array([it["n"] % 10 - 1 if it["n"] % 10 in (1, 2, 3) else 3 for it in items])
    return lg[np.arange(len(items)), idx] - lg[:, 3]


def auroc(pos, neg):
    """P(pos > neg), ties count one half (pair counting)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def summary(lg, items):
    m = mvals(lg, items)
    cls = np.array([it["cls"] for it in items])
    t, e = m[cls == "teen"], m[cls == "end123"]
    ok = np.argmax(lg, 1) == np.array([SUFFIXES.index(suffix(it["n"])) for it in items])
    return dict(teen=float(e.mean() - t.mean()) if len(t) and len(e) else float("nan"),
                pull=float(e.mean()) if len(e) else float("nan"), auroc=auroc(e, t),
                acc_teen=float(ok[cls == "teen"].mean()) if len(t) else float("nan"),
                acc_end=float(ok[cls == "end123"].mean()) if len(e) else float("nan"))


# ------------------------------------------------------------------ hooked model
def _sub(m, path):
    for k in path.split("."): m = getattr(m, k)
    return m


class Hooked:
    def __init__(self, name):
        self.name = name
        self.tok, self.model = load(name)
        self.cols = suffix_ids(self.tok)
        self.BL = blocks(self.model); self.L = len(self.BL)
        cfg = self.model.config
        self.H = cfg.num_attention_heads
        # head size per layer (Gemma-4 mixes layer types with different head sizes)
        self.dhl = [_sub(b, "self_attn.o_proj").in_features // self.H for b in self.BL]
        self.dh = self.dhl[0]
        self.post = "olmo" in name   # OLMo-2: norms after the blocks
        self.abl, self.add, self.keep = {}, {}, {}
        self.mabl, self.rpatch, self.scale = {}, {}, {}
        self.offsets = (1,)
        self.cap = set(); self.c = {}
        self.last = None
        for l, b in enumerate(self.BL):
            _sub(b, "self_attn.o_proj").register_forward_pre_hook(self._oproj(l))
            b.mlp.register_forward_hook(self._mlpout(l))
            b.register_forward_hook(self._block(l))
            if self.post:
                b.self_attn.register_forward_pre_hook(self._attn_in_kw(l), with_kwargs=True)
                b.mlp.register_forward_pre_hook(self._pre(("mlp", l)))
            else:
                b.input_layernorm.register_forward_pre_hook(self._pre(("attn", l)))
                b.post_attention_layernorm.register_forward_pre_hook(self._pre(("mlp", l)))
        final_norm(self.model).register_forward_pre_hook(self._pre(("final",)))
        self.model.get_input_embeddings().register_forward_hook(self._emb)

    def _emb(self, m, a, o):
        if "emb" in self.cap:
            r, p = self._rows(o); self.c["emb"] = o[r, p].detach().float().clone()

    def _rows(self, x):
        return torch.arange(x.shape[0], device=x.device), self.last.to(x.device)

    def _pre(self, key):
        def f(m, a):
            x = a[0]
            if key[0] == "mlp" and "mid" in self.cap:
                r, p = self._rows(x); self.c.setdefault("mid", {})[key[1]] = x[r, p].detach().float().clone()
            if key in self.add:
                x = x.clone(); r, p = self._rows(x)
                x[r, p] = x[r, p] + self.add[key].to(x.dtype).to(x.device)
                return (x,) + tuple(a[1:])
        return f

    def _attn_in_kw(self, l):
        def f(m, a, kw):
            if ("attn", l) not in self.add: return None
            x = kw["hidden_states"].clone(); r, p = self._rows(x)
            x[r, p] = x[r, p] + self.add[("attn", l)].to(x.dtype).to(x.device)
            kw = dict(kw); kw["hidden_states"] = x
            return a, kw
        return f

    def _oproj(self, l):
        def f(m, a):
            x = a[0]; r, p = self._rows(x); dh = self.dhl[l]
            if "oin" in self.cap: self.c.setdefault("oin", {})[l] = x[r, p].detach().float().clone()
            ch = False
            if l in self.abl:
                x = x.clone(); ch = True
                for h, v in self.abl[l]:
                    x[r, p, h * dh:(h + 1) * dh] = v.to(x.dtype).to(x.device)
            if l in self.scale:
                if not ch: x = x.clone(); ch = True
                for h, k in self.scale[l]:
                    x[r, p, h * dh:(h + 1) * dh] = x[r, p, h * dh:(h + 1) * dh] * k
            if l in self.keep:
                if not ch: x = x.clone(); ch = True
                h, clean = self.keep[l]
                cur = x[r, p].clone()
                new = clean.to(x.dtype).to(x.device).clone()
                new[:, h * dh:(h + 1) * dh] = cur[:, h * dh:(h + 1) * dh]
                x[r, p] = new
            return ((x,) + tuple(a[1:])) if ch else None
        return f

    def _mlpout(self, l):
        def f(m, a, o):
            r, p = self._rows(o)
            if "mlpout" in self.cap:
                self.c.setdefault("mlpout", {})[l] = o[r, p].detach().float().clone()
            if l in self.mabl:
                o = o.clone(); o[r, p] = self.mabl[l].to(o.dtype).to(o.device); return o
        return f

    def _block(self, l):
        def f(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            r, p = self._rows(h)
            if "resid" in self.cap:
                self.c.setdefault("resid", {})[l] = h[r, p].detach().float().clone()
            if "residoff" in self.cap:
                for k in self.offsets:
                    self.c.setdefault(f"resid-{k}", {})[l] = h[r, p - k].detach().float().clone()
            if l in self.rpatch:
                k, v = self.rpatch[l]
                h = h.clone(); h[r, p - k] = v.to(h.dtype).to(h.device)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return f

    def encode_batch(self, prefixes):
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        enc = [encode(self.tok, p) for p in prefixes]
        T = max(len(e) for e in enc)
        ids = torch.full((len(enc), T), pad, dtype=torch.long); att = torch.zeros((len(enc), T), dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, :len(e)] = torch.tensor(e); att[j, :len(e)] = 1
        return ids.to(DEVICE), att.to(DEVICE), torch.tensor([len(e) - 1 for e in enc], device=DEVICE)

    @torch.no_grad()
    def run(self, items, cap=(), bs=192, final_cap=False):
        """Suffix logits [N, 4] for items; captures (concatenated over batches) in self.c when cap is given.
        Interventions in abl/add/keep that hold per-row tensors must be sized for the whole item list; they are
        sliced per batch here."""
        outs, caps = [], {}
        full_add, full_keep, full_rp = self.add, self.keep, self.rpatch
        for i in range(0, len(items), bs):
            ids, att, last = self.encode_batch([it["prefix"] for it in items[i:i + bs]])
            self.last = last; self.cap = set(cap); self.c = {}
            self.add = {k: v[i:i + bs] for k, v in full_add.items()}
            self.keep = {l: (h, c[i:i + bs]) for l, (h, c) in full_keep.items()}
            self.rpatch = {l: (k, v[i:i + bs]) for l, (k, v) in full_rp.items()}
            hs = self.model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
            hl = hs[torch.arange(len(last), device=DEVICE), last]
            lg = unembed(self.model, hl)
            outs.append(lg[:, self.cols].float().cpu())
            if final_cap: self.c["final"] = hl.float()
            for k, v in self.c.items():
                if isinstance(v, dict):
                    for l, t in v.items(): caps.setdefault(k, {}).setdefault(l, []).append(t.cpu())
                else: caps.setdefault(k, []).append(v.cpu())
        self.add, self.keep, self.rpatch, self.cap = full_add, full_keep, full_rp, set()
        self.c = {k: ({l: torch.cat(t) for l, t in v.items()} if isinstance(v, dict) else torch.cat(v)) for k, v in caps.items()}
        return torch.cat(outs).numpy()

    def clear(self):
        self.abl, self.add, self.keep = {}, {}, {}
        self.mabl, self.rpatch, self.scale = {}, {}, {}

    # ---- helpers
    def head_means(self, items):
        """Mean o_proj input at the last token over items: {layer: [H*dh]}."""
        self.clear(); self.run(items, cap=("oin",))
        return {l: v.mean(0) for l, v in self.c["oin"].items()}

    def ablate_heads(self, heads, means):
        self.abl = {}
        for l, h in heads:
            self.abl.setdefault(l, []).append((h, means[l][h * self.dhl[l]:(h + 1) * self.dhl[l]]))

    def lens(self, resid):
        """Logit lens on captured block outputs [N, d] -> suffix logits [N, 4]."""
        with torch.no_grad():
            x = resid.to(DEVICE).to(next(self.model.parameters()).dtype)
            return unembed(self.model, final_norm(self.model)(x))[:, self.cols].float().cpu().numpy()

    def wdir(self):
        """Unembedding rows of the four suffixes [4, d], float32."""
        W = self.model.get_output_embeddings().weight
        return W[self.cols].float()


def probe_cv(X, y, folds=5, lam=1e-2, seed=0):
    """Cross-validated accuracy of an L2 logistic-regression probe (binary y). X [N, d] float32 tensor."""
    X = X.float(); y = torch.as_tensor(y, dtype=torch.float32)
    N = len(y); g = torch.Generator().manual_seed(seed); perm = torch.randperm(N, generator=g)
    accs = []
    for f in range(folds):
        te = perm[f::folds]; tr = torch.tensor(sorted(set(perm.tolist()) - set(te.tolist())))
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            with torch.enable_grad():
                loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, y[tr]) + lam * (w ** 2).sum()
                loss.backward()
            return loss
        with torch.enable_grad(): opt.step(closure)
        with torch.no_grad(): accs.append(float((((B @ w + b) > 0).float() == y[te]).float().mean()))
    return float(np.mean(accs))
