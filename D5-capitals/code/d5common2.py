"""D5 round 2 helpers (CHARTER-2.md). Re-exports the frozen round-1 helpers (d5common) unchanged and adds:
 * load_part2: paragraph parts; 'reserve' (held out for round 2) only once PROTOCOL-2.md exists and is
   read-only; 'explore' and 'conf' (exploration material in round 2) always;
 * load_headings: heading items; 'reserve' under the same rule;
 * heading_text: the text of a heading item with its heading in capitals or in its original case;
 * Writes: per-component writes at the last position (attention heads, MLPs, embedding, Gemma's
   per-layer-input blocks), captured with forward hooks, for decomposing the residual along a direction.
"""
import stat
from d5common import *  # noqa: F401,F403
from d5common import D5, OUT, DEV, torch, np, json, random, load_part as _load_part


def _frozen2():
    p = D5 / "PROTOCOL-2.md"
    return p.exists() and not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def load_part2(part, seed=0):
    if part == "reserve":
        if not _frozen2(): raise SystemExit("refusing to read reserve: PROTOCOL-2.md is not frozen")
        rows = [json.loads(l) for l in open(D5 / "data" / "part-reserve.jsonl")]
        random.Random(seed).shuffle(rows); return rows
    return _load_part(part, seed)


def load_headings(part, seed=0):
    if part == "reserve" and not _frozen2(): raise SystemExit("refusing to read reserve headings: PROTOCOL-2.md is not frozen")
    rows = [json.loads(l) for l in open(D5 / "data" / f"headings-{part}.jsonl")]
    random.Random(seed).shuffle(rows); return rows


def heading_text(it, caps=True):
    """(text, start of heading, start of next paragraph)."""
    h = it["heading"].upper() if caps else it["heading"]
    a = it["prev"] + "\n\n"; b = a + h + "\n\n"
    return b + it["next"], len(a), len(b)


class Writes:
    """Captures, for the last position of a batch row 0, every additive term of the residual stream:
    'emb' (scaled token embedding), ('attn', l, h) per head, ('mlp', l), ('ple', l) for Gemma's per-layer
    input block; and each layer's scalar (Gemma). Terms are returned already passed through the post-norm
    scale of their block (exact: RMSNorm is a vector scale of its input times a fixed weight)."""
    def __init__(self, M):
        self.M = M; m = M.m; self.m = m; self.L = m.L; self.H = m.H; self.on = False; self.t = {}
        raw = M.raw; base = getattr(raw.model, "language_model", raw.model)
        self.gemma = hasattr(m.layers[0], "post_attention_layernorm") and hasattr(m.layers[0], "post_feedforward_layernorm")
        self.hooks = []
        emb = base.embed_tokens
        self.hooks.append(emb.register_forward_hook(lambda mod, a, o: self._put("emb", o[0, -1]) if self.on else None))
        for l, lay in enumerate(m.layers):
            self.hooks.append(lay.self_attn.o_proj.register_forward_pre_hook(self._zh(l)))
            if self.gemma:
                self.hooks.append(lay.post_attention_layernorm.register_forward_hook(self._nh(("attn_n", l))))
                self.hooks.append(lay.post_feedforward_layernorm.register_forward_hook(self._nh(("mlp", l))))
                if getattr(lay, "post_per_layer_input_norm", None) is not None:
                    self.hooks.append(lay.post_per_layer_input_norm.register_forward_hook(self._nh(("ple", l))))
            else:
                self.hooks.append(lay.mlp.register_forward_hook(self._nh(("mlp", l))))

    def _put(self, k, v): self.t[k] = v.detach().float().clone()

    def _zh(self, l):
        def f(mod, a):
            if not self.on: return None
            z = a[0][0, -1].float(); W = mod.weight.float(); dh = self.m.dhl[l]
            o = torch.stack([z[h * dh:(h + 1) * dh] @ W[:, h * dh:(h + 1) * dh].T for h in range(self.H)])
            self.t[("attn_raw", l)] = o  # (H, d) per-head o_proj outputs (no bias in these models)
            return None
        return f

    def _nh(self, k):
        def f(mod, a, o):
            if not self.on: return None
            self._put(k, o[0, -1])
            if k[0] == "attn_n": self._put(("attn_in", k[1]), a[0][0, -1])
            return None
        return f

    def terms(self):
        """Dict of additive terms at the last position, in the residual after the last layer's scale
        is NOT applied; use scale_to(l) to bring term from layer j to the residual after layer l."""
        out = {"emb": self.t["emb"]}
        for l in range(self.L):
            raw = self.t[("attn_raw", l)]
            if self.gemma:
                s = self.t[("attn_n", l)] / torch.where(self.t[("attn_in", l)].abs() > 0, self.t[("attn_in", l)], torch.ones_like(raw[0]))
                # recompute exactly: RMSNorm(x) = x * c(x) * w, so per-head share = raw_h * (normed / x)
                ratio = torch.where(self.t[("attn_in", l)].abs() > 1e-6, s, torch.zeros_like(s))
                for h in range(self.H): out[("attn", l, h)] = raw[h] * ratio
            else:
                for h in range(self.H): out[("attn", l, h)] = raw[h]
            out[("mlp", l)] = self.t[("mlp", l)]
            if ("ple", l) in self.t: out[("ple", l)] = self.t[("ple", l)]
        return out

    def scalars(self):
        if not self.gemma: return [1.0] * self.L
        return [float(getattr(lay, "layer_scalar", torch.tensor(1.0)).float().reshape(-1)[0]) for lay in self.m.layers]


class LastBatch:
    """Run the last token of a prefix n times in one batch on top of a cached prefix (the earlier
    positions are computed once). Hooks that edit position -1 (m.patch, m.rpatch, own hooks) act on
    that last token only, exactly as in a full forward pass; the earlier positions are untouched.
    Every prefix must be primed (prime(ids)) while no patch is set; a call on an unprimed prefix fails."""
    def __init__(self, M_):
        import copy
        self.M_ = M_; self.past = {}; self._copy = copy.deepcopy

    def prime(self, ids):
        m = self.M_.m
        assert not m.patch and not m.rpatch, "prime with patches set"
        x = ([self.M_.bos] if self.M_.bos is not None else []) + list(ids[:-1])
        with torch.no_grad(): o = self.M_.raw(input_ids=torch.tensor([x], device=DEV), use_cache=True, logits_to_keep=1)
        self.past[tuple(ids)] = o.past_key_values

    def clear(self): self.past = {}

    def __call__(self, ids, n):
        """Next-token logits (n, V), float32, for the last position of `ids`, batch of n."""
        p = self._copy(self.past[tuple(ids)]); p.batch_repeat_interleave(n)
        with torch.no_grad():
            o = self.M_.raw(input_ids=torch.tensor([[ids[-1]]] * n, device=DEV), past_key_values=p, use_cache=True, logits_to_keep=1)
        return o.logits[:, -1].float()
