"""Residual-stream capture and patching hooks for D4 (exploratory)."""
import torch
from common import blocks, DEVICE


def _out(o):
    return o[0] if isinstance(o, tuple) else o


def _set(o, h):
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h


class Resid:
    """Capture block outputs (resid after layer l) and optionally patch rows at given positions.
    patch: dict layer -> (pos_index_tensor [B], values [B, d]) replaced in the block output."""

    def __init__(self, model):
        self.model, self.bl = model, blocks(model)
        self.cap, self.patch, self.capture = {}, {}, False
        self.hs = [b.register_forward_hook(self._hook(l)) for l, b in enumerate(self.bl)]

    def _hook(self, l):
        def f(m, a, o):
            h = _out(o)
            if l in self.patch:
                pos, val = self.patch[l]
                h = h.clone()
                h[torch.arange(h.shape[0], device=h.device), pos] = val.to(h.dtype)
                o = _set(o, h)
            if self.capture:
                self.cap[l] = h.detach()
            return o
        return f

    def remove(self):
        for h in self.hs:
            h.remove()


def batch_ids(tok, prefixes):
    from common import encode
    enc = [encode(tok, p) for p in prefixes]
    L = max(len(e) for e in enc)
    assert all(len(e) == L for e in enc), "patching batches need equal lengths"
    return torch.tensor(enc, device=DEVICE)
