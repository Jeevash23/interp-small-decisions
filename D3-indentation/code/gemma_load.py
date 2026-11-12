"""Gemma-4-E2B loader for D3, copied unchanged in substance from D1-X (xmodel/code/xm3_common.py, frozen)
so that D3 does not import D1 modules whose names clash with its own (both have a common.py)."""
from pathlib import Path
import torch

class _RowTable(torch.nn.Module):
    """Gemma-4's per-layer token-embedding lookup, reading only the needed rows from the safetensors
    file (memory-mapped). Same output as Gemma4TextScaledWordEmbedding: row * embed_scale in bfloat16."""
    def __init__(self, path, key, scale, dt):
        super().__init__()
        from safetensors import safe_open
        self._f = safe_open(str(path), "pt", device="cpu"); self._sl = self._f.get_slice(key); self._dt = dt
        self.scale = scale; self.embedding_dim = self._sl.get_shape()[1]
    def forward(self, ids):
        u, inv = torch.unique(ids.cpu(), return_inverse=True)
        rows = torch.cat([self._sl[int(i):int(i) + 1] for i in u]).to(self._dt)
        out = rows[inv] * torch.tensor(self.scale).to(self._dt)
        return out.to(ids.device)



def load_gemma(repo, dt, attn, DEV="cuda"):
    """Text decoder on the GPU in bfloat16; nothing is quantised. Built directly: an empty model,
    the image and audio encoders removed (never run), the per-layer embedding table (a lookup,
    262,144 x 8,960, 4.7 GB) replaced by _RowTable, which reads the rows of the tokens in each input
    from the weights file (same values, same scale, same bfloat16 arithmetic), and every other
    text weight read from the weights file straight onto the GPU. The output layer is tied to the
    token embedding, as in the checkpoint. The key and value weights stored for layers 15-34 are
    not used by the model (those layers reuse earlier layers' keys and values) and are not loaded."""
    from transformers import AutoModelForCausalLM, AutoConfig
    from huggingface_hub import snapshot_download
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    cfg = AutoConfig.from_pretrained(repo)
    st = Path(snapshot_download(repo, local_files_only=True)) / "model.safetensors"
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, dtype=dt)
    base = model.model
    for n in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"): setattr(base, n, None)
    lm = base.language_model
    lm.embed_tokens_per_layer = _RowTable(st, "model.language_model.embed_tokens_per_layer.weight",
                                          float(lm.embed_tokens_per_layer.scalar_embed_scale), dt)
    dev = 0 if DEV == "cuda" else DEV
    with safe_open(str(st), "pt", device="cpu") as f:
        have = set(f.keys())
        for k in list(model.state_dict()):
            if k == "lm_head.weight": continue
            assert k in have, k
            t = f.get_tensor(k)
            set_module_tensor_to_device(model, k, dev, value=t.to(dt) if t.is_floating_point() else t)
    model.lm_head.weight = lm.embed_tokens.weight
    model = model.to(DEV).eval()
    assert not any(p.is_meta for p in model.parameters()) and not any(b.is_meta for b in model.buffers())
    return model

