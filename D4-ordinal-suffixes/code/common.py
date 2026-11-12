"""Shared helpers for D4 (ordinal suffixes). Exploratory code under CHARTER.md (frozen 7 Nov 2026).

load_part() reads the exploration part freely and refuses 'conf' and 'reserve' until PROTOCOL.md
exists and is read-only (frozen). Data files are checked against data/SHA256SUMS on load.
"""
import os, json, random, stat, hashlib
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
DEVICE = os.environ.get("D4_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "pythia-410m": ("EleutherAI/pythia-410m", torch.float32),
    "pythia-1b": ("EleutherAI/pythia-1b", torch.bfloat16),
    "qwen3-06b": ("Qwen/Qwen3-0.6B-Base", torch.float32),
    "qwen3-17b": ("Qwen/Qwen3-1.7B-Base", torch.bfloat16),
    "qwen25-15b": ("Qwen/Qwen2.5-1.5B", torch.bfloat16),
    "olmo2-1b": ("allenai/OLMo-2-0425-1B", torch.bfloat16),
    "smollm2-17b": ("HuggingFaceTB/SmolLM2-1.7B", torch.bfloat16),
    "gemma4-e2b": ("google/gemma-4-E2B", torch.bfloat16),
    "gpt2": ("openai-community/gpt2", torch.float32),
}
SUFFIXES = ["st", "nd", "rd", "th"]


def suffix(n):
    if n % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _check_digest(name):
    want = dict(l.split()[::-1] for l in open(DATA / "SHA256SUMS"))
    got = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    assert want[name] == got, f"{name} differs from its frozen digest"


def load_part(part):
    if part != "explore":
        p = ROOT / "PROTOCOL.md"
        if not p.exists() or (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
            raise SystemExit(f"refusing to read part {part}: PROTOCOL.md is not frozen")
    name = f"part-{part}.jsonl"
    _check_digest(name)
    return [json.loads(l) for l in open(DATA / name)]


def explore_numbers():
    """Numbers usable in constructed exploration inputs: 1-99 and the exploration numbers >= 100."""
    _check_digest("numbers.json")
    m = json.load(open(DATA / "numbers.json"))
    return sorted(int(n) for n, p in m.items() if p in ("all", "explore"))


def explore_frames():
    _check_digest("frames.jsonl")
    return [f for f in (json.loads(l) for l in open(DATA / "frames.jsonl")) if f["part"] == "explore"]


# ------------------------------------------------------------------ models
def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path, dtype = MODELS[name]
    if path.startswith("google/gemma-4"):
        return load_gemma4_text(path, dtype)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").to(DEVICE).eval()
    return tok, model


def suffix_ids(tok):
    ids = []
    for s in SUFFIXES:
        e = tok.encode(s, add_special_tokens=False)
        assert len(e) == 1, (s, e)
        ids.append(e[0])
    return ids


def encode(tok, prefix):
    """Token ids of a prefix, with the tokeniser's own special tokens (BOS where it adds one)."""
    return tok(prefix)["input_ids"]


def number_pieces(tok, prefix, n):
    """Positions (indices into encode(prefix)) of the tokens that spell the number, and their text."""
    ids = encode(tok, prefix)
    s = str(n)
    acc, pos = "", []
    for i in range(len(ids) - 1, -1, -1):
        t = tok.decode([ids[i]])
        acc = t + acc
        pos.insert(0, i)
        if acc.strip() == s or (acc.strip().endswith(s) and len(acc.strip()) >= len(s)):
            break
    pieces = [tok.decode([ids[i]]).strip() for i in pos]
    pieces = [p for p in pieces if p]
    pos = [i for i in pos if tok.decode([ids[i]]).strip()]
    assert "".join(pieces) == s, (prefix, pieces)
    return pos, pieces


@torch.no_grad()
def last_logits(model, tok, prefixes, cols, bs=64):
    """Logits of the columns `cols` and the log-sum-exp of all logits, at the last token of each prefix."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    out_c, out_l = [], []
    cols_t = torch.tensor(cols, device=DEVICE)
    for i in range(0, len(prefixes), bs):
        enc = [encode(tok, p) for p in prefixes[i:i + bs]]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), pad, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, :len(e)] = torch.tensor(e); att[j, :len(e)] = 1
        ids, att = ids.to(DEVICE), att.to(DEVICE)
        last = att.sum(1) - 1
        h = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
        hl = h[torch.arange(len(enc), device=DEVICE), last]
        lg = unembed(model, hl)
        out_c.append(lg[:, cols_t].cpu()); out_l.append(torch.logsumexp(lg, -1).cpu())
    return torch.cat(out_c).numpy(), torch.cat(out_l).numpy()


def unembed(model, h):
    """Final-normed hidden state -> float32 logits (with Gemma's soft cap)."""
    lg = model.get_output_embeddings()(h).float()
    c = getattr(model.config, "final_logit_softcapping", None)
    if c:
        lg = torch.tanh(lg / c) * c
    return lg


def final_norm(model):
    b = model.base_model
    for k in ("final_layer_norm", "ln_f", "norm"):
        if hasattr(b, k):
            return getattr(b, k)
    raise AttributeError("no final norm")


def blocks(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer"):
        return model.transformer.h
    return model.model.layers


def save_json(obj, name):
    json.dump(obj, open(OUT / name, "w"), indent=1, default=float)


# ------------------------------------------------------------------ Gemma 4 (copied from D2 code/common.py, 7 Nov)
class MmapPerLayerEmbedding(torch.nn.Module):
    def __init__(self, table, scale):
        super().__init__()
        self.table, self.scale = table, scale

    def forward(self, ids):
        rows = self.table[ids.reshape(-1).cpu()]
        return (rows.to(ids.device) * self.scale).reshape(*ids.shape, self.table.shape[1])


def load_gemma4_text(path, dtype):
    """Gemma 4 E2B text decoder only, per-layer table memory-mapped from disk (D2's loader)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    d = snapshot_download(path, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(d)
    cfg = AutoConfig.from_pretrained(d).text_config
    cfg._attn_implementation = "eager"
    nv, width = cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
    cfg.vocab_size_per_layer_input = 1
    with torch.device(DEVICE):
        model = Gemma4ForCausalLM._from_config(cfg, dtype=dtype)
    cfg.vocab_size_per_layer_input = nv
    f = os.path.join(d, "model.safetensors")
    with open(f, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
    pre, ple = "model.language_model.", "model.language_model.embed_tokens_per_layer.weight"
    params = dict(model.named_parameters()); params.update(dict(model.named_buffers()))
    loaded = set()
    with safe_open(f, "pt", device="cpu") as so:
        for k in so.keys():
            if not k.startswith(pre) or k == ple:
                continue
            tgt = "model." + k[len(pre):]
            if tgt not in params:
                continue
            t = so.get_tensor(k)
            with torch.no_grad():
                params[tgt].copy_(t.to(params[tgt].dtype))
            loaded.add(tgt)
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = model.model.embed_tokens.weight
    missing = [k for k in dict(model.named_parameters()) if k not in loaded
               and k not in ("lm_head.weight", "model.embed_tokens_per_layer.weight")]
    assert not missing, f"weights not loaded: {missing[:10]}"
    off = hdr[ple]["data_offsets"][0]
    mm = np.memmap(f, dtype=np.uint16, mode="r", offset=8 + n + off, shape=(nv, width))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = torch.from_numpy(mm).view(torch.bfloat16)
    model.model.embed_tokens_per_layer = MmapPerLayerEmbedding(table, float(cfg.hidden_size_per_layer_input) ** 0.5)
    return tok, model.eval()


def is_explore_number(n):
    """Charter split, extended to 10,000-9,999,999 by AMENDMENTS.md Amendment 1 (same hash rule)."""
    if n < 100:
        return True
    r = int(hashlib.sha256(f"D4-num-{n}".encode()).hexdigest(), 16) % 10
    return r < 6


def number_part_ext(n):
    """'all' for 1-99, else explore/conf/reserve by the charter's hash rule (AMENDMENTS.md Amendment 1 for > 9,999)."""
    if n < 100:
        return "all"
    r = int(hashlib.sha256(f"D4-num-{n}".encode()).hexdigest(), 16) % 10
    return "explore" if r < 6 else "conf" if r < 9 else "reserve"


def require_frozen_protocol(part):
    if part == "explore":
        return
    p = ROOT / "PROTOCOL.md"
    if not p.exists() or (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
        raise SystemExit(f"refusing to use part {part}: PROTOCOL.md is not frozen")


def part_frames(part):
    """Frames of one part. Non-exploration parts only once PROTOCOL.md is frozen."""
    require_frozen_protocol(part)
    _check_digest("frames.jsonl")
    return [f for f in (json.loads(l) for l in open(DATA / "frames.jsonl")) if f["part"] == part]


def in_part(n, part):
    return number_part_ext(n) in ("all", part)
