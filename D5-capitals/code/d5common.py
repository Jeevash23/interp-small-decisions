"""D5 shared helpers. Models come from D1-X's loader (xm3_common.M: Gemma-4-E2B's text decoder in
bfloat16 with its per-layer table read from disk; Qwen3, SmolLM2 and Pythia through transformers), so
head and residual hooks (m.patch, m.rpatch, m.cap, m.rcap) work as in D1-X.

Definitions follow CHARTER.md: words are maximal letter runs; a word start is a letter run preceded by
a space whose first letter is the first non-space character of a token; only words after the first
five of a paragraph count; inside-word positions are tokens whose first character lies inside a run."""
import json, os, random, re, stat, sys
from pathlib import Path

D5 = Path(__file__).resolve().parent.parent
XM = D5.parent / "D1-line-breaking" / "xmodel" / "code"
sys.path.insert(0, str(XM))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch
import xm3_common as X  # noqa: E402

OUT = D5 / "outputs"; OUT.mkdir(exist_ok=True)
DEV = X.DEV
NAMES = {"gemma": "gemma4-e2b", "qwen3-17b": "qwen3-17b", "smollm2": "smollm2-17b", "pythia-1b": "pythia-1b",
         "qwen3-06b": "qwen3-06b", "pythia-410m": "pythia-410m"}
MAX_CHARS = 1200  # paragraphs cut at the last space before this (keeps Gemma's logits within 6 GB)


def load_part(part, seed=0):
    """Paragraph records of a D5 part, shuffled with `seed`. 'conf' only once PROTOCOL.md exists and
    is read-only; 'reserve' never (CHARTER.md)."""
    if part == "reserve": raise SystemExit("refusing to read reserve: kept for a later round")
    if part == "conf":
        p = D5 / "PROTOCOL.md"
        if not p.exists() or (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
            raise SystemExit("refusing to read conf: PROTOCOL.md is not frozen")
    rows = [json.loads(l) for l in open(D5 / "data" / f"part-{part}.jsonl")]
    random.Random(seed).shuffle(rows)
    return rows


def cut(text, n=MAX_CHARS):
    if len(text) <= n: return text
    return text[:text.rfind(" ", 0, n)]


def title(text):
    return re.sub(r"[^\W\d_]+", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), text)


def versions(text):
    return {"upper": text.upper(), "lower": text.lower(), "title": title(text), "orig": text}


def runs(text):
    """Letter runs as (start, end) character spans."""
    return [m.span() for m in re.finditer(r"[^\W\d_]+", text)]


def _bytes_to_unicode():
    """GPT-2's byte-to-character table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def vocab_strings(tok):
    """Surface string of every token id (byte-level BPE bytes decoded; SentencePiece '▁' -> ' ')."""
    toks = tok.convert_ids_to_tokens(list(range(len(tok))))
    if any(t and "Ġ" in t for t in toks[:5000]):
        bd = {v: k for k, v in _bytes_to_unicode().items()}
        out = []
        for t in toks:
            if t is None: out.append(""); continue
            try: out.append(bytes(bd[c] for c in t).decode("utf-8", errors="replace"))
            except KeyError: out.append(t)
        return out
    out = []
    for t in toks:
        if t is None: out.append(""); continue
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t): out.append("�"); continue
        out.append(t.replace("▁", " "))
    return out


def case_of(s):
    up = any(c.isupper() for c in s); lo = any(c.islower() for c in s)
    if not up and not lo: return "caseless"
    return "upper" if up and not lo else "lower" if lo and not up else "mixed"


def first_cased_upper(s):
    for c in s:
        if c.isupper(): return True
        if c.islower(): return False
    return None


def encode(tok, text):
    e = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = e["input_ids"]; first = []
    for a, b in e["offset_mapping"]:
        k = a
        while k < b and text[k].isspace(): k += 1
        first.append(k if k < b else -1)
    return ids, first


def positions(text, first, skip_words=5):
    """Word-start token indices {k: run index} and inside-word token indices {k: (run index, rest)}."""
    rs = runs(text); starts = {s: i for i, (s, e) in enumerate(rs)}
    owner = {}
    for i, (s, e) in enumerate(rs):
        for c in range(s + 1, e): owner[c] = i
    ws, iw = {}, {}
    for k, c in enumerate(first):
        if k == 0 or c < 0: continue
        if c in starts:
            i = starts[c]
            if i >= skip_words and c > 0 and text[c - 1] == " ": ws[k] = i
        elif c in owner:
            i = owner[c]
            if i >= skip_words: iw[k] = (i, text[c:rs[i][1]])
    return ws, iw


class Model:
    """Thin wrapper over xm3_common.M with top-1 over all positions, greedy word decoding and the
    raw HF model for cached decoding."""
    def __init__(self, short, attn="sdpa"):
        self.short = short
        # Pythia (GPT-NeoX) uses D1-X's first loader (xm_common.M), which has no BOS and no _logit_chunks
        self.m = X._xc.M(NAMES[short], attn=attn) if short.startswith("pythia") else X.M(NAMES[short], attn=attn)
        if not hasattr(self.m, "off"): self.m.off = 0
        self.tok = self.m.tok
        self.raw = self.m.model._m if self.m.off else self.m.model
        self.bos = self.tok.bos_token_id if self.m.off else None
        self.vs = vocab_strings(self.tok)
        nv = self.raw.get_output_embeddings().weight.shape[0]  # Qwen3's output is wider than its tokeniser
        self.vs += [""] * (nv - len(self.vs))

    def chunks(self, ids):
        if hasattr(self.m, "_logit_chunks"): yield from self.m._logit_chunks(ids); return
        for c in self.m.model(torch.tensor([ids], device=DEV)).logits[0].split(128): yield c.float()

    def top1(self, ids):
        with torch.no_grad():
            return torch.cat([c.argmax(-1) for c in self.chunks(ids)]).cpu().numpy()

    def greedy_word(self, prefix, max_new=8):
        """Greedy-decode one word after `prefix` (token ids). Returns (word, first token string):
        word = the letters of the decoded text after the leading space, up to the first non-letter;
        '' if the first token does not start with a space followed by a letter."""
        x = ([self.bos] if self.bos is not None else []) + list(prefix)
        x = torch.tensor([x], device=DEV); past = None; text = ""; first = None
        with torch.no_grad():
            for step in range(max_new):
                o = self.raw(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = o.past_key_values
                t = int(o.logits[0, -1].argmax()); s = self.vs[t]
                if step == 0:
                    first = s
                    if not (s.startswith(" ") and len(s) > 1 and s[1].isalpha()): return "", first
                    text = s[1:]
                else:
                    if not (s and s[0].isalpha()): break
                    text += s
                m = re.match(r"[^\W\d_]*", text)
                if m.end() < len(text): text = text[:m.end()]; break
                x = torch.tensor([[t]], device=DEV)
        return re.match(r"[^\W\d_]*", text).group(0), first


def save(obj, name):
    with open(OUT / name, "w") as f: json.dump(obj, f, indent=1, ensure_ascii=False)
