"""D3 round 2 (CHARTER-2.md): shared helpers for exploration. Imports round 1's common.py unchanged.
- r2_sources(name): exploration material only (round-1 explore parts, X1's code and generated parts, gen2 explore).
- Tap: captures, for chosen layers, the attention rows of every head at chosen query positions (eager attention
  only) and the value vectors, so that a head's output can be recomputed with some source positions removed.
- line_roles(text, offs, p): for a decision point, maps every token to a role relative to the point.
- point_class(p): header / same / term / dedent (as round 1)."""
import json, random
from pathlib import Path
from common import *

R2 = Path(__file__).resolve().parent.parent
X1D = R2.parent / "X1-shared-parts" / "data"
OUT2 = R2 / "outputs" / "r2"; OUT2.mkdir(parents=True, exist_ok=True)


def r2_sources(name):
    rd = lambda p: [json.loads(l) for l in open(p)]
    if name == "r1code": return rd(DATA / "newcode" / "part-explore.jsonl")
    if name == "r1gen": return rd(DATA / "gen" / "part-explore.jsonl")
    if name == "stdlib": return rd(DATA / "stdlib" / "part-explore.jsonl")
    if name == "x1code": return rd(X1D / "code" / "part-explore.jsonl") + rd(X1D / "code" / "part-held.jsonl")
    if name == "x1gen": return rd(X1D / "gen" / "part-explore.jsonl") + rd(X1D / "gen" / "part-held.jsonl")
    if name == "gen2": return rd(DATA / "r2gen2" / "part-explore.jsonl")
    raise SystemExit(f"not an exploration source: {name}")


def point_class(p):
    return "header" if p["header"] else "same" if p["cls"] == "same" else "term" if p["terminal"] else "dedent"


def hn(s):
    l, h = s[1:].split("H"); return int(l), int(h)


class Tap:
    """Capture attention rows A[l] (H, nq, T) at query positions `rows` and values V[l] (T, KVH, dh) for layers
    in `layers`, batch row 0. Requires the model loaded with attn='eager'."""
    def __init__(self, m, layers):
        self.m = m; self.layers = sorted(layers); self.rows = None; self.A = {}; self.V = {}; self.on = False
        c = m.model.config; c = getattr(c, "text_config", None) or c
        self.KVH = getattr(c, "num_key_value_heads", m.H); self.dh = m.dhl[0]
        for l in self.layers:
            att = m.layers[l].self_attn
            att.register_forward_hook(self._ah(l))
            att.v_proj.register_forward_hook(self._vh(l))

    def _ah(self, l):
        def f(mod, a, o):
            if self.on and o[1] is not None:
                self.A[l] = o[1][0][:, self.m._sh(torch.as_tensor(self.rows)), :][:, :, self.m.off:].float()
            return (o[0], None) + tuple(o[2:])
        return f

    def _vh(self, l):
        def f(mod, a, o):
            if self.on: self.V[l] = o[0, self.m.off:].view(-1, self.KVH, self.dh).float()
        return f

    def head_out(self, l, h, qi, keep=None):
        """Recompute head h's output at query row index qi; keep: bool mask over source positions (others
        removed, attention renormalised over the kept ones)."""
        a = self.A[l][h, qi]
        if keep is not None:
            a = a * keep.to(a.device); a = a / a.sum().clamp_min(1e-9)
        g = self.m.H // self.KVH
        return a @ self.V[l][:, h // g, :]


def line_roles(text, offs, p, d):
    """Roles of tokens 0..d for decision point p (with p['d'] == d). Returns dict role -> list of positions:
    self (d), sink (0), prev (the previous logical line's other tokens), prev_start (its first token),
    encl (the header line that opened the block holding the previous line: the nearest earlier logical line
    with a smaller indent than the previous line), encl_start, target (the nearest earlier logical line whose
    indent equals the true next indent, other than the previous line), target_start, other."""
    lines = text.split("\n"); lo = [0]
    for ln in lines: lo.append(lo[-1] + len(ln) + 1)
    starts = sorted(P._logical_starts(text)); prior = [n for n in starts if n < p["line"]]
    tokline = []
    for i in range(d + 1):
        a = offs[i][0]; n = max(j for j in range(len(lo) - 1) if lo[j] <= a) + 1 if a < lo[-1] else len(lines)
        tokline.append(n)
    ind = lambda n: len(lines[n - 1]) - len(lines[n - 1].lstrip(" "))
    # logical line spans: from a logical start to the next one
    span = {}
    for i, n in enumerate(prior):
        nxt = prior[i + 1] if i + 1 < len(prior) else p["line"]
        span[n] = (n, nxt - 1)
    which = lambda n: next((s for s, (a, b) in span.items() if a <= n <= b), None)
    R = {k: [] for k in ["self", "sink", "prev", "prev_start", "encl", "encl_start", "target", "target_start", "other"]}
    if not prior: return R
    pv = prior[-1]; encl = next((n for n in reversed(prior[:-1]) if ind(n) < ind(pv) and lines[n - 1].rstrip().endswith(":")), None)
    tg = next((n for n in reversed(prior[:-1]) if ind(n) == p["true"]), None) if p["cls"] != "deeper" else None
    first = {}
    for i in range(d + 1):
        s = which(tokline[i])
        if s is not None and s not in first and offs[i][1] > offs[i][0]: first[s] = i
    for i in range(d + 1):
        s = which(tokline[i])
        if i == d: r = "self"
        elif i == 0: r = "sink"
        elif s == pv: r = "prev_start" if first.get(s) == i else "prev"
        elif encl is not None and s == encl: r = "encl_start" if first.get(s) == i else "encl"
        elif tg is not None and s == tg: r = "target_start" if first.get(s) == i else "target"
        else: r = "other"
        R[r].append(i)
    return R


def open_headers(text, offs, p):
    """Headers whose blocks are still open just before decision point p (after its previous logical line):
    list of (header_indent, token index of the header's ':\\n' token, i.e. the token covering the newline that
    ends the header's logical line), innermost last. Uses Python's tokenize."""
    import tokenize, io
    lines = text.split("\n"); lo = [0]
    for ln in lines: lo.append(lo[-1] + len(ln) + 1)
    logical = []  # (start_row, indent, nl_char, is_header)
    start = None; last_op = None
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING): continue
            if t.type == tokenize.NEWLINE:
                if start is not None:
                    r = start; ind = len(lines[r - 1]) - len(lines[r - 1].lstrip(" "))
                    logical.append((r, ind, lo[t.start[0] - 1] + t.start[1], last_op == ":"))
                start = None; last_op = None; continue
            if t.type == tokenize.ENDMARKER: break
            if start is None: start = t.start[0]
            last_op = t.string if t.type == tokenize.OP else None
    except (tokenize.TokenError, IndentationError):
        return []
    st = []
    for r, ind, nlc, hdr in logical:
        if r >= p["line"]: break
        while st and st[-1][0] >= ind: st.pop()
        if hdr: st.append((ind, nlc))
    tokat = lambda ch: next((i for i, (a, b) in enumerate(offs) if a <= ch < b), None)
    return [(ind, tokat(nlc)) for ind, nlc in st]


class EdgeBlock:
    """Exact attention-edge removal (eager attention, batch row 0): while self.edges is a non-empty list of
    (layers, q_positions, k_positions), every head of those layers gets -inf added to the scores from those query
    positions to those key positions before the softmax. Text positions (shifted by m.off)."""
    def __init__(self, m):
        self.m = m; self.edges = []
        for l, lay in enumerate(m.layers):
            lay.self_attn.register_forward_pre_hook(self._h(l), with_kwargs=True)

    def _h(self, l):
        def f(mod, args, kwargs):
            es = [e for e in self.edges if l in e[0]]
            if not es: return None
            hs = kwargs.get("hidden_states", args[0] if args else None); T = hs.shape[1]
            am = kwargs.get("attention_mask")
            neg = torch.finfo(hs.dtype).min
            if am is None:
                am = torch.full((1, 1, T, T), neg, dtype=hs.dtype, device=hs.device).triu(1)
            else:
                am = am.clone()
                if am.dtype == torch.bool:
                    am = torch.where(am, torch.zeros((), dtype=hs.dtype, device=hs.device), torch.tensor(neg, dtype=hs.dtype, device=hs.device))
            for _, qs, ks in es:
                qi = torch.as_tensor(qs, device=hs.device) + self.m.off; ki = torch.as_tensor(ks, device=hs.device) + self.m.off
                am[0, :, qi[:, None], ki[None, :]] = neg
            kwargs["attention_mask"] = am
            return args, kwargs
        return f


# Question 5 (added 9 Nov 09:20): OLMo-2-0425-1B training checkpoints in the shared Hugging Face cache, registered
# as models "olmo1b-<tokens>" (bf16, as the final model "olmo1b"). Only checkpoints already on disk are registered.
OLMO_CKPTS = {"1B": "stage1-step300-tokens1B", "21B": "stage1-step10000-tokens21B", "84B": "stage1-step40000-tokens84B",
              "420B": "stage1-step200000-tokens420B", "1007B": "stage1-step480000-tokens1007B",
              "4001B": "stage1-step1907359-tokens4001B"}
_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--OLMo-2-0425-1B"
for _tag, _rev in OLMO_CKPTS.items():
    _ref = _hub / "refs" / _rev
    if _ref.exists():
        MODELS[f"olmo1b-{_tag}"] = (str(_hub / "snapshots" / _ref.read_text().strip()), torch.bfloat16)
