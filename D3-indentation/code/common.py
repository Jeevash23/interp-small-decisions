"""D3 model-side helpers (exploratory). Loading the charter models with head, MLP and residual hooks,
and the indent distribution of CHARTER.md (Measures).

Material loader: load_chunks(source, part). A sealed confirmation part ('conf') loads only once
PROTOCOL.md exists and is read-only (frozen), as in D1-X."""
import os, sys, json, stat, hashlib, random
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch

D3 = Path(__file__).resolve().parent.parent
DATA, OUT = D3 / "data", D3 / "outputs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyindent as P

DEV = os.environ.get("D3_DEVICE", "cuda")
MODELS = {"pythia-410m": ("EleutherAI/pythia-410m", torch.float32), "gpt2": ("openai-community/gpt2", torch.float32),
          "qwen3-06b": ("Qwen/Qwen3-0.6B-Base", torch.float32), "gemma4-e2b": ("google/gemma-4-E2B", torch.bfloat16),
          "pythia-1b": ("EleutherAI/pythia-1b", torch.bfloat16), "qwen05b": ("Qwen/Qwen2.5-0.5B", torch.float32),
          "olmo1b": ("allenai/OLMo-2-0425-1B", torch.bfloat16), "smollm2-17b": ("HuggingFaceTB/SmolLM2-1.7B", torch.bfloat16),
          "qwen3-17b": ("Qwen/Qwen3-1.7B-Base", torch.bfloat16), "pythia-160m-dd": ("EleutherAI/pythia-160m-deduped", torch.float32),
          "pythia-410m-dd": ("EleutherAI/pythia-410m-deduped", torch.float32),
          # bfloat16 copy of the main model for fast exploratory scans (8 Nov 00:30; checked against float32)
          "qwen3-06b-bf16": ("Qwen/Qwen3-0.6B-Base", torch.bfloat16)}
MAXCTX = {"gpt2": 1024}


def load_chunks(source, part):
    """source in {'gen', 'newcode', 'stdlib'}; part in {'explore', 'conf'}. Returns list of dicts with 'text'."""
    if part == "conf":
        p = D3 / "PROTOCOL.md"
        if not p.exists() or (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
            raise SystemExit(f"refusing to read {source}/conf: PROTOCOL.md is not frozen")
    return [json.loads(l) for l in open(DATA / source / f"part-{part}.jsonl")]


# ------------------------------------------------------------------ models
class M:
    """A loaded model with hooks.
      self.zpatch[l] = list of (head, pos, value[, row]) applied to the input of layer l's attention output
        projection (per-head outputs); self.mpatch[l] = list of (pos, value[, row]) replacing layer l's MLP
        output; self.rpatch[l] = list of (pos, value[, row]) replacing layer l's residual output.
      self.capture=True stores zcap[l] (T, H*dh), mcap[l], rcap[l] (T, d) for batch row 0.
    Positions are text positions; for models with a BOS token (Gemma) everything is shifted by self.off."""
    def __init__(self, name, attn="eager"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        repo, dt = MODELS[name]
        self.name = name; self.tok = AutoTokenizer.from_pretrained(repo)
        if name == "gemma4-e2b":
            from gemma_load import load_gemma
            self.model = load_gemma(repo, dt, "sdpa" if attn == "sdpa" else "eager", DEV)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation=attn).to(DEV).eval()
        self.off = 1 if name == "gemma4-e2b" else 0
        m = self.model
        if hasattr(m, "gpt_neox"):
            self.layers = m.gpt_neox.layers; self.oproj = [l.attention.dense for l in self.layers]
            self.mlps = [l.mlp for l in self.layers]
        elif hasattr(m, "transformer"):  # GPT-2
            self.layers = m.transformer.h; self.oproj = [l.attn.c_proj for l in self.layers]; self.mlps = [l.mlp for l in self.layers]
        else:
            base = m.model.language_model if hasattr(m.model, "language_model") else m.model
            self.layers = base.layers; self.oproj = [l.self_attn.o_proj for l in self.layers]; self.mlps = [l.mlp for l in self.layers]
        c = m.config; c = getattr(c, "text_config", None) or c
        self.L = len(self.layers); self.H = c.num_attention_heads if hasattr(c, "num_attention_heads") else c.n_head
        ins = [(o.in_features if hasattr(o, "in_features") else o.weight.shape[0]) for o in self.oproj]  # GPT-2 Conv1D: (in, out)
        self.dhl = [i // self.H for i in ins]
        self.zpatch, self.mpatch, self.rpatch = {}, {}, {}
        self.capture = False; self.zcap, self.mcap, self.rcap = {}, {}, {}
        for l in range(self.L):
            self.oproj[l].register_forward_pre_hook(self._zhook(l))
            self.mlps[l].register_forward_hook(self._mhook(l))
            self.layers[l].register_forward_hook(self._rhook(l))
        self.vocab_strings()

    def _sh(self, pos):
        if not self.off: return pos
        if isinstance(pos, slice): return slice(None if pos.start is None else pos.start + self.off,
                                                None if pos.stop is None else pos.stop + self.off)
        return torch.as_tensor(pos) + self.off if not isinstance(pos, int) else pos + self.off

    def _zhook(self, l):
        def f(mod, a):
            z = a[0]
            if self.capture: self.zcap[l] = z[0, self.off:].detach().clone()
            lst = self.zpatch.get(l)
            if not lst: return None
            z = z.clone(); w = self.dhl[l]
            for e in lst:
                h, pos, val = e[:3]; row = e[3] if len(e) > 3 else slice(None)
                z[row, self._sh(pos), h * w:(h + 1) * w] = val.to(z.dtype).to(z.device)
            return (z,) + tuple(a[1:])
        return f

    def _mhook(self, l):
        def f(mod, a, o):
            x = o[0] if isinstance(o, tuple) else o
            if self.capture: self.mcap[l] = x[0, self.off:].detach().clone()
            lst = self.mpatch.get(l)
            if not lst: return None
            x = x.clone()
            for e in lst:
                pos, val = e[:2]; row = e[2] if len(e) > 2 else slice(None)
                x[row, self._sh(pos)] = val.to(x.dtype).to(x.device)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return f

    def _rhook(self, l):
        def f(mod, a, o):
            x = o[0] if isinstance(o, tuple) else o
            if self.capture: self.rcap[l] = x[0, self.off:].detach().clone()
            lst = self.rpatch.get(l)
            if not lst: return None
            x = x.clone()
            for e in lst:
                pos, val = e[:2]; row = e[2] if len(e) > 2 else slice(None)
                x[row, self._sh(pos)] = val.to(x.dtype).to(x.device)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return f

    # -------------------------------------------------------------- vocabulary
    def vocab_strings(self):
        n = len(self.tok)
        self.vs = [self.tok.decode([i]) for i in range(n)]
        self.ws_only = np.array([len(s) > 0 and s.strip(" \n") == "" for s in self.vs])
        self._sets = {}

    def set_ids(self, pre):
        """Tokens whose string is `pre` followed by a non-whitespace character."""
        if pre not in self._sets:
            L = len(pre)
            ids = [i for i, s in enumerate(self.vs) if len(s) > L and s.startswith(pre) and not s[L].isspace()]
            self._sets[pre] = torch.tensor(ids, device=DEV)
        return self._sets[pre]

    # -------------------------------------------------------------- forward
    def logprobs(self, ids, positions=None, batch=None, out_attn=False):
        """Log-softmax over the vocabulary at the given text positions (float32, on the GPU).
        ids: list[int] (one sequence) or LongTensor (B, T) with `positions` a list of (row, pos)."""
        x = torch.tensor([ids], device=DEV) if not torch.is_tensor(ids) else ids.to(DEV)
        if self.off:
            b = torch.full((x.shape[0], 1), self.tok.bos_token_id, dtype=x.dtype, device=x.device)
            x = torch.cat([b, x], 1)
        with torch.no_grad():
            # base model's final hidden states, then the output layer only at the wanted positions
            h = self.model.base_model(input_ids=x).last_hidden_state[:, self.off:]
            if positions is not None:
                rows = torch.tensor([r for r, _ in positions], device=DEV); cols = torch.tensor([p for _, p in positions], device=DEV)
                hh = h[rows, cols]
            else:
                hh = h[0]
            head = self.model.get_output_embeddings()
            cap = getattr(self.model.config.get_text_config(), "final_logit_softcapping", None)
            outs = []
            for c in hh.split(128):
                lg = head(c)
                if cap is not None: lg = torch.tanh(lg / cap) * cap
                outs.append(torch.log_softmax(lg.float(), -1))
            return torch.cat(outs)


# ------------------------------------------------------------------ indent distribution
def prepare(m, text, kmax_extra=8, kcap=40):
    """Tokenise a chunk, find its decision points, and build each point's candidate paths.
    Returns (ids, pts) where every kept point has: d (decision position), K (candidate indents), paths
    (list of element lists; an element is ('t', token_id) or ('s', pre)), u (index of the first element
    that identifies the candidate). Points whose candidate tokenisations disturb the context are dropped."""
    tok = m.tok
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > MAXCTX.get(m.name, 4096) - 48: return ids, None
    pts = []
    for p in P.decision_points(text):
        s = p["s"]
        d = max([i for i, (a, b) in enumerate(offs) if b <= s and b > a], default=None)
        if d is None: continue
        end_d = offs[d][1]
        content = text[s + p["true"]:].split("\n", 1)[0]
        K = list(range(0, min(kcap, max(p["prev"], p["true"]) + kmax_extra) + 1))
        if p["true"] not in K: continue
        # Amended 7 Nov 22:30 (NOTEBOOK): the decision position is the last token shared by every candidate's
        # tokenisation (some tokenisers split a blank-line run differently before column 0), and a
        # candidate's factors run at least past its last token that contains a newline.
        toks, okp = [], True
        for k in K:
            t = text[:s] + " " * k + content
            e2 = tok(t, return_offsets_mapping=True, add_special_tokens=False)
            toks.append((t, e2["input_ids"], e2["offset_mapping"], s + k))
        cp = 0
        while all(len(x[1]) > cp for x in toks) and len({x[1][cp] for x in toks}) == 1: cp += 1
        d = min(d, cp - 1)
        if d < 0 or any(x[1][:d + 1] != ids[:d + 1] for x in toks): continue
        paths = []
        for t, ii, oo, cpos in toks:
            el = []
            for j in range(d + 1, len(ii)):
                a, b = oo[j]
                if a <= cpos < b:
                    el.append(("s", t[a:cpos])); break
                el.append(("t", ii[j]))
            else:
                okp = False; break
            paths.append(el)
        if not okp: continue
        us = []
        for i, el in enumerate(paths):
            u = 0
            while any(j != i and len(o) > u and o[:u + 1] == el[:u + 1] for j, o in enumerate(paths)):
                u += 1
            nl = [j for j, e in enumerate(el) if e[0] == "t" and "\n" in m.vs[e[1]]]
            if nl: u = max(u, nl[-1] + 1)
            if u >= len(el): okp = False; break
            us.append(u)
        if not okp: continue
        p.update(d=d, K=K, paths=paths, u=us)
        pts.append(p)
    return ids, pts


def indent_dist(m, ids, pts, root_lp=None, bs=8):
    """Fill p['mass'] (raw probability per candidate in p['K']) for each prepared point. root_lp: optional
    precomputed log-probs at every position (T, V) from the same (possibly patched) forward pass."""
    if not pts: return pts
    if root_lp is None:
        root_lp = m.logprobs(ids, positions=[(0, p["d"]) for p in pts])
        rootmap = {p["d"]: i for i, p in enumerate(pts)}
    else:
        rootmap = None
    # nodes beyond the root: token prefixes of length >= 1 that some candidate needs
    need = []
    for pi, p in enumerate(pts):
        nodes = set()
        for el, u in zip(p["paths"], p["u"]):
            for j in range(1, u + 1):
                nodes.add(tuple(e[1] for e in el[:j]))
        maxi = [n for n in nodes if not any(len(o) > len(n) and o[:len(n)] == n for o in nodes)]
        for n in maxi: need.append((pi, n))
    nodelp = {}
    for i in range(0, len(need), bs):
        grp = need[i:i + bs]
        seqs = [ids[:pts[pi]["d"] + 1] + list(n) for pi, n in grp]
        T = max(len(s) for s in seqs)
        pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
        x = torch.tensor([s + [pad] * (T - len(s)) for s in seqs])
        positions = []
        for r, (pi, n) in enumerate(grp):
            for j in range(1, len(n) + 1): positions.append((r, pts[pi]["d"] + j))
        lp = m.logprobs(x, positions=positions)
        q = 0
        for r, (pi, n) in enumerate(grp):
            for j in range(1, len(n) + 1):
                nodelp[(pi, n[:j])] = lp[q]; q += 1
    for pi, p in enumerate(pts):
        r0 = root_lp[rootmap[p["d"]]] if rootmap is not None else root_lp[p["d"]]
        mass = []
        for el, u in zip(p["paths"], p["u"]):
            tot = 0.0
            for j in range(u + 1):
                lp = r0 if j == 0 else nodelp[(pi, tuple(e[1] for e in el[:j]))]
                kind, v = el[j]
                tot += float(lp[v]) if kind == "t" else float(torch.logsumexp(lp[m.set_ids(v)], 0))
            mass.append(float(np.exp(tot)))
        p["mass"] = mass
    return pts


def summarise_point(p):
    """Derived quantities for one point with p['mass'] set."""
    K, mass = p["K"], np.array(p["mass"])
    tot = mass.sum(); q = mass / tot
    k = int(np.argmax(q)); st = p["stack"]; cur = st[-1]
    open_sh = [x for x in st[:-1]]
    return dict(top1=K[k] == p["true"], pred=K[k], lp_true=float(np.log(q[K.index(p["true"])])), other=float(1 - tot),
                p_deeper=float(q[[i for i, x in enumerate(K) if x > cur]].sum()),
                p_same=float(q[K.index(cur)]) if cur in K else 0.0,
                p_shallow=float(q[[i for i, x in enumerate(K) if x < cur]].sum()),
                p_shallow_open=float(q[[i for i, x in enumerate(K) if x in open_sh]].sum()),
                p_shallow_notopen=float(q[[i for i, x in enumerate(K) if x < cur and x not in open_sh]].sum()),
                copy_ok=p["true"] == p["prev"])


class RootMasses:
    """Vectorised root-token candidate masses for a fixed list of points (all with max(u) == 0), reused
    across many forward passes (ablation scans). __call__(lp) with lp (n_points, V) returns a list of
    renormalised numpy arrays, one per point. Added 8 Nov 00:05 (speed); same arithmetic as indent_dist."""
    def __init__(self, m, pts):
        self.m = m; self.pres = sorted({el[0][1] for p in pts for el in p["paths"] if el[0][0] == "s"})
        rt, it, ft, rs, ps, fs = [], [], [], [], [], []; self.seg = [0]; f = 0
        for r, p in enumerate(pts):
            for el in p["paths"]:
                kind, v = el[0]
                if kind == "t": rt.append(r); it.append(v); ft.append(f)
                else: rs.append(r); ps.append(self.pres.index(v)); fs.append(f)
                f += 1
            self.seg.append(f)
        T = lambda x: torch.tensor(x, device=DEV, dtype=torch.long)
        self.rt, self.it, self.ft, self.rs, self.ps, self.fs, self.n = T(rt), T(it), T(ft), T(rs), T(ps), T(fs), f
        self.sets = [m.set_ids(pr) for pr in self.pres]

    def __call__(self, lp):
        out = torch.empty(self.n, device=lp.device)
        if len(self.ft): out[self.ft] = lp[self.rt, self.it]
        if len(self.fs):
            S = torch.stack([torch.logsumexp(lp[:, s], 1) for s in self.sets])  # (n_pres, rows)
            out[self.fs] = S[self.ps, self.rs]
        v = out.double().cpu().numpy(); res = []
        for a, b in zip(self.seg[:-1], self.seg[1:]):
            x = np.exp(v[a:b] - v[a:b].max()); res.append(x / x.sum())
        return res
