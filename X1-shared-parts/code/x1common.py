"""X1 shared helpers (exploratory unless run under a frozen PROTOCOL.md on the held parts).

Reuses the four studies' code by importing it (nothing copied, nothing changed):
  D1-X  xm_common / xm3_common (wrap, decision_points, wrapped_items, width_items, logodds, auroc, newline_id)
  D3    common (model class M with per-head hooks, prepare, RootMasses) and pyindent
  D4    common (suffix, suffix_ids)
  D5    d5common (cut, versions, encode, positions, case_of)
Each study's own module is called `common`, so they are loaded under unique names.

One model object (D3's M: per-head hooks on the input of each attention output projection, zpatch) serves
every task. Ablation = per-head mean ablation; means are taken over all text positions of a task's
exploration material (task_means) and saved, so a confirmation run can load them unchanged.

load_part(src, part): 'held' parts load only once X1's PROTOCOL.md exists and is read-only (frozen)."""
import os, sys, json, stat, random, hashlib, importlib.util, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch

X1 = Path(__file__).resolve().parent.parent
AREA = X1.parent
DATA, OUT = X1 / "data", X1 / "outputs"
OUT.mkdir(exist_ok=True)

# ---- D1-X (loads D1's `common` as a side effect; it is renamed so the next studies can load theirs)
sys.path.insert(0, str(AREA / "D1-line-breaking" / "xmodel" / "code"))
import xm_common as XC  # noqa: E402
D1C = sys.modules.pop("common"); sys.modules["d1common"] = D1C


def _load(name, path):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); sys.path.pop(0); return mod


D3 = _load("d3common", AREA / "D3-indentation" / "code" / "common.py")
D4 = _load("d4common", AREA / "D4-ordinal-suffixes" / "code" / "common.py")
D5 = _load("d5common_x1", AREA / "D5-capitals" / "code" / "d5common.py")
P = D3.P  # D3's pyindent
DEV = D3.DEV

# ---- head groups (CHARTER.md)
GROUPS = {
    "COUNT": "L4H0,L4H1,L5H13,L6H2,L7H15", "CORR": "L4H1", "WRITE": "L5H12,L5H13,L6H2",
    "READ": "L7H14,L15H3,L17H6,L22H3",
    "DED": "L23H2,L17H3,L17H4,L17H5,L18H6", "CARRY": "L17H4,L11H9,L5H6,L17H6,L16H2,L6H5",
    "TERM": "L11H9,L5H6,L17H6,L16H2,L23H6,L22H8,L23H3,L25H9,L19H6",
    "TEEN": "L0H2,L1H6", "L0H2": "L0H2", "L1H6": "L1H6", "L0H3": "L0H3",
    "L17H6": "L17H6", "L11H9": "L11H9", "L17H4": "L17H4",
    "READ-L17H6": "L7H14,L15H3,L22H3", "CARRY-L17H6": "L17H4,L11H9,L5H6,L16H2,L6H5",
    "TERM-L17H6": "L11H9,L5H6,L16H2,L23H6,L22H8,L23H3,L25H9,L19H6",
}
parse = lambda s: [tuple(int(x) for x in t[1:].split("H")) for t in s.split(",")] if s else []
NAMED = {h for v in GROUPS.values() for h in parse(v)}
nm = lambda S: ",".join(f"L{l}H{h}" for l, h in S)


def random_sets(heads, n, seed, H=16, exclude=NAMED):
    """n layer-matched random groups: each head replaced by a head of the same layer drawn uniformly from heads
    not named in any group (and not already in the draw)."""
    rng = random.Random(seed); out = []
    for _ in range(n):
        r = []
        for l, _h in heads:
            r.append((l, rng.choice([h for h in range(H) if (l, h) not in exclude and (l, h) not in r])))
        out.append(r)
    return out


# ---- material
def load_part(src, part):
    if part == "held":
        p = X1 / "PROTOCOL.md"
        if not p.exists() or (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
            raise SystemExit(f"refusing to read {src}/held: PROTOCOL.md is not frozen")
    if src == "ord":
        return json.loads((DATA / "ord" / f"part-{part}.json").read_text())
    return [json.loads(l) for l in open(DATA / src / f"part-{part}.jsonl")]


# ---- model
class Model:
    """D3's M (Qwen3 in bfloat16, SDPA) plus what D1-X's helpers expect (tok, nlpos)."""
    def __init__(self, name="qwen3-06b-bf16"):
        self.m = D3.M(name, attn="sdpa"); self.tok = self.m.tok; self.L, self.H = self.m.L, self.m.H
        self.nl = D1C.newline_id(self.tok)
        self.head = self.m.model.get_output_embeddings()
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    nlpos = XC.M.nlpos

    # ablation of `heads` with per-head mean vectors `means`; scope 'all' = every position, 'dec' = only the
    # task's decision positions passed to logits_at(..., dec=[(row, pos), ...])
    abl = None

    def set_ablation(self, heads, means, scope="all"):
        self.abl = (heads, means, scope) if heads else None

    def clear(self): self.abl = None; self.m.zpatch = {}

    def _apply(self, dec):
        self.m.zpatch = {}
        if not self.abl: return
        heads, means, scope = self.abl
        if scope == "dec":
            if not dec: return
            rows = torch.tensor([r for r, _ in dec]); cols = torch.tensor([p for _, p in dec])
        for l, h in heads:
            w = self.m.dhl[l]; v = means[l][h * w:(h + 1) * w]
            e = (h, slice(None), v) if scope == "all" else (h, cols, v.expand(len(cols), -1), rows)
            self.m.zpatch.setdefault(l, []).append(e)

    @torch.no_grad()
    def hidden(self, seqs, dec=None):
        self._apply(dec)
        T = max(len(s) for s in seqs)
        x = torch.tensor([s + [self.pad] * (T - len(s)) for s in seqs], device=DEV)
        return self.m.model.base_model(input_ids=x).last_hidden_state

    @torch.no_grad()
    def logits_at(self, seqs, positions, dec=None):
        """positions: list of (row, pos); dec: decision positions for scope 'dec'. Returns float32 logits (n, V)."""
        h = self.hidden(seqs, dec)
        rows = torch.tensor([r for r, _ in positions], device=DEV); cols = torch.tensor([p for _, p in positions], device=DEV)
        hh = h[rows, cols]
        return torch.cat([self.head(c).float() for c in hh.split(256)])

    @torch.no_grad()
    def nll_all(self, ids):
        """Next-token loss at positions 0..T-2 of one sequence (numpy)."""
        h = self.hidden([ids], [(0, p) for p in range(len(ids))])[0]; t = torch.tensor(ids[1:], device=DEV); out = []
        for k in range(0, len(ids) - 1, 256):
            lg = self.head(h[k:min(k + 256, len(ids) - 1)]).float()
            tt = t[k:k + len(lg)]
            out.append(torch.logsumexp(lg, -1) - lg.gather(1, tt[:, None])[:, 0])
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def capture_sum(self, ids, pos=None):
        """Sum over positions (all, or the list `pos`) of every layer's o-proj input for one sequence (no
        ablation); returns ({l: sum}, count)."""
        ab = self.abl; self.abl = None
        self.m.capture = True; self.hidden([ids]); self.m.capture = False; self.abl = ab
        sel = slice(None) if pos is None else torch.tensor(pos)
        return {l: self.m.zcap[l][sel].float().sum(0) for l in range(self.L)}, (len(ids) if pos is None else len(pos))


def task_means(model, seqs, pos=None):
    """Per-layer mean o-proj input over all positions of `seqs`, or over pos[i] in seqs[i]."""
    acc = None; n = 0
    for i, ids in enumerate(seqs):
        s, t = model.capture_sum(ids, None if pos is None else pos[i])
        acc = s if acc is None else {l: acc[l] + s[l] for l in acc}; n += t
    return {l: (acc[l] / n) for l in acc}


# =========================================================== tasks
class LB:
    """Line-breaking (D1-X). Width items (xm_common.width_items, width 60, seed 200) and natural paragraphs
    (xm_common.wrapped_items, width 60, >= 5 lines, <= 700 tokens)."""
    name = "LB"

    def __init__(self, model, paras, n_width=120, n_nat=50):
        self.M = model; tok = model.tok
        self.w = XC.width_items(paras, model, n_width, W=60, seed=200)
        self.nat = XC.wrapped_items(paras, 60, n_nat, tok=tok)

    def seqs_for_means(self):
        return [ids for ids, _, _ in self.nat]

    def dec_for_means(self):
        return [sorted(set(self.M.nlpos(ids)) | {r["i"] for r in recs}) for ids, recs, _ in self.nat]

    def run(self):
        M = self.M; nl = M.nl; E = []
        for i in range(0, len(self.w), 16):
            grp = self.w[i:i + 16]; seqs = []
            for ib, is_, _, _ in grp: seqs += [ib, is_]
            dec = [(r, p) for r, s in enumerate(seqs) for p in sorted(set(M.nlpos(s)) | {len(s) - 1})]
            lg = M.logits_at(seqs, [(r, len(s) - 1) for r, s in enumerate(seqs)], dec)
            lp = (lg[:, nl] - torch.logsumexp(lg, -1)).cpu().numpy()
            E += list(lp[1::2] - lp[0::2])
        brk, non, sc, lab, per = [], [], [], [], []
        for ids, recs, _ in self.nat:
            lg = M.logits_at([ids], [(0, r["i"]) for r in recs], [(0, p) for p in sorted(set(M.nlpos(ids)) | {r["i"] for r in recs})])
            lp = (lg[:, nl] - torch.logsumexp(lg, -1)).cpu().numpy()
            b_, n_ = [], []
            for r, v in zip(recs, lp):
                v = float(XC.logodds(v)); sc.append(v); lab.append(r["label"])
                if r["label"] == 1: b_.append(v)
                elif r["remaining"] <= 12: n_.append(v)
            per.append((b_, n_))
        sep = np.mean([v for b, _ in per for v in b]) - np.mean([v for _, n in per for v in n])
        return dict(width_effect=float(np.mean(E)), sep=float(sep), auroc=float(D1C.auroc(sc, lab)), n_width=len(E),
                    n_nat=len(per)), dict(E=np.array(E), nat=np.array([[sum(b), len(b), sum(n), len(n)] for b, n in per], float))


class IND:
    """Indentation (D3). Points with max(u) == 0 (as c01); classes header/same/term/dedent; renormalised
    candidate distribution q via RootMasses; per class mean log q(true) and mass on the current level."""
    name = "IND"

    def __init__(self, model, chunks_by_source):
        self.M = model; m = model.m; self.batches = []; self.cls = []; self.src = []
        for source, chunks in chunks_by_source.items():
            items = []
            for c in chunks:
                ids, pts = D3.prepare(m, c["text"])
                if not pts: continue
                pts = [p for p in pts if max(p["u"]) == 0]
                for p in pts: p["_c"] = f"{source}:{c.get('repo', c.get('seed', ''))}"
                if pts: items.append((ids, pts))
            for i in range(0, len(items), 8):
                grp = items[i:i + 8]
                pos = [(r, p["d"]) for r, (_, pts) in enumerate(grp) for p in pts]
                bp = [p for _, pts in grp for p in pts]
                self.batches.append(([ids for ids, _ in grp], pos, bp, D3.RootMasses(m, bp)))
                for p in bp:
                    self.cls.append("header" if p["header"] else "same" if p["cls"] == "same" else "term" if p["terminal"] else "dedent")
                    self.src.append(source)
        self.cls, self.src = np.array(self.cls), np.array(self.src)
        self.seq_list = [s for b in self.batches for s in b[0]]

    def seqs_for_means(self):
        return self.seq_list[::3]

    def dec_for_means(self):
        dp = [sorted({p for r, p in pos if r == j}) for seqs, pos, _, _ in self.batches for j in range(len(seqs))]
        return dp[::3]

    def run(self):
        lq, ps = [], []
        for seqs, pos, pts, rm in self.batches:
            lg = self.M.logits_at(seqs, pos, pos)
            lp = torch.log_softmax(lg, -1)
            for q, p in zip(rm(lp), pts):
                K = p["K"]; cur = p["stack"][-1]
                lq.append(float(np.log(q[K.index(p["true"])]))); ps.append(float(q[K.index(cur)]) if cur in K else 0.0)
        lq, ps = np.array(lq), np.array(ps); out = {}
        for s in sorted(set(self.src)):
            for c in ("header", "same", "term", "dedent"):
                k = (self.src == s) & (self.cls == c)
                out[f"{s}|{c}|lq"] = float(lq[k].mean()) if k.any() else None
                out[f"{s}|{c}|p_same"] = float(ps[k].mean()) if k.any() else None
                out[f"{s}|{c}|n"] = int(k.sum())
        return out, dict(lq=lq, ps=ps, cls=self.cls, src=self.src,
                         cluster=np.array([str(p.get("_c", "")) for b in self.batches for p in b[2]]))


class ORD:
    """Ordinal suffixes (D4). m = logit(last-digit suffix) - logit('th') at the number's last token.
    Teen effect = mean m (non-teen ending 1-3) - mean m (teen x11-x13); last-digit pull = mean m (non-teen
    ending 1-3); accuracy by class. Items: per digit count, up to `per` (number, frame) pairs per class."""
    name = "ORD"

    def __init__(self, model, part, digits=(4, 5), per=96, seed=0):
        self.M = model; tok = model.tok; self.cols = D4.suffix_ids(tok); rng = random.Random(f"X1-ord-items-{seed}")
        frames = part["frames"]; self.items = []
        for d in digits:
            for cls in ("teen", "end123"):
                nums = [x["n"] for x in part["numbers"] if x["digits"] == d and x["cls"] == cls]
                fr = lambda n: [f for f in frames if f["hi"] >= n] or [f for f in frames if f["family"] in ("rankings", "positions")]
                pairs = [(n, f["fid"]) for n in nums for f in fr(n)]
                rng.shuffle(pairs)
                fmap = {f["fid"]: f for f in frames}
                for n, fid in sorted(pairs[:per]):
                    self.items.append((d, cls, n, D4.encode(tok, fmap[fid]["text"].replace("{N}", str(n)))))
        self.digits = digits

    def seqs_for_means(self):
        return [ids for *_, ids in self.items[::2]]

    def dec_for_means(self):
        return [[len(ids) - 1] for *_, ids in self.items[::2]]

    def run(self):
        M = self.M; lgs = []
        for i in range(0, len(self.items), 64):
            seqs = [it[3] for it in self.items[i:i + 64]]
            pl = [(r, len(s) - 1) for r, s in enumerate(seqs)]
            lgs.append(M.logits_at(seqs, pl, pl)[:, self.cols].cpu().numpy())
        lg = np.concatenate(lgs); out = {}; mm = []
        for (d, cls, n, _), l in zip(self.items, lg): mm.append(l[n % 10 - 1] - l[3])
        mm = np.array(mm)
        ok = np.array([D4.SUFFIXES[int(np.argmax(l))] == D4.suffix(n) for (d, cls, n, _), l in zip(self.items, lg)])
        D = np.array([it[0] for it in self.items]); C = np.array([it[1] for it in self.items])
        for d in self.digits:
            t, e = (D == d) & (C == "teen"), (D == d) & (C == "end123")
            out[f"d{d}|teen_effect"] = float(mm[e].mean() - mm[t].mean()); out[f"d{d}|pull"] = float(mm[e].mean())
            out[f"d{d}|acc_teen"] = float(ok[t].mean()); out[f"d{d}|acc_end123"] = float(ok[e].mean())
            out[f"d{d}|auroc"] = float(D1C.auroc(np.r_[mm[e], mm[t]], np.r_[np.ones(e.sum()), np.zeros(t.sum())]))
            out[f"d{d}|n_teen"] = int(t.sum()); out[f"d{d}|n_end123"] = int(e.sum())
        return out, dict(m=mm, ok=ok.astype(float), digits=D, cls=C, n=np.array([it[2] for it in self.items]))


class CAPS:
    """Capitals (D5 definitions) and GEN. Paragraphs cut at 1,200 characters (D5's cut). At word starts present
    in both the upper- and the lower-case version (D5 positions: word index >= 5): U = upper-case first piece,
    Lc = lower-case first piece; D_up = log p(U) - log p(Lc) predicting the word in the upper passage, D_low =
    log p(Lc) - log p(U) in the lower passage; word_lp = log p(Lc) in the lower passage; case_kept = share of
    upper-passage word starts whose top-1 token starts with an upper-case letter. GEN: mean next-token loss
    over the original paragraphs."""
    name = "CAPS"

    def __init__(self, model, rows):
        self.M = model; tok = model.tok; self.items = []; self.orig = []
        self.vs = D5.vocab_strings(tok)
        for r in rows:
            text = D5.cut(r["text"]); V = D5.versions(text)
            eu, el = D5.encode(tok, V["upper"]), D5.encode(tok, V["lower"])
            wsu, _ = D5.positions(V["upper"], eu[1]); wsl, _ = D5.positions(V["lower"], el[1])
            inv_u = {i: k for k, i in wsu.items()}; inv_l = {i: k for k, i in wsl.items()}
            both = sorted(set(inv_u) & set(inv_l))
            pk = [(inv_u[i], inv_l[i], eu[0][inv_u[i]], el[0][inv_l[i]]) for i in both if eu[0][inv_u[i]] != el[0][inv_l[i]]]
            if pk: self.items.append((eu[0], el[0], pk))
            self.orig.append(tok(text, add_special_tokens=False)["input_ids"])

    def seqs_for_means(self):
        return [x for u, l, _ in self.items for x in (u, l)][::2] + self.orig[::2]

    def dec_for_means(self):
        d = [p for u, l, pk in self.items for p in ([ku - 1 for ku, _, _, _ in pk], [kl - 1 for _, kl, _, _ in pk])][::2]
        return d + [list(range(len(ids))) for ids in self.orig[::2]]

    def run(self):
        M = self.M; du, dl, wl, ck, par = [], [], [], [], []
        for pi, (u, l, pk) in enumerate(self.items):
            pl = [(0, ku - 1) for ku, _, _, _ in pk] + [(1, kl - 1) for _, kl, _, _ in pk]
            lg = M.logits_at([u, l], pl, pl)
            lp = torch.log_softmax(lg, -1); n = len(pk)
            U = torch.tensor([x[2] for x in pk], device=DEV); Lc = torch.tensor([x[3] for x in pk], device=DEV)
            a = torch.arange(n, device=DEV)
            du += (lp[a, U] - lp[a, Lc]).cpu().tolist(); dl += (lp[n + a, Lc] - lp[n + a, U]).cpu().tolist()
            wl += lp[n + a, Lc].cpu().tolist(); par += [pi] * n
            top = lp[:n].argmax(-1).cpu().tolist()
            ck += [bool(D5.first_cased_upper(self.vs[t])) for t in top]
        gen = np.concatenate([M.nll_all(ids) for ids in self.orig]) if (M.abl is None or M.abl[2] == "all") else np.array([np.nan])
        out = dict(D_up=float(np.mean(du)), D_low=float(np.mean(dl)), word_lp=float(np.mean(wl)), case_kept=float(np.mean(ck)),
                   n_pos=len(du), n_par=len(self.items), gen_loss=float(gen.mean()), gen_tokens=int(len(gen)))
        return out, dict(du=np.array(du), dl=np.array(dl), wl=np.array(wl), par=np.array(par))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1)); return path
