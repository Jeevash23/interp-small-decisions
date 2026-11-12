"""D3 e09 (exploratory): false-dedent test. If the dedent signal is read from the previous line's FIRST TOKEN,
then a line whose first token is a terminal keyword used as an identifier prefix (`return_value = ...`,
`pass_count = ...`) should raise the dedent mass, while a keyword inside a string (`x = "return early"`)
should not. Items: same-level points in generated exploration programs where the previous line is a simple
assignment `<name> = <rhs>`, the stack has >= 2 levels and no blank line intervenes. Variants of the
previous line (indent and right-hand side kept):
  kw:<w>   <w>_<name> = <rhs>          w in return, pass, break, continue, raise
  ctl:<c>  <c>_<name> = <rhs>          c in result, past, brick, contain, raised (paired with w by index)
  str:kw   <name> = "<w> <name>"       str:ctl  <name> = "<c> <name>"
  real     return <name> / pass / break / continue / raise ValueError(<name>)   (true terminal statements)
Measure: renormalised mass on shallower open levels at the next line (p_sh). Optional mediation: patch the
outputs of a head group at the decision position from the ctl:<c> run into the kw:<w> run.
Usage: python e09_falsekw.py <model> <n_items> [GROUP=L5H6,...]"""
import sys, json, re
from common import *

name, nitems = sys.argv[1], int(sys.argv[2])
group = None
for a in sys.argv[3:]:
    if "=" in a: gname, hs = a.split("="); group = [tuple(int(x) for x in h[1:].split("H")) for h in hs.split(",")]
m = M(name, attn="sdpa")
KW = ["return", "pass", "break", "continue", "raise"]; CTL = ["result", "past", "brick", "contain", "raised"]
progs = load_chunks("gen", "explore"); random.Random(9).shuffle(progs)
items = []
for pr in progs:
    text = pr["text"]; lines = text.split("\n")
    for p in P.decision_points(text):
        if p["cls"] != "same" or p["blank"] or len(p["stack"]) < 2: continue
        L = p["line"]; prev = lines[L - 2]
        mt = re.match(r"^(\s*)([a-z]+) = (.+)$", prev)
        if not mt or mt.group(2) in KW + CTL: continue
        items.append((text, L, mt.group(1), mt.group(2), mt.group(3))); break
    if len(items) >= nitems: break


def variant_text(text, L, line):
    ls = text.split("\n"); ls[L - 2] = line; return "\n".join(ls)


def p_sh(text, L, patch=None):
    ids, pts = prepare(m, text)
    if not pts: return None
    q = [p for p in pts if p["line"] == L]
    if not q or max(q[0]["u"]): return None
    p = q[0]; rm = RootMasses(m, [p])
    m.capture = patch is None
    if patch is not None:
        m.zpatch = {}
        for (l, h), val in patch.items():
            w = m.dhl[l]; m.zpatch.setdefault(l, []).append((h, p["d"], val))
    lp = m.logprobs(ids, positions=[(0, p["d"])]); m.zpatch = {}; m.capture = False
    qv = rm(lp)[0]; st = p["stack"]
    return float(sum(qv[i] for i, k in enumerate(p["K"]) if k in st[:-1])), p["d"]


rows = []
for text, L, ind, nm, rhs in items:
    r = {}
    for i, (w, c) in enumerate(zip(KW, CTL)):
        a = p_sh(variant_text(text, L, f"{ind}{w}_{nm} = {rhs}"), L)
        b = p_sh(variant_text(text, L, f"{ind}{c}_{nm} = {rhs}"), L)
        if a is None or b is None: continue
        r[f"kw:{w}"], r[f"ctl:{w}"] = a[0], b[0]
        if group is not None:
            capb = {(l, h): m.zcap[l][b[1]][h * m.dhl[l]:(h + 1) * m.dhl[l]].clone() for l, h in group}
            # capb is from the ctl run (last captured); patch into kw run
            pa = p_sh(variant_text(text, L, f"{ind}{w}_{nm} = {rhs}"), L, patch=capb)
            if pa is not None: r[f"kw_patched:{w}"] = pa[0]
        sa = p_sh(variant_text(text, L, f'{ind}{nm} = "{w} {nm}"'), L); sb = p_sh(variant_text(text, L, f'{ind}{nm} = "{c} {nm}"'), L)
        if sa and sb: r[f"str:kw:{w}"], r[f"str:ctl:{w}"] = sa[0], sb[0]
        real = {"return": f"{ind}return {nm}", "pass": f"{ind}pass", "break": f"{ind}break", "continue": f"{ind}continue",
                "raise": f"{ind}raise ValueError({nm})"}[w]
        rr = p_sh(variant_text(text, L, real), L)
        if rr: r[f"real:{w}"] = rr[0]
    base = p_sh(text, L)
    if base: r["base"] = base[0]
    rows.append(r)
summ = {}
for k in sorted({k for r in rows for k in r}):
    v = np.array([r[k] for r in rows if k in r]); summ[k] = dict(n=len(v), mean=float(v.mean()))
for w in KW:
    d = np.array([r[f"kw:{w}"] - r[f"ctl:{w}"] for r in rows if f"kw:{w}" in r])
    ds = np.array([r[f"str:kw:{w}"] - r[f"str:ctl:{w}"] for r in rows if f"str:kw:{w}" in r])
    summ[f"diff:{w}"] = dict(n=len(d), mean=float(d.mean()), frac_pos=float((d > 0).mean()))
    summ[f"strdiff:{w}"] = dict(n=len(ds), mean=float(ds.mean()), frac_pos=float((ds > 0).mean()))
    if group is not None:
        dp = np.array([r[f"kw_patched:{w}"] - r[f"ctl:{w}"] for r in rows if f"kw_patched:{w}" in r])
        if len(dp): summ[f"patched_diff:{w}"] = dict(n=len(dp), mean=float(dp.mean()))
tag = f"_{gname}" if group is not None else ""
(OUT / f"e09_{name}_{nitems}{tag}.json").write_text(json.dumps(dict(summary=summ, rows=rows)))
for k, v in summ.items(): print(k, json.dumps(v))
