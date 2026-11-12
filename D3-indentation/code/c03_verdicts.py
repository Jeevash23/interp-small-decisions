"""D3 confirmation c03 (PROTOCOL.md): verdicts from the saved confirmation outputs, with the frozen thresholds.
Reads outputs/conf/c01_newcode.json, c01_gen.json, c02_gen.json (or outputs/conf_smoke/ with --smoke), and the
per-point arrays for repository-clustered bootstrap intervals (descriptive only; verdicts use the frozen rules).
Writes outputs/conf/verdicts.json and prints a Markdown table.
Usage: python c03_verdicts.py [--smoke]"""
import sys, json
import numpy as np
from pathlib import Path

smoke = "--smoke" in sys.argv
OD = Path(__file__).resolve().parent.parent / "outputs" / ("conf_smoke" if smoke else "conf")


def load(n): return json.loads((OD / n).read_text())


def rand(res, g, meas, c):
    return np.array([v[meas][c] for k, v in res.items() if k.startswith(g + "_rand")])


def delta(res, g, meas, c): return res[g][meas][c] - res["base"][meas][c]


def boot(source, g, row, c, B=2000, seed=0):
    """95% interval of the change (group minus base) in a per-point measure, resampling clusters (repositories,
    or program seeds for generated code)."""
    meta = np.load(OD / f"c01_{source}_meta.npy", allow_pickle=True); cls, cl = meta[0], meta[1]
    a = np.load(OD / f"c01_{source}_points_base.npy")[row]; b = np.load(OD / f"c01_{source}_points_{g}.npy")[row]
    sel = cls == c; d = (b - a)[sel]; cs = cl[sel]; u = np.unique(cs)
    idx = {k: np.where(cs == k)[0] for k in u}; rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        pick = rng.choice(u, len(u)); ii = np.concatenate([idx[k] for k in pick]); out.append(d[ii].mean())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


V = []
n = load("c01_newcode.json")
dd = delta(n, "DED", "lq", "dedent"); rd = rand(n, "DED", "lq", "dedent") - n["base"]["lq"]["dedent"]
V.append(("H1", "DED, natural, other dedents: change in mean log q(true) <= -0.40 and below every random group", dd,
          f"random min {rd.min():.3f}", dd <= -0.40 and dd < rd.min(), boot("newcode", "DED", 0, "dedent")))
ds = delta(n, "DED", "p_same", "dedent"); rs = rand(n, "DED", "p_same", "dedent") - n["base"]["p_same"]["dedent"]
V.append(("H2", "DED, natural, other dedents: change in mass on the current level >= +0.10 and above every random group", ds,
          f"random max {rs.max():.3f}", ds >= 0.10 and ds > rs.max(), boot("newcode", "DED", 1, "dedent")))
po = n["DED"]["p_open"]["term"]; dt = delta(n, "DED", "lq", "term")
V.append(("H3", "DED, natural, terminal dedents: shallower-open mass >= 0.95 under ablation and |change in log q| <= 0.25 x |H1 change|",
          po, f"terminal change {dt:.3f}; ratio {abs(dt) / abs(dd):.3f}", po >= 0.95 and abs(dt) <= 0.25 * abs(dd), boot("newcode", "DED", 0, "term")))
dsm = delta(n, "DED", "lq", "same")
V.append(("H4", "DED, natural, same-level lines: change in mean log q(true) >= -0.01", dsm, "", dsm >= -0.01, boot("newcode", "DED", 0, "same")))
tt = delta(n, "TERM", "lq", "term"); rt = rand(n, "TERM", "lq", "term") - n["base"]["lq"]["term"]
V.append(("H5", "TERM, natural, terminal dedents: change in mean log q(true) <= -0.03 and below every random group", tt,
          f"random min {rt.min():.3f}", tt <= -0.03 and tt < rt.min(), boot("newcode", "TERM", 0, "term")))
td = delta(n, "TERM", "lq", "dedent")
V.append(("H6", "TERM, natural, other dedents: |change in mean log q(true)| <= 0.10", td, "", abs(td) <= 0.10, boot("newcode", "TERM", 0, "dedent")))
tj = delta(n, "TERMDED", "lq", "term")
V.append(("H7", "TERM+DED, natural, terminal dedents: change in mean log q(true) more negative than the sum of the two single changes", tj,
          f"sum of singles {tt + dt:.3f}", tj < tt + dt, boot("newcode", "TERMDED", 0, "term")))
g = load("c01_gen.json")
gd = delta(g, "DED", "lq", "dedent"); grd = rand(g, "DED", "lq", "dedent") - g["base"]["lq"]["dedent"]
V.append(("H8", "DED, generated, other dedents: change in mean log q(true) <= -0.25 and below every random group", gd,
          f"random min {grd.min():.3f}", gd <= -0.25 and gd < grd.min(), boot("gen", "DED", 0, "dedent")))
p = load("c02_gen.json")["comps"]
rn = np.array([v["nec"] for k, v in p.items() if k.startswith("CARRY_rand")]); rsu = np.array([v["suf"] for k, v in p.items() if k.startswith("CARRY_rand")])
V.append(("H9", "CARRY, generated pairs: necessity >= 0.40 and sufficiency >= 0.20, each above every random group",
          p["CARRY"]["nec"], f"sufficiency {p['CARRY']['suf']:.3f}; random max nec {rn.max():.3f}, suf {rsu.max():.3f}",
          p["CARRY"]["nec"] >= 0.40 and p["CARRY"]["suf"] >= 0.20 and p["CARRY"]["nec"] > rn.max() and p["CARRY"]["suf"] > rsu.max(), None))
out = [dict(id=i, claim=c, value=float(v), context=x, verdict="pass" if ok else "fail", ci95=ci) for i, c, v, x, ok, ci in V]
desc = dict(n_points_newcode=n["n_points"], n_chunks_newcode=n["n_chunks_used"], n_points_gen=g["n_points"], n_chunks_gen=g["n_chunks_used"],
            base_newcode=n["base"], base_gen=g["base"], pairs=load("c02_gen.json")["n"], pairs_gap=load("c02_gen.json")["gap"],
            DED_pairs=dict(nec=p["DED"]["nec"], suf=p["DED"]["suf"]))
(OD / "verdicts.json").write_text(json.dumps(dict(verdicts=out, descriptive=desc), indent=1))
print("| part | claim | value | context | 95% interval (clustered bootstrap, descriptive) | verdict |\n|---|---|---|---|---|---|")
for o in out:
    ci = "-" if o["ci95"] is None else f"[{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]"
    print(f"| {o['id']} | {o['claim']} | {o['value']:.3f} | {o['context']} | {ci} | {o['verdict']} |")
print(json.dumps(desc))
