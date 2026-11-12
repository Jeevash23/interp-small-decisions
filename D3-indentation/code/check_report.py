"""D3: independent check of every number in REPORT.md.
- Confirmation numbers are recomputed from the per-point arrays (outputs/conf/c01_*_points_*.npy and the meta
  arrays), not from the JSON summaries the report was built from; random-group extremes from the raw run JSON.
- Gate numbers are recomputed from the raw e01 per-point outputs (post-fix: outputs/e01_*.jsonl; pre-fix:
  outputs/v1-before-2230-fix/), not from the gate summaries.
- Counts of chunks, repositories and pairs from the sealed data and the raw outputs; times from systemd and logs.
- The "under a tenth" statement from the e09 outputs.
Each expected string must occur in REPORT.md. Prints the number of checks and failures; exit 1 on any failure.
Usage: python check_report.py"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

D3 = Path(__file__).resolve().parent.parent; O = D3 / "outputs"; C = O / "conf"
R = (D3 / "REPORT.md").read_text(); checks = []; fails = []
f3 = lambda x: f"{x:.3f}"


def expect(s, what):
    checks.append(what)
    if s not in R: fails.append(f"{what}: '{s}' not found")


def pts(src, grp):
    meta = np.load(C / f"c01_{src}_meta.npy", allow_pickle=True); a = np.load(C / f"c01_{src}_points_{grp}.npy")
    return meta[0], meta[1], a


def mean(src, grp, row, c):
    cls, _, a = pts(src, grp); return float(a[row][cls == c].mean())


def dl(src, grp, row, c): return mean(src, grp, row, c) - mean(src, "base", row, c)


# confirmation, recomputed from per-point arrays
H1 = dl("newcode", "DED", 0, "dedent"); expect(f"| {f3(H1)} |", "H1 value")
expect(f"falls by {f3(-H1)} nats", "H1 prose")
H2 = dl("newcode", "DED", 1, "dedent"); expect(f"+{f3(H2)}", "H2 value")
expect(f"rises from {f3(mean('newcode','base',1,'dedent'))} to {f3(mean('newcode','DED',1,'dedent'))}", "H2 prose")
po = mean("newcode", "DED", 2, "term"); H3t = dl("newcode", "DED", 0, "term")
expect(f"{f3(po)}; change {f3(H3t)} (ratio {f3(abs(H3t)/abs(H1))})", "H3 value")
expect(f"{f3(po)} of the mass stays", "H3 prose")
H4 = dl("newcode", "DED", 0, "same"); expect(f"+{f3(H4)}", "H4 value")
H5 = dl("newcode", "TERM", 0, "term"); expect(f"| {f3(H5)} |", "H5 value"); expect(f"costs terminal-statement dedents {f3(-H5)}", "H5 prose")
H6 = dl("newcode", "TERM", 0, "dedent"); expect(f"| {f3(H6)} |", "H6 value"); expect(f"nearly unchanged ({f3(H6)})", "H6 prose")
H7 = dl("newcode", "TERMDED", 0, "term"); expect(f"| {f3(H7)} | sum of singles {f3(H5 + H3t)} |", "H7 value")
expect(f"costs terminal-statement dedents {f3(-H7)}, more than the {f3(-(H5 + H3t))}", "H7 prose")
H8 = dl("gen", "DED", 0, "dedent"); expect(f"| {f3(H8)} |", "H8 value")
for src in ["newcode", "gen"]:
    for c in ["header", "same", "term", "dedent"]:
        expect(f3(mean(src, "base", 0, c)), f"base lq {src} {c}")
    expect(f3(mean(src, "base", 1, "dedent")), f"base p_same {src}")
# random extremes and verdict thresholds, from raw run JSON
n = json.loads((C / "c01_newcode.json").read_text()); g = json.loads((C / "c01_gen.json").read_text())
rr = lambda r, grp, meas, c: np.array([v[meas][c] for k, v in r.items() if k.startswith(grp + "_rand")]) - r["base"][meas][c]
assert len(rr(n, "DED", "lq", "dedent")) == 20 and len(rr(n, "TERM", "lq", "term")) == 20 and len(rr(g, "DED", "lq", "dedent")) == 20
expect(f"lowest {f3(rr(n,'DED','lq','dedent').min())}", "H1 random"); expect(f"highest +{f3(rr(n,'DED','p_same','dedent').max())}", "H2 random")
expect(f"lowest {f3(rr(n,'TERM','lq','term').min())}", "H5 random"); expect(f"lowest {f3(rr(g,'DED','lq','dedent').min())}", "H8 random")
ok = [H1 <= -0.40 and H1 < rr(n, "DED", "lq", "dedent").min(), H2 >= 0.10 and H2 > rr(n, "DED", "p_same", "dedent").max(),
      po >= 0.95 and abs(H3t) <= 0.25 * abs(H1), H4 >= -0.01, H5 <= -0.03 and H5 < rr(n, "TERM", "lq", "term").min(), abs(H6) <= 0.10,
      H7 < H5 + H3t, H8 <= -0.25 and H8 < rr(g, "DED", "lq", "dedent").min()]
p = json.loads((C / "c02_gen.json").read_text()); cm = p["comps"]
rn = [v["nec"] for k, v in cm.items() if k.startswith("CARRY_rand")]; rs = [v["suf"] for k, v in cm.items() if k.startswith("CARRY_rand")]
assert len(rn) == 20
ok.append(cm["CARRY"]["nec"] >= 0.40 and cm["CARRY"]["suf"] >= 0.20 and cm["CARRY"]["nec"] > max(rn) and cm["CARRY"]["suf"] > max(rs))
expect(f"necessity {f3(cm['CARRY']['nec'])}, sufficiency {f3(cm['CARRY']['suf'])}", "H9 value"); expect(f"highest {f3(max(rn))} and {f3(max(rs))}", "H9 random")
checks.append("all nine verdicts pass on recomputation")
if not all(ok): fails.append(f"verdict recomputation: {ok}")
expect("All nine frozen predictions pass", "verdict sentence")
# counts
cls_n, cl_n, _ = pts("newcode", "base"); cls_g, cl_g, _ = pts("gen", "base")
for c in ["header", "same", "term", "dedent"]:
    expect(str(int((cls_n == c).sum())), f"count newcode {c}"); expect(str(int((cls_g == c).sum())), f"count gen {c}")
conf_rows = [json.loads(l) for l in open(D3 / "data" / "newcode" / "part-conf.jsonl")]
expect(f"{len(conf_rows)} chunks of Python from {len({r['repo'] for r in conf_rows})} GitHub repositories", "natural chunks and repositories")
expect(f"{sum(1 for _ in open(D3 / 'data' / 'gen' / 'part-conf.jsonl'))} programs", "generated programs")
expect(f"{p['n']} return/assignment pairs", "pairs"); expect(f"gap {p['gap']:.2f} nats", "pair gap")
# times
jl = subprocess.run(["journalctl", "--user", "-u", "d3-conf", "--no-pager", "-o", "short-iso"], capture_output=True, text=True).stdout
st = [l for l in jl.split("\n") if "Started" in l and "queue_conf.sh" in l]
if st: expect(f"between {st[0].split('T')[1][:5]}", "confirmation start time (unit start, journal)")
else: fails.append("confirmation start time not found in the journal")
endlog = (O / "c02_pairs_timelimit_1000.log").read_text().strip().split("\n")[-1]   # "exit 0 HH:MM:SS"
hh, mm, ss = (int(x) for x in endlog.split()[-1].split(":")); mm += ss >= 30
expect(f"and {hh:02d}:{mm:02d} IST", "confirmation end time")


# gate, recomputed from raw per-point outputs
def gstats(path, gen_dir=None):
    xs = [json.loads(l) for l in open(path) if '"mass"' in l]
    a = lambda k, f: float(np.mean([x[k] for x in xs if f(x)]))
    s = dict(n=len(xs), h=a("top1", lambda x: x["header"]), tt=a("top1", lambda x: x["terminal"]), ts=a("p_shallow_open", lambda x: x["terminal"]),
             d=a("top1", lambda x: True) - a("copy_ok", lambda x: True))
    return s


def openlvl(path):
    xs = [json.loads(l) for l in open(path) if '"mass"' in l]
    return max(float(np.mean([x["p_shallow_notopen"] for x in xs if str(x["width"]) == w and x["cls"] == "shallower"])) for w in ["2", "3", "mixed"])


for mname, lab in [("pythia-410m", "Pythia-410M"), ("qwen3-06b", "Qwen3-0.6B-Base"), ("gemma4-e2b", "Gemma-4-E2B"), ("gpt2", "GPT-2")]:
    for tag, base in [("pre-fix", O / "v1-before-2230-fix"), ("post-fix", O)]:
        pth = base / f"e01_{mname}_newcode.jsonl"
        if not pth.exists(): continue
        s = gstats(pth); gp = base / f"e01_{mname}_gen.jsonl"
        ol = f3(openlvl(gp)) if gp.exists() else "–"
        expect(f"| {lab} | {tag} | {s['n']} | {f3(s['h'])} | {f3(s['tt'])} | {f3(s['ts'])} | {f3(s['d'])} | {ol} |", f"gate row {lab} {tag}")
# e09: keyword-prefix effect under a tenth of a real statement's, in all three models
mx = 0
for f in ["e09_qwen3-06b-bf16_250_READ.json", "e09_pythia-410m_250.json", "e09_gemma4-e2b_250.json"]:
    s = json.loads((O / f).read_text())["summary"]
    for w in ["return", "pass", "break", "continue", "raise"]:
        mx = max(mx, s[f"diff:{w}"]["mean"] / s[f"real:{w}"]["mean"])
checks.append("e09 under a tenth")
if not mx < 0.1: fails.append(f"e09 largest ratio {mx:.3f} not under 0.1")
# the report is exactly what build_report.py produces
before = R
subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "build_report.py")], check=True, capture_output=True)
checks.append("report reproducible")
if (D3 / "REPORT.md").read_text() != before: fails.append("REPORT.md differs from a fresh build")
print(f"{len(checks)} checks, {len(fails)} failed (largest e09 ratio {mx:.3f})")
for f in fails: print("FAIL", f)
sys.exit(1 if fails else 0)
