"""Independent check of every number, claim, text rule and reference in the D5 write-up, draft v2 (rounds 1 and 2).
Standard library only, plus torch to read the round-2 record files (.pt).

  python3 -B scripts/check_numbers.py

Shares no code with make_numbers.py, with the study's code/, with conf_verdicts.json or conf_verdicts2.json. It recomputes every held-out
and exploration measure from the per-position records of the run files (lens values, top-1 tokens after each edit,
recased passages, head patches, changed-word ranks, gate decodes), with its own case and same-word rules written
from the frozen charter and protocol. It then compares them with numbers.tex and the figure data, recomputes the
bootstrap intervals with its own resampling (different seed; intervals must agree to within 0.03), re-derives every
pass or fail from thresholds that it checks against PROTOCOL.md and the table, recomputes the exploration-to-held-out
differences, checks the claims the prose makes in words, and checks the text (every \\nv key defined, no hand-typed
decimals, British spelling, no product names, no repo paths in captions, labels) and the references.
Writes checks/NUMBER-CHECK.md and checks/number-check.json; exits 1 if anything fails.
"""
import json, random, re, statistics, sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
D5 = PAPER.parents[1]
OUT = D5 / "outputs"
RES = []
MODELS = {"g": dict(m="gemma", l=24, L=35, late=30), "q": dict(m="qwen3-17b", l=22, L=28, late=23)}


def check(name, ok, detail=""):
    RES.append(dict(name=name, ok=bool(ok), detail=str(detail)))


def readj(p):
    with open(p) as f:
        return json.load(f)


NV = {}
for line in (PAPER / "numbers.tex").read_text().splitlines():
    mm = re.match(r"\\expandafter\\def\\csname nv@(.+?)\\endcsname\{(.*)\}$", line)
    if mm: NV[mm.group(1)] = mm.group(2)


def num(s):
    return float(s.replace("\u2212", "-").replace(",", ""))


def same(key, x, dec=3, label="recount"):
    if key not in NV:
        check(f"{label}: {key}", False, "missing from numbers.tex"); return
    got = num(NV[key])
    ok = abs(got - round(x, dec)) < 10 ** -dec / 2 + 1e-9 or abs(got - x) <= 10 ** -dec / 2 + 1e-9
    check(f"{label}: {key}", ok, f"text {NV[key]} vs recomputed {x:.6f}")


# ------------------------------------------------------------------ own rules, from the charter's definitions
def tcase(s):
    """Token case: upper if it has a cased letter and no lower-case letter, lower if the reverse; else mixed or caseless."""
    up = any(c.isupper() for c in s); lo = any(c.islower() for c in s)
    if up and lo: return "mixed"
    return "upper" if up else "lower" if lo else "caseless"


def sameword(a, b):
    a = a.strip().casefold(); b = b.strip().casefold()
    if not a or not b or not a.isalpha() or not b.isalpha(): return False
    return a[:len(b)] == b or b[:len(a)] == a


def share(pairs):
    return sum(x for _, x in pairs) / len(pairs)


def boot(pairs, stat, seed, B=1000):
    """pairs: (paragraph, item). Resample paragraphs with replacement; returns (lo, hi) percentiles."""
    by = {}
    for p, x in pairs: by.setdefault(p, []).append(x)
    ps = sorted(by); rng = random.Random(seed); vals = []
    for _ in range(B):
        xs = []
        for _ in ps: xs.extend(by[ps[rng.randrange(len(ps))]])
        try: vals.append(stat(xs))
        except ZeroDivisionError: pass
    vals.sort()
    q = lambda f: vals[min(len(vals) - 1, max(0, int(round(f * (len(vals) - 1)))))]
    return q(0.025), q(0.975)


MEAS = {"c": {}, "e": {}}      # stage -> model key -> measure -> (point, pairs, stat)


def put(stage, k, key, pairs, stat=None):
    stat = stat or (lambda xs: sum(xs) / len(xs))
    MEAS[stage].setdefault(k, {})[key] = (stat([x for _, x in pairs]), pairs, stat)


LAYERS = {}   # per-layer lens shares (held out), for Figure 2
for stage, pre in (("c", "conf_"), ("e", "")):
    for k, M in MODELS.items():
        m, l, L = M["m"], M["l"], M["L"]
        # lens (exp01): per position, per layer [U, Lc, Lr, Ur] log-probabilities
        x = readj(OUT / f"{pre}exp01_{m}.json")
        R = x["records"]

        def lens(run, lay, a, b):
            keep = [r for r in R if not (0 in (a, b) and r[run]["lens"][lay][0] == r[run]["lens"][lay][1])]
            return [(r["para"], 1.0 if r[run]["lens"][lay][a] > r[run]["lens"][lay][b] else 0.0) for r in keep]
        put(stage, k, "lens_Lc_gt_U_at_l", lens("upper", l, 1, 0))
        put(stage, k, "lens_Lr_gt_U_at_l", lens("upper", l, 2, 0))
        put(stage, k, "lens_Lc_gt_U_last", lens("upper", L - 1, 1, 0))
        put(stage, k, "lens_Lc_gt_Lr_upper_at_l", lens("upper", l, 1, 2))
        put(stage, k, "lens_Lc_gt_Lr_lower_at_l", lens("lower", l, 1, 2))
        prof = [share(lens("upper", i, 1, 0)) for i in range(L)]
        firstlow = min(i for i in range(L) if all(p < 0.5 for p in prof[i:]))
        same(f"{stage}.{k}.firstlow" if stage == "c" else f"e.{k}.firstlow", firstlow, 0)
        check(f"claim: {k} {stage} capitals win only in the last three layers", L - firstlow == 3, (firstlow, L))
        if stage == "c":
            LAYERS[k] = {"LcU": prof, "LrU": [share(lens("upper", i, 2, 0)) for i in range(L)],
                         "LcLr": [share(lens("upper", i, 1, 2)) for i in range(L)]}
            for grp, sel in (("seen", [r for r in R if r["seen"]]), ("new", [r for r in R if not r["seen"]])):
                for v in ("upper", "lower"):
                    same(f"c.{k}.{grp}.{v}", sum(1 for r in sel if r[v]["top1"]) / len(sel))
                same(f"c.{k}.{grp}.n", len(sel), 0)
        del x, R

        # direction edits (exp02): per layer, [top0, add/sub d a=1, add/sub r a=1, ...]
        x = readj(OUT / f"{pre}exp02_{m}.json")
        for lay in (l, 16):
            for v, sgn, want in (("lower", "add", "upper"), ("upper", "sub", "lower")):
                el = [r for r in x["records"] if tcase(r[v][str(lay)][0]) == v]
                for j, kind in ((1, "dir"), (2, "rand")):
                    put(stage, k, f"{sgn}_{kind}_L{lay}_flip", [(r["para"], float(tcase(r[v][str(lay)][j]) == want)) for r in el])
                    put(stage, k, f"{sgn}_{kind}_L{lay}_same_word",
                        [(r["para"], float(tcase(r[v][str(lay)][j]) == want and sameword(r[v][str(lay)][0], r[v][str(lay)][j]))) for r in el])
        if stage == "e":   # Figure 1: the whole exploration sweep
            dat = (PAPER / f"fig-sweep-{k}.dat").read_text().split("\n")[1:]
            dat = {int(s.split()[0]): [float(t) for t in s.split()[2:]] for s in dat if s.strip()}
            ok = True
            for lay in x["layers"]:
                vals = []
                for v, want, j in (("lower", "upper", 1), ("upper", "lower", 1), ("lower", "upper", 2), ("upper", "lower", 2)):
                    el = [r for r in x["records"] if tcase(r[v][str(lay)][0]) == v]
                    vals.append(sum(tcase(r[v][str(lay)][j]) == want for r in el) / len(el))
                ok &= all(abs(a - b) < 6e-5 for a, b in zip(vals, dat[lay]))
                same(f"sweep.{k}.add.{lay}", vals[0]); same(f"sweep.{k}.sub.{lay}", vals[1])
            check(f"figure 1 data {k}: every layer recomputed from records", ok and set(dat) == set(x["layers"]))
            same(f"sweep.{k}.npara", x["rows"][1] - x["rows"][0], 0)
        del x

        # removal and prose (exp07)
        x = readj(OUT / f"{pre}exp07_{m}.json")
        el = [r for r in x["records"] if tcase(r["upper"]["top0"]) == "upper"]
        for kind in ("dir", "rand"):
            put(stage, k, f"rm_{kind}_upper_share", [(r["para"], float(tcase(r["upper"][f"rm_{kind}_{l}"]) == "upper")) for r in el])
            put(stage, k, f"rm_{kind}_same_word_any_case", [(r["para"], float(sameword(r["upper"]["top0"], r["upper"][f"rm_{kind}_{l}"]))) for r in el])
        eo = [r for r in x["records"] if tcase(r["orig"]["top0"]) == "lower"]
        for kind in ("dir", "rand"):
            nw = [(r, r["orig"][f"add_{kind}_a1_{l}"]) for r in eo]
            put(stage, k, f"orig_add_{kind}_to_upper", [(r["para"], float(tcase(t) == "upper")) for r, t in nw])
            put(stage, k, f"orig_add_{kind}_to_upper_same_word", [(r["para"], float(tcase(t) == "upper" and sameword(r["orig"]["top0"], t))) for r, t in nw])
        del x

        # recasing the last words (exp04)
        x = readj(OUT / f"{pre}exp04_{m}.json")
        for f in ("L+u1", "L+u2", "L+u8", "U+l1", "U+l2", "U+l8"):
            put(stage, k, f"upper_top1_{f}", [(r["para"], float(r[f]["case"] == "upper")) for r in x["records"] if r[f]["case"] != "caseless"])
        del x

        # changed words (exp08)
        x = readj(OUT / f"{pre}exp08_{m}.json")
        up = [o for o in x["records"] if o["kind"] == "upper"]
        put(stage, k, "changed_word_top5", [(o["para"], float(o["rank_word"] <= 5)) for o in up])
        put(stage, k, "changed_word_top5_diff_first", [(o["para"], float(o["rank_word"] <= 5)) for o in up if not o["same_first"]])
        put(stage, k, "ctrl_word_top5", [(o["para"], float(o["rank_ctrl"] <= 5)) for o in up if o["rank_ctrl"] is not None])
        if stage == "c":
            dd = [o for o in up if not o["same_first"]]
            same(f"c.{k}.e1.medrank", statistics.median(o["rank_word"] for o in dd), 1)
            same(f"c.{k}.e1.ctrlrank", statistics.median(o["rank_ctrl"] for o in dd if o["rank_ctrl"] is not None), 1)
        del x

        # head L0H3 (exp06), Qwen3 only
        if k == "q":
            x = readj(OUT / f"{pre}exp06_{m}_L0H3.json")
            items, recs = x["items"], x["records"]
            for v in ("upper", "lower"):
                put(stage, k, f"L0H3_self_attention_{v}", [(it["para"], it[v]["att_off"]["0"]) for it in items])
            idx = [n for n, it in enumerate(items) if tcase(it["upper"]["top0"]) == "upper"]
            for name in ("same_lower", "other_lower", "other_upper", "ctrl_L0H2_same_lower", "ctrl_L0H0_same_lower"):
                trip = [(items[n]["para"], (items[n]["upper"]["D0"], items[n]["lower"]["D0"], recs[n]["patched"]["upper"][name][0])) for n in idx]

                def moved(xs):
                    a = sum(t[0] for t in xs) / len(xs); w = sum(t[1] for t in xs) / len(xs); p = sum(t[2] for t in xs) / len(xs)
                    return (a - p) / (a - w)
                put(stage, k, f"L0H3_{name}_moved", trip, moved)
                put(stage, k, f"L0H3_{name}_flip", [(items[n]["para"], float(tcase(recs[n]["patched"]["upper"][name][1]) == "lower")) for n in idx])
            del x

# ------------------------------------------------------------------ compare measures with numbers.tex; intervals
for stage in ("c", "e"):
    for k in MODELS:
        for key, (pt, pairs, stat) in MEAS[stage][k].items():
            same(f"{stage}.{k}.{key}", pt)
            if stage == "c":
                same(f"c.{k}.{key}.n", len(pairs), 0, "count")
                same(f"c.{k}.{key}.np", len({p for p, _ in pairs}), 0, "count")
                lo, hi = boot(pairs, stat, f"check-{k}-{key}")
                okci = abs(num(NV[f"c.{k}.{key}.lo"]) - lo) <= 0.03 and abs(num(NV[f"c.{k}.{key}.hi"]) - hi) <= 0.03
                check(f"interval: c.{k}.{key} agrees with an independent bootstrap (0.03)", okci,
                      f"text [{NV[f'c.{k}.{key}.lo']}, {NV[f'c.{k}.{key}.hi']}] vs [{lo:.3f}, {hi:.3f}]")
                check(f"interval: c.{k}.{key} contains its point", num(NV[f"c.{k}.{key}.lo"]) <= num(NV[f"c.{k}.{key}"]) <= num(NV[f"c.{k}.{key}.hi"]))

# Figure 2 data and the layer values quoted in the text
for k, M in MODELS.items():
    dat = [s.split() for s in (PAPER / f"fig-lens-{k}.dat").read_text().split("\n")[1:] if s.strip()]
    ok = len(dat) == M["L"] and all(abs(float(d[2]) - LAYERS[k]["LcU"][i]) < 6e-5 and abs(float(d[3]) - LAYERS[k]["LrU"][i]) < 6e-5
                                    and abs(float(d[4]) - LAYERS[k]["LcLr"][i]) < 6e-5 for i, d in enumerate(dat))
    check(f"figure 2 data {k}: every layer recomputed from records", ok)
    for key in [kk for kk in NV if kk.startswith(f"fig2.{k}.")]:
        _, _, nm, lay = key.split(".")
        same(key, LAYERS[k][nm][int(lay)], 2)

# ------------------------------------------------------------------ gate (exploration, four models; held out, two)
for k, m in (("g", "gemma"), ("q", "qwen3-17b"), ("s", "smollm2"), ("p", "pythia-1b")):
    for stage, pre in (("gate", ""), ("c", "conf_")):
        if stage == "c" and k in ("s", "p"): continue
        g = readj(OUT / f"{pre}gate_{m}.json")
        alpha = [r for r in g["records"] if r["lower"] and r["lower"].isalpha()]
        c2 = sum(r["upper"] == r["lower"].upper() for r in alpha) / len(alpha)
        ceil = sum(r["orig"].lower() == r["lower"] for r in alpha) / len(alpha)
        tf = g["teacher_forced"]["upper"]
        c1 = tf["ws_case"].get("upper", 0) / sum(v for c, v in tf["ws_case"].items() if c != "caseless")
        c3 = tf["iw_ok"] / tf["iw_n"]
        if stage == "gate":
            same(f"gate.{k}.c1", c1); same(f"gate.{k}.c2", c2); same(f"gate.{k}.c3", c3); same(f"gate.{k}.ceil", ceil)
            same(f"gate.{k}.n", g["n_paragraphs"], 0)
            passed = c1 >= 0.95 and c2 >= 0.60 and c3 >= 0.70
            check(f"gate {k}: {'passes' if passed else 'fails'} as the text says", passed == (k in ("g", "q")))
            if k in ("s", "p"): check(f"gate {k}: fails on the same-word condition only", c2 < 0.60 and c1 >= 0.95 and c3 >= 0.70)
        else:
            same(f"c.{k}.gate.c1", c1); same(f"c.{k}.gate.c2", c2); same(f"c.{k}.gate.c3", c3)
            MEAS["c"][k]["_G1"] = c1 >= 0.95 and c2 >= 0.60 and c3 >= 0.70
for c, lo_k, hi_k in (("c1", "g", "p"), ("c3", "p", "g"), ("ceil", "p", "g")):
    vals = {k: num(NV[f"gate.{k}.{c}"]) for k in "gqsp"}
    check(f"claim: gate {c} range is {lo_k} to {hi_k}", min(vals.values()) == vals[lo_k] and max(vals.values()) == vals[hi_k], vals)

# ------------------------------------------------------------------ material
for part in ("explore", "conf"):
    rows = [json.loads(s) for s in open(D5 / "data" / f"part-{part}.jsonl")]
    same(f"mat.{part}.paras", len(rows), 0); same(f"mat.{part}.articles", len({r["pageid"] for r in rows}), 0)

# ------------------------------------------------------------------ verdicts from thresholds checked against PROTOCOL.md
proto = (D5 / "PROTOCOL.md").read_text()
def section(pid):
    mm = re.search(rf"^- \*\*{pid}[ .]", proto, flags=re.M)
    return proto[mm.start():].split("\n- **")[0] if mm else ""
sect = {pid: section(pid) for pid in ("L1", "L2", "D1", "D2", "D3", "D4", "R1", "H1", "E1", "G1")}
for pid, needs in {"L1": ["**≥ 0.60**", "**≤ 0.30**", "**≤ 0.05**", "Gemma ≥ 30, Qwen3 ≥ 23"], "L2": ["**≥ 0.75**", "**≤ 0.05**"],
                   "D1": ["**≥ 0.80**", "**≥ 0.45**", "**≤ 0.05**"], "D2": ["**≤ 0.70**", "**≥ 0.60**", "**≥ 0.95**"],
                   "D3": ["**≥ 0.75**", "**≥ 0.50**", "**≤ 0.05**"], "D4": ["**≤ 0.15**", "**≥ 0.40**"],
                   "R1": ["**≥ 0.50**", "**≥ 0.85**", "**≤ 0.25**", "**≤ 0.10**"],
                   "H1": ["**≥ 0.90**", "**≥ 0.20**", "**≥ 0.25**", "**≤ 0.08**", "**≤ 0.05**"], "E1": ["**≥ 0.50**", "**≤ 0.10**"]}.items():
    for t in needs: check(f"threshold {pid} {t} is in PROTOCOL.md under {pid}", t in sect[pid])
check("gate thresholds 95% / 60% / 70% are in the frozen charter",
      all(t in (D5 / "CHARTER.md").read_text() for t in ("at least 95%", "at least 60%", "at least 70%")))
P = lambda k, key: MEAS["c"][k][key][0]
verd = {}
for k, M in MODELS.items():
    l = M["l"]; v = {}
    v["G1"] = MEAS["c"][k]["_G1"]
    v["L1"] = (P(k, "lens_Lc_gt_U_at_l") >= 0.60 and P(k, "lens_Lr_gt_U_at_l") <= 0.30 and P(k, "lens_Lc_gt_U_last") <= 0.05
               and num(NV[f"c.{k}.firstlow"]) >= M["late"])
    a, b = P(k, "lens_Lc_gt_Lr_upper_at_l"), P(k, "lens_Lc_gt_Lr_lower_at_l")
    v["L2"] = a >= 0.75 and b >= 0.75 and abs(a - b) <= 0.05
    v["D1"] = all(P(k, f"{s}_dir_L{l}_flip") >= 0.80 and P(k, f"{s}_dir_L{l}_same_word") >= 0.45 and P(k, f"{s}_rand_L{l}_flip") <= 0.05 for s in ("add", "sub"))
    v["D2"] = P(k, "rm_dir_upper_share") <= 0.70 and P(k, "rm_dir_same_word_any_case") >= 0.60 and P(k, "rm_rand_upper_share") >= 0.95
    v["D3"] = P(k, "orig_add_dir_to_upper") >= 0.75 and P(k, "orig_add_dir_to_upper_same_word") >= 0.50 and P(k, "orig_add_rand_to_upper") <= 0.05
    v["D4"] = P(k, "add_dir_L16_flip") <= 0.15 and P(k, "sub_dir_L16_flip") >= 0.40
    v["R1"] = P(k, "upper_top1_L+u2") >= 0.50 and P(k, "upper_top1_L+u8") >= 0.85 and P(k, "upper_top1_U+l2") <= 0.25 and P(k, "upper_top1_U+l8") <= 0.10
    v["E1"] = P(k, "changed_word_top5_diff_first") >= 0.50 and P(k, "ctrl_word_top5") <= 0.10
    if k == "q":
        v["H1"] = (P(k, "L0H3_self_attention_upper") >= 0.90 and P(k, "L0H3_self_attention_lower") >= 0.90
                   and P(k, "L0H3_other_lower_moved") >= 0.20 and P(k, "L0H3_other_lower_flip") >= 0.25
                   and P(k, "L0H3_other_upper_moved") <= 0.08 and P(k, "L0H3_other_upper_flip") <= 0.05
                   and P(k, "L0H3_ctrl_L0H2_same_lower_moved") <= 0.08 and P(k, "L0H3_ctrl_L0H0_same_lower_moved") <= 0.08)
    core = all(v[x] for x in ("L1", "L2", "D1", "D2", "D3"))
    acct = "supported" if core else ("partly supported" if v["D1"] and v["D3"] else "not supported")
    for pid, ok in v.items():
        check(f"verdict {k} {pid}: recomputed {'passed' if ok else 'failed'} matches the text", NV.get(f"c.{k}.{pid}") == ("passed" if ok else "failed"))
    check(f"verdict {k}: account '{acct}' matches the text", NV.get(f"c.{k}.account") == acct)
    same(f"c.{k}.npass", sum(v.values()), 0); same(f"c.{k}.ntests", len(v), 0)
    verd[k] = v
same("c.npass", sum(sum(v.values()) for v in verd.values()), 0); same("c.ntests", sum(len(v) for v in verd.values()), 0)
check("claim: every prediction passed in both models", all(all(v.values()) for v in verd.values()))

# ------------------------------------------------------------------ exploration-to-held-out differences
gaps = sorted(((abs(MEAS["c"][k][key][0] - MEAS["e"][k][key][0]), k, key) for k in MODELS for key in MEAS["c"][k]
               if not key.startswith("_") and key in MEAS["e"][k]), reverse=True)
same("gap.n", len(gaps), 0); same("gap.max", gaps[0][0]); same("gap.second", gaps[1][0]); same("gap.over05", sum(g > 0.05 for g, _, _ in gaps), 0)
check("claim: the largest difference is Qwen3's share of words kept after removal", gaps[0][1:] == ("q", "rm_dir_same_word_any_case"), gaps[0])

# ------------------------------------------------------------------ claims made in words
fl = {(k, s): P(k, f"{s}_dir_L{MODELS[k]['l']}_flip") for k in MODELS for s in ("add", "sub")}
sw = {(k, s): P(k, f"{s}_dir_L{MODELS[k]['l']}_same_word") for k in MODELS for s in ("add", "sub")}
check("claim: abstract ranges ordered (Gemma subtract-add, Qwen3 add-subtract)", fl["g", "sub"] <= fl["g", "add"] and fl["q", "add"] <= fl["q", "sub"])
check("claim: the edit flips the case at more than 0.95 of positions", min(fl.values()) > 0.95, fl)
ch = [(fl[x] - sw[x]) / fl[x] for x in fl]
check("claim: the edit changes the word at a fifth to a third of flipped positions", all(0.15 <= c <= 0.36 for c in ch), [round(c, 3) for c in ch])
check("claim: same-word ranges ordered as typeset", sw["q", "add"] <= sw["q", "sub"] and sw["g", "sub"] <= sw["g", "add"])
rnd = [P(k, f"{s}_rand_L{MODELS[k]['l']}_flip") for k in MODELS for s in ("add", "sub")]
check("claim: random flips at most the quoted value", round(max(rnd), 3) == num(NV["c.g.sub_rand_L24_flip"]), rnd)
check("claim: removal leaves about half in capitals", all(0.45 <= P(k, "rm_dir_upper_share") <= 0.60 for k in MODELS))
check("claim: L+u1 single word not enough (below one half)", all(P(k, "upper_top1_L+u1") < 0.5 for k in MODELS))
check("claim: same median rank in both models", NV["c.g.e1.medrank"] == NV["c.q.e1.medrank"])
check("claim: removing capitals works earlier (layer 16 subtract above add)", all(num(NV[f"sweep.{k}.sub.16"]) > num(NV[f"sweep.{k}.add.16"]) for k in MODELS))
check("claim: Qwen3 adding switches on between layers 17 and 19", num(NV["sweep.q.add.17"]) < 0.1 and num(NV["sweep.q.add.19"]) > 0.8)
check("claim: layer 0, another word's token beats U at most positions", all(LAYERS[k]["LrU"][0] > 0.5 for k in MODELS))
check("claim: layer 0, Lc beats Lr no more often than chance", all(LAYERS[k]["LcLr"][0] <= 0.5 for k in MODELS))
check("claim: Gemma's word stands out from about layer 20", LAYERS["g"]["LcLr"][19] < 0.6 < LAYERS["g"]["LcLr"][20])
drops = [LAYERS["q"]["LrU"][i - 1] - LAYERS["q"]["LrU"][i] for i in range(1, 28)]
check("claim: Qwen3's Lr-above-U share falls most sharply at layer 19", drops.index(max(drops)) + 1 == 19)
check("claim: two other layer-0 heads move almost nothing (below 0.02)", all(abs(P("q", f"L0H3_ctrl_L0H{h}_same_lower_moved")) < 0.02 for h in (0, 2)))
check("claim: copying, the upper-lower gap is small for seen and new words (below 0.05)",
      all(abs(num(NV[f"c.{k}.{g}.upper"]) - num(NV[f"c.{k}.{g}.lower"])) < 0.05 for k in MODELS for g in ("seen", "new")))

# ================================================================== round 2 (recomputed from the raw records)
import torch  # only to read the saved .pt record files
R2 = OUT / "r2"
def tload(p): return torch.load(p, weights_only=False)
def klass(s):
    t = [c for c in s.strip() if c.isalpha()]
    if not t: return "none"
    if len(t) >= 2 and all(c.isupper() for c in t): return "ALL"
    if t[0].isupper() and any(c.islower() for c in t): return "Init"
    if len(t) == 1 and t[0].isupper(): return "One"
    return "low" if all(c.islower() for c in t) else "mix"
def ratio_stat(xs): return sum(n for n, _ in xs) / len(xs) / (sum(d for _, d in xs) / len(xs))
def mean_stat(xs): return sum(xs) / len(xs)
R2FILES = {
    "c": {"q": dict(sets="exp03_qwen3-17b_reserve_0_conf.pt", single="exp03_qwen3-17b_reserve_0_l17h6.pt", attn="exp06_qwen3-17b_reserve_0.json",
                    edit="exp05_qwen3-17b_reserve_0.json", other=["exp04_qwen3-17b_reserve_0.json", "exp04_qwen3-17b_reserve_264.json"]),
          "g": dict(sets="exp03_gemma_reserve_0_conf.pt", single=None, attn="exp06_gemma_reserve_0.json",
                    edit="exp05_gemma_reserve_0.json", other=["exp04_gemma_reserve_0.json", "exp04_gemma_reserve_264.json"])},
    "e": {"q": dict(sets="exp03_qwen3-17b_conf_440_a.pt", single="exp03_qwen3-17b_conf_440_l17h6.pt", attn="exp06_qwen3-17b_conf_0.json",
                    edit="exp05_qwen3-17b_conf_100_pieces.json", other=["exp04_qwen3-17b_explore_0.json", "exp04_qwen3-17b_conf_0.json"]),
          "g": dict(sets="exp03_gemma_conf_440_a.pt", single=None, attn="exp06_gemma_conf_0.json",
                    edit="exp05_gemma_conf_100.json", other=["exp04_gemma_conf_0.json"])}}
M2 = {"c": {}, "e": {}}; X2 = {"c": {}, "e": {}}   # measures (point, pairs, stat); extras (single values)
def p2(st, k, key, pairs, stat): M2[st].setdefault(k, {})[key] = (stat([x for _, x in pairs]), pairs, stat)
for st in ("c", "e"):
    for k, F in R2FILES[st].items():
        X = X2[st].setdefault(k, {})
        s = tload(R2 / F["sets"]); rc = s["records"]; nr = len(s["random"]["writers"])
        if st == "c": check(f"round 2 {k}: held-out sets read the reserve part", s["part"] == "reserve")
        up = [r for r in rc if tcase(r["U"]["top"]) == "upper"]; lo = [r for r in rc if tcase(r["L"]["top"]) == "lower"]
        p2(st, k, "W_none_upper", [(r["pageid"], float(tcase(r["W"][0][0]) == "upper")) for r in up], mean_stat)
        p2(st, k, "CA_named_upper", [(r["pageid"], float(tcase(r["CA"][1][0]) == "upper")) for r in rc], mean_stat)
        p2(st, k, "W_named_upper", [(r["pageid"], float(tcase(r["W"][1][0]) == "upper")) for r in up], mean_stat)
        p2(st, k, "W_named_same_word", [(r["pageid"], float(sameword(r["U"]["top"], r["W"][1][0]))) for r in up], mean_stat)
        p2(st, k, "W_named_proj_removed", [(r["pageid"], (r["W"][0][2] - r["W"][1][2], r["W"][0][2] - r["L"]["proj"])) for r in rc], ratio_stat)
        X["W_random_upper.min"] = min(mean_stat([float(tcase(r["W"][j][0]) == "upper") for r in up]) for j in range(2, 2 + nr))
        cap = lambda key, j, c, ref: [(r["pageid"], (r[key][0][c] - r[key][j][c], r[key][0][c] - r["L"][ref])) for r in rc]
        p2(st, k, "CA_named_proj_removed", cap("CA", 1, 2, "proj"), ratio_stat)
        p2(st, k, "CA_named_D_removed", cap("CA", 1, 1, "D"), ratio_stat)
        X["CA_random_proj_removed.max"] = max(ratio_stat([x for _, x in cap("CA", j, 2, "proj")]) for j in range(2, 2 + nr))
        p2(st, k, "CU_named_upper", [(r["pageid"], float(tcase(r["CU"][1][0]) == "upper")) for r in up], mean_stat)
        X["CU_random_upper.min"] = min(mean_stat([float(tcase(r["CU"][j][0]) == "upper") for r in up]) for j in range(2, 2 + nr))
        p2(st, k, "R_none_flip", [(r["pageid"], float(tcase(r["R"][0][0]) == "upper")) for r in lo], mean_stat)
        p2(st, k, "R_named_flip", [(r["pageid"], float(tcase(r["R"][1][0]) == "upper")) for r in lo], mean_stat)
        X["R_random_flip.min"] = min(mean_stat([float(tcase(r["R"][j][0]) == "upper") for r in lo]) for j in range(2, 2 + nr))
        X["context_share"] = mean_stat([r["A"]["proj"] - r["L"]["proj"] for r in rc]) / mean_stat([r["U"]["proj"] - r["L"]["proj"] for r in rc])
        if F["single"]:
            s1 = tload(R2 / F["single"]); r1 = s1["records"]
            f1 = lambda j: [(r["pageid"], (r["CA"][0][2] - r["CA"][j][2], r["CA"][0][2] - r["L"]["proj"])) for r in r1]
            p2(st, k, "L17H6_proj_removed", f1(1), ratio_stat)
            X["L17H6_random_proj_removed.max"] = max(ratio_stat([x for _, x in f1(j)]) for j in range(2, 2 + len(s1["random"]["carriers"])))
        a = readj(R2 / F["attn"])
        for h in [x for x in a["records"][0] if x.startswith("L")]:
            p2(st, k, f"attn_{h}_recent8", [(r["pageid"], (sum(r[h][0:4]), sum(r[h][0:5]))) for r in a["records"]], ratio_stat)
            X[f"attn_{h}"] = [mean_stat([r[h][b] for r in a["records"]]) for b in range(len(a["bins"]))]
        ed = readj(R2 / F["edit"])["records"]
        fl2 = [r for r in ed if tcase(r["lo"]) == "lower" and tcase(r["edit"]) == "upper"]
        chg = lambda r: float(not sameword(r["lo"], r["edit"]))
        X["E_flips"] = len(fl2)
        p2(st, k, "E_change", [(r["pageid"], chg(r)) for r in fl2], mean_stat)
        p2(st, k, "E_change_if_up_changes", [(r["pageid"], chg(r)) for r in fl2 if not sameword(r["lo"], r["up"])], mean_stat)
        p2(st, k, "E_change_if_up_keeps", [(r["pageid"], chg(r)) for r in fl2 if sameword(r["lo"], r["up"])], mean_stat)
        p2(st, k, "E_change_not_by_up", [(r["pageid"], chg(r) if sameword(r["lo"], r["up"]) else 0.0) for r in fl2], mean_stat)
        p2(st, k, "E_changed_in_lower_top5", [(r["pageid"], float(r["edit_lower_rank"] is not None and r["edit_lower_rank"] <= 5)) for r in fl2 if chg(r)], mean_stat)
        p2(st, k, "E_change_extra0", [(r["pageid"], chg(r)) for r in fl2 if r["pieces_up"] == r["pieces_lo"]], mean_stat)
        p2(st, k, "E_change_extra1p", [(r["pageid"], chg(r)) for r in fl2 if r["pieces_up"] >= r["pieces_lo"] + 1], mean_stat)
        X["E_random_flip"] = mean_stat([float(tcase(r["rand"]) == "upper") for r in fl2])
        paras, heads = [], []
        for fn in F["other"]:
            o = readj(R2 / fn); paras += o["para"]; heads += o["head"]; X["pu"], X["pl"] = o["proj_upper_mean"], o["proj_lower_mean"]
        gap = X["pu"] - X["pl"]
        kp = lambda rs, kind, base, j, want: [(r["pageid"], float(klass(r["tops"][j]) == want)) for r in rs if r["kind"] == kind and klass(r["tops"][0]) == base]
        p2(st, k, "A_acr_rm_keeps", kp(paras, "acr", "ALL", 1, "ALL"), mean_stat)
        p2(st, k, "A_acr_rm_rand_keeps", kp(paras, "acr", "ALL", 2, "ALL"), mean_stat)
        p2(st, k, "A_init_rm_keeps", kp(paras, "init", "Init", 1, "Init"), mean_stat)
        p2(st, k, "A_init_rm_rand_keeps", kp(paras, "init", "Init", 2, "Init"), mean_stat)
        hc = [r for r in heads if r["caps"]]
        p2(st, k, "A_head_rm_keeps", kp(hc, "hin", "ALL", 1, "ALL"), mean_stat)
        p2(st, k, "A_head_rm_rand_keeps", kp(hc, "hin", "ALL", 2, "ALL"), mean_stat)
        def pr2(kind):
            x = {(r["item"], r["word"].lower()): r for r in heads if r["kind"] == kind and r["caps"]}
            y = {(r["item"], r["word"].lower()): r for r in heads if r["kind"] == kind and not r["caps"]}
            return [(x[q]["pageid"], (x[q]["proj"] - y[q]["proj"]) / gap) for q in x if q in y]
        p2(st, k, "A_head_inside_excess", pr2("hin"), mean_stat)
        p2(st, k, "A_head_after_excess", pr2("hbody"), mean_stat)
        p2(st, k, "A_head_after_allcaps_top", [(r["pageid"], float(klass(r["tops"][0]) == "ALL")) for r in heads if r["kind"] == "hbody" and r["caps"]], mean_stat)
# compare with numbers.tex; intervals (held out)
for st, pre in (("c", "r2"), ("e", "r2e")):
    for k in ("q", "g"):
        for key, (pt, pairs, stat) in M2[st][k].items():
            same(f"{pre}.{k}.{key}", pt)
            if st == "c":
                same(f"r2.{k}.{key}.n", len(pairs), 0, "count")
                lo_, hi_ = boot(pairs, stat, f"check-r2-{k}-{key}")
                okci = abs(num(NV[f"r2.{k}.{key}.lo"]) - lo_) <= 0.03 and abs(num(NV[f"r2.{k}.{key}.hi"]) - hi_) <= 0.03
                check(f"interval: r2.{k}.{key} agrees with an independent bootstrap (0.03)", okci, f"text [{NV[f'r2.{k}.{key}.lo']}, {NV[f'r2.{k}.{key}.hi']}] vs [{lo_:.3f}, {hi_:.3f}]")
        for key, v in X2[st][k].items():
            if key.startswith("attn_"):
                for b, x in enumerate(v): same(f"{pre}.{k}.{key}.{b}", x, 2)
            elif key in ("pu", "pl", "context_share", "E_random_flip"): continue
            elif key == "E_flips": same(f"{pre}.{k}.E_flips", v, 0, "count") if st == "c" else None
            else: same(f"{pre}.{k}.{key}", v)
    for k in ("q", "g"):
        if st == "c":
            same(f"r2.{k}.proj_upper_mean", X2["c"][k]["pu"], 1); same(f"r2.{k}.proj_lower_mean", X2["c"][k]["pl"], 1)
# figure data (attention)
for k in ("q", "g"):
    rows = [s.split() for s in (PAPER / f"fig-attn-{k}.dat").read_text().split("\n") if s.strip()]
    hdr = rows[0][1:]
    ok = all(abs(float(rows[1 + b][1 + i]) - X2["c"][k][f"attn_{h}"][b]) < 6e-5 for i, h in enumerate(hdr) for b in range(6))
    check(f"figure 4 data {k}: every bar recomputed from records", ok)
# thresholds in PROTOCOL-2.md and verdicts
proto2 = (D5 / "PROTOCOL-2.md").read_text()
def sect2(pid):
    mm = re.search(rf"^- \*\*{pid}[ .]", proto2, flags=re.M)
    return proto2[mm.start():].split("\n- **")[0] if mm else ""
for pid, needs in {"W1": ["**≤ 0.30**", "**≥ 0.55**", "**≥ 0.85**"], "R1": ["**≥ 0.80**", "**≤ 0.20**", "**0.15**"],
                   "C1": ["**≥ 0.60**", "**≥ 0.30**", "**0.25**"], "C2": ["**≤ 0.60**", "**≥ 0.90**"], "C3": ["**≥ 0.20**"],
                   "C4": ["**≥ 0.60**", "**≤ 0.50**"], "E1": ["**≥ 0.30**", "**≥ 0.40**"], "E2": ["**0.10**"],
                   "A1": ["**≥ 0.85**", "**≥ 0.90**"], "A2": ["**≥ 0.90**", "**≥ 0.95**"], "A3": ["**≤ 0.85**", "**≥ 0.95**"],
                   "A4": ["**≥ 0.40**", "**≤ 0.25**", "**≤ 0.10**"]}.items():
    for t in needs: check(f"threshold {pid} {t} is in PROTOCOL-2.md under {pid}", t in sect2(pid))
LOC = {"q": ("L17H6", "L19H0"), "g": ("L14H7", "L24H6")}; V2 = {}
for k in ("q", "g"):
    Q = lambda key: M2["c"][k][key][0]; XX = X2["c"][k]; v = {}
    v["W1"] = Q("W_named_upper") <= 0.30 and Q("W_named_same_word") >= 0.55 and XX["W_random_upper.min"] >= 0.85
    v["R1"] = Q("R_none_flip") >= 0.80 and Q("R_named_flip") <= 0.20 and XX["R_random_flip.min"] >= Q("R_none_flip") - 0.15
    v["C1"] = Q("CA_named_proj_removed") >= 0.60 and Q("CA_named_D_removed") >= 0.30 and XX["CA_random_proj_removed.max"] <= Q("CA_named_proj_removed") - 0.25
    v["C2"] = Q("CU_named_upper") <= 0.60 and XX["CU_random_upper.min"] >= 0.90
    if k == "q": v["C3"] = Q("L17H6_proj_removed") >= 0.20 and XX["L17H6_random_proj_removed.max"] < Q("L17H6_proj_removed")
    v["C4"] = Q(f"attn_{LOC[k][0]}_recent8") >= 0.60 and Q(f"attn_{LOC[k][1]}_recent8") <= 0.50
    v["E1"] = Q("E_change_if_up_changes") - Q("E_change_if_up_keeps") >= 0.30 and Q("E_changed_in_lower_top5") >= 0.40
    v["E2"] = Q("E_change_extra1p") - Q("E_change_extra0") >= 0.10
    v["A1"] = Q("A_acr_rm_keeps") >= 0.85 and Q("A_acr_rm_rand_keeps") >= 0.90
    v["A2"] = Q("A_init_rm_keeps") >= 0.90 and Q("A_init_rm_rand_keeps") >= 0.95
    v["A3"] = Q("A_head_rm_keeps") <= 0.85 and Q("A_head_rm_rand_keeps") >= 0.95
    v["A4"] = Q("A_head_inside_excess") >= 0.40 and Q("A_head_after_excess") <= 0.25 and Q("A_head_after_allcaps_top") <= 0.10
    for pid, ok in v.items():
        check(f"round 2 verdict {k} {pid}: recomputed {'passed' if ok else 'failed'} matches the text", NV.get(f"r2.{k}.{pid}") == ("passed" if ok else "failed"))
    vv = lambda a, b: "supported" if v[a] and v[b] else ("partly supported" if v[a] or v[b] else "not supported")
    for qk, val in {"Q1_parts": vv("W1", "R1"), "Q2_carriers": vv("C1", "C2"), "Q3_word_changes": vv("E1", "E2"),
                    "Q4_acronyms_initials_separate": "supported" if v["A1"] and v["A2"] else "not supported",
                    "Q4_headings_same_direction_reset": "supported" if v["A3"] and v["A4"] else "not supported"}.items():
        check(f"round 2 verdict {k} {qk} matches the text", NV.get(f"r2.{k}.v.{qk}") == val)
    same(f"r2.{k}.v.n_pass", sum(v.values()), 0); same(f"r2.{k}.v.n_tests", len(v), 0)
    V2[k] = v
same("r2.npass", sum(sum(v.values()) for v in V2.values()), 0); same("r2.ntests", sum(len(v) for v in V2.values()), 0)
check("claim: every round-2 prediction passed and every verdict is supported", all(all(v.values()) for v in V2.values())
      and all(NV[x] == "supported" for x in NV if x.startswith("r2.") and ".v.Q" in x))
gaps2 = sorted(((abs(M2["c"][k][key][0] - M2["e"][k][key][0]), k, key) for k in ("q", "g") for key in M2["c"][k] if key in M2["e"][k]), reverse=True)
same("r2gap.n", len(gaps2), 0); same("r2gap.max", gaps2[0][0]); same("r2gap.second", gaps2[1][0]); same("r2gap.over05", sum(x > 0.05 for x, _, _ in gaps2), 0)
check("claim: largest round-2 difference is Qwen3's top-5 share of changed words; next, Qwen3's heading removal",
      gaps2[0][1:] == ("q", "E_changed_in_lower_top5") and gaps2[1][1:] == ("q", "A_head_rm_keeps"), gaps2[:2])
check("claim: both were estimated from few exploration positions (70 or fewer)", len(M2["e"]["q"]["E_changed_in_lower_top5"][1]) <= 70 and len(M2["e"]["q"]["A_head_rm_keeps"][1]) <= 70)
check("claim: round-2 differences larger than round 1's (largest)", gaps2[0][0] > num(NV["gap.max"]))
# claims in words, round 2
sq, sg = readj(D5 / "code" / "r2_sets_qwen.json"), readj(D5 / "code" / "r2_sets_gemma.json")
cnt = lambda s, w: (sum(x.startswith("MLP") for x in s[w]), sum(x.startswith("L") for x in s[w]))
check("claim: writers are four MLPs and one head (Qwen3), three MLPs and one head (Gemma)", cnt(sq, "writers") == (4, 1) and cnt(sg, "writers") == (3, 1))
check("claim: four or five carrier heads", len(sq["carriers"]) == 4 and len(sg["carriers"]) == 5 and cnt(sq, "carriers")[0] == 0 == cnt(sg, "carriers")[0])
check("claim: readers are MLPs only, all after the direction layer", all(x.startswith("MLP") and int(x[3:]) > l for s, l in ((sq, 22), (sg, 24)) for x in s["readers"]))
check("claim: Gemma's random carrier sets reach higher than Qwen3's", X2["c"]["g"]["CA_random_proj_removed.max"] > X2["c"]["q"]["CA_random_proj_removed.max"])
check("claim (exploration): context alone carried about half of the full gap", all(0.35 <= X2["e"][k]["context_share"] <= 0.65 for k in ("q", "g")),
      [round(X2["e"][k]["context_share"], 3) for k in ("q", "g")])
check("claim: the first word after a capitalised heading is almost never all capitals (below 0.05)", all(M2["c"][k]["A_head_after_allcaps_top"][0] < 0.05 for k in ("q", "g")))
check("claim: the writers' edit removes more than the whole gap", all(M2["c"][k]["W_named_proj_removed"][0] > 1 for k in ("q", "g")))
check("claim: random directions flipped none of the edit positions", all(X2["c"][k]["E_random_flip"] == 0 for k in ("q", "g")))
ex1 = tload(R2 / "exp01_qwen3-17b_conf_300.pt")["records"]
emb = mean_stat([r["U"]["emb_proj"] - r["L"]["emb_proj"] for r in ex1]) / mean_stat([r["U"]["resid_proj"] - r["L"]["resid_proj"] for r in ex1])
check("claim (exploration): the token embedding wrote almost nothing along u (under 1% of the gap, Qwen3)", abs(emb) < 0.01, emb)
# OLMo gate (exploration, Q5)
g = readj(OUT / "gate_olmo1b.json")
alpha = [r for r in g["records"] if r["lower"] and r["lower"].isalpha()]
tf = g["teacher_forced"]["upper"]
oc1 = tf["ws_case"].get("upper", 0) / sum(v for c, v in tf["ws_case"].items() if c != "caseless"); oc2 = sum(r["upper"] == r["lower"].upper() for r in alpha) / len(alpha); oc3 = tf["iw_ok"] / tf["iw_n"]
same("gate.o.c1", oc1); same("gate.o.c2", oc2); same("gate.o.c3", oc3)
check("gate o: OLMo-2-1B fails on the same-word condition only", oc2 < 0.60 and oc1 >= 0.95 and oc3 >= 0.70)
# material
rows = [json.loads(s) for s in open(D5 / "data" / "part-reserve.jsonl")]
same("mat.reserve.paras", len(rows), 0); same("mat.reserve.articles", len({r["pageid"] for r in rows}), 0)
hr = range(json.loads((D5.parent / "SEALED-MATERIAL.json").read_text())["files"]["D5-capitals/data/headings-reserve.jsonl"]["lines"])  # release: the unread sealed file is not shipped; its line count is in SEALED-MATERIAL.json
same("mat.head.reserve.items", len(hr), 0)
# X1 values quoted (from X1's held-out records summary)
x1 = readj(D5.parent / "X1-shared-parts" / "outputs" / "conf" / "verdicts.json")["tests"]
same("x1.l0h3.dlow", x1["C7"]["values"]["Dl"]); same("x1.l0h3.dup", x1["C7"]["values"]["Du"]); same("x1.l0h3.wl", x1["C7"]["values"]["wl"], 2)
sv = [x1["C8"]["values"][t]["L0H3_self"] for t in ("LB", "IND", "ORD", "CAPS")]
same("x1.l0h3.self.min", min(sv), 2); same("x1.l0h3.self.max", max(sv), 2); same("x1.l17h6.dup", x1["A6"]["values"]["Du"])
check("claim: X1 tests C7, C8 and A6 passed", all(x1[t]["passed"] for t in ("C7", "C8", "A6")))

# ------------------------------------------------------------------ text
tex = (PAPER / "main.tex").read_text()
body = tex[tex.index("\\begin{document}"):tex.index("\\end{document}")]
expand = {"G": "c.g.", "Q": "c.q.", "GI": "c.g.", "QI": "c.q.", "EG": "e.g.", "EQ": "e.q.", "nv": "",
          "RQ": "r2.q.", "RG": "r2.g.", "RQI": "r2.q.", "RGI": "r2.g.", "XQ": "r2e.q.", "XG": "r2e.g."}
MAC = r"\\(RQI|RGI|RQ|RG|XQ|XG|GI|QI|EG|EQ|G|Q|nv)\{([^}]+)\}"
keys = [expand[a] + b for a, b in re.findall(MAC, body)]
keys += [expand[a] + b + s for a, b in re.findall(r"\\(RQI|RGI|GI|QI)\{([^}]+)\}", body) for s in (".lo", ".hi")]
check("every number macro used in the text is defined", all(k in NV for k in keys), sorted({k for k in keys if k not in NV}))
covered = {r["name"].split(": ", 1)[1] for r in RES if r["name"].split(": ", 1)[0] in ("recount", "count") and r["ok"]}
covered |= {f"c.{k}.{p}" for k in MODELS for p in list(verd[k]) + ["account"]}
covered |= {f"r2.{k}.{p}" for k in V2 for p in V2[k]}
covered |= {f"r2.{k}.v.{q}" for k in V2 for q in ("Q1_parts", "Q2_carriers", "Q3_word_changes", "Q4_acronyms_initials_separate", "Q4_headings_same_direction_reset")}
covered |= {re.match(r"interval: (\S+) agrees", r["name"]).group(1) + e for r in RES if r["ok"] and re.match(r"interval: (\S+) agrees", r["name"]) for e in (".lo", ".hi")}
covered |= {"sweep.g.rows", "sweep.q.rows"} if all(NV[f"sweep.{k}.rows"] == f"{a}--{b}" for k, (a, b) in
            (("g", readj(OUT / "exp02_gemma.json")["rows"]), ("q", readj(OUT / "exp02_qwen3-17b.json")["rows"]))) else set()
missing = sorted(set(keys) - covered)
check("every number used in the text was recomputed by this script", not missing, missing)
plain = re.sub(MAC, "", body)
plain = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", plain, flags=re.S)
dec = re.findall(r"(?<![\w.{])\d+\.\d+", plain)
thresholds = {"0.60", "0.30", "0.05", "0.75", "0.80", "0.45", "0.70", "0.95", "0.50", "0.15", "0.40", "0.85", "0.25", "0.10", "0.90", "0.20", "0.08", "0.55", "1.7"}
check("no hand-typed decimals except frozen thresholds and model names", all(d in thresholds for d in dec), sorted(set(dec) - thresholds))
check("model-name decimal only in model names", "1.7" not in re.sub(r"(Qwen3-1\.7B|SmolLM2-1\.7B)", "", re.sub(r"\\newcommand.*", "", tex)))
low = plain.lower()
for w in ("claude", "anthropic", "codex", "openai", "chatgpt", "gpt-4", "gemini"):
    check(f"no product or company name: {w}", w not in low)
us = ["behavior", "tokenization", "tokenizer", "normalization", "analyze", "color", "modeling", "favor", "labeled", "summarize",
      "recognize", "organize", "generalize", "center ", "characterize", "capitalization", "capitalize"]
bodytext = re.sub(r"\\cite[pt]?\{[^}]*\}", "", low)
check("British spelling (no US forms)", not [w for w in us if w in bodytext], [w for w in us if w in bodytext])
caps = re.findall(r"\\caption\{(.*?)\}\s*\n", body)
check("no repo paths in captions", caps and not any(re.search(r"outputs/|code/|\.py|\.json|\.jsonl|\.md\b|\.dat", c) for c in caps))
check("AI-use statement says 'language-model assistants'", "language-model assistants" in body)
check("author line is a placeholder", "[Author line to be added]" in tex)
for lab in ("exploratory", "prospective", "post hoc", "descriptive"):
    check(f"label used: {lab}", lab in low)
check("states that SmolLM2 and Pythia-1B failed the gate", re.search(r"SmolLM2-1\.7B.*?and Pythia-1B.*?failed it", body, flags=re.S) is not None)
check("states the reserve part was read only after the second protocol was frozen", "The reserve part was read only after the second round's protocol was frozen" in body)
check("reports the exploration-to-held-out differences (both rounds)", all(f"\\nv{{{x}}}" in body for x in ("gap.max", "gap.n", "r2gap.max", "r2gap.n", "r2gap.over05")))
tab = body[body.index("\\label{tab:preds}"):body.index("\\end{tabularx}", body.index("\\label{tab:preds}"))]
rowthr = {"lens_Lc_gt_U_at_l": "\\ge 0.60", "lens_Lr_gt_U_at_l": "\\le 0.30", "lens_Lc_gt_U_last": "\\le 0.05", "lens_Lc_gt_Lr_upper_at_l": "\\ge 0.75",
          "lens_Lc_gt_Lr_lower_at_l": "\\ge 0.75", "add_dir_L24_flip": "\\ge 0.80", "add_dir_L24_same_word": "\\ge 0.45", "sub_dir_L24_flip": "\\ge 0.80",
          "sub_dir_L24_same_word": "\\ge 0.45", "add_rand_L24_flip": "\\le 0.05", "sub_rand_L24_flip": "\\le 0.05", "rm_dir_upper_share": "\\le 0.70",
          "rm_dir_same_word_any_case": "\\ge 0.60", "rm_rand_upper_share": "\\ge 0.95", "orig_add_dir_to_upper": "\\ge 0.75",
          "orig_add_dir_to_upper_same_word": "\\ge 0.50", "orig_add_rand_to_upper": "\\le 0.05", "add_dir_L16_flip": "\\le 0.15", "sub_dir_L16_flip": "\\ge 0.40",
          "upper_top1_L+u2": "\\ge 0.50", "upper_top1_L+u8": "\\ge 0.85", "upper_top1_U+l2": "\\le 0.25", "upper_top1_U+l8": "\\le 0.10",
          "changed_word_top5": "\\ge 0.50", "ctrl_word_top5": "\\le 0.10", "gate.c2": "\\ge 0.60"}
for key, thr in rowthr.items():
    row = [r for r in tab.split("\\\\") if f"{{{key}}}" in r]
    check(f"table 1: row {key} carries threshold {thr}", len(row) == 1 and thr in row[0])
check("table 1: E1 row uses the frozen filtered measure (identical to the unfiltered one)",
      all(MEAS["c"][k]["changed_word_top5"][0] == MEAS["c"][k]["changed_word_top5_diff_first"][0] for k in MODELS))
htab = body[body.index("\\label{tab:head}"):body.index("\\end{tabularx}", body.index("\\label{tab:head}"))]
for key, thr in {"L0H3_self_attention_upper": "\\ge 0.90", "L0H3_self_attention_lower": "\\ge 0.90", "L0H3_other_lower_moved": "\\ge 0.20",
                 "L0H3_other_lower_flip": "\\ge 0.25", "L0H3_other_upper_moved": "\\le 0.08", "L0H3_other_upper_flip": "\\le 0.05",
                 "L0H3_ctrl_L0H2_same_lower_moved": "\\le 0.08", "L0H3_ctrl_L0H0_same_lower_moved": "\\le 0.08"}.items():
    row = [r for r in htab.split("\\\\") if f"{{{key}}}" in r]
    check(f"table 2: row {key} carries threshold {thr}", len(row) == 1 and thr in row[0])

tab2 = body[body.index("\\label{tab:preds2}"):body.index("\\end{tabularx}", body.index("\\label{tab:preds2}"))]
for key, thr in {"W_named_upper": "\\le 0.30", "W_named_same_word": "\\ge 0.55", "W_random_upper.min": "\\ge 0.85", "R_none_flip": "\\ge 0.80",
                 "R_named_flip": "\\le 0.20", "CA_named_proj_removed": "\\ge 0.60", "CA_named_D_removed": "\\ge 0.30", "CU_named_upper": "\\le 0.60",
                 "CU_random_upper.min": "\\ge 0.90", "L17H6_proj_removed": "\\ge 0.20", "attn_L17H6_recent8": "\\ge 0.60", "attn_L19H0_recent8": "\\le 0.50",
                 "E_changed_in_lower_top5": "\\ge 0.40", "A_acr_rm_keeps": "\\ge 0.85", "A_acr_rm_rand_keeps": "\\ge 0.90", "A_init_rm_keeps": "\\ge 0.90",
                 "A_init_rm_rand_keeps": "\\ge 0.95", "A_head_rm_keeps": "\\le 0.85", "A_head_rm_rand_keeps": "\\ge 0.95", "A_head_inside_excess": "\\ge 0.40",
                 "A_head_after_excess": "\\le 0.25", "A_head_after_allcaps_top": "\\le 0.10"}.items():
    row = [r for r in tab2.split("\\\\") if f"{{{key}}}" in r]
    check(f"table 3: row {key} carries threshold {thr}", len(row) == 1 and thr in row[0], row[:1])
tab2n = body[body.index("\\label{tab:preds2}"):body.index("\\end{table}", body.index("\\label{tab:preds2}"))]
check("table 3 note: R1, C1, E1 and E2 margins", all(t in tab2n for t in ("minus 0.15", "minus 0.25", "at least 0.30", "at least 0.10")))

# ------------------------------------------------------------------ references
bib = (PAPER / "refs.bib").read_text()
entries = dict(re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, flags=re.S))
cited = set(k.strip() for grp in re.findall(r"\\cite[pt]?\{([^}]+)\}", body) for k in grp.split(","))
check("every cited key is in refs.bib", cited <= set(entries), cited - set(entries))
check("every refs.bib entry is cited", set(entries) <= cited, set(entries) - cited)
abs_ = readj(PAPER / "sources" / "arxiv-abs-pages.json")
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
for k, e in entries.items():
    mm = re.search(r"(?:eprint\s*=\s*\{|arXiv:)(\d{4}\.\d{5})", e)
    if not mm:
        src = readj(PAPER / "sources" / "logit-lens-page.json")
        t = re.search(r"title\s*=\s*\{(.*)\},\n", e).group(1)
        check(f"reference {k}: non-arXiv entry matches its saved page record", k == "nostalgebraist2020logitlens" and norm(t) == norm(src["title"])
              and src["author"] in e and src["date"][:4] in e)
        continue
    a = abs_.get(mm.group(1))
    t = re.search(r"title\s*=\s*\{(.*)\},\n", e).group(1)
    first = re.search(r"author\s*=\s*\{(.*?)\},\n", e, flags=re.S).group(1).split(" and ")[0]
    sur = first.split(",")[0].strip("{} ").split()[-1]
    yr = re.search(r"year\s*=\s*\{(\d{4})\}", e).group(1)
    ok = a is not None and norm(t) == norm(a["title"]) and norm(sur) in norm(" ".join(a["authors"][:2]))
    check(f"reference {k}: arXiv {mm.group(1)} title and first author match the arXiv record", ok, f"{t} | {a['title'] if a else None} | {sur}")
    check(f"reference {k}: year {yr} matches the first arXiv version or the venue", a is not None and (a["date"][:4] == yr or "booktitle" in e),
          f"arXiv {a['date'] if a else None}")

# ------------------------------------------------------------------ report
nok = sum(r["ok"] for r in RES); nbad = len(RES) - nok
(PAPER / "checks").mkdir(exist_ok=True)
json.dump(RES, open(PAPER / "checks" / "number-check.json", "w"), indent=1, ensure_ascii=False)
with open(PAPER / "checks" / "NUMBER-CHECK.md", "w") as f:
    f.write(f"# Number check, D5 write-up v2\n\nGenerated by `scripts/check_numbers.py`. {len(RES)} checks, {nok} passed, {nbad} failed.\n\n")
    for r in RES:
        if not r["ok"]: f.write(f"- **FAILED** {r['name']}: {r['detail']}\n")
    f.write("\n## All checks\n\n")
    for r in RES: f.write(f"- {'ok' if r['ok'] else 'FAILED'}: {r['name']}" + (f" ({r['detail']})" if r["detail"] and not r["ok"] else "") + "\n")
print(f"{len(RES)} checks, {nok} passed, {nbad} failed")
for r in RES:
    if not r["ok"]: print("FAILED", r["name"], r["detail"])
sys.exit(1 if nbad else 0)
