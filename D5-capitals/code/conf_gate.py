"""CONFIRMATION COPY (PROTOCOL.md): identical to gate.py except that it reads the held-out `conf` part
and writes conf_* outputs. The case directions are still read from the exploration means (exp01, explore).
"""
"""D5 competence gate (CHARTER.md, "Gate"). Exploratory part only.
Usage: gate.py <model short name> [n_paragraphs=150]
Teacher-forced top-1 at every position for the upper, lower, title and original versions of each
paragraph (conditions 1 and 3 and their controls), then greedy one-word decoding at four word starts
per paragraph (chosen with seed 1, the same words for every model) in the upper, lower and original
versions (condition 2 and its ceiling). Writes outputs/gate_<model>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rows = load_part("conf", seed=0)[:N]
rng = random.Random(1)
picks = []
for r in rows:
    t = cut(r["text"]); rs = runs(t)
    cand = [i for i, (s, e) in enumerate(rs) if i >= 5 and s > 0 and t[s - 1] == " "]
    picks.append(sorted(rng.sample(cand, min(4, len(cand)))))

t0 = time.time()
M_ = Model(short); vs = M_.vs
tf = {v: {"ws_case": {}, "ws_n": 0, "ws_first_upper": 0, "ws_cased": 0, "iw_n": 0, "iw_ok": 0, "iw_strict": 0}
      for v in ("upper", "lower", "title", "orig")}
dec = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); V = versions(text)
    enc = {}
    for v, s in V.items():
        ids, first = encode(M_.tok, s); enc[v] = (ids, first)
        top = M_.top1(ids); ws, iw = positions(s, first); d = tf[v]
        for k in ws:
            c = case_of(vs[top[k - 1]]); d["ws_case"][c] = d["ws_case"].get(c, 0) + 1; d["ws_n"] += 1
            fu = first_cased_upper(vs[top[k - 1]])
            if fu is not None: d["ws_cased"] += 1; d["ws_first_upper"] += int(fu)
        for k, (i, rest) in iw.items():
            s1 = vs[top[k - 1]]; d["iw_n"] += 1
            d["iw_ok"] += int(len(s1) > 0 and rest.startswith(s1)); d["iw_strict"] += int(top[k - 1] == enc[v][0][k])
    rs = runs(text)
    for i in picks[pi]:
        c = rs[i][0]; rec = {"para": pi, "pageid": r["pageid"], "run": i, "truth": text[rs[i][0]:rs[i][1]]}
        okall = True
        for v in ("upper", "lower", "orig"):
            ids, first = enc[v]
            if c not in first: okall = False; break
            k = first.index(c)
            w, f = M_.greedy_word(ids[:k]); rec[v] = w; rec[v + "_first"] = f
        if okall: dec.append(rec)
    if pi % 25 == 0: print(pi, round(time.time() - t0), flush=True)

res = {"model": short, "n_paragraphs": len(rows), "teacher_forced": tf}
for v, d in tf.items():
    cased = d["ws_n"] - d["ws_case"].get("caseless", 0)
    d["share_upper_of_cased"] = d["ws_case"].get("upper", 0) / max(cased, 1)
    d["share_lower_of_cased"] = d["ws_case"].get("lower", 0) / max(cased, 1)
    d["share_first_upper_of_cased"] = d["ws_first_upper"] / max(d["ws_cased"], 1)
    d["iw_ok_rate"] = d["iw_ok"] / max(d["iw_n"], 1); d["iw_strict_rate"] = d["iw_strict"] / max(d["iw_n"], 1)
alpha = [x for x in dec if x["lower"]]
res["decode"] = {"n_positions": len(dec), "n_lower_alpha": len(alpha),
                 "upper_matches_lower": sum(x["upper"] == x["lower"].upper() for x in alpha) / max(len(alpha), 1),
                 "orig_matches_lower": sum(x["orig"].lower() == x["lower"] for x in alpha) / max(len(alpha), 1),
                 "upper_is_upper_alpha": sum(bool(x["upper"]) and x["upper"].isupper() for x in alpha) / max(len(alpha), 1),
                 "upper_lc_matches_lower": sum(x["upper"].lower() == x["lower"] for x in alpha) / max(len(alpha), 1)}
u = tf["upper"]
res["gate"] = {"c1_case_kept": u["share_upper_of_cased"], "c1_pass": u["share_upper_of_cased"] >= 0.95,
               "c2_same_word": res["decode"]["upper_matches_lower"], "c2_pass": res["decode"]["upper_matches_lower"] >= 0.60,
               "c3_inside_word": u["iw_ok_rate"], "c3_pass": u["iw_ok_rate"] >= 0.70}
res["gate"]["pass"] = res["gate"]["c1_pass"] and res["gate"]["c2_pass"] and res["gate"]["c3_pass"]
res["records"] = dec; res["seconds"] = round(time.time() - t0)
save(res, f"conf_gate_{short}.json")
print(json.dumps(res["gate"]), json.dumps(res["decode"]))
