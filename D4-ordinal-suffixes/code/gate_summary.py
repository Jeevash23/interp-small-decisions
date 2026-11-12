"""Summarise gate01 outputs (no model weights; tokenisers only). Exploratory.
Usage: gate_summary.py <model> [<model> ...]  -> outputs/gate_summary_<model>.json and a printed table."""
import sys, json
from collections import defaultdict
from common import *
from transformers import AutoTokenizer


def split_pattern(tok, prefix, n):
    _, pieces = number_pieces(tok, prefix, n)
    return "+".join(str(len(p)) for p in pieces)


def ending(n):
    if n % 100 in (11, 12, 13):
        return "teen"
    return {1: "1", 2: "2", 3: "3"}.get(n % 10, "th")


def summarise(name):
    tok = AutoTokenizer.from_pretrained(MODELS[name][0]) if not name.startswith("gemma") else \
        AutoTokenizer.from_pretrained(MODELS[name][0])
    rows = [json.loads(l) for l in open(OUT / f"gate01_{name}.jsonl")]
    frames = {f["fid"]: f["text"] for f in explore_frames()}
    G = defaultdict(list)
    for r in rows:
        n, lg = r["n"], np.array(r["logits"])
        choice = SUFFIXES[int(np.argmax(lg))]
        ok = choice == suffix(n)
        p4 = float(np.exp(np.logaddexp.reduce(lg) - r["lse"]))
        e = ending(n); d = len(str(n))
        rec = (ok, p4, choice)
        G["all"].append(rec)
        if r["family"] == "dates" and n <= 31:
            G["G1_dates_1_31"].append(rec)
        if r["family"] != "dates" and 32 <= n <= 999 and e in ("1", "2", "3"):
            G["G2_32_999_end123"].append(rec)
        G[f"ending={e}|digits={d}"].append(rec)
        G[f"family={r['family']}"].append(rec)
        if d >= 3:
            pat = split_pattern(tok, frames[r["fid"]].replace("{N}", str(n)), n)
            G[f"split={pat}|ending={'teen' if e == 'teen' else 'other'}"].append(rec)
        if e == "teen":
            G[f"teen_choice|digits={d}|{choice}"].append(rec)
    out = {}
    for k, v in sorted(G.items()):
        out[k] = dict(n=len(v), acc=float(np.mean([x[0] for x in v])), p_four=float(np.mean([x[1] for x in v])))
    g1, g2 = out["G1_dates_1_31"]["acc"], out["G2_32_999_end123"]["acc"]
    out["_gate"] = dict(G1=g1, G2=g2, G1_pass=g1 >= 0.90, G2_pass=g2 >= 0.80, qualifies=(g1 >= 0.90 and g2 >= 0.80))
    save_json(out, f"gate_summary_{name}.json")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        o = summarise(name)
        print(f"== {name}: G1 {o['_gate']['G1']:.3f}  G2 {o['_gate']['G2']:.3f}  qualifies {o['_gate']['qualifies']}")
        for k, v in o.items():
            if k.startswith("_") or k.startswith("teen_choice"):
                continue
            print(f"   {k:40s} n={v['n']:5d} acc={v['acc']:.3f} p4={v['p_four']:.3f}")
        print("   teen choices:", {k.split("|", 1)[1]: v["n"] for k, v in o.items() if k.startswith("teen_choice")})
