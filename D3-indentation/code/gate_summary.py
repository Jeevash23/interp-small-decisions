"""D3 gate (CHARTER.md, Gate): summarise e01 outputs. Reads outputs/e01_<model>_<source>.jsonl for the
exploration parts and writes outputs/gate_summary.json and a Markdown table to stdout.
Items: (1) header top-1 >= 0.90; (2) terminal-statement points: mean shallower-open mass >= 0.7 and top-1
>= 0.5; (3) overall top-1 - copy-baseline top-1 >= 0.10 (all on newcode); open-level part: on generated
2-space, 3-space and mixed code, mean not-open shallower mass at dedent points <= 0.10.
Usage: python gate_summary.py model [model ...]"""
import sys, json
import numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "outputs"


def rows(model, source):
    p = OUT / f"e01_{model}_{source}.jsonl"
    return [json.loads(l) for l in open(p) if '"mass"' in l] if p.exists() else []


def stats(xs):
    if not xs: return None
    a = lambda k, f=None: float(np.mean([x[k] for x in xs if (f is None or f(x))])) if any(f is None or f(x) for x in xs) else None
    n = lambda f: sum(1 for x in xs if f(x))
    hd, tm, sh = (lambda x: x["header"]), (lambda x: x["terminal"]), (lambda x: x["cls"] == "shallower")
    return dict(n=len(xs), n_header=n(hd), n_terminal=n(tm), n_dedent=n(sh), other=a("other"),
                top1=a("top1"), copy=a("copy_ok"), header_top1=a("top1", hd), terminal_top1=a("top1", tm),
                terminal_shopen=a("p_shallow_open", tm), terminal_shnotopen=a("p_shallow_notopen", tm),
                same_top1=a("top1", lambda x: x["cls"] == "same"),
                dedent_top1=a("top1", sh), dedent_shopen=a("p_shallow_open", sh), dedent_shnotopen=a("p_shallow_notopen", sh),
                nonterm_dedent_top1=a("top1", lambda x: sh(x) and not tm(x)))


def main(models):
    out = {}
    for mname in models:
        r = {}
        nc = rows(mname, "newcode"); r["newcode"] = stats(nc)
        g = rows(mname, "gen")
        for w in ["2", "3", "4", "mixed"]: r[f"gen_w{w}"] = stats([x for x in g if str(x["width"]) == w])
        r["stdlib"] = stats(rows(mname, "stdlib"))
        s = r["newcode"]
        if s:
            r["gate"] = dict(item1=s["header_top1"] >= 0.90, item2=(s["terminal_shopen"] >= 0.7 and s["terminal_top1"] >= 0.5),
                             item3=(s["top1"] - s["copy"]) >= 0.10)
        ol = [r[f"gen_w{w}"]["dedent_shnotopen"] for w in ["2", "3", "mixed"] if r[f"gen_w{w}"]]
        if len(ol) == 3: r.setdefault("gate", {})["open_level"] = max(ol) <= 0.10
        out[mname] = r
    (OUT / "gate_summary.json").write_text(json.dumps(out, indent=1))
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    print("| model | source | n | header top-1 | terminal top-1 | terminal shallower-open | top-1 | copy | dedent not-open | other |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mname, r in out.items():
        for src in ["newcode", "gen_w2", "gen_w3", "gen_w4", "gen_wmixed", "stdlib"]:
            s = r.get(src)
            if s: print(f"| {mname} | {src} | {s['n']} | {f(s['header_top1'])} | {f(s['terminal_top1'])} | {f(s['terminal_shopen'])} | {f(s['top1'])} | {f(s['copy'])} | {f(s['dedent_shnotopen'])} | {f(s['other'])} |")
        print(f"gate {mname}: {r.get('gate')}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
