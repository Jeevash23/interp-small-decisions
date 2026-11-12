"""D3 round 2: check that REPORT-2.md cites the independently recomputed values (checker code).

  python3 -B code/r2_check_report.py [--values outputs/r2/independent_values.json] [--report REPORT-2.md]

Reads the flat key -> string file written by code/r2_independent.py (and its .meta.json) and REPORT-2.md.
  1. REQUIRED below is a list of (key, required) pairs; extend it as the report grows. A required key whose exact
     string (as in independent_values.json, e.g. "−0.135", "+2.69", "2,712") does not occur in REPORT-2.md fails.
     A string counts as present only when it is not part of a longer number.
  2. Every other key of independent_values.json whose exact string occurs in REPORT-2.md is listed as found
     (information only: short strings can occur by chance).
  3. The verdict table is re-derived from the PROTOCOL-2.md thresholds, using the value strings in the file
     (the rounded values), and compared with the verdicts computed from the unrounded values (c2.v.* in the same
     file); a difference marks a case on a rounding boundary. Each prediction's verdict word (pass, fail, not
     evaluable) must then appear in its REPORT-2.md table row ("| Q1 |" or "| **Q1** |"), and each question's
     verdict (confirmed / not confirmed) on a line naming the question.
If REPORT-2.md does not exist yet, it says so and exits 0. Exits 1 if anything required is missing or disagrees.
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_independent as R2I  # noqa: E402  (checker code)

D3 = R2I.D3
# (key, required). Defaults: the value behind each of the 25 predictions and the material counts.
REQUIRED = [(k, True) for k in [
    "c2.q.n.chunks", "c2.q.n.repos", "c2.qg.n.programs",
    "c2.q.att_inner_ded", "c2.q.att_inner_same", "c2.q.ko_inner_ded", "c2.q.others_lo", "c2.q.others_hi",
    "c2.q.ko_inner_same", "c2.q.ko_inner_term", "c2.q.ko_closed_ded", "c2.q.ko_inner_ded_closedsub",
    "c2.q.pick_not_inner", "c2.q.pick_all_multi", "c2.q.innermost_rule_all_multi",
    "c2.q.steer_dedent_head_dlq", "c2.q.steer_dedent_ctrl_dlq", "c2.q.steer_dedent_head_top", "c2.q.steer_dedent_ctrl_top",
    "c2.q.ko_true_ded",
    "c2.qg.steer_term_head_dlq", "c2.qg.steer_term_ctrl_dlq", "c2.qg.steer_term_head_top", "c2.qg.steer_term_ctrl_top",
    "c2.qg.share_blk_kw_return", "c2.qg.share_layer_return_max", "c2.qg.l0h2_line_dqcur_term",
    "c2.o.att_inner_ded", "c2.o.att_inner_same", "c2.o.ko_inner_ded", "c2.o.others_lo", "c2.o.others_hi",
    "c2.o.ko_inner_same", "c2.o.ko_inner_term", "c2.o.ko_closed_ded", "c2.o.ko_inner_ded_closedsub",
    "c2.o.pick_not_inner", "c2.o.pick_inner_attended_not_inner",
    "c2.o.steer_dedent_head_dlq", "c2.o.steer_dedent_ctrl_dlq", "c2.o.steer_dedent_head_top", "c2.o.steer_dedent_ctrl_top",
    "c2.o.ko_true_ded",
    "c2.og.steer_term_head_dlq", "c2.og.steer_term_ctrl_dlq", "c2.og.steer_term_head_top", "c2.og.steer_term_ctrl_top",
    "c2.og.share_blk_kw_return", "c2.og.share_layer_return_max",
    "c2.g.d_lq_dedent", "c2.g.rand_d_lq_dedent_min", "c2.g.rand_d_lq_dedent_max", "c2.g.d_lq_term", "c2.g.p_open_term_ablated",
    "c2.g.d_lq_same", "c2.g.d_psame_dedent",
]]


def present(s, text):
    return re.search(r"(?<![\d.,])" + re.escape(s) + r"(?![\d])", text) is not None


def raw_from_strings(V, meta):
    """Value strings -> floats, grouped as r2_independent.verdicts expects (rounded values)."""
    f = lambda x: float(x.replace(R2I.MINUS, "-").replace(",", "").replace("+", ""))
    blk = {}
    for t in ("q", "qg", "o", "og", "g"):
        d = {}
        for k, v in V.items():
            if k.startswith(f"c2.{t}.") and not k.startswith(f"c2.{t}.ci."):
                try: d[k[len(f"c2.{t}."):]] = f(v)
                except ValueError: pass
        blk[t] = d or None
    if blk["g"] is not None: blk["g"]["n.rand"] = meta.get("consistency", {}).get("r2c03 random groups", 0)
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default=str(R2I.OUT2 / "independent_values.json"))
    ap.add_argument("--report", default=str(D3 / "REPORT-2.md"))
    a = ap.parse_args()
    vp = Path(a.values); V = json.loads(vp.read_text())
    mp = vp.with_suffix(".meta.json"); meta = json.loads(mp.read_text()) if mp.exists() else {}
    print(f"values: {vp} (records: {meta.get('records', 'unknown')})")
    if meta.get("records") != "conf":
        print("note: these values are not from the confirmation records; REPORT-2 should cite conf values")
    fails = []
    # verdict table from thresholds, on the rounded values
    B = raw_from_strings(V, meta)
    D = R2I.verdicts(B["q"], B["qg"], B["o"], B["og"], B["g"])
    rows = []
    for i in R2I.PRED_IDS + list(R2I.QUESTIONS):
        fromfile = V.get(f"c2.v.{i}"); der = D[i][0]
        if fromfile is not None and fromfile != der:
            fails.append(f"{i}: verdict from rounded values ({der}) differs from unrounded ({fromfile}): rounding boundary")
        rows.append((i, der, fromfile))
    print("\nverdict table re-derived from the PROTOCOL-2.md thresholds (rounded values | unrounded values):")
    for i, der, ff in rows: print(f"  {i:<11} {der:<14} | {ff}")

    rp = Path(a.report)
    if not rp.exists():
        print(f"\n{rp.name} not yet written: nothing to check against.")
        if fails: print("\n".join("FAIL " + x for x in fails))
        sys.exit(1 if fails else 0)
    text = rp.read_text()
    req = dict(REQUIRED)
    missing = [k for k, r in REQUIRED if r and (k not in V or not present(V[k], text))]
    for k in missing:
        hint = ""
        if k in V and V[k].startswith("+") and present(V[k][1:], text): hint = " (found without its + sign)"
        fails.append(f"required value {k} = {V.get(k)} not in {rp.name}{hint}")
    found = sorted(k for k, v in V.items() if not k.startswith("c2.v.") and present(v, text))
    print(f"\nrequired values: {sum(1 for _, r in REQUIRED if r) - len(missing)} of {sum(1 for _, r in REQUIRED if r)} present")
    print(f"values of independent_values.json found in {rp.name}: {len(found)} of {len(V)}"
          f" ({len([k for k in found if k not in req])} beyond the required list)")
    # verdict words in the report's table rows and question lines
    lines = text.splitlines()
    for i, der, ff in rows:
        want = ff if ff is not None else der
        if i in R2I.QUESTIONS:
            lab = i.replace("question", "")
            ql = [l for l in lines if re.search(rf"\bQuestion {lab}\b", l, re.I)]
            if not ql: fails.append(f"question {lab}: no line naming it in {rp.name}"); continue
            hit = any(re.search(r"\bnot confirmed\b", l, re.I) for l in ql) if want == "not confirmed" else \
                any(re.search(r"(?<!not )\bconfirmed\b", l, re.I) for l in ql)
            if not hit: fails.append(f"question {lab}: verdict '{want}' not stated on its line in {rp.name}")
            continue
        row = next((l for l in lines if re.match(rf"^\|\s*(\*\*)?{i}(\*\*)?\s*\|", l)), None)
        if row is None: fails.append(f"{i}: no table row in {rp.name}"); continue
        cells = [c.strip().strip("*").lower() for c in row.strip().strip("|").split("|")]
        if want not in cells and not re.search(rf"\b{want}\b", cells[-1]):
            fails.append(f"{i}: verdict '{want}' not in its row: {row[:100]}")
    print()
    for x in fails: print("FAIL", x)
    print(f"{len(fails)} problem(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
