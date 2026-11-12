"""D5 round 2 (exploratory; CHARTER-2.md Q3): why does adding d change the word? Re-reads round-1 records
(exp02 and conf_exp02, a = 1 at the direction layer); loads tokenisers only. For lower-run positions with
a lower-case top-1 where adding d gives an upper-case top-1: kept (same-word rule) or changed; whether the
edited top-1 equals the upper-case passage's own top-1; and the number of pieces of the lower-case top-1's
word written in capitals, against kept positions."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
OUT = Path(__file__).resolve().parent.parent / "outputs"
REPO = {"gemma": "google/gemma-4-E2B", "qwen3-17b": "Qwen/Qwen3-1.7B-Base"}; LAY = {"gemma": "24", "qwen3-17b": "22"}

def case_of(s):
    up = any(c.isupper() for c in s); lo = any(c.islower() for c in s)
    return "caseless" if not (up or lo) else "upper" if up and not lo else "lower" if lo and not up else "mixed"
def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))

for mdl in ("gemma", "qwen3-17b"):
    tok = AutoTokenizer.from_pretrained(REPO[mdl]); l = LAY[mdl]
    npieces = lambda w: len(tok(" " + w, add_special_tokens=False)["input_ids"])
    rows = []
    for f in (f"exp02_{mdl}.json", f"conf_exp02_{mdl}.json"):
        for rc in json.load(open(OUT / f))["records"]:
            lo = rc["lower"][l]; up = rc["upper"][l]
            if case_of(lo[0]) != "lower" or case_of(lo[1]) != "upper": continue
            w = lo[0].strip()
            rows.append({"src": f[:4], "lo": lo[0], "edit": lo[1], "uptop": up[0], "kept": same_word(lo[0], lo[1]),
                         "eq_up": lo[1].strip() == up[0].strip(), "up_kept": same_word(lo[0], up[0]),
                         "pl": npieces(w), "pu": npieces(w.upper()), "alpha": w.isalpha(), "len": len(w)})
    k = [r for r in rows if r["kept"]]; c = [r for r in rows if not r["kept"]]
    print(f"\n{mdl}: flipped {len(rows)}, kept {len(k)}, changed {len(c)} ({len(c)/len(rows):.3f})")
    print(" changed: edited top-1 = upper passage's own top-1: %.3f; upper passage also changed the word: %.3f" % (
        np.mean([r["eq_up"] for r in c]), np.mean([not r["up_kept"] for r in c])))
    print(" kept: upper passage kept the word too: %.3f" % np.mean([r["up_kept"] for r in k]))
    for name, g in (("kept", k), ("changed", c)):
        print(f" {name}: extra pieces in capitals mean %.2f; lower word non-alpha %.3f; len mean %.1f; pieces in caps: %s" % (
            np.mean([r["pu"] - r["pl"] for r in g]), np.mean([not r["alpha"] for r in g]), np.mean([r["len"] for r in g]),
            dict(sorted(Counter(min(r["pu"], 4) for r in g).items()))))
    # change rate by number of extra pieces
    for e in range(0, 4):
        g = [r for r in rows if min(r["pu"] - r["pl"], 3) == e]
        if g: print(f"  extra pieces {e}{'+' if e == 3 else ''}: n {len(g)}, changed {np.mean([not r['kept'] for r in g]):.3f}, upper passage changed {np.mean([not r['up_kept'] for r in g]):.3f}")
    print(" changed examples:", [(r["lo"], r["edit"], r["uptop"]) for r in c[:25]])
