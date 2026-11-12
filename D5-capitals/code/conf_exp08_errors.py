"""CONFIRMATION COPY (PROTOCOL.md): identical to exp08_errors.py except that it reads the held-out `conf` part
and writes conf_* outputs. The case directions are still read from the exploration means (exp01, explore).
"""
"""D5 exploration 8 (exploratory): where do capital-letter word changes come from? (charter, candidate (b))
Usage: exp08_errors.py <model>
Uses the gate's greedy words (outputs/gate_<model>.json; explore paragraphs 0-150, four word starts each).
At positions where the upper-case passage gives a different word from the lower-case passage
(upper word alphabetic and != lower word in capitals), take the lower-case passage's next-token
distribution at the same word start and find the rank of the first lower-case piece of the upper word
(" " + word.lower(), first token). If the word is chosen in a lower-case-like form and then capitalised
(route R2), the changed word should be one of the lower run's own top alternatives. Controls: another
word of the same paragraph (a letter run 20 or more words away, seed 5); and, as a reference, the same
measure for words that change between the original passage and the lower-case passage.
Writes outputs/exp08_<model>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short = sys.argv[1]
G = json.load(open(OUT / f"conf_gate_{short}.json"))
rows = load_part("conf", seed=0)[:G["n_paragraphs"]]
M_ = Model(short); m = M_.m; tok = M_.tok


def first_piece(w):
    return tok(" " + w.lower(), add_special_tokens=False)["input_ids"][0]


rng = random.Random(5); t0 = time.time(); out = []
cache = {}
for rec in G["records"]:
    lw, uw, ow = rec["lower"], rec["upper"], rec["orig"]
    if not lw or not lw.isalpha(): continue
    kinds = []
    if uw and uw.isalpha() and uw != lw.upper(): kinds.append(("upper", uw))
    if ow and ow.isalpha() and ow.lower() != lw: kinds.append(("orig", ow))
    if not kinds: continue
    pi = rec["para"]
    if pi not in cache:
        text = cut(rows[pi]["text"]); V = versions(text); cache = {pi: (text, encode(tok, V["lower"]))}
    text, (ids, first) = cache[pi]
    rs = runs(text); c = rs[rec["run"]][0]; k = first.index(c)
    with torch.no_grad(): lg = m.model(torch.tensor([ids[:k]], device=DEV), logits_to_keep=1).logits[0, -1].float()
    order = torch.argsort(lg, descending=True); rank = torch.empty_like(order); rank[order] = torch.arange(len(order), device=order.device)
    far = [j for j in range(len(rs)) if abs(j - rec["run"]) >= 20 and text[rs[j][0]:rs[j][1]].lower() not in (lw, uw.lower(), ow.lower())]
    ctrl = text[slice(*rs[rng.choice(far)])] if far else None
    for kind, w in kinds:
        out.append({"para": pi, "run": rec["run"], "kind": kind, "lower": lw, "word": w, "ctrl": ctrl,
                    "rank_word": int(rank[first_piece(w)]) + 1, "rank_ctrl": int(rank[first_piece(ctrl)]) + 1 if ctrl else None,
                    "rank_lower_first": int(rank[first_piece(lw)]) + 1})

summ = {}
for o in out: o["same_first"] = first_piece(o["word"]) == first_piece(o["lower"])
for kind in ("upper", "orig", "upper_diff_first", "orig_diff_first"):
    x = [o for o in out if o["kind"] == kind.split("_")[0] and (not kind.endswith("diff_first") or not o["same_first"])]
    xc = [o for o in x if o["rank_ctrl"] is not None]
    rw = np.array([o["rank_word"] for o in x]); rc = np.array([o["rank_ctrl"] for o in xc])
    summ[kind] = {"n": len(x), "n_ctrl": len(xc),
                  "word_top1": float(np.mean(rw <= 1)), "word_top2": float(np.mean(rw <= 2)), "word_top5": float(np.mean(rw <= 5)), "word_top20": float(np.mean(rw <= 20)),
                  "word_median_rank": float(np.median(rw)),
                  "ctrl_top2": float(np.mean(rc <= 2)), "ctrl_top5": float(np.mean(rc <= 5)), "ctrl_top20": float(np.mean(rc <= 20)),
                  "ctrl_median_rank": float(np.median(rc)),
                  "same_first_piece_as_lower": float(np.mean([o["same_first"] for o in x]))}
save({"model": short, "summary": summ, "records": out, "seconds": round(time.time() - t0)}, f"conf_exp08_{short}.json")
print(json.dumps(summ, indent=1))
