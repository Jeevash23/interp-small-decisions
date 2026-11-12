"""D5 exploration 4 (exploratory): what sets the case of the next word, the last words or the passage?
Usage: exp04_switch.py <model> [first=300] [n=100]
At up to four word starts per paragraph (seed 3), the prefix up to the word is built in mixed forms:
  U: all upper;  L: all lower;
  L+u1, L+u2, L+u4, L+u8: lower passage whose last 1, 2, 4 or 8 words (letter runs) are upper;
  U+l1, U+l2, U+l4, U+l8: upper passage whose last 1, 2, 4 or 8 words are lower;
  T: title-case passage (control).
Records the case of the top-1 next token (and P(top-1 is upper) mass: summed probability of all
upper-case tokens at the position). Writes outputs/exp04_<model>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short = sys.argv[1]; FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 300; N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rows = load_part("explore", seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok
cls = [case_of(s) for s in vs]
NV = M_.raw.get_output_embeddings().weight.shape[0]  # output width can exceed the tokeniser's (Qwen3 pads)
cls += ["caseless"] * (NV - len(cls)); vs = vs + [""] * (NV - len(vs))
UPM = torch.tensor([c == "upper" for c in cls], device=DEV); LOM = torch.tensor([c == "lower" for c in cls], device=DEV)
FORMS = ["U", "L", "L+u1", "L+u2", "L+u4", "L+u8", "U+l1", "U+l2", "U+l4", "U+l8", "T"]


def build(text, rs, i, form):
    pre = text[:rs[i][0] - 1]  # up to (not including) the space before word i
    if form == "U": return pre.upper()
    if form == "L": return pre.lower()
    if form == "T": return title(pre)
    base, sw, n = (pre.lower(), str.upper, int(form[3:])) if form.startswith("L+") else (pre.upper(), str.lower, int(form[3:]))
    cut0 = rs[max(i - n, 0)][0]
    return base[:cut0] + sw(pre[cut0:])


rng = random.Random(3); t0 = time.time(); out = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); rs = runs(text)
    cand = [i for i, (s, e) in enumerate(rs) if i >= 12 and text[s - 1] == " "]
    for i in sorted(rng.sample(cand, min(4, len(cand)))):
        rec = {"para": FIRST + pi, "run": i}
        for f in FORMS:
            ids = tok(build(text, rs, i, f), add_special_tokens=False)["input_ids"]
            with torch.no_grad(): lg = m.model(torch.tensor([ids], device=DEV), logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(lg, -1)
            rec[f] = {"top": vs[int(lg.argmax())], "case": cls[int(lg.argmax())],
                      "p_upper": round(float(p[UPM].sum()), 4), "p_lower": round(float(p[LOM].sum()), 4)}
        out.append(rec)
    if pi % 20 == 0: print(pi, round(time.time() - t0), flush=True)

summ = {}
for f in FORMS:
    cased = [x[f] for x in out if x[f]["case"] in ("upper", "lower", "mixed")]
    summ[f] = {"n": len(out), "n_cased": len(cased), "top_upper": sum(c["case"] == "upper" for c in cased) / max(len(cased), 1),
               "mean_p_upper": float(np.mean([x[f]["p_upper"] for x in out])), "mean_p_lower": float(np.mean([x[f]["p_lower"] for x in out]))}
save({"model": short, "rows": [FIRST, FIRST + N], "summary": summ, "records": out, "seconds": round(time.time() - t0)}, f"exp04_{short}.json")
for f in FORMS: print(f, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summ[f].items()}))
