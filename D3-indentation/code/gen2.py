"""D3 round 2: targeted generated programs (CHARTER-2.md, Material). Written 8 Nov 2026 before its first run.
Each program has a function at level 0 with a chain of D nested blocks (D in 2..4; kinds if/for/while/with),
so that at the end of the innermost block D + 1 shallower levels are open (0, w, ..., D*w). The innermost
block's last line is either a plain statement (an unmarked dedent follows) or a terminal statement
(return/pass/raise, and break/continue inside a loop). The next line (the target line) returns to a level
drawn uniformly from the open shallower levels and holds a plain statement; one more plain line at that level
ends the program. A short helper function may come first. Width 4 or 2.
For terminal programs, text_neutral is the same program with the terminal line replaced by
"<indent>value = value" (round 1's neutral line), for return/assignment pair patching.
Fields: seed, width, D, kinds, last ("plain" or "terminal"), term_kw, levels (open shallower levels),
target (true indent of the target line), target_line (1-based line number), text, text_neutral (or null).
Split: part = int(sha256("D3R2-gen2-" + str(seed)), 16) % 2 -> 0 explore, 1 held.
Usage: python gen2.py [n]   (default 3000 seeds; writes data/r2gen2/part-{explore,held}.jsonl)"""
import ast, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_programs import NAMES, FUNCS, ERRS

OUT = Path(__file__).resolve().parent.parent / "data" / "r2gen2"


def plain(r):
    a, b = r.choice(NAMES), r.choice(NAMES)
    return r.choice([f"{a} = {r.choice(FUNCS)}({b})", f"{a} += {r.randint(1, 9)}", f"{a}.append({b})",
                     f"{a} = {b} + {r.randint(1, 9)}", f"{a} = {b}[{r.randint(0, 3)}]", f"{r.choice(FUNCS)}({a}, {b})",
                     f"{a} = len({b})", f"{a} = {b} * {r.randint(2, 5)}"])


def header(r, kind):
    a, b = r.choice(NAMES), r.choice(NAMES)
    return {"if": f"if {a} > {r.randint(0, 9)}:", "for": f"for {a} in {b}:", "while": f"while {a} < {r.randint(2, 50)}:",
            "with": f"with {r.choice(FUNCS)}({a}) as {b}:"}[kind]


def program(seed):
    r = random.Random(seed * 7 + 3); w = r.choice([4, 2]); D = r.randint(2, 4)
    kinds = [r.choice(["if", "for", "while", "with"]) for _ in range(D)]
    L = []
    if r.random() < 0.5:
        L.append(f"def {r.choice(FUNCS)}_{r.choice(NAMES)}({r.choice(NAMES)}):")
        for _ in range(r.randint(1, 2)): L.append(" " * w + plain(r))
        L.append(" " * w + f"return {r.choice(NAMES)}"); L.append(""); L.append("")
    L.append(f"def {r.choice(FUNCS)}({r.choice(NAMES)}, {r.choice(NAMES)}):")
    loop = False
    for i, k in enumerate(kinds):
        lev = w * (i + 1)
        for _ in range(r.randint(0, 2)): L.append(" " * lev + plain(r))
        L.append(" " * lev + header(r, k)); loop = loop or k in ("for", "while")
    inner = w * (D + 1)
    for _ in range(r.randint(1, 3)): L.append(" " * inner + plain(r))
    last = r.choice(["plain", "terminal"]); kw = None
    if last == "terminal":
        opts = ["return", "pass", "raise"] + (["break", "continue"] if loop else [])
        kw = r.choice(opts)
        term = {"return": f"return {r.choice(NAMES)}", "pass": "pass", "raise": f'raise {r.choice(ERRS)}("{r.choice(FUNCS)} failed")',
                "break": "break", "continue": "continue"}[kw]
        L.append(" " * inner + term)
    levels = [w * i for i in range(D + 1)]
    target = r.choice(levels)
    tl = len(L) + 1
    for _ in range(2): L.append(" " * target + (plain(r) if target else f"{r.choice(NAMES)} = {r.choice(FUNCS)}({r.choice(NAMES)})"))
    text = "\n".join(L) + "\n"
    ast.parse(text)
    neutral = None
    if last == "terminal":
        N = list(L); N[tl - 2] = " " * inner + "value = value"; neutral = "\n".join(N) + "\n"; ast.parse(neutral)
    return dict(seed=seed, width=w, D=D, kinds=kinds, last=last, term_kw=kw, levels=levels, target=target,
                target_line=tl, text=text, text_neutral=neutral)


def main(n):
    OUT.mkdir(parents=True, exist_ok=True); parts = {"explore": [], "held": []}
    for seed in range(n):
        p = int(hashlib.sha256(f"D3R2-gen2-{seed}".encode()).hexdigest(), 16) % 2
        parts[["explore", "held"][p]].append(program(seed))
    for k, v in parts.items():
        with open(OUT / f"part-{k}.jsonl", "w") as fh:
            for x in v: fh.write(json.dumps(x) + "\n")
    print({k: len(v) for k, v in parts.items()})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
