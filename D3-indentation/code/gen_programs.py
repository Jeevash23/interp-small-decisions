"""D3 generated Python programs (CHARTER.md, Material). Seeded, valid (ast.parse), with controlled indent
width (2, 3, 4, or mixed: each nesting level draws its own width from 2-6 and 8, so that some multiples of
the unit are never opened), block types, terminal statements ending blocks, blank lines, comments, and
distractors (a terminal keyword inside a string or a comment).
Split, fixed before generation: part = sha256("D3-gen-" + str(seed)) % 2 -> 0 explore, 1 conf.
Usage: python gen_programs.py [n_programs]   writes data/gen/part-{explore,conf}.jsonl
"""
import ast, hashlib, json, random, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "gen"
NAMES = ["items", "total", "result", "value", "count", "name", "data", "config", "path", "line", "key", "node",
         "record", "score", "limit", "offset", "buffer", "index", "label", "status", "parts", "row", "token",
         "entry", "level", "size", "batch", "target", "source", "weight"]
FUNCS = ["load", "parse", "update", "check", "build", "merge", "send", "fetch", "render", "split", "scale",
         "clean", "store", "apply", "resolve", "convert", "collect", "format", "compute", "validate"]
CLASSES = ["Parser", "Cache", "Worker", "Report", "Loader", "Session", "Builder", "Queue", "Store", "Tracker"]
ERRS = ["ValueError", "KeyError", "RuntimeError", "TypeError"]
DISTRACT = ['msg = "return early if empty"', 'note = "pass the value on"', '# return here once done',
            'label = "break"', '# continue with the next item', 'hint = "raise an error"',
            'text = "skip and continue"', '# pass through unchanged']


class G:
    def __init__(self, seed, width):
        self.r = random.Random(seed); self.width = width; self.lines = []

    def unit(self, depth):
        if self.width == "mixed": return self.levels[depth]
        return self.width

    def n(self): return self.r.choice(NAMES)
    def f(self): return self.r.choice(FUNCS)

    def simple(self, ctx):
        r = self.r; a, b = self.n(), self.n()
        opts = [f"{a} = {self.f()}({b})", f"{a} += {r.randint(1, 9)}", f"{a}.append({b})", f"{a} = {b} + {r.randint(1, 9)}",
                f"{a} = {b}[{r.randint(0, 3)}]", f"{self.f()}({a}, {b})", f'{a} = "{self.f()}"', f"{a} = len({b})",
                f"{a} = {b} * {r.randint(2, 5)}", f"{a}, {b} = {b}, {a}"]
        if ctx.get("cls"): opts += [f"self.{a} = {b}", f"self.{a} = self.{self.f()}({b})"]
        if r.random() < 0.12: return r.choice(DISTRACT)
        return r.choice(opts)

    def terminal(self, ctx):
        r = self.r; opts = ["pass", f'raise {r.choice(ERRS)}("{self.f()} failed")']
        if ctx.get("func"): opts += [f"return {self.n()}", "return", f"return {self.n()} + {r.randint(1, 9)}"] * 2
        if ctx.get("loop"): opts += ["break", "continue"] * 2
        return r.choice(opts)

    def emit(self, ind, s): self.lines.append(" " * ind + s)

    def body(self, ind, depth, ctx, nmin=1, nmax=4):
        r = self.r; k = r.randint(nmin, nmax)
        for i in range(k):
            last = i == k - 1
            if depth < self.maxdepth and r.random() < 0.38:
                self.block(ind, depth, ctx)
            else:
                self.emit(ind, self.simple(ctx))
            if not last and r.random() < 0.08: self.lines.append("")
        if r.random() < 0.4: self.emit(ind, self.terminal(ctx))

    def block(self, ind, depth, ctx):
        r = self.r; u = self.unit(depth + 1); inner = ind + u
        kind = r.choice(["if", "if", "for", "while", "try", "with", "ifelse", "def"] if not ctx.get("cls") or ctx.get("func")
                        else ["if", "for", "ifelse"])
        if kind == "if":
            self.emit(ind, f"if {self.n()} {r.choice(['>', '<', '==', 'is not'])} {r.choice(['None', '0', self.n()])}:")
            self.body(inner, depth + 1, ctx)
        elif kind == "ifelse":
            self.emit(ind, f"if {self.n()}:"); self.body(inner, depth + 1, ctx)
            if r.random() < 0.5:
                self.emit(ind, f"elif {self.n()} in {self.n()}:"); self.body(inner, depth + 1, ctx)
            self.emit(ind, "else:"); self.body(inner, depth + 1, ctx)
        elif kind in ("for", "while"):
            self.emit(ind, f"for {self.n()} in {self.n()}:" if kind == "for" else f"while {self.n()} < {self.n()}:")
            self.body(inner, depth + 1, dict(ctx, loop=True))
        elif kind == "try":
            self.emit(ind, "try:"); self.body(inner, depth + 1, ctx, 1, 3)
            self.emit(ind, f"except {r.choice(ERRS)}:"); self.body(inner, depth + 1, ctx, 1, 2)
            if r.random() < 0.3:
                self.emit(ind, "finally:"); self.body(inner, depth + 1, ctx, 1, 2)
        elif kind == "with":
            self.emit(ind, f"with open({self.n()}) as {self.n()}:"); self.body(inner, depth + 1, ctx)
        elif kind == "def":
            self.emit(ind, f"def {self.f()}_{self.n()}({self.n()}):"); self.body(inner, depth + 1, dict(ctx, func=True, loop=False))

    def program(self):
        r = self.r
        self.maxdepth = r.choice([2, 3, 3, 4, 4, 5])
        if self.width == "mixed": self.levels = [0] + [r.choice([2, 3, 4, 5, 6, 8]) for _ in range(8)]
        units = r.randint(2, 4)
        for i in range(units):
            if i: self.lines += ["", ""] if r.random() < 0.7 else [""]
            if r.random() < 0.3:
                u = self.unit(1); c = r.choice(CLASSES)
                self.emit(0, f"class {c}:")
                for j in range(r.randint(1, 3)):
                    if j and r.random() < 0.8: self.lines.append("")
                    self.emit(u, f"def {self.f()}(self, {self.n()}):")
                    self.body(u + self.unit(2), 2, dict(cls=True, func=True))
            else:
                self.emit(0, f"def {self.f()}_{self.n()}({self.n()}, {self.n()}):")
                self.body(self.unit(1), 1, dict(func=True))
        return "\n".join(self.lines) + "\n"


def main(n=4000):
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {"explore": [], "conf": []}
    for seed in range(n):
        width = [2, 3, 4, "mixed"][seed % 4]
        for attempt in range(20):
            src = G(seed * 100 + attempt, width).program()
            if not (400 <= len(src) <= 1400): continue
            try: ast.parse(src)
            except SyntaxError: continue
            break
        else:
            continue
        p = int(hashlib.sha256(f"D3-gen-{seed}".encode()).hexdigest(), 16) % 2
        parts[["explore", "conf"][p]].append(dict(seed=seed, attempt=attempt, width=width, text=src))
    for k, v in parts.items():
        with open(OUT / f"part-{k}.jsonl", "w") as fh:
            for x in v: fh.write(json.dumps(x) + "\n")
    print({k: len(v) for k, v in parts.items()})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
