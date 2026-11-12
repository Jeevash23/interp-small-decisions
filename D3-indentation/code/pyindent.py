"""D3 helpers that need no model: normalising Python source, cutting it into chunks that start at indent 0,
and listing decision points with their truth (CHARTER.md, Material and Measures). Exploratory code."""
import ast, io, re, tokenize

TERMINAL = {"return", "pass", "raise", "break", "continue"}


def normalise(src):
    """Unix newlines, trailing whitespace removed, a final newline. None if the file has a tab, a non-ASCII
    character, a form feed, or does not parse."""
    if isinstance(src, bytes):
        try: src = src.decode("utf-8")
        except UnicodeDecodeError: return None
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    if "\t" in src or "\f" in src or not src.isascii(): return None
    src = "\n".join(l.rstrip() for l in src.split("\n")).strip("\n") + "\n"
    try: ast.parse(src)
    except (SyntaxError, ValueError, MemoryError, RecursionError): return None
    return src


def _logical_starts(src):
    """Physical line numbers (1-based) that begin a logical line, with INDENT/DEDENT counts before them."""
    starts, ind, ded = {}, 0, 0
    expect = True
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.INDENT: ind += 1; continue
        if t.type == tokenize.DEDENT: ded += 1; continue
        if t.type in (tokenize.NL, tokenize.COMMENT, tokenize.ENCODING): continue
        if t.type == tokenize.NEWLINE: expect = True; continue
        if t.type == tokenize.ENDMARKER: break
        if expect:
            starts[t.start[0]] = (ind, ded, t.string); ind = ded = 0; expect = False
    return starts


def chunks(src, maxc=1400, minc=300):
    """Cut a normalised file into chunks of whole top-level units (a unit runs from one indent-0 logical
    line start to the next), packed greedily up to maxc characters. A unit longer than maxc is cut at the
    last logical-line boundary before maxc and the rest of it dropped. Chunks under minc are dropped."""
    lines = src.split("\n")[:-1]
    try: starts = _logical_starts(src)
    except (tokenize.TokenError, IndentationError, SyntaxError): return []
    tops = [n for n in sorted(starts) if not lines[n - 1].startswith(" ")]
    if not tops: return []
    # attach blank/comment lines just above a top-level start to that unit
    bounds = []
    for n in tops:
        b = n
        while b - 1 >= 1 and (lines[b - 2].strip() == "" or lines[b - 2].startswith("#")) and (b - 1) not in starts:
            b -= 1
        bounds.append(b)
    bounds[0] = 1
    units = []
    for i, b in enumerate(bounds):
        e = bounds[i + 1] - 1 if i + 1 < len(bounds) else len(lines)
        units.append((b, e))
    out, cur, cur_len = [], [], 0

    def flush():
        if cur:
            text = "\n".join(lines[cur[0][0] - 1:cur[-1][1]]).strip("\n") + "\n"
            if len(text) >= minc: out.append(text)

    for b, e in units:
        text = "\n".join(lines[b - 1:e]).strip("\n") + "\n"
        if len(text) > maxc:
            flush(); cur, cur_len = [], 0
            # cut at a logical-line start inside the unit
            acc = 0; cut = None
            for n in range(b, e + 1):
                acc += len(lines[n - 1]) + 1
                if acc > maxc: break
                if (n + 1) in starts and n + 1 <= e: cut = n
            if cut is not None:
                t2 = "\n".join(lines[b - 1:cut]).strip("\n") + "\n"
                if len(t2) >= minc and _parses_tokenwise(t2): out.append(t2)
            continue
        if cur_len + len(text) + 1 > maxc:
            flush(); cur, cur_len = [], 0
        cur.append((b, e)); cur_len += len(text) + 1
    flush()
    return [c for c in out if _parses_tokenwise(c)]


def _parses_tokenwise(text):
    try: _logical_starts(text); return True
    except (tokenize.TokenError, IndentationError, SyntaxError): return False


def decision_points(text):
    """Decision points of a chunk (CHARTER.md, Measures). Each is a dict with
      s: character offset where the line's indentation begins; e: offset just after the previous code
      line's last character (the start of the newline run); line: 1-based line number;
      true: indent of the line; prev: indent of the previous logical line's first line; stack: open
      indent levels before the line (Python's indentation stack; last item = prev unless deeper);
      cls: 'deeper' | 'same' | 'shallower'; ndedent; header: previous logical line opens a block;
      terminal: previous logical line starts with a terminal keyword; prevword; blank: blank lines between;
      word: first token of the line."""
    lines = text.split("\n")[:-1]
    starts = _logical_starts(text)
    offs = [0]
    for l in lines: offs.append(offs[-1] + len(l) + 1)
    stack = [0]; pts = []; prev_start = None; last_code_line = None
    # last physical line of each logical line: the line before the next logical start, skipping blanks/comments
    order = sorted(starts)
    for idx, n in enumerate(order):
        ind, ded, word = starts[n]
        indent = len(lines[n - 1]) - len(lines[n - 1].lstrip(" "))
        st_before = list(stack)
        if ind: stack.append(indent)
        for _ in range(ded): stack.pop()
        if prev_start is not None:
            # previous non-blank physical line
            j = n - 1
            while j >= 1 and lines[j - 1].strip() == "": j -= 1
            prevline = lines[j - 1]
            ok = not prevline.lstrip().startswith("#") and j >= prev_start
            if ok:
                pw = starts[prev_start][2]
                pind = len(lines[prev_start - 1]) - len(lines[prev_start - 1].lstrip(" "))
                cls = "deeper" if ind else ("shallower" if ded else "same")
                pts.append(dict(s=offs[n - 1], e=offs[j - 1] + len(prevline), line=n, true=indent, prev=pind,
                                stack=st_before, cls=cls, ndedent=ded, header=bool(ind),
                                terminal=pw in TERMINAL, prevword=pw, blank=n - 1 - j, word=word))
        prev_start = n
    return pts
