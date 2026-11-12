"""Exploratory: where teen errors fall, by hundreds digit / leading digits and family. No model weights."""
import sys, json
from collections import defaultdict
from common import *
for name in sys.argv[1:]:
    rows=[json.loads(l) for l in open(OUT/f"gate01_{name}.jsonl")]
    by=defaultdict(list); fam=defaultdict(list); lastd=defaultdict(list)
    for r in rows:
        n=r["n"]
        if n%100 not in (11,12,13) or n<100: continue
        ok=SUFFIXES[int(np.argmax(r["logits"]))]=="th"
        by[(len(str(n)), (n//100)%10)].append(ok); fam[(len(str(n)),r["family"])].append(ok); lastd[(len(str(n)),n%10)].append(ok)
    print("==",name)
    for k in sorted(by): print("  digits,hundreds",k,len(by[k]),round(np.mean(by[k]),3))
    for k in sorted(fam): print("  digits,family",k,len(fam[k]),round(np.mean(fam[k]),3))
    for k in sorted(lastd): print("  digits,last",k,len(lastd[k]),round(np.mean(lastd[k]),3))
