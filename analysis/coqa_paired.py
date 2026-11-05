#!/usr/bin/env python
"""CPU: paired per-conversation F1 gaps to native on the saved CoQA outputs, bootstrapped over conversations."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW, SUMMARY  # noqa: E402
import glob, json, os
import numpy as np
R = f"{RAW}/extra/coqa"
out = {}
for d in ("4Bto9B", "9Bto4B", "9Bto27B", "27Bto9B"):
    rows = {}
    for f in sorted(glob.glob(f"{R}/coqa_{d}_sh*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            assert r["doc"] not in rows
            rows[r["doc"]] = r
    rows = [rows[i] for i in sorted(rows)]
    assert len(rows) == 100
    conv = lambda c: np.array([np.mean([t[c]["f1"] for t in r["turns"]]) * 100 for r in rows])
    nat = conv("native")
    rng = np.random.default_rng(20261103)
    idx = rng.integers(0, 100, (10000, 100))
    out[d] = {"native": float(nat.mean())}
    for c in ("once_web", "each_web", "each_copy_web"):
        g = conv(c) - nat
        out[d][c] = [float(g.mean()), *[float(x) for x in np.percentile(g[idx].mean(1), [2.5, 97.5])]]
    print(d, {k: (round(v, 1) if isinstance(v, float) else [round(x, 1) for x in v]) for k, v in out[d].items()})
json.dump(out, open(os.path.join(SUMMARY, "coqa_paired.json"), "w"), indent=1)
