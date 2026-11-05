#!/usr/bin/env python
"""CPU check of calibration/evaluation document overlap in the cached token data (exact and first-256-token matches)."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import CACHE, SUMMARY  # noqa: E402
import glob, hashlib, json, os
import torch
C = CACHE
rep = {}
for f in sorted(glob.glob(f"{C}/data_s*_c*_e*.pt") + glob.glob(f"{C}/*/data_s*_c*_e*.pt")):
    d = torch.load(f)
    h = lambda t, n=None: hashlib.sha1(t[:n].numpy().tobytes()).hexdigest()
    ca, ev = d["calib"], d["eval"]
    hc, he = {h(x) for x in ca}, {h(x) for x in ev}
    pc, pe = {h(x, 256) for x in ca}, {h(x, 256) for x in ev}
    rep[os.path.relpath(f, C)] = {"keys": list(d.keys()), "calib": list(ca.shape), "eval": list(ev.shape), "exact_overlap": len(hc & he),
                                  "first256_overlap": len(pc & pe), "calib_unique": len(hc), "calib_unique_first256": len(pc), "eval_unique": len(he)}
for k, v in rep.items():
    print(k, v)
json.dump(rep, open(os.path.join(SUMMARY, "data_overlap.json"), "w"), indent=1)
