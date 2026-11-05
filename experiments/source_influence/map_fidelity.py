#!/usr/bin/env python
"""FineWeb short-prefix fidelity (NLL retained vs the no-prefix floor) of the chat maps, all directions."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import RAW  # noqa: E402
import glob, json, sys
R = RAW + "/"
SRC = [(sorted(glob.glob(R + f"phase3_*_{tag}_chatcal.json"))[-1].split("/")[-1], pair) for tag, pair in (("9b27b", ("9B", "27B")), ("4b9b", ("4B", "9B")), ("4b27b", ("4B", "27B")))]
out = {}
for f, pair in SRC:
    for k, v in json.load(open(R + f))["results"].items():
        s, t = k.split(">"); lab = {"small": pair[0], "large": pair[1]}
        c = v["configs"]; nat, flo = c["native"]["nll"], c["no_prefix"]["nll"]
        ret = lambda x: 100 * (flo - c[x]["nll"]) / (flo - nat)
        out[f"{lab[s]}>{lab[t]}"] = {"file": f, **{x: round(ret(x), 1) for x in ("attn_mapped", "delta_best", "combined_best", "combined_best_tail64") if x in c},
                                     "combined_kl": round(c["combined_best"]["kl"], 4), "combined_top1": round(c["combined_best"]["top1"], 3)}
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
