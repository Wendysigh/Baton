#!/usr/bin/env python
"""Component-ablation block of results/summary/ablation.json, with bootstrap 95% CIs.

NLL-retained = (NLL_noprefix - NLL_cfg) / (NLL_noprefix - NLL_native), bootstrapped over eval documents."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW, SUMMARY  # noqa: E402
import glob, json, os, random, re

import numpy as np

R = RAW
OUT = os.path.join(SUMMARY, "ablation.json")
B = 2000
rng = np.random.default_rng(0)


def latest(pat):
    fs = sorted(glob.glob(os.path.join(R, pat)))
    return fs[-1] if fs else None


def ci(vals_fn, n):
    idx = rng.integers(0, n, size=(B, n))
    bs = np.array([vals_fn(i) for i in idx])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def phase3_block(path, pair, calib):
    d = json.load(open(path))
    out = {}
    names = {"small>large": f"{pair[0]}>{pair[1]}", "large>small": f"{pair[1]}>{pair[0]}"}
    for dkey, res in d["results"].items():
        cf = res["configs"]
        nat, flo = cf["native"]["nll"], cf["no_prefix"]["nll"]
        rows = {}
        pd = res.get("per_doc")
        for name, r in cf.items():
            row = {k: r[k] for k in ("nll", "kl", "top1") if k in r}
            row["acos_mean"] = r.get("acos_mean"); row["dcos_mean"] = r.get("dcos_mean")
            row["ret"] = (flo - r["nll"]) / (flo - nat)
            if pd and pd.get(name, {}).get("nll"):
                a = np.array(pd[name]["nll"]); n0 = np.array(pd["native"]["nll"]); f0 = np.array(pd["no_prefix"]["nll"])
                row["ret_ci"] = ci(lambda i: (f0[i].mean() - a[i].mean()) / (f0[i].mean() - n0[i].mean()), len(a))
            rows[name] = row
        # fit diagnostics
        rr = res.get("refine_report", {})
        diag = {}
        if rr:
            for key in ("lift", "mix"):
                diag[f"state_cos_{key}"] = float(np.mean([v[key]["cos"] for v in rr.values() if isinstance(v, dict) and key in v]))
            diag["state_cos_best"] = float(np.mean([v[v["best"]]["cos"] for v in rr.values() if isinstance(v, dict) and "best" in v]))
            diag["state_cos_best_min"] = float(np.min([v[v["best"]]["cos"] for v in rr.values() if isinstance(v, dict) and "best" in v]))
        ck = res.get("checks", {})
        for md in ("lift", "best", "stream_full", "copy"):
            vals = [v for k, v in ck.items() if k.startswith(f"evalstate_cos_{md}_L")]
            if vals:
                diag[f"evalstate_cos_{md}"] = float(np.mean(vals))
        out[names[dkey]] = {"calib": calib, "file": os.path.basename(path), "rows": rows, "diag": diag,
                            "copy_src": res.get("copy_src")}
    return out


def main():
    data = {"nll": {}}
    blocks = [("phase3_*_9b27b.json", ("9B", "27B"), "fineweb"),
              ("phase3_*_4b9b.json", ("4B", "9B"), "fineweb"),
              # prior-work baselines on the same harness (experiments/plain_als.py)
              ("baselines_*_9b27b.json", ("9B", "27B"), "baseline"),
              ("baselines_*_4b9b.json", ("4B", "9B"), "baseline")]
    for pat, pair, calib in blocks:
        p = latest(pat)
        if p:
            for k, v in phase3_block(p, pair, calib).items():
                data["nll"].setdefault(k, {})[calib] = v
    json.dump(data, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
