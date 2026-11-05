#!/usr/bin/env python
"""Compare zero-shot configurations across result directories: accuracy per config on the common documents and the
paired difference to a reference config with a 95% bootstrap interval.
usage: ll_compare.py <direction 4B>9B> <task> [--ref copy_web] [--dirs ll ll_v1 ...]"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW  # noqa: E402
import argparse, glob, json, os
import numpy as np

R = f"{RAW}/extra"


def load(dirs, task, tag):
    rows = {}
    for d in dirs:
        for f in glob.glob(os.path.join(R, d, f"{task}_{tag}_sh*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                rows.setdefault(r["doc_id"], {}).update({k: v for k, v in r.items() if k not in ("doc_id", "n_ctx_tokens")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("direction"); ap.add_argument("task")
    ap.add_argument("--ref", default="copy_web")
    ap.add_argument("--dirs", nargs="+", default=sorted(os.path.basename(p) for p in glob.glob(os.path.join(R, "ll*"))))
    ap.add_argument("--json", default="", help="also write {config: {acc, n, diff, ci95}} to this path")
    args = ap.parse_args()
    rows = load(args.dirs, args.task, args.direction.replace(">", "to"))
    cfgs = sorted({c for r in rows.values() for c in r})
    key = "acc_norm" if any("acc_norm" in r.get("native", {}) for r in rows.values()) else "acc"
    common = sorted(d for d, r in rows.items() if all(c in r for c in cfgs))
    print(f"{args.task} {args.direction}: {len(common)} docs with every config ({len(rows)} in all), metric {key}")
    rng = np.random.default_rng(0)
    ref = np.array([rows[d][args.ref][key] for d in common]) if args.ref in cfgs else None
    out = {"task": args.task, "direction": args.direction, "metric": key, "n": len(common), "ref": args.ref, "configs": {}}
    for c in cfgs:
        a = np.array([rows[d][c][key] for d in common])
        line = f"{c:14s} {100 * a.mean():5.1f}"
        out["configs"][c] = {"acc": float(100 * a.mean())}
        if ref is not None and c != args.ref:
            diff = a - ref
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            lo, hi = 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
            line += f"   vs {args.ref}: {100 * diff.mean():+5.1f} [{lo:+5.1f}, {hi:+5.1f}]"
            out["configs"][c].update({"diff": float(100 * diff.mean()), "ci95": [float(lo), float(hi)]})
        print(line)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
