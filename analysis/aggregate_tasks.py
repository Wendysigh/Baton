#!/usr/bin/env python
"""Aggregate the downstream evaluations (raw/extra/*) -> results/summary/tasks.json."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW, SUMMARY  # noqa: E402
import glob, json, math, os, re
import numpy as np

R = f"{RAW}/extra"
OUT = os.path.join(SUMMARY, "tasks.json")
DIRS = ["4Bto9B", "9Bto4B", "9Bto27B", "27Bto9B"]
rng = np.random.default_rng(0)


def boot(f, n, B=2000):
    vals = [f(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(R, pat))):
        for l in open(p):
            r = json.loads(l)
            out[r.get("doc_id", r.get("chunk", r.get("idx", r.get("doc"))))] = r
    return list(out.values())


def ppl(sub="ppl_detok"):
    """sub: ppl_detok (lm-eval detokenized test text, reported) or ppl (raw text, Heo et al.'s protocol)."""
    res = {}
    for d in DIRS:
        rs = load(f"{sub}/wikitext2_{d}_sh*.jsonl")
        if not rs:
            continue
        cfgs = [k for k in rs[0] if k != "chunk" and not k.endswith("@seg")]
        A = {c: np.array([r[c] for r in rs]) for c in cfgs}
        e = {"n": len(rs)}
        for c in cfgs:
            e[c] = {"ppl": float(math.exp(A[c].mean()))}
            if c not in ("native", "none"):
                f = lambda i, c=c: (A["none"][i].mean() - A[c][i].mean()) / (A["none"][i].mean() - A["native"][i].mean()) * 100
                e[c]["ret"] = float(f(np.arange(len(rs)))); e[c]["ret_ci"] = boot(f, len(rs))
            if c + "@seg" in rs[0]:
                e[c]["seg_nll"] = np.array([r[c + "@seg"] for r in rs]).mean(0).tolist()
        res[d] = e
    return res


LL_METRIC = {"arc_challenge": ("acc_norm", 25.0), "hellaswag": ("acc_norm", 25.0), "winogrande": ("acc", 50.0)}


def ll():
    res = {}
    for task, (met, floor) in LL_METRIC.items():
        for d in DIRS:
            rs = load(f"ll/{task}_{d}_sh*.jsonl")
            if not rs:
                continue
            e = {"n": len(rs)}
            nat = np.array([r["native"][met] for r in rs]) * 100
            for c in [k for k in rs[0] if isinstance(rs[0][k], dict)]:
                a = np.array([r[c][met] for r in rs]) * 100
                e[c] = {"acc": float(a.mean()), "ci": boot(lambda i: a[i].mean(), len(a)),
                        "ret": float(a.mean() / nat.mean() * 100), "ret_fn": float((a.mean() - floor) / (nat.mean() - floor) * 100),
                        "acc_raw": float(np.mean([r[c]["acc"] for r in rs]) * 100)}
            e["ctx_tokens"] = float(np.mean([np.mean(r["n_ctx_tokens"]) for r in rs]))
            res.setdefault(task, {})[d] = e
    return res


def gsm():
    res = {}
    for d in DIRS:
        rs = load(f"gsm8k/gsm8k_{d}_sh*.jsonl")
        if not rs:
            continue
        e = {"n": len(rs)}
        for c in [k for k in rs[0] if isinstance(rs[0][k], dict)]:
            ok = np.array([r[c]["ok"] for r in rs], float)
            af = sum(bool(re.match(r"\s*(The answer is)", r[c]["text"] or "")) for r in rs)
            e[c] = {"acc": float(ok.mean() * 100), "ci": boot(lambda i: ok[i].mean() * 100, len(ok)), "answer_first": int(af),
                    "answer_first_pct": 100 * af / len(rs), "mean_tokens": float(np.mean([r[c]["n_tokens"] for r in rs])),
                    "capped": int(sum(r[c]["n_tokens"] >= 512 for r in rs))}
        res[d] = e
    return res


def coqa():
    res = {}
    for d in DIRS:
        rs = load(f"coqa/coqa_{d}_sh*.jsonl")
        if not rs:
            continue
        T = len(rs[0]["turns"])
        cfgs = [k for k in rs[0]["turns"][0] if k != "len"]
        e = {"n": len(rs), "turns": T, "mean_len": [float(np.mean([r["turns"][t]["len"] for r in rs])) for t in range(T)]}
        F = {c: np.array([[r["turns"][t][c]["f1"] for t in range(T)] for r in rs]) * 100 for c in cfgs}
        for c in cfgs:
            per = F[c].mean(0)
            e[c] = {"f1": float(F[c].mean()), "ci": boot(lambda i, c=c: F[c][i].mean(), len(rs)), "per_turn": per.tolist()}
            if c != "native":
                gap = F["native"].mean(0) - per
                slope = float(np.polyfit(np.arange(1, T + 1), gap, 1)[0])
                e[c]["drift_slope"] = slope
                e[c]["slope_ci"] = boot(lambda i, c=c: np.polyfit(np.arange(1, T + 1), F["native"][i].mean(0) - F[c][i].mean(0), 1)[0], len(rs), 1000)
                e[c]["gap_first"], e[c]["gap_last"] = float(gap[0]), float(gap[-1])
        res[d] = e
    return res


if __name__ == "__main__":
    data = {"ppl": ppl(), "ppl_raw": ppl("ppl"), "ll": ll(), "gsm8k": gsm(), "coqa": coqa()}
    json.dump(data, open(OUT, "w"), indent=1)
    for k, v in data.items():
        print(k, {d: {c: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in x.items() if kk in ("ppl", "ret", "acc", "f1", "drift_slope", "answer_first", "mean_tokens")}
                     for c, x in e.items() if isinstance(x, dict)} if isinstance(e, dict) and "n" in e else
                  {dd: {c: round(x.get("acc", 0), 1) for c, x in ee.items() if isinstance(x, dict)} for dd, ee in e.items()}
                  for d, e in v.items()})
