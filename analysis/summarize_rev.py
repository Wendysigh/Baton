#!/usr/bin/env python
"""Subset means and paired comparisons for the paper, from the same rows as aggregate_rev.py. The first-line RULER score is a
post hoc sensitivity analysis added after run-on outputs were seen; the primary metric stays string_match_all."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW, SUMMARY  # noqa: E402
import collections, glob, json, os
import numpy as np
from aggregate_rev import read_rows, boot, T, f1, longbench  # noqa: F401

STD = ("vt", "fwe", "cwe", "niah_single_1", "niah_multikey_2")
ALL7 = STD + ("vt_hard", "niah_multikey_3")
FIRST = ("vt", "vt_hard", "niah_single_1", "niah_multikey_2", "niah_multikey_3")
sc = lambda text, refs: sum(x.lower() in text.lower() for x in refs) / len(refs)
first = lambda text: text.split("\n", 1)[0]


def means(R, d, L, tasks, cfg, base="native", scorer=lambda t: t):
    """Unweighted mean over tasks, bootstrapped within task over prompts, with the paired difference from base."""
    vals, bases = [], []
    for t in tasks:
        rs = [r for r in R.values() if r["task"] == t and r["length"] == L and cfg in r["configs"]]
        vals.append(np.array([sc(scorer(r["configs"][cfg]["text"]), r["refs"]) for r in rs]))
        bases.append(np.array([sc(scorer(r["configs"][base]["text"]), r["refs"]) for r in rs]))
    rng = np.random.default_rng(20261103)
    bd, bv = [], []
    for _ in range(2000):
        a = b = 0
        for v, u in zip(vals, bases):
            i = rng.integers(0, len(v), len(v)); a += v[i].mean(); b += u[i].mean()
        bv.append(100 * a / len(vals)); bd.append(100 * (a - b) / len(vals))
    m = 100 * np.mean([v.mean() for v in vals]); dm = 100 * np.mean([(v - u).mean() for v, u in zip(vals, bases)])
    return {"mean": float(m), "ci": [float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))], "diff": float(dm), "diff_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}


out = {"ruler": {}, "ruler_first_line": {}, "panel_baton_minus_copy": {}, "longbench_macro": {}}
for d in ("9B>27B", "27B>9B"):
    R = read_rows(f"{RAW}/ruler/ruler_test_{T(d)}_sh*.jsonl")
    for L in (4096, 16384):
        for name, tasks in (("standard", STD), ("all7", ALL7)):
            for c in ("native", "all_web", "all_chat", "ctx_web", "ctx_chat"):
                out["ruler"][f"{d}|{L}|{name}|{c}"] = means(R, d, L, tasks, c)
        for t in FIRST:
            for c in ("native", "all_web", "ctx_web", "all_chat", "ctx_chat"):
                out["ruler_first_line"][f"{d}|{L}|{t}|{c}"] = means(R, d, L, (t,), c, scorer=first)
        for t in ("vt", "vt_hard", "niah_single_1", "niah_multikey_3"):
            rs = [r for r in R.values() if r["task"] == t and r["length"] == L and "all_copy_web" in r["configs"]]
            a = np.array([sc(r["configs"]["all_web"]["text"], r["refs"]) for r in rs]); b = np.array([sc(r["configs"]["all_copy_web"]["text"], r["refs"]) for r in rs]); n = np.array([sc(r["configs"]["no_context"]["text"], r["refs"]) for r in rs])
            out["panel_baton_minus_copy"][f"{d}|{L}|{t}"] = {"n": len(rs), "baton": 100 * a.mean(), "copy": 100 * b.mean(), "no_context": 100 * n.mean(), "diff": boot(100 * (a - b))}
for d, v in longbench().items():
    ds = [k for k in v if k != "coverage"]
    rep = {}
    for c in ("native", "no_document", "source", "web", "chat", "copy_web"):
        rep[c] = float(np.mean([v[k][c][0] for k in ds]))
    for c in ("web", "chat", "copy_web"):
        rep[c + "_gap"] = 100 * (rep[c] - rep["no_document"]) / (rep["native"] - rep["no_document"])
        rep[c + "_diff"] = rep[c] - rep["native"]
    out["longbench_macro"][d] = rep
json.dump(out, open(f"{SUMMARY}/rev_summary.json", "w"), indent=1)
f = lambda x: f"{x['mean']:5.1f} d{x['diff']:+5.1f}({x['diff_ci'][0]:.1f},{x['diff_ci'][1]:.1f})"
for k, v in out["ruler"].items():
    print(k, f(v))
print("first line")
for k, v in out["ruler_first_line"].items():
    if "chat" not in k: print(k, f(v))
print("panel")
for k, v in out["panel_baton_minus_copy"].items():
    print(k, v["n"], round(v["baton"], 1), round(v["copy"], 1), round(v["no_context"], 1), [round(x, 1) for x in v["diff"]])
print(json.dumps(out["longbench_macro"], indent=0))
