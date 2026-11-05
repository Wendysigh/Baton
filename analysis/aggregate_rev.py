#!/usr/bin/env python
"""Strict aggregation of the fidelity, RULER and LongBench runs: every manifest row exactly once, no duplicates, otherwise a
coverage error. Uncertainty: paired bootstrap over RULER prompts, LongBench document clusters, and fidelity documents.
RULER score is the official string_match_all (per-prompt fractional credit); LongBench uses the official qa_f1_score."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW, MAN, SUMMARY  # noqa: E402
import collections, glob, json, os, re, string, sys

import numpy as np

B = 4000
DIRS = ("4B>9B", "9B>4B", "9B>27B", "27B>9B")
T = lambda d: d.replace(">", "to")


def read_rows(pat):
    rows = {}
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            r = json.loads(l)
            assert r["id"] not in rows, ("duplicate", r["id"], f)
            rows[r["id"]] = r
    return rows


def boot(mat, clusters=None, seed=20261103):
    """mat: [n] per-unit values; returns mean and 95% interval (cluster bootstrap when clusters is given)."""
    x = np.asarray(mat, float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        i = rng.integers(0, len(x), (B, len(x)))
        m = x[i].mean(1)
    else:
        ids = sorted(set(clusters)); idx = {c: np.where(np.array(clusters) == c)[0] for c in ids}
        sums, cnt = np.array([x[idx[c]].sum() for c in ids]), np.array([len(idx[c]) for c in ids])
        i = rng.integers(0, len(ids), (B, len(ids)))
        m = sums[i].sum(1) / cnt[i].sum(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    c = collections.Counter(p) & collections.Counter(g)
    k = sum(c.values())
    if k == 0:
        return 0.0
    a, b = k / len(p), k / len(g)
    return 2 * a * b / (a + b)


def ruler(man="ruler_test.jsonl", sub="ruler"):
    M = {json.loads(l)["id"]: json.loads(l) for l in open(f"{MAN}/{man}")}
    out = {}
    for d in ("9B>27B", "27B>9B"):
        R = read_rows(f"{RAW}/{sub}/{man[:-6]}_{T(d)}_sh*.jsonl")
        missing = set(M) - set(R)
        out[d] = {"coverage": {"rows": len(R), "missing": len(missing)}}
        if not R:
            continue
        sc = lambda r, c: sum(x.lower() in r["configs"][c]["text"].lower() for x in r["refs"]) / len(r["refs"])
        cells = collections.defaultdict(list)
        for r in R.values():
            cells[(r["task"], r["length"])].append(r)
        cellout = {}
        for (t, L), rs in sorted(cells.items()):
            rep = {"n": len(rs)}
            base = np.array([sc(r, "native") for r in rs])
            rep["native"] = boot(100 * base)
            for c in ("all_web", "all_chat", "ctx_web", "ctx_chat"):
                v = np.array([sc(r, c) for r in rs])
                rep[c] = boot(100 * v); rep[c + "_minus_native"] = boot(100 * (v - base))
                rep[c + "_max_new_rate"] = float(np.mean([r["configs"][c]["stop"] == "max_new" for r in rs]))
            for c in ("all_copy_web", "no_context"):
                pr = [r for r in rs if c in r["configs"]]
                if pr:
                    rep[c] = boot([100 * sc(r, c) for r in pr]); rep[c + "_n"] = len(pr)
                    rep["all_web_on_panel"] = boot([100 * sc(r, "all_web") for r in pr])
            cellout[f"{t}@{L}"] = rep
        out[d]["cells"] = cellout
    return out


def longbench(man="longbench_test.jsonl", sub="longbench"):
    M = {json.loads(l)["id"]: json.loads(l) for l in open(f"{MAN}/{man}")}
    out = {}
    for d in DIRS:
        R = read_rows(f"{RAW}/{sub}/{man[:-6]}_{T(d)}_sh*.jsonl")
        out[d] = {"coverage": {"rows": len(R), "missing": len(set(M) - set(R))}}
        for ds in sorted({m["dataset"] for m in M.values()}):
            rs = [r for r in R.values() if r["dataset"] == ds]
            if not rs:
                continue
            cl = [r["doc"] for r in rs]
            S = {c: np.array([max(f1(r["configs"][c]["text"], a) for a in r["answers"]) for r in rs]) for c in rs[0]["configs"]}
            rep = {"n": len(rs), "documents": len(set(cl))}
            for c, v in S.items():
                rep[c] = boot(100 * v, cl)
                rep[c + "_max_new_rate"] = float(np.mean([r["configs"][c]["stop"] == "max_new" for r in rs]))
            for c in ("web", "chat", "copy_web"):
                rep[c + "_minus_native"] = boot(100 * (S[c] - S["native"]), cl)
            n0, nn = S["no_document"], S["native"]
            ids = sorted(set(cl)); idx = {c: np.where(np.array(cl) == c)[0] for c in ids}
            rng = np.random.default_rng(20261103)
            for c in ("web", "chat", "copy_web"):
                bs = []
                for _ in range(B):
                    ii = np.concatenate([idx[ids[j]] for j in rng.integers(0, len(ids), len(ids))])
                    den = nn[ii].mean() - n0[ii].mean()
                    bs.append((S[c][ii].mean() - n0[ii].mean()) / den if abs(den) > 1e-9 else np.nan)
                rep[c + "_gap_recovered"] = [float((S[c].mean() - n0.mean()) / (nn.mean() - n0.mean())) * 100, float(np.nanpercentile(bs, 2.5)) * 100, float(np.nanpercentile(bs, 97.5)) * 100]
            rep["native_minus_no_document"] = boot(100 * (nn - n0), cl)
            out[d][ds] = rep
    return out


def fidelity(sub="fidelity"):
    out = {}
    for f_tag, d in [("", d) for d in DIRS] + [(f"_fit{s}", d) for s in (1, 2, 3) for d in ("4B>9B", "9B>27B", "27B>9B")]:
        R = read_rows(f"{RAW}/{sub}/fid_{T(d)}{f_tag}_sh*.jsonl")
        if not R:
            continue
        blk = out.setdefault(d + f_tag, {})
        cells = collections.defaultdict(list)
        for r in R.values():
            cells[(r["set"], r["prefix"])].append(r)
        for (st, P), rs in sorted(cells.items()):
            nat = np.array([r["configs"]["native"]["nll"] for r in rs]); no = np.array([r["configs"]["no_prefix"]["nll"] for r in rs])
            rep = {"n": len(rs), "native_nll": boot(nat), "no_prefix_nll": boot(no), "source_nll": boot([r["configs"]["source"]["nll"] for r in rs])}
            rng = np.random.default_rng(20261103)
            idx = rng.integers(0, len(rs), (B, len(rs)))
            for c in rs[0]["configs"]:
                if c in ("native", "no_prefix", "source"):
                    continue
                v = np.array([r["configs"][c]["nll"] for r in rs])
                ret = (no[idx].mean(1) - v[idx].mean(1)) / (no[idx].mean(1) - nat[idx].mean(1))
                rep[c] = {"nll": float(v.mean()), "excess_nll": boot(v - nat), "retained": [float((no.mean() - v.mean()) / (no.mean() - nat.mean())) * 100, float(np.percentile(ret, 2.5)) * 100, float(np.percentile(ret, 97.5)) * 100],
                          "kl": float(np.mean([r["configs"][c]["kl"] for r in rs])), "top1": float(np.mean([r["configs"][c]["top1"] for r in rs]))}
                if c != "baton" and "baton" in rs[0]["configs"]:
                    bat = np.array([r["configs"]["baton"]["nll"] for r in rs])
                    rep[c]["baton_minus_this_nll"] = boot(bat - v)
            blk[f"{st}@{P}"] = rep
    return out


if __name__ == "__main__":
    res = {"ruler": ruler(), "longbench": longbench(), "fidelity": fidelity()}
    json.dump(res, open(f"{SUMMARY}/rev_aggregate.json", "w"), indent=1)
    print("coverage", {k: {d: v.get("coverage") for d, v in res[k].items() if isinstance(v, dict) and "coverage" in v} for k in ("ruler", "longbench")})
