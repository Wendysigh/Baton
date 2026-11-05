#!/usr/bin/env python
"""CPU-only rescoring of the saved full-test GSM8K outputs (raw/extra/gsm8k) with lm-eval's exact gsm8k_cot filters and
exact_match settings (lm_eval 0.4.13, tasks/gsm8k/gsm8k-cot.yaml, metadata version 3.0). Saved text was generated with the
chat template, 512 tokens and a stop at newline + 'Q:'; lm-eval's own stop list ('Q:', '</s>', '<|im_end|>') is applied to it
before filtering, so this is a rescoring of those outputs, not an lm-eval run. The original score is kept as stored."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE, RAW, SUMMARY  # noqa: E402
import glob, json, os, re
import numpy as np
import pyarrow.parquet as pq
from lm_eval.filters.extraction import RegexFilter
from lm_eval.api.metrics import exact_match_hf_evaluate as em

R = f"{RAW}/extra/gsm8k"
OUT = os.path.join(SUMMARY, "gsm8k_rescored.json")
STRICT = RegexFilter(regex_pattern=r"The answer is (\-?[0-9\.\,]+).")
FLEX = RegexFilter(regex_pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)
IGN = [",", r"\$", r"(?s).*#### ", r"\.$"]
test = pq.read_table(f"{CACHE}/data/gsm8k_test.parquet").to_pylist()
B = 2000


def until(t):
    for s in ("Q:", "</s>", "<|im_end|>"):
        t = t.split(s)[0]
    return t


def match(pred, ref):
    return bool(em([pred], [ref], regexes_to_ignore=IGN, ignore_case=True, ignore_punctuation=False)["exact_match"])


def repeats(t):
    """True if some sentence of at least 20 characters occurs three or more times."""
    c = {}
    for s in re.split(r"(?<=[.!?])\s+|\n+", t):
        s = s.strip()
        if len(s) >= 20:
            c[s] = c.get(s, 0) + 1
    return bool(c) and max(c.values()) >= 3


def ci(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(20261103)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1) * 100
    return [float(v) for v in np.percentile(m, [2.5, 97.5])]


rep = {}
for d in ("4Bto9B", "9Bto4B", "9Bto27B", "27Bto9B"):
    rows = {}
    for p in sorted(glob.glob(f"{R}/gsm8k_{d}_sh*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            assert r["idx"] not in rows, ("duplicate", d, r["idx"])
            rows[r["idx"]] = r
    assert sorted(rows) == list(range(len(test))) and len(rows) == 1319, d
    rows = [rows[i] for i in sorted(rows)]
    rep[d] = {}
    for c in [k for k, v in rows[0].items() if isinstance(v, dict)]:
        out = {k: [] for k in ("legacy", "strict", "flex", "missing_strict", "disagree", "first_answer", "several_answers",
                               "contradict", "repeat", "cap", "ntok", "early_wrong_late_right", "early_right_late_wrong")}
        for r in rows:
            ref = test[r["idx"]]["answer"].split("####")[-1].strip()
            t = until(r[c]["text"])
            s_ = STRICT.apply([[t]], [None])[0][0]
            f_ = FLEX.apply([[t]], [None])[0][0]
            so, fo = match(s_, ref), match(f_, ref)
            ans = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", t, flags=re.I)
            ansn = [a.replace(",", "") for a in ans]
            out["legacy"].append(r[c]["ok"]); out["strict"].append(so); out["flex"].append(fo)
            out["missing_strict"].append(s_ == "[invalid]"); out["disagree"].append(so != fo)
            out["first_answer"].append(bool(re.match(r"\s*(The answer is)", t)))
            out["several_answers"].append(len(ans) > 1); out["contradict"].append(len(set(ansn)) > 1)
            out["repeat"].append(repeats(t)); out["cap"].append(r[c]["n_tokens"] >= 512); out["ntok"].append(r[c]["n_tokens"])
            out["early_wrong_late_right"].append(fo and not so); out["early_right_late_wrong"].append(so and not fo)
        rep[d][c] = {"n": len(rows), **{k: 100 * float(np.mean(out[k])) for k in ("legacy", "strict", "flex")},
                     "strict_ci": ci(out["strict"]), "flex_ci": ci(out["flex"]),
                     **{k: int(np.sum(out[k])) for k in ("missing_strict", "disagree", "first_answer", "several_answers", "contradict",
                                                         "repeat", "cap", "early_wrong_late_right", "early_right_late_wrong")},
                     "ntok_median": float(np.median(out["ntok"])), "ntok_mean": float(np.mean(out["ntok"]))}
    for c, v in rep[d].items():
        print(d, f"{c:9s}", f"legacy {v['legacy']:.1f} strict {v['strict']:.1f} flex {v['flex']:.1f} cap {v['cap']} several {v['several_answers']} contra {v['contradict']} repeat {v['repeat']} first {v['first_answer']} medtok {v['ntok_median']:.0f}")
json.dump({"scope": __doc__, "lm_eval": "0.4.13", "directions": rep}, open(OUT, "w"), indent=1)
