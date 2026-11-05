#!/usr/bin/env python
"""Exploratory (post hoc, not in the frozen protocol): donor-strength contrast. For each receiver, compare the same
condition between its two donors on identical items: if handoff conveys donor capability, the stronger donor should
help more, also in setting B. Paired bootstrap over questions."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
import glob, json, os, re, sys
from collections import defaultdict
from analyze import boot_ci, binom_two_sided  # noqa: E402

SIZE = {"4B": 0, "9B": 1, "27B": 2}


def correct_mmlu(r, d, c, tgt):
    vals = []
    for var in ("orig", "shift2"):
        if var not in r:
            continue
        v = r[var]
        gold = r["gold"] if var == "orig" else r["shift2_gold"]
        s = v["native"][tgt] if c == "R-native" else v["dirs"][d].get(c)
        if isinstance(s, dict) and "pred" in s:
            vals.append(s["pred"] == gold)
    return sum(vals) / len(vals) if vals else None


def correct_gsm(r, d, c, tgt):
    src = d.split(">")[0]
    rows = r["recv"][tgt]["rows"]
    x = rows.get("R-native") if c == "R-native" else rows.get(f"{c}|{src}")
    return float(x["flex_ok"]) if x else None


def main():
    runs, task = sys.argv[1], sys.argv[2]
    data = defaultdict(dict)
    for f in glob.glob(os.path.join(runs, f"{task}__chat__*__s*of*.jsonl")):
        g = re.search(r"__(\d+B-\d+B)", f).group(1)
        for line in open(f):
            r = json.loads(line)
            if r.get("status") == "ok":
                data[g][r["item_id"]] = r
    corr = correct_mmlu if task.startswith("mmlu") else correct_gsm
    conds = ["H-all", "H-tail", "H-ctx", "C-swap"]
    out = {}
    for tgt in ("4B", "9B", "27B"):
        donors = sorted([m for m in SIZE if m != tgt], key=SIZE.get)
        groups = {dn: "-".join(sorted((dn, tgt), key=SIZE.get)) for dn in donors}
        if not all(g in data for g in groups.values()):
            continue
        weak, strong = donors
        common = sorted(set(data[groups[weak]]) & set(data[groups[strong]]))
        for c in conds:
            diffs, w, l = [], 0, 0
            for i in common:
                a = corr(data[groups[strong]][i], f"{strong}>{tgt}", c, tgt)
                b = corr(data[groups[weak]][i], f"{weak}>{tgt}", c, tgt)
                if a is None or b is None:
                    continue
                diffs.append(a - b); w += int(a > b); l += int(b > a)
            if diffs:
                m, lo, hi = boot_ci(diffs, seed=3)
                out[f"{tgt}<-{strong} minus {tgt}<-{weak} | {c}"] = {"n": len(diffs), "diff": round(m, 2), "ci95": [round(lo, 2), round(hi, 2)],
                                                                    "wins": w, "losses": l, "p_sign": binom_two_sided(w, w + l)}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 3:
        json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
