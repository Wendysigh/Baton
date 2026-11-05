#!/usr/bin/env python
"""Cross-run variation of R-native. Each model is evaluated natively in two worker groups (different processes, GPUs and
allocation histories) on the same frozen prompts; compare them item by item."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
import glob, json, os, re, sys
from collections import defaultdict


def load(runs, task, calib="chat"):
    out = defaultdict(dict)  # group -> item -> rec
    for f in sorted(glob.glob(os.path.join(runs, f"{task}__{calib}__*__s*of*.jsonl"))):
        g = re.search(r"__(\d+B-\d+B)", os.path.basename(f)).group(1)
        for line in open(f):
            r = json.loads(line)
            if r.get("status") == "ok":
                out[g][r["item_id"]] = r
    return out


def mmlu_native(r, m):
    return r["orig"]["native"].get(m)


def gsm_native(r, m):
    return r["recv"].get(m, {}).get("rows", {}).get("R-native")


def main():
    runs = sys.argv[1]
    res = {}
    for task, get in (("mmlu", mmlu_native), ("gsm8k", gsm_native)):
        data = load(runs, task)
        for m in ("4B", "9B", "27B"):
            gs = [g for g in data if m in g.split("-")]
            if len(gs) < 2:
                continue
            a, b = data[gs[0]], data[gs[1]]
            common = sorted(set(a) & set(b))
            if not common:
                continue
            if task == "mmlu":
                same_pred = sum(get(a[i], m)["pred"] == get(b[i], m)["pred"] for i in common)
                same_lp = sum(get(a[i], m)["lp"] == get(b[i], m)["lp"] for i in common)
                maxd = max(max(abs(x - y) for x, y in zip(get(a[i], m)["lp"], get(b[i], m)["lp"])) for i in common)
                acc = [sum(get(d[i], m)["pred"] == d[i]["gold"] for i in common) / len(common) * 100 for d in (a, b)]
                res[f"mmlu/{m}"] = {"groups": gs, "n": len(common), "same_pred": same_pred, "bit_identical_lp": same_lp,
                                    "max_abs_lp_diff": maxd, "acc_each": acc}
            else:
                same_tok = sum(get(a[i], m)["token_ids"] == get(b[i], m)["token_ids"] for i in common)
                same_ok = sum(get(a[i], m)["flex_ok"] == get(b[i], m)["flex_ok"] for i in common)
                acc = [sum(get(d[i], m)["flex_ok"] for i in common) / len(common) * 100 for d in (a, b)]
                res[f"gsm8k/{m}"] = {"groups": gs, "n": len(common), "token_identical": same_tok, "same_correctness": same_ok, "acc_each": acc}
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 2:
        json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
