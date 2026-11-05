#!/usr/bin/env python
"""Compare lm-eval (vLLM backend) native results with the study's reimplemented-format natives, item by item.
MMLU items are matched by (subject, question, choices); GSM8K items by question text."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import RAW, MAN, SUMMARY, MMLU_SNAPSHOTS, GSM8K_ARROW  # noqa: E402
import glob, json, os, re, sys
import pyarrow as pa, pyarrow.parquet as pq
snap = os.listdir(MMLU_SNAPSHOTS)[0]
mtest = pq.read_table(f"{MMLU_SNAPSHOTS}/{snap}/all/test-00000-of-00001.parquet").to_pylist()
gtest = pa.ipc.open_stream(GSM8K_ARROW.format("test")).read_all().to_pylist()
man_m = {json.loads(l)["item_id"]: json.loads(l)["orig_index"] for l in open(f"{MAN}/source_influence/mmlu_base.jsonl")}
man_g = {json.loads(l)["item_id"]: json.loads(l)["orig_index"] for l in open(f"{MAN}/source_influence/gsm8k.jsonl")}

def study_native(task, size):
    out = {}
    for f in sorted(glob.glob(f"{RAW}/source_influence/confirm/{task}__chat__*.jsonl")):
        if size not in os.path.basename(f).split("__")[2].split("-"):
            continue
        for l in open(f):
            r = json.loads(l)
            if r["item_id"] in out or r.get("status") != "ok":
                continue
            if task == "mmlu":
                n = r["orig"]["native"].get(size)
                if n:
                    q = mtest[man_m[r["item_id"]]]
                    out[(q["subject"], q["question"].strip(), tuple(c.strip() for c in q["choices"]))] = (n["pred"], int(r["gold"]))
            else:
                n = r["recv"].get(size)
                if n:
                    q = gtest[man_g[r["item_id"]]]
                    row = n["rows"]["R-native"]
                    out[q["question"].strip()] = (row["flex_ok"], row["strict_ok"], row["n_tokens"], row["stop"])
    return out

def lmeval_samples(size, task):
    fs = sorted(glob.glob(f"{RAW}/source_influence/lmeval/{size}/**/samples_{task}*.jsonl", recursive=True))
    recs = []
    for f in fs:
        recs += [json.loads(l) for l in open(f)]
    return recs

res = {}
for size in ("4B", "9B", "27B"):
    # MMLU
    sm, recs = study_native("mmlu", size), lmeval_samples(size, "mmlu_")
    if recs:
        allacc = sum(r["acc"] for r in recs) / len(recs)
        both = agree = s_ok = l_ok = 0
        for r in recs:
            d = r["doc"]; k = (d["subject"], d["question"].strip(), tuple(c.strip() for c in d["choices"]))
            if k in sm:
                lpred = max(range(4), key=lambda i: float(r["filtered_resps"][i][0]))
                both += 1; agree += lpred == sm[k][0]; s_ok += sm[k][0] == sm[k][1]; l_ok += r["acc"]
        res[f"{size}|mmlu"] = {"lmeval_all_n": len(recs), "lmeval_all_acc": 100 * allacc, "matched": both,
                              "lmeval_acc_matched": 100 * l_ok / max(both, 1), "study_acc_matched": 100 * s_ok / max(both, 1),
                              "pred_agreement": 100 * agree / max(both, 1)}
    # GSM8K: stock gsm8k_cot, and gsm8k_cot_nothink (<think> banned, 1024 tokens: the study's decoding settings)
    sg = study_native("gsm8k", size)
    for task, key in (("gsm8k_cot_2", "gsm8k"), ("gsm8k_cot_nothink_", "gsm8k_nothink")):
        recs = lmeval_samples(size, task)
        if not recs:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["doc"]["question"].strip(), {})[r.get("filter", "x")] = r
        flex = [v["flexible-extract"]["exact_match"] for v in by.values() if "flexible-extract" in v]
        strict = [v["strict-match"]["exact_match"] for v in by.values() if "strict-match" in v]
        m = [(q, v) for q, v in by.items() if q in sg]
        lf = [v["flexible-extract"]["exact_match"] for q, v in m]
        sf = [int(sg[q][0] in (True, "True")) for q, v in m]
        res[f"{size}|{key}"] = {"lmeval_all_n": len(by), "lmeval_flex": 100 * sum(flex) / len(flex), "lmeval_strict": 100 * sum(strict) / len(strict),
                               "matched": len(m), "lmeval_flex_matched": 100 * sum(lf) / max(len(m), 1),
                               "study_flex_matched": 100 * sum(sf) / max(len(m), 1),
                               "correct_agreement": 100 * sum(a == b for a, b in zip(lf, sf)) / max(len(m), 1)}
json.dump(res, open(f"{SUMMARY}/lmeval_compare.json", "w"), indent=1)
for k, v in res.items():
    print(k, {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()})
