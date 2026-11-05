#!/usr/bin/env python
"""Timing aggregate: median, IQR and min/max of timed requests, paired handoff/native ratio with a percentile bootstrap
over documents, components, memory and cold maps. Warm-up rows are dropped. Rows with other GPU processes are counted, not dropped."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW, SUMMARY  # noqa: E402
import glob, json, os, statistics as st, sys

import numpy as np



def spread(x):
    x = np.asarray(x, float) * 1e3
    q = np.percentile(x, [25, 50, 75])
    return {"median": float(q[1]), "q1": float(q[0]), "q3": float(q[2]), "min": float(x.min()), "max": float(x.max())}


def boot_ratio(a, b, B=10000, seed=20261103):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(a), (B, len(a)))
    r = np.median(a[i] / b[i], axis=1)
    return {"median": float(np.median(a / b)), "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}


def main(name="timing"):
    R = os.path.join(RAW, name)
    out = {}
    for path in sorted(glob.glob(os.path.join(R, "timing_*.jsonl"))):
        tag = os.path.basename(path)[7:-6]
        kern, place, d = tag.split("_")
        rows = [json.loads(l) for l in open(path)]
        meta = json.load(open(os.path.join(R, f"meta_{tag}.json")))
        c = out.setdefault(tag, {"kernels": kern, "placement": place, "direction": d.replace("to", ">"), "lengths": {}})
        if meta["cold_maps"]:
            c["cold"] = {"disk_read_s": st.median(x["disk_read_s"] for x in meta["cold_maps"]), "to_gpu_s": st.median(x["to_gpu_s"] for x in meta["cold_maps"]), "reps": len(meta["cold_maps"])}
        c["maps_on_gpu_gib"], c["map_files_gib"] = meta["maps_on_gpu_gib"], sum(meta["map_files_gib"].values())
        c["kernels_wrapped"] = meta["kernels_wrapped_by_accelerated_impl"]
        for L in sorted({r["length"] for r in rows}):
            t = [r for r in rows if r["length"] == L and not r["warmup"]]
            nat = [r["native"]["ttft"] for r in t]
            hd = [r["handoff_e2e"]["ttft"] for r in t]
            tail = [r["handoff_tail_e2e"]["ttft"] for r in t]
            src = [r["source_prefill"]["ttft"] for r in t]
            srct = [r["source_prefill_tail"]["ttft"] for r in t]
            parts = {k: float(np.median([r["handoff_parts"][k] for r in t]) * 1e3) for k in t[0]["handoff_parts"]}
            c["lengths"][str(L)] = {
                "n": len(t), "docs": len({r["doc"] for r in t}), "foreign_rows": sum(bool(r["foreign_apps"]) for r in t),
                "native": spread(nat), "handoff": spread(hd), "tail": spread(tail), "source": spread(src), "source_tail": spread(srct), "scratch": spread(np.add(src, hd)),
                "ratio_handoff_native": boot_ratio(hd, nat), "ratio_tail_native": boot_ratio(tail, nat),
                "ratio_scratch_native": boot_ratio(np.add(src, hd), nat),
                "parts_ms": parts, "same_token": sum(r["native_tok"] == r["handoff_tok"] for r in t),
                "native_first": sum(r["order"][0] == "native" for r in t),
                "native_peak_gib": {k: float(np.median([r["native_peak_gib"][k] for r in t])) for k in t[0]["native_peak_gib"]},
                "handoff_peak_gib": {k: float(np.median([r["handoff_peak_gib"][k] for r in t])) for k in t[0]["handoff_peak_gib"]},
            }
    json.dump(out, open(os.path.join(SUMMARY, f"{name}_summary.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    o = main(sys.argv[1] if len(sys.argv) > 1 else "timing")
    for tag, c in o.items():
        print(tag, f"maps {c['maps_on_gpu_gib']:.1f} GiB files {c['map_files_gib']:.1f} wrapped {c['kernels_wrapped']}")
        for L, v in c["lengths"].items():
            r = v["ratio_handoff_native"]; p = v["parts_ms"]
            print(f" {L:>5} n={v['n']} foreign={v['foreign_rows']} nat1st={v['native_first']} same={v['same_token']} native {v['native']['median']:7.1f} [{v['native']['q1']:.0f},{v['native']['q3']:.0f}] handoff {v['handoff']['median']:6.1f} [{v['handoff']['q1']:.0f},{v['handoff']['q3']:.0f}] tail {v['tail']['median']:6.1f} src {v['source']['median']:7.1f} ratio {r['median']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] scratch {v['ratio_scratch_native']['median']:.2f}  parts " + " ".join(f"{k}={x:.1f}" for k, x in p.items()))
            print("       peak native", {k: round(x, 2) for k, x in v["native_peak_gib"].items()}, "handoff", {k: round(x, 2) for k, x in v["handoff_peak_gib"].items()})
