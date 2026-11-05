#!/usr/bin/env python
"""LaTeX table bodies and flat number files from results/summary/*.json -> results/tables/."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import ROOT, SUMMARY  # noqa: E402
import json
import numpy as np, os

D = json.load(open(os.path.join(SUMMARY, "ablation.json")))
OUT = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)
DIRS = ["4B>9B", "9B>4B", "9B>27B", "27B>9B"]


def ret_val(d, cfg, calib="fineweb"):
    r = D["nll"].get(d, {}).get(calib, {}).get("rows", {}).get(cfg)
    return None if r is None else 100 * r["ret"]


def fmt_ret(v):
    return "--" if v is None else (f"${v:.0f}$" if v < -99 else f"{v:.1f}")


def bold_best(vals, fmt, lower=False, skip=()):
    """Format a column; bold the best entry (ties at display precision all bold), ignoring rows in `skip`."""
    cand = [fmt(v) for i, v in enumerate(vals) if v is not None and i not in skip]
    key = (lambda x: -x) if lower else (lambda x: x)
    best = None
    if cand:
        bv = max((v for i, v in enumerate(vals) if v is not None and i not in skip), key=key)
        best = fmt(bv)
    return [("\\textbf{" + fmt(v) + "}") if (v is not None and i not in skip and fmt(v) == best) else fmt(v)
            for i, v in enumerate(vals)]


def tab_nll():
    """Bold: best per column within each block; the 64-native-token row uses target prefill, so it is not compared."""
    blocks = [[("Zero recurrent state$^\\dagger$", "rec_zero", "fineweb"),
               ("Copy GDN cache$^\\dagger$", "delta_copy", "fineweb"),
               ("Plain ALS, GDN$^\\dagger$", "delta_plainals", "baseline"),
               ("Baton, GDN$^\\dagger$", "delta_best", "fineweb")],
              [("KV only; fresh GDN", "attn_only", "baseline"),
               ("KV + copied GDN", "combined_copy", "fineweb"),
               ("KV + plain-ALS GDN", "combined_plainals", "baseline"),
               ("Baton", "combined_best", "fineweb"),
               ("Baton + 64 native tokens", "combined_best_tail64", "fineweb")]]
    lines = []
    for bi, rows in enumerate(blocks):
        if bi:
            lines.append("\\midrule")
            lines.append("\\multicolumn{5}{l}{\\emph{Full cache translated}} \\\\")
        skip = {i for i, (_, cfg, _) in enumerate(rows) if cfg.endswith("_tail64")}
        cols = [bold_best([ret_val(d, cfg, cal) for _, cfg, cal in rows], fmt_ret, skip=skip) for d in DIRS]
        for i, (lab, _, _) in enumerate(rows):
            lines.append(lab + " & " + " & ".join(c[i] for c in cols) + " \\\\")
    lines.append("\\midrule")
    def one(d, key, fmt):
        r = D["nll"].get(d, {}).get("fineweb", {}).get("rows", {}).get("combined_best")
        return "--" if r is None else fmt.format(r[key])
    lines.append("Baton: KL (nats) & " + " & ".join(one(d, "kl", "{:.3f}") for d in DIRS) + " \\\\")
    lines.append("Baton: top-1 agreement & " + " & ".join(one(d, "top1", "{:.2f}") for d in DIRS) + " \\\\")
    open(os.path.join(OUT, "nll.tex"), "w").write("\n".join(lines) + "\n")
    # The appendix needs only the isolated GDN comparison; retain the full
    # diagnostic table above as an analysis artifact.
    gdn = [line.replace("$^\\dagger$", "") for line in lines[:4]]
    open(os.path.join(OUT, "nll_gdn.tex"), "w").write("\n".join(gdn) + "\n")




def _rev():
    return json.load(open(f"{SUMMARY}/rev_aggregate.json")), json.load(open(f"{SUMMARY}/rev_summary.json"))


def _ci(v, d=1):
    return f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]".replace("-", "$-$")


def _n(x, d=0):
    return f"{x:.{d}f}".replace("-", "$-$")


RULER_TASKS = [("vt", "Variable tracking"), ("fwe", "Freq.\\ words"), ("cwe", "Common words"), ("niah_single_1", "NIAH single"),
               ("niah_multikey_2", "NIAH multi-key"), ("vt_hard", "VT, 3 chains"), ("niah_multikey_3", "NIAH UUID")]
RULER_DIRS = ("9B>27B", "27B>9B")
RULER_CFG = ("all_web", "all_chat", "ctx_web", "ctx_chat")
FID_CELLS = ("web@1024", "web@4096", "book@1024", "book@4096")
FID_SHORT = {"web@1024": "Web, 1K", "web@4096": "Web, 4K", "book@1024": "Books, 1K", "book@4096": "Books, 4K"}


def tab_ruler_web():
    """Compact main-paper view using the default web calibration throughout.

    The full web/chat results remain in ruler.tex and the original aggregates.
    No task or direction selects its calibration based on its own score.
    """
    A, S = _rev()
    configs = ("native", "all_web", "ctx_web")
    lines = []
    for L in (4096, 16384):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{L // 1024}K tokens}}}} \\\\")
        for task, label in RULER_TASKS:
            cells = [_n(A["ruler"][d]["cells"][f"{task}@{L}"][c][0])
                     for d in RULER_DIRS for c in configs]
            lines.append(label + " & " + " & ".join(cells) + " \\\\")
        cells = [_n(S["ruler"][f"{d}|{L}|all7|{c}"]["mean"], 1)
                 for d in RULER_DIRS for c in configs]
        lines.append("\\emph{Mean} & " + " & ".join(cells) + " \\\\")
        if L == 4096:
            lines.append("\\midrule")
    open(os.path.join(OUT, "ruler_web.tex"), "w").write("\n".join(lines) + "\n")


def tab_fidelity_rev():
    A, _ = _rev()
    F = A["fidelity"]
    rows = [("Baton", "baton"), ("Plain ALS", "plainals"), ("Anchor $\\alpha{=}0$", "anchor0"), ("Copy", "copy"), ("KV only", "kv_only")]
    lines = []
    for lab, c in rows:
        lines.append(lab + " & " + " & ".join(_n(F[d][x][c]["retained"][0]) for d in DIRS for x in FID_CELLS) + " \\\\")
    cells = []
    for d in DIRS:
        for x in FID_CELLS:
            cells.append("--" if f"{d}_fit1" not in F else _n(np.mean([F[f"{d}_fit{s}"][x]["baton"]["retained"][0] for s in (1, 2, 3)])))
    lines.insert(1, "Refits, mean of 3 & " + " & ".join(cells) + " \\\\")
    open(os.path.join(OUT, "fid_rev.tex"), "w").write("\n".join(lines) + "\n")
    lines = []
    for d in DIRS:
        for x in FID_CELLS:
            r = F[d][x]; b = r["baton"]
            lines.append(f"{d.replace('>', chr(92) + 'ra ')} & {FID_SHORT[x]} & {r['native_nll'][0]:.3f} & {r['no_prefix_nll'][0]:.3f} & {_ci(b['excess_nll'], 3)} & {_ci(b['retained'])} & {_ci(r['anchor0']['baton_minus_this_nll'], 3)} \\\\")
        if d != DIRS[-1]:
            lines.append("\\midrule")
    open(os.path.join(OUT, "fid_rev_ci.tex"), "w").write("\n".join(lines) + "\n")
    lines = []
    for d in ("4B>9B", "9B>27B", "27B>9B"):
        for x in FID_CELLS:
            lines.append(f"{d.replace('>', chr(92) + 'ra ')} & {FID_SHORT[x]} & {F[d][x]['baton']['retained'][0]:.1f} & " + " & ".join(f"{F[f'{d}_fit{s}'][x]['baton']['retained'][0]:.1f}" for s in (1, 2, 3)) + " \\\\")
        if d != "27B>9B":
            lines.append("\\midrule")
    open(os.path.join(OUT, "fit_var.tex"), "w").write("\n".join(lines) + "\n")


def tab_fidelity_comparison():
    """Method comparison on the matched 128-document run.

    The main table shows one stated condition, web@1024. The full corpus/length
    grid remains in fid_rev.tex and the existing fidelity interval table.
    Named prior-work rows are component adaptations, not complete reproductions.
    """
    A, _ = _rev()
    F = A["fidelity"]
    condition = "web@1024"
    rows = [
        ("Heo et al. (KV only)$^\\dagger$", "kv_only"),
        ("LatentPort (copy)$^\\dagger$", "copy"),
        ("LatentPort (ALS)$^\\dagger$", "plainals"),
        ("Baton, $\\alpha{=}0$", "anchor0"),
        ("Baton", "baton"),
    ]
    for d in DIRS:
        assert F[d][condition]["n"] == 128
    columns = [bold_best([F[d][condition][c]["retained"][0] for _, c in rows],
                         lambda v: _n(v, 1)) for d in DIRS]
    lines = [label + " & " + " & ".join(col[i] for col in columns) + " \\\\"
             for i, (label, _) in enumerate(rows)]
    lines.append("\\midrule")
    lines.append("Baton: KL (nats) & " + " & ".join(
        f"{F[d][condition]['baton']['kl']:.3f}" for d in DIRS) + " \\\\")
    lines.append("Baton: agreement (\\%) & " + " & ".join(
        f"{100 * F[d][condition]['baton']['top1']:.1f}" for d in DIRS) + " \\\\")
    open(os.path.join(OUT, "fid_comparison.tex"), "w").write("\n".join(lines) + "\n")


def tab_longbench():
    A, S = _rev()
    LB, M = A["longbench"], S["longbench_macro"]
    sets = ("qasper", "hotpotqa", "multifieldqa_en")
    lines = []
    for d in DIRS:
        cells = []
        for s in sets:
            r = LB[d][s]
            cells += [_n(r[c][0]) for c in ("native", "no_document", "web", "copy_web")]
        cells += [_n(M[d]["web_gap"]), _n(M[d]["copy_web_gap"])]
        lines.append(d.replace(">", "\\ra ") + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(OUT, "longbench.tex"), "w").write("\n".join(lines) + "\n")
    names = {"qasper": "Qasper", "hotpotqa": "HotpotQA", "multifieldqa_en": "MultiFieldQA"}
    lines = []
    for d in DIRS:
        for s in sets:
            r = LB[d][s]
            lines.append(f"{d.replace('>', chr(92) + 'ra ')} & {names[s]} & {_ci(r['web_minus_native'])} & {_ci(r['web_gap_recovered'], 0)} & {_ci(r['copy_web_minus_native'])} \\\\")
        if d != DIRS[-1]:
            lines.append("\\midrule")
    open(os.path.join(OUT, "longbench_ci.tex"), "w").write("\n".join(lines) + "\n")


CT = os.path.join(SUMMARY, "source_influence.json")


def tab_capacity():
    """Source-influence study (separate controlled protocol): accuracy per condition, wrong-donor adoption."""
    if not os.path.exists(CT):
        return
    S = json.load(open(CT))
    mb, mc, gs = S["mmlu/chat"]["dirs"], S["mmlu_chat/chat"]["dirs"], S["gsm8k/chat"]["dirs"]
    groups = [("Larger donor", ["27B>9B", "27B>4B", "9B>4B"]), ("Smaller donor", ["4B>9B", "9B>27B", "4B>27B"])]
    a = lambda blk, c: f"{blk['acc'][c][0]:.1f}"

    def adopt(blk):
        w = blk["wrong_donor_adoption"]["H-all"]
        return f"{100 * w['adopt_cond']:.0f}/{100 * w['adopt_comp']:.0f}"

    lines, flat = [], {}
    for gi, (lab, dirs) in enumerate(groups):
        if gi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{13}}{{l}}{{\\emph{{{lab}}}}} \\\\")
        for d in dirs:
            b, c, g = mb[d], mc[d], gs[d]
            cells = [a(b, k) for k in ("R-native", "D-native", "H-all", "C-swap")] + [adopt(b)]
            cells += [a(c, k) for k in ("R-native", "D-native", "H-all")]
            cells += [a(g, k) for k in ("R-native", "D-native", "H-all", "C-swap")]
            lines.append(d.replace(">", "\\ra ") + " & " + " & ".join(cells) + " \\\\")
            for task, blk in (("mmlu_base", b), ("mmlu_chat", c), ("gsm8k", g)):
                for k, v in blk["vs_native"].items():
                    flat[f"{d}|{task}|{k}-minus-native"] = [round(v["diff"], 1)] + [round(x, 1) for x in v["ci95"]]
                for k, v in blk.get("exploratory_vs_donor", {}).items():
                    flat[f"{d}|{task}|{k}-minus-donor"] = round(v["diff"], 1)
                for k, v in blk.get("exploratory_vs_ensemble", {}).items():
                    flat[f"{d}|{task}|{k}-minus-ensemble"] = round(v["diff"], 1)
            for k in ("H-attn", "H-delta", "H-copy"):
                if b["acc"].get(k, [0, 0])[1]:
                    flat[f"{d}|mmlu_base|{k}"] = round(b["acc"][k][0], 1)
    open(os.path.join(OUT, "capacity.tex"), "w").write("\n".join(lines) + "\n")
    json.dump(flat, open(os.path.join(OUT, "capacity_flat.json"), "w"), indent=0, sort_keys=True)


LMEVAL = os.path.join(SUMMARY, "lmeval_compare.json")


def tab_posthoc():
    """Post hoc checks (wrong-question control, lm-eval cross-check): ranges over directions, for the text."""
    if not os.path.exists(CT):
        return
    S = json.load(open(CT))
    rng = lambda xs: [round(min(xs), 1), round(max(xs), 1)]
    flat = {}
    for task, blk in S["wrongq"].items():
        ds = blk["dirs"].values()
        flat[f"wrongq|{task}|C-wrongq-acc"] = rng([d["acc"]["C-wrongq"] for d in ds])
        flat[f"wrongq|{task}|matches-partner-H-all"] = rng([d["matches_partner_H-all"] for d in ds])
        for c in ("R-native", "H-all"):
            rv = [d["repeat_vs_confirm"][c] for d in ds]
            flat[f"repeat|{task}|{c}|same-prediction"] = rng([r["same_prediction"] for r in rv])
            flat[f"repeat|{task}|{c}|abs-acc-change"] = round(max(abs(r["acc_now"] - r["acc_recorded"]) for r in rv), 1)
    for task in ("mmlu/chat", "mmlu_chat/chat"):
        for d, b in S[task]["dirs"].items():
            m, a = b["acc_macro"], b["acc"]
            flat[f"macro|{task}|{d}|H-all-minus-native"] = [round(a["H-all"][0] - a["R-native"][0], 1),
                                                           round(m["H-all"][0] - m["R-native"][0], 1)]
            flat[f"macro|{task}|{d}|H-all-minus-donor"] = [round(a["H-all"][0] - a["D-native"][0], 1),
                                                          round(m["H-all"][0] - m["D-native"][0], 1)]
    if os.path.exists(LMEVAL):
        L = json.load(open(LMEVAL))
        # GSM8K: stock task for 4B/9B; the 27B only with <think> banned and 1,024 tokens (the study's settings),
        # since with the stock task it opens <think> and is cut at 256 tokens
        pick = {"mmlu": {s: f"{s}|mmlu" for s in ("4B", "9B", "27B")},
                "gsm8k": {"4B": "4B|gsm8k", "9B": "9B|gsm8k", "27B": "27B|gsm8k_nothink"}}
        for task, stat in (("mmlu", "pred_agreement"), ("gsm8k", "correct_agreement")):
            v = [L[k] for k in pick[task].values() if k in L]
            lm = "lmeval_acc_matched" if task == "mmlu" else "lmeval_flex_matched"
            st = "study_acc_matched" if task == "mmlu" else "study_flex_matched"
            flat[f"lmeval|{task}|{stat}"] = rng([x[stat] for x in v])
            flat[f"lmeval|{task}|abs-acc-gap"] = round(max(abs(x[lm] - x[st]) for x in v), 1)
            flat[f"lmeval|{task}|n-sizes"] = len(v)
        flat["lmeval|27B|gsm8k-stock-acc"] = round(L["27B|gsm8k"]["lmeval_flex"], 1)
    json.dump(flat, open(os.path.join(OUT, "posthoc_flat.json"), "w"), indent=0, sort_keys=True)


def tab_std():
    """Standard benchmarks (data_extra.json): WikiText-2 perplexity and zero-shot ARC-C / HellaSwag / WinoGrande."""
    X = json.load(open(os.path.join(SUMMARY, "tasks.json")))
    lines = []
    for d in DIRS:
        k = d.replace(">", "to")
        p = X["ppl"].get(k, {})
        # bold: best configuration without the target's own prefill (Native is the reference, not compared)
        cells = bold_best([p[c]["ppl"] if c in p else None for c in ("native", "none", "web", "copy_web")],
                          lambda v: f"{v:.2f}", lower=True, skip={0})
        for t in ("arc_challenge", "hellaswag", "winogrande"):
            e = X["ll"].get(t, {}).get(k, {})
            cells += bold_best([e[c]["acc"] if c in e else None for c in ("native", "web", "copy_web")],
                               lambda v: f"{v:.1f}", skip={0})
        lines.append(d.replace(">", "\\ra ") + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(OUT, "std.tex"), "w").write("\n".join(lines) + "\n")


def tab_appendix_summaries():
    """Essential appendix evidence, without repeated main-table transformations."""

    extra = json.load(open(os.path.join(SUMMARY, "tasks.json")))
    continuation = []
    for key, label in (("native", "Native"), ("none", "No prefix"), ("web", "Baton"), ("copy_web", "Copy")):
        row = extra["ppl"]["9Bto27B"][key]
        continuation.append(label + " & " + " & ".join(f"{v:.3f}" for v in row["seg_nll"])
                            + f" & {row['ppl']:.2f}" + " \\\\")
    open(os.path.join(OUT, "continuation_summary.tex"), "w").write("\n".join(continuation) + "\n")

    path = os.path.join(SUMMARY, "copy_ablation_arc_4Bto9B.json")
    configs = json.load(open(path))["configs"]
    selected = [
        ("Copy", "copy_web"),
        ("Copy, energy matched", "copyN"),
        ("Head-diagonal, $\\lambda=10$", "mld10"),
        ("Copy-anchored, $\\lambda=3$", "mlc3"),
        ("Baton", "web"),
    ]
    rows = []
    for label, config in selected:
        row = configs[config]
        delta = "--" if "diff" not in row else (
            f"{row['diff']:+.1f} [{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}]")
        rows.append(f"{label} & {row['acc']:.1f} & {delta}" + " \\\\")
    open(os.path.join(OUT, "copy_summary.tex"), "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    tab_nll(); tab_ruler_web(); tab_fidelity_rev(); tab_fidelity_comparison(); tab_longbench(); tab_capacity(); tab_posthoc(); tab_std()
    tab_appendix_summaries()
    for f in sorted(os.listdir(OUT)):
        print("==", f); print(open(os.path.join(OUT, f)).read()[:1500])
