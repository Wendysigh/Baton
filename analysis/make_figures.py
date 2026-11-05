#!/usr/bin/env python
"""Paper figures (Times New Roman) from results/summary/*.json -> results/figures/*.pdf."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import ROOT, SUMMARY  # noqa: E402
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
for f in glob.glob(os.path.join(os.environ.get("BATON_FONTS", ""), "*.ttf")):
    fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
})
# Native/zero are neutral references, not series.
GRAY, GRAY_D = "#b4b3ad", "#52514e"
ORANGE = "#eb6834"
BLUE_L, BLUE, BLUE_D = "#86b6ef", "#2a78d6", "#184f95"
TEXT2 = "#52514e"

DIRS = [("4B>9B", "4B$\\to$9B"), ("9B>4B", "9B$\\to$4B"), ("9B>27B", "9B$\\to$27B"), ("27B>9B", "27B$\\to$9B")]


def bars(ax, groups, series, ylim=(0, 100), clip_label=True, width=0.8):
    """series: list of (label, color, hatch, fn(group)->(val, ci) or None)"""
    n = len(series)
    w = width / n
    x = np.arange(len(groups))
    for i, (lab, col, hatch, fn) in enumerate(series):
        labeled = False
        for gi, g in enumerate(groups):
            v = fn(g)
            if v is None:
                continue
            val, ci = v
            xx = x[gi] - width / 2 + w * (i + 0.5)
            shown = max(val, ylim[0])
            ax.bar(xx, shown - ylim[0], w * 0.86, bottom=ylim[0], color=col, hatch=hatch, edgecolor="white" if not hatch else col,
                   linewidth=0.0 if not hatch else 0.0, label=None if labeled else lab, zorder=3,
                   **({"facecolor": "white"} if hatch else {}))
            labeled = True
            if hatch:
                ax.bar(xx, shown - ylim[0], w * 0.86, bottom=ylim[0], facecolor="none", edgecolor=col, hatch=hatch, linewidth=0.6, zorder=3)
            if val >= ylim[0] and ci and ci[0] != ci[1]:
                ax.errorbar(xx, val, yerr=[[val - max(ci[0], ylim[0])], [ci[1] - val]], fmt="none", ecolor=GRAY_D,
                            elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=4)
            if val < ylim[0] and clip_label:
                ax.annotate(f"{val:.0f}%", (xx, ylim[0]), xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=5.6, color=TEXT2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in groups])
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


# ----------------------------------------------------------------------------- Fig 2: fidelity ladder


# ----------------------------------------------------------------------------- Fig 3: tasks


DX = json.load(open(os.path.join(SUMMARY, "tasks.json")))


GSM = json.load(open(os.path.join(SUMMARY, "gsm8k_rescored.json")))["directions"]


def gsm_val(d, cfg):
    e = GSM.get(d.replace(">", "to"), {}).get(cfg)
    return None if e is None else (e["flex"], e["flex_ci"])


def fig_tasks():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 1.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    ser = [("Copy (chat)", ORANGE, None, lambda g: gsm_val(g[0], "copy_chat")),
           ("Baton (web)", BLUE_L, None, lambda g: gsm_val(g[0], "web")),
           ("Baton + 12 native tokens", BLUE, None, lambda g: gsm_val(g[0], "tail_web")),
           ("Baton (chat)", BLUE_D, None, lambda g: gsm_val(g[0], "chat"))]
    bars(ax, DIRS, ser, ylim=(0, 100))
    for gi, (g, _) in enumerate(DIRS):
        v = gsm_val(g, "native")
        if v:
            ax.hlines(v[0], gi - 0.42, gi + 0.42, colors=GRAY_D, linestyles=(0, (2.5, 1.5)), linewidth=0.9, zorder=5,
                      label="Native target" if gi == 0 else None)
    ax.set_title("(a) GSM8K", loc="left")
    ax.set_ylabel("Accuracy (%)")
    # (b) CoQA F1 by turn, mean over the four directions
    ax = axs[1]
    T = DX["coqa"]["9Bto27B"]["turns"]
    t = np.arange(1, T + 1)
    for cfg, lab, col, ls, mk in (("native", "Native target", GRAY_D, (0, (2.5, 1.5)), None),
                                  ("once_web", "Baton (story)", BLUE, "-", "o"),
                                  ("each_web", "Baton (full prompt)", BLUE_L, "-", "s"),
                                  ("each_copy_web", "Copy (full prompt)", ORANGE, "-", "^")):
        y = np.mean([DX["coqa"][d.replace(">", "to")][cfg]["per_turn"] for d, _ in DIRS], 0)
        ax.plot(t, y, color=col, ls=ls, lw=1.1, marker=mk, ms=2.6, label=lab if cfg != "native" else None, zorder=4)
    ax.set_xticks(t); ax.set_xlabel("Conversation turn", labelpad=1); ax.set_ylabel("F1")
    ax.set_ylim(0, 100); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b) CoQA", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h + h2, l + l2, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.99), handlelength=1.4,
               columnspacing=1.0)
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(os.path.join(FIG, "tasks.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig 4: real handoff latency
def fig_latency():
    T = json.load(open(os.path.join(SUMMARY, "timing_summary.json")))
    long_T = json.load(open(os.path.join(SUMMARY, "timing_long_summary.json")))
    Ls = [1024, 4096, 16384, 32768, 65536, 131072]
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.55), sharex=True, sharey=True)
    series = (("native", "Native prefill", GRAY_D, "s"),
              ("handoff", "Baton (cache ready)", BLUE, "o"))
    for ax, (d, name) in zip(axes.flat, DIRS):
        tag = f"fla_same_{d.replace('>', 'to')}"
        # Preserve the original 1K--32K measurements; extend with the long run.
        # Its separate 32K controls are checked in review/audit_long_timing_20261104.py.
        c = dict(T[tag]["lengths"])
        c.update({str(L): long_T[tag]["lengths"][str(L)] for L in Ls[-2:]})
        for key, lab, col, mk in series:
            rows = [c[str(L)][key] for L in Ls]
            med = np.array([r["median"] for r in rows]) / 1000
            err = np.array([[r["median"] - r["q1"] for r in rows],
                            [r["q3"] - r["median"] for r in rows]]) / 1000
            ax.errorbar(Ls, med, yerr=err, color=col, marker=mk, ms=2.5,
                        lw=1.0, elinewidth=0.9, capsize=2.7, capthick=0.9,
                        barsabove=True, label=lab, clip_on=False)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ls); ax.set_xticklabels(["1", "4", "16", "32", "64", "128"]); ax.minorticks_off()
        ax.set_xlim(800, 160000); ax.set_ylim(0.045, 30)
        ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
        ax.tick_params(axis="both", labelsize=7, pad=2, length=2)
        ax.set_title(name, fontsize=8, pad=3, loc="left")
        ax.yaxis.grid(True, which="major"); ax.set_axisbelow(True)
        ratio = c["131072"]["ratio_handoff_native"]["median"]
        ax.text(0.03, 0.96, f"128K: {ratio:.2f}" + r"$\times$ native",
                transform=ax.transAxes, va="top", fontsize=6.5, color=TEXT2)
    fig.supylabel("Time to first token (s)", fontsize=8, x=0.01)
    fig.supxlabel("Prompt length (K tokens)", fontsize=8, y=0.025)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.53, 1.01), fontsize=7, handlelength=1.7,
               columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.80, bottom=0.20,
                        wspace=0.22, hspace=0.46)
    fig.savefig(os.path.join(FIG, "latency.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig 1: method overview
_savefig = matplotlib.figure.Figure.savefig


def _save_both(self, path, *a, **k):
    _savefig(self, path, *a, **k)
    if path.endswith(".pdf"):
        os.makedirs(os.path.join(FIG, "png"), exist_ok=True)
        _savefig(self, os.path.join(FIG, "png", os.path.basename(path)[:-4] + ".png"), dpi=220)


matplotlib.figure.Figure.savefig = _save_both


def box(ax, x, y, w, h, fc, ec, text=None, fs=7, tc="#0b0b0b", lw=0.8, r=0.02, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, **kw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x0, y0, x1, y1, col=GRAY_D, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=col, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def model_stack(ax, x, y0, n_groups, label, h_layer, w=0.13):
    """Draw a hybrid stack: repeating [GDN,GDN,GDN,Attn] blocks, bottom to top."""
    yy = y0
    for gi in range(n_groups):
        for k in range(4):
            attn = k == 3
            box(ax, x, yy, w, h_layer * 0.86, "#fde3d6" if attn else "#dce9fa", ORANGE if attn else BLUE, lw=0.5, r=0.006)
            yy += h_layer
    ax.text(x + w / 2, yy + 0.012, label, ha="center", va="bottom", fontsize=7.5, weight="bold")
    return yy


def fig_overview():
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_xlim(0, 1); ax.set_ylim(-0.01, 0.43); ax.axis("off")
    hl = 0.0245
    y_kv, y_st, y_cv = 0.315, 0.19, 0.075          # row centres
    # model stacks
    model_stack(ax, 0.0, 0.03, 3, "Source (e.g. 9B)", hl, w=0.1)
    ax.text(0.05, 0.0, "prefills the prompt", ha="center", fontsize=6.4, color=TEXT2)
    model_stack(ax, 0.9, 0.03, 3, "Target (e.g. 27B)", hl, w=0.1)
    ax.text(0.95, 0.0, "continues decoding", ha="center", fontsize=6.4, color=TEXT2)
    ax.text(0.05, 0.345, "", fontsize=1)

    def cache(x0, n_tok, n_state, title, state_note):
        ax.text(x0 + 0.095, 0.41, title, ha="center", fontsize=7.5, weight="bold")
        for i in range(n_tok):
            ax.add_patch(Rectangle((x0 + i * 0.0172, y_kv - 0.022), 0.014, 0.044, fc="#fde3d6", ec=ORANGE, lw=0.4))
        ax.text(x0 + 0.095, y_kv - 0.032, "attention K/V, one row per token", ha="center", va="top", fontsize=6.2)
        sw = 0.19 / n_state
        for i in range(n_state):
            ax.add_patch(Rectangle((x0 + i * sw + 0.002, y_st - 0.026), sw * 0.82, 0.052, fc="#dce9fa", ec=BLUE, lw=0.5))
        ax.text(x0 + 0.095, y_st - 0.036, state_note, ha="center", va="top", fontsize=6.2)
        for i in range(4):
            ax.add_patch(Rectangle((x0 + 0.06 + i * 0.018, y_cv - 0.02), 0.015, 0.04, fc="#eef0f3", ec=GRAY_D, lw=0.4))
        ax.text(x0 + 0.095, y_cv - 0.03, "conv buffer, last 4 tokens", ha="center", va="top", fontsize=6.2)

    cache(0.135, 11, 4, "Source cache", "recurrent state, 32 heads")
    cache(0.676, 11, 6, "Target cache", "recurrent state, 48 heads")
    # maps
    mx, mw = 0.355, 0.29
    box(ax, mx, y_kv - 0.04, mw, 0.08, "white", ORANGE)
    ax.text(mx + mw / 2, y_kv + 0.017, "Ridge map per attention layer", ha="center", va="center", fontsize=6.6)
    ax.text(mx + mw / 2, y_kv - 0.017, "$k_t=[k_s^{(1)},\\dots,k_s^{(m)}]\\,W_K$, pre-RoPE", ha="center", va="center", fontsize=6.6)
    box(ax, mx, y_st - 0.078, mw, 0.156, "white", BLUE)
    ax.text(mx + mw / 2, y_st + 0.05, "Bilinear head-pair map per layer", ha="center", va="center", fontsize=6.6)
    ax.text(mx + mw / 2, y_st - 0.002, "$S_{t,i}=\\sum_j A_{ij}\\,S_{s,j}\\,B_{ij}^{\\top}$", ha="center", va="center", fontsize=7.4)
    ax.text(mx + mw / 2, y_st - 0.054, "init. from per-token maps, refined by ALS", ha="center", va="center", fontsize=6.1, color=TEXT2)
    box(ax, mx, y_cv - 0.027, mw, 0.054, "white", GRAY_D)
    ax.text(mx + mw / 2, y_cv, "Ridge map on the pre-conv projection", ha="center", va="center", fontsize=6.6)
    for y in (y_kv, y_st, y_cv):
        arrow(ax, 0.327, y, mx - 0.004, y)
        arrow(ax, mx + mw + 0.004, y, 0.673, y)
    arrow(ax, 0.103, 0.19, 0.13, 0.19)
    arrow(ax, 0.869, 0.19, 0.897, 0.19)
    ax.text(0.5, 0.0, "Optional: the target prefills the last 12 prompt tokens itself", ha="center", va="center", fontsize=6.0, color=TEXT2, style="italic")
    # every label drawn inside a map box must fit inside it, with a margin
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for p in ax.patches:
        if not isinstance(p, FancyBboxPatch) or p.get_x() < mx - 1e-6 or p.get_x() > mx + 1e-6:
            continue
        pb = p.get_window_extent(rend)
        for t in ax.texts:
            tb = t.get_window_extent(rend)
            if pb.y0 < (tb.y0 + tb.y1) / 2 < pb.y1 and pb.x0 < (tb.x0 + tb.x1) / 2 < pb.x1:
                assert tb.x0 >= pb.x0 + 3 and tb.x1 <= pb.x1 - 3 and tb.y0 >= pb.y0 + 1 and tb.y1 <= pb.y1 - 1, \
                    f"label overflows its box: {t.get_text()!r}"
    fig.savefig(os.path.join(FIG, "overview.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_overview()
    fig_tasks()
    fig_latency()
    print("figures in", FIG)
