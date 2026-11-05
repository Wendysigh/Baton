#!/usr/bin/env python
"""Paired analysis of source-influence runs. Reads raw evaluate.py outputs only.

usage: analyze.py --runs DIR [--task mmlu|mmlu_chat|gsm8k] [--calib chat] --out summary.json --md tables.md"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
import argparse, glob, json, math, os, random, re
from collections import defaultdict

GROUPS = {("4B", "9B"), ("9B", "27B"), ("4B", "27B")}
DIRS = ["4B>9B", "9B>4B", "9B>27B", "27B>9B", "4B>27B", "27B>4B"]
FAMILY_A = ["H-all", "H-tail"]
FAMILY_B = ["H-ctx"]
THRESH, EQUIV = 2.0, 2.0
B_BOOT = 10000


def binom_two_sided(k, n):
    """exact two-sided McNemar / sign test p-value"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def boot_ci(diffs, alpha=0.05, seed=0, B=B_BOOT):
    """diffs: per-question values (already averaged over orderings). Returns mean, lo, hi (percent points)."""
    import numpy as np
    x = np.asarray(diffs, dtype=np.float64)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = np.concatenate([x[rng.integers(0, n, (1000, n))].mean(1) for _ in range(B // 1000)])
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return 100 * x.mean(), 100 * lo, 100 * hi


def load(runs, task, calib):
    """-> {group: {item_id: record}} keeping only ok items; reports failures"""
    out, fails = {}, defaultdict(int)
    for f in sorted(glob.glob(os.path.join(runs, f"{task}__{calib}__*__s*of*.jsonl"))):
        g = tuple(re.search(r"_(\d+B-\d+B)", os.path.basename(f)).group(1).split("-"))
        for line in open(f):
            r = json.loads(line)
            if r.get("status") != "ok":
                fails[g] += 1
                continue
            out.setdefault(g, {})[r["item_id"]] = r
    return out, dict(fails)


# --------------------------------------------------------------------------- per-task extraction
def mmlu_table(groups):
    """-> per direction: list of per-question dicts {cond: [correct_orig, correct_shift2 or None], 'pred': {...}}"""
    tab = {}
    for g, recs in groups.items():
        a, b = g
        for src, tgt in ((a, b), (b, a)):
            d = f"{src}>{tgt}"
            rows = []
            for iid, r in recs.items():
                q = {"item_id": iid, "subject": r.get("subject"), "corr": defaultdict(list), "pred": defaultdict(dict), "mass": defaultdict(dict), "lp": defaultdict(dict), "gold": {}}
                for var in ("orig", "shift2"):
                    if var not in r:
                        continue
                    v = r[var]
                    gold = r["gold"] if var == "orig" else r["shift2_gold"]
                    conds = {"R-native": v["native"][tgt], "D-native": v["native"][src]}
                    if v.get("split"):
                        conds["R-split"] = v["split"][tgt]; conds["R-self"] = v["self"][tgt]
                    for c, s in v["dirs"][d].items():
                        if isinstance(s, dict) and "pred" in s:
                            conds[c] = s
                    q["gold"][var] = gold
                    for c, s in conds.items():
                        q["corr"][c].append((var, s["pred"] == gold))
                        q["pred"][c][var] = s["pred"]
                        q["mass"][c][var] = s.get("mass")
                        q["lp"][c][var] = s.get("lp")
                rows.append(q)
            tab[d] = rows
    return tab


def gsm_table(groups):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scoring import norm
    tab = {}
    for g, recs in groups.items():
        a, b = g
        for src, tgt in ((a, b), (b, a)):
            d = f"{src}>{tgt}"
            rows = []
            for iid, r in recs.items():
                rv, dv = r["recv"][tgt]["rows"], r["recv"][src]["rows"]
                q = {"item_id": iid, "corr": defaultdict(list), "pred": defaultdict(dict), "diag": {}}
                conds = {"R-native": rv["R-native"], "D-native": dv["R-native"]}
                for lab, x in rv.items():
                    if "|" in lab and lab.split("|")[1] == src:
                        conds[lab.split("|")[0]] = x
                    elif lab in ("R-split", "R-self"):
                        conds[lab] = x
                for c, x in conds.items():
                    q["corr"][c].append(("orig", bool(x["flex_ok"])))
                    q["pred"][c]["orig"] = norm(x["flex"])
                    q["diag"][c] = {k: x[k] for k in ("strict_ok", "conflicting", "answer_first", "no_answer", "n_tokens", "stop")}
                rows.append(q)
            tab[d] = rows
    return tab


def per_q(q, c, var=None):
    xs = [ok for v, ok in q["corr"].get(c, []) if var is None or v == var]
    return sum(xs) / len(xs) if xs else None


def compare(rows, c1, c0, var=None, seed=0):
    """paired difference c1 - c0 over questions having both; McNemar on the original ordering"""
    diffs, b01, b10 = [], 0, 0
    for q in rows:
        x1, x0 = per_q(q, c1, var), per_q(q, c0, var)
        if x1 is None or x0 is None:
            continue
        diffs.append(x1 - x0)
        o1, o0 = per_q(q, c1, "orig"), per_q(q, c0, "orig")
        if o1 is not None and o0 is not None:
            b10 += int(o1 > o0); b01 += int(o0 > o1)
    m, lo, hi = boot_ci(diffs, seed=seed)
    _, lo90, hi90 = boot_ci(diffs, alpha=0.10, seed=seed + 1)
    return {"diff": m, "ci95": [lo, hi], "ci90": [lo90, hi90], "n": len(diffs), "wins": b10, "losses": b01,
            "p_mcnemar": binom_two_sided(b10, b10 + b01)}


def adoption(rows, cond, comp="C-swap"):
    """items where D-native and R-native differ (original ordering): does cond's answer equal the donor's more often
    than the comparator's?"""
    a_c, a_k, n, b10, b01 = 0, 0, 0, 0, 0
    for q in rows:
        p = q["pred"]
        try:
            dn, rn, x, k = p["D-native"]["orig"], p["R-native"]["orig"], p[cond]["orig"], p[comp]["orig"]
        except KeyError:
            continue
        if dn == rn:
            continue
        n += 1
        ac, ak = int(x == dn), int(k == dn)
        a_c += ac; a_k += ak
        b10 += int(ac > ak); b01 += int(ak > ac)
    return {"n_disagree": n, "adopt_cond": a_c / n if n else float("nan"), "adopt_comp": a_k / n if n else float("nan"),
            "p_mcnemar": binom_two_sided(b10, b10 + b01), "comparator": comp}


def wrong_donor_adoption(rows, cond, comp="C-swap"):
    """items where D-native is wrong and differs from R-native: rate of outputting the donor's specific wrong answer"""
    a_c, a_k, n, b10, b01 = 0, 0, 0, 0, 0
    for q in rows:
        p = q["pred"]
        try:
            dn, rn, x, k = p["D-native"]["orig"], p["R-native"]["orig"], p[cond]["orig"], p[comp]["orig"]
        except KeyError:
            continue
        if dn == rn or per_q(q, "D-native", "orig") != 0:
            continue
        n += 1
        ac, ak = int(x == dn), int(k == dn)
        a_c += ac; a_k += ak; b10 += int(ac > ak); b01 += int(ak > ac)
    return {"n": n, "adopt_cond": a_c / n if n else float("nan"), "adopt_comp": a_k / n if n else float("nan"),
            "p_mcnemar": binom_two_sided(b10, b10 + b01)}


def rescue_damage(rows, cond):
    res = {"rescue": 0, "rescue_den": 0, "damage": 0, "damage_den": 0}
    for q in rows:
        d, r, h = per_q(q, "D-native", "orig"), per_q(q, "R-native", "orig"), per_q(q, cond, "orig")
        if None in (d, r, h):
            continue
        if d == 1 and r == 0:
            res["rescue_den"] += 1; res["rescue"] += int(h == 1)
        if d == 0 and r == 1:
            res["damage_den"] += 1; res["damage"] += int(h == 0)
    return res


def add_ensemble(rows):
    """exploratory: output-level ensemble of the two native models (mean of letter probabilities renormalised over
    the four letters), stored as pseudo-condition 'E-ens'"""
    import math as _m
    for q in rows:
        for var, gold in q["gold"].items():
            a, b = q["lp"]["R-native"].get(var), q["lp"]["D-native"].get(var)
            if a is None or b is None:
                continue
            pa = [_m.exp(x) for x in a]; pb = [_m.exp(x) for x in b]
            pa = [x / sum(pa) for x in pa]; pb = [x / sum(pb) for x in pb]
            pe = [(x + y) / 2 for x, y in zip(pa, pb)]
            pred = max(range(4), key=lambda i: pe[i])
            q["corr"]["E-ens"].append((var, pred == gold)); q["pred"]["E-ens"][var] = pred


def acc_macro(rows, c, var=None):
    """subject-macro accuracy: mean over subjects of the per-subject mean of per-question accuracy"""
    by = defaultdict(list)
    for q in rows:
        x = per_q(q, c, var)
        if x is not None:
            by[q.get("subject")].append(x)
    return (100 * sum(sum(v) / len(v) for v in by.values()) / len(by), len(by)) if by else (float("nan"), 0)


def acc(rows, c, var=None):
    xs = [per_q(q, c, var) for q in rows]
    xs = [x for x in xs if x is not None]
    return (100 * sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--task", default="mmlu")
    ap.add_argument("--calib", default="chat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()
    groups, fails = load(args.runs, args.task, args.calib)
    tab = (gsm_table if args.task == "gsm8k" else mmlu_table)(groups)
    dirs = [d for d in DIRS if d in tab]
    is_mmlu = args.task.startswith("mmlu")
    res = {"task": args.task, "calib": args.calib, "failures": fails, "dirs": {}}
    famA, famB = [], []
    for d in dirs:
        rows = tab[d]
        if is_mmlu:
            add_ensemble(rows)
        conds = sorted({c for q in rows for c in q["corr"]})
        R = {"n_questions": len(rows), "acc": {c: acc(rows, c) for c in conds}}
        if is_mmlu:
            R["acc_macro"] = {c: acc_macro(rows, c) for c in conds}
        if args.task == "mmlu":
            R["acc_orig"] = {c: acc(rows, c, "orig") for c in conds}
            R["acc_shift2"] = {c: acc(rows, c, "shift2") for c in conds}
        R["vs_native"] = {c: compare(rows, c, "R-native", seed=sum(map(ord, d + c))) for c in conds if c != "R-native"}
        if args.task == "mmlu":
            R["vs_native_shift2"] = {c: compare(rows, c, "R-native", var="shift2", seed=7) for c in conds if c != "R-native" and R["acc_shift2"][c][1]}
        R["exploratory_vs_donor"] = {c: compare(rows, c, "D-native", seed=5) for c in FAMILY_A + FAMILY_B if c in conds}
        if "E-ens" in conds:
            R["exploratory_vs_ensemble"] = {c: compare(rows, c, "E-ens", seed=6) for c in FAMILY_A + FAMILY_B if c in conds}
        R["vs_controls"] = {c: {k: compare(rows, c, k, seed=11) for k in ("C-noise", "C-swap") if k in conds} for c in FAMILY_A + FAMILY_B if c in conds}
        R["adoption"] = {c: adoption(rows, c) for c in FAMILY_A + ["C-noise"] if c in conds and "C-swap" in conds}
        R["wrong_donor_adoption"] = {c: wrong_donor_adoption(rows, c) for c in FAMILY_A + ["C-noise"] if c in conds and "C-swap" in conds}
        R["rescue_damage"] = {c: rescue_damage(rows, c) for c in FAMILY_A + FAMILY_B + ["C-noise", "C-swap"] if c in conds}
        Dn, Rn = R["acc"]["D-native"][0], R["acc"]["R-native"][0]
        R["gap_recovered"] = {c: ((R["acc"][c][0] - Rn) / (Dn - Rn) if abs(Dn - Rn) > 1e-9 else None) for c in FAMILY_A + FAMILY_B if c in conds}
        if is_mmlu:
            R["pred_letter_dist"] = {c: [sum(1 for q in rows if q["pred"][c].get("orig") == L) for L in range(4)] for c in conds}
            R["letter_mass_mean"] = {c: sum(q["mass"][c]["orig"] for q in rows if q["mass"][c].get("orig") is not None) / n_m
                                     for c in conds for n_m in [sum(1 for q in rows if q["mass"][c].get("orig") is not None)] if n_m}
        else:
            R["format"] = {c: {k: sum(int(q["diag"][c][k]) if not isinstance(q["diag"][c][k], str) else int(q["diag"][c][k] == "length") for q in rows if c in q["diag"]) for k in ("strict_ok", "conflicting", "answer_first", "no_answer", "stop")} for c in conds}
            R["acc_strict"] = {c: 100 * sum(q["diag"][c]["strict_ok"] for q in rows if c in q["diag"]) / max(1, sum(1 for q in rows if c in q["diag"])) for c in conds}
            R["mean_tokens"] = {c: sum(q["diag"][c]["n_tokens"] for q in rows if c in q["diag"]) / max(1, sum(1 for q in rows if c in q["diag"])) for c in conds}
        for c in FAMILY_A:
            if c in R["vs_native"]:
                famA.append((d, c, R["vs_native"][c]["p_mcnemar"]))
        for c in FAMILY_B:
            if c in R["vs_native"]:
                famB.append((d, c, R["vs_native"][c]["p_mcnemar"]))
        res["dirs"][d] = R
    for fam in (famA, famB):
        adj = holm([p for _, _, p in fam])
        for (d, c, _), a in zip(fam, adj):
            res["dirs"][d]["vs_native"][c]["p_holm"] = a
    # decision per cell
    for d in dirs:
        R = res["dirs"][d]
        R["verdict"] = {}
        for c in FAMILY_A + FAMILY_B:
            if c not in R["vs_native"]:
                continue
            v = R["vs_native"][c]
            c1 = v.get("p_holm", 1) < 0.05 and v["diff"] >= THRESH
            ctl = R["vs_controls"].get(c, {})
            c2 = all(x["ci95"][0] > 0 for x in ctl.values()) if c in FAMILY_A and ctl else None
            c3 = (R["vs_native_shift2"][c]["ci95"][0] > 0) if args.task == "mmlu" and c in R.get("vs_native_shift2", {}) else None
            ad = R["adoption"].get(c)
            c4 = (ad["adopt_cond"] > ad["adopt_comp"] and ad["p_mcnemar"] < 0.05) if (c in FAMILY_A and ad) else None
            equiv = v["ci90"][0] > -EQUIV and v["ci90"][1] < EQUIV
            if c1 and c in FAMILY_A and c2 is None and c4 is None:
                verdict = "gain over native (controls not run)"
            elif c1 and all(x in (True, None) for x in (c2, c3, c4)):
                verdict = "evidence of context-specific transfer" if c in FAMILY_A else "question-independent gain"
            elif c1:
                verdict = "gain not attributable to donor information"
            elif v["ci95"][1] < 0:
                verdict = "worse than native"
            elif equiv:
                verdict = "equivalent to native (+/-2)"
            else:
                verdict = "no detectable gain"
            R["verdict"][c] = {"c1_gain": c1, "c2_beats_controls": c2, "c3_permutation": c3, "c4_adoption": c4, "verdict": verdict}
    json.dump(res, open(args.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_md(res, args.md, is_mmlu)


def f1(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def write_md(res, path, is_mmlu):
    L = [f"## {res['task']} ({res['calib']} maps)", ""]
    if res["failures"]:
        L.append(f"Failed items per group: {res['failures']}"); L.append("")
    conds_order = ["R-native", "D-native", "E-ens", "H-all", "H-tail", "H-ctx", "C-noise", "C-swap", "H-attn", "H-delta", "H-copy", "R-split", "R-self"]
    L.append("Accuracy (%" + (", mean over both option orderings" if res["task"] == "mmlu" else "") + "):")
    L.append("")
    present = [c for c in conds_order if any(c in R["acc"] for R in res["dirs"].values())]
    L.append("| direction | n | " + " | ".join(present) + " |")
    L.append("|---|---|" + "---|" * len(present))
    for d, R in res["dirs"].items():
        L.append(f"| {d} | {R['n_questions']} | " + " | ".join(f1(R["acc"].get(c, (None,))[0]) for c in present) + " |")
    if is_mmlu:
        nsub = max((v[1] for R in res["dirs"].values() for v in R.get("acc_macro", {}).values()), default=0)
        L.append("")
        L.append(f"Subject-macro accuracy (%, unweighted mean over {nsub} subjects of per-subject accuracy" + (", both option orderings" if res["task"] == "mmlu" else "") + "):")
        L.append("")
        L.append("| direction | " + " | ".join(present) + " |")
        L.append("|---|" + "---|" * len(present))
        for d, R in res["dirs"].items():
            L.append(f"| {d} | " + " | ".join(f1(R["acc_macro"].get(c, (None,))[0]) for c in present) + " |")
    if res["task"] == "mmlu":
        L.append("")
        L.append("Accuracy on the original option ordering only (%):")
        L.append("")
        L.append("| direction | " + " | ".join(present) + " |")
        L.append("|---|" + "---|" * len(present))
        for d, R in res["dirs"].items():
            L.append(f"| {d} | " + " | ".join(f1(R["acc_orig"].get(c, (None,))[0]) for c in present) + " |")
        L.append("")
        L.append("Mean probability mass on the four answer letters (original ordering):")
        L.append("")
        L.append("| direction | " + " | ".join(present) + " |")
        L.append("|---|" + "---|" * len(present))
        for d, R in res["dirs"].items():
            L.append(f"| {d} | " + " | ".join((f"{R['letter_mass_mean'][c]:.2f}" if c in R["letter_mass_mean"] else "n/a") for c in present) + " |")
    L.append("")
    L.append("Paired difference vs receiver native (points, 95% bootstrap CI; wins/losses on the original ordering; Holm within family):")
    L.append("")
    L.append("| direction | condition | diff | 95% CI | wins/losses | p (McNemar) | p (Holm) | permuted-order diff [95% CI] | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d, R in res["dirs"].items():
        for c in FAMILY_A + FAMILY_B:
            if c not in R["vs_native"]:
                continue
            v = R["vs_native"][c]
            ps = R.get("vs_native_shift2", {}).get(c)
            pstxt = f"{ps['diff']:+.1f} [{ps['ci95'][0]:+.1f}, {ps['ci95'][1]:+.1f}]" if ps else "-"
            L.append(f"| {d} | {c} | {v['diff']:+.1f} | [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] | {v['wins']}/{v['losses']} | {v['p_mcnemar']:.2g} | {v.get('p_holm', float('nan')):.2g} | {pstxt} | {R['verdict'][c]['verdict']} |")
    L.append("")
    L.append("Controls and donor adoption (setting A):")
    L.append("")
    L.append("| direction | condition | vs C-noise [95% CI] | vs C-swap [95% CI] | donor-disagree items | adopts donor answer: cond / C-swap | p | donor-wrong items | adopts donor's wrong answer: cond / C-swap | p | rescue | damage |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d, R in res["dirs"].items():
        for c in FAMILY_A:
            if c not in R["vs_native"]:
                continue
            ctl = R["vs_controls"].get(c, {})
            cn, cs = ctl.get("C-noise"), ctl.get("C-swap")
            ad = R["adoption"].get(c)
            rd = R["rescue_damage"].get(c, {})
            fmt = lambda x: f"{x['diff']:+.1f} [{x['ci95'][0]:+.1f}, {x['ci95'][1]:+.1f}]" if x else "-"
            wd = R.get("wrong_donor_adoption", {}).get(c)
            adt = f"{ad['n_disagree']} | {100 * ad['adopt_cond']:.1f} / {100 * ad['adopt_comp']:.1f} | {ad['p_mcnemar']:.2g}" if ad else "- | - | -"
            wdt = f"{wd['n']} | {100 * wd['adopt_cond']:.1f} / {100 * wd['adopt_comp']:.1f} | {wd['p_mcnemar']:.2g}" if wd and wd["n"] else "- | - | -"
            L.append(f"| {d} | {c} | {fmt(cn)} | {fmt(cs)} | {adt} | {wdt} | {rd.get('rescue')}/{rd.get('rescue_den')} | {rd.get('damage')}/{rd.get('damage_den')} |")
    L.append("")
    L.append("Exploratory (post hoc): handoff vs the donor model itself and vs an output-level ensemble of the two natives:")
    L.append("")
    L.append("| direction | condition | vs D-native [95% CI] | vs ensemble E-ens [95% CI] |")
    L.append("|---|---|---|---|")
    for d, R in res["dirs"].items():
        for c in FAMILY_A + FAMILY_B:
            vd = R.get("exploratory_vs_donor", {}).get(c)
            ve = R.get("exploratory_vs_ensemble", {}).get(c)
            if not vd:
                continue
            fm = lambda x: f"{x['diff']:+.1f} [{x['ci95'][0]:+.1f}, {x['ci95'][1]:+.1f}]" if x else "-"
            L.append(f"| {d} | {c} | {fm(vd)} | {fm(ve)} |")
    L.append("")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
