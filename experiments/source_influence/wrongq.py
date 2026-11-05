#!/usr/bin/env python
"""Post hoc wrong-question donor control and treatment repeat.

Reads the wrongq and confirm runs. For each task and direction reports: accuracy against the current item's gold, paired
H-all minus C-wrongq and C-wrongq minus R-native, how often C-wrongq gives the partner item's gold answer and H-all's
prediction on the partner, and the per-item agreement of this run's R-native and H-all with the recorded confirmatory run.
usage: wrongq.py <out.json> <out.md>"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import RAW  # noqa: E402
import json, os, sys
from analyze import load, mmlu_table, gsm_table, compare, acc, DIRS  # noqa: E402
from scoring import norm  # noqa: E402

WQ = os.path.join(RAW, "source_influence", "wrongq")
CONF = os.path.join(RAW, "source_influence", "confirm")
TASKS = [("mmlu", "MMLU-base (both option orderings)"), ("mmlu_chat", "MMLU-chat protocol"), ("gsm8k", "GSM8K (flexible extract)")]


def partner_of(r, task, var, src, tgt):
    if task == "gsm8k":
        return r["recv"][tgt]["map_relerr"]["wrongq_from"][src]
    return r[var]["dirs"][f"{src}>{tgt}"]["C-wrongq_from"]


def gold_of(r, task, var):
    if task == "gsm8k":
        return norm(r["gold"])
    return r["gold"] if var == "orig" else r["shift2_gold"]


def analyse(task):
    groups, fails = load(WQ, task, "chat")
    if not groups:
        return None
    cgroups, _ = load(CONF, task, "chat")
    table = (gsm_table if task == "gsm8k" else mmlu_table)
    tab, ctab = table(groups), table(cgroups)
    out = {"failures": fails, "dirs": {}}
    for d in [x for x in DIRS if x in tab]:
        src, tgt = d.split(">")
        g = tuple(sorted((src, tgt), key=["4B", "9B", "27B"].index))
        recs = groups[g]
        rows = tab[d]
        byid = {q["item_id"]: q for q in rows}
        R = {"n": len(rows), "acc": {c: acc(rows, c)[0] for c in ("R-native", "H-all", "C-wrongq", "D-native")}}
        R["H-all_minus_C-wrongq"] = compare(rows, "H-all", "C-wrongq", seed=21)
        R["C-wrongq_minus_R-native"] = compare(rows, "C-wrongq", "R-native", seed=22)
        n = m_gold = m_hall = m_hall_n = 0
        for q in rows:
            r = recs[q["item_id"]]
            for var, p in q["pred"]["C-wrongq"].items():
                j = partner_of(r, task, var, src, tgt)
                if j not in recs:
                    continue
                n += 1
                m_gold += int(p == gold_of(recs[j], task, var))
                hj = byid[j]["pred"]["H-all"].get(var)
                if hj is not None:
                    m_hall_n += 1; m_hall += int(p == hj)
        R["matches_partner_gold"] = 100 * m_gold / max(1, n)
        R["matches_partner_H-all"] = 100 * m_hall / max(1, m_hall_n)
        R["n_variants"] = n
        # repeat of the recorded confirmatory run on the same items
        rep = {}
        cby = {q["item_id"]: q for q in ctab.get(d, [])}
        for c in ("R-native", "H-all"):
            same = tot = 0
            a_now = a_rec = 0
            for q in rows:
                cq = cby.get(q["item_id"])
                if cq is None:
                    continue
                for var, p in q["pred"][c].items():
                    pr = cq["pred"][c].get(var)
                    if pr is None:
                        continue
                    tot += 1; same += int(p == pr)
                ok_now = [ok for _, ok in q["corr"][c]]; ok_rec = [ok for _, ok in cq["corr"][c]]
                a_now += sum(ok_now) / len(ok_now); a_rec += sum(ok_rec) / len(ok_rec)
            nq = sum(1 for q in rows if q["item_id"] in cby)
            rep[c] = {"same_prediction": 100 * same / max(1, tot), "n": tot, "acc_now": 100 * a_now / max(1, nq), "acc_recorded": 100 * a_rec / max(1, nq)}
        R["repeat_vs_confirm"] = rep
        out["dirs"][d] = R
    return out


def fmt_ci(v):
    return f"{v['diff']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"


def main():
    res, md = {}, []
    for task, title in TASKS:
        r = analyse(task)
        if r is None:
            continue
        res[task] = r
        md += [f"### {title}", ""]
        if r["failures"]:
            md += [f"Failed items per group: {r['failures']}", ""]
        md += ["| direction | n | R-native | H-all | C-wrongq | H-all minus C-wrongq [95% CI] | C-wrongq minus R-native [95% CI] | C-wrongq = partner's gold | C-wrongq = H-all on partner |",
               "|---|---|---|---|---|---|---|---|---|"]
        for d, R in r["dirs"].items():
            a = R["acc"]
            md.append(f"| {d} | {R['n']} | {a['R-native']:.1f} | {a['H-all']:.1f} | {a['C-wrongq']:.1f} | {fmt_ci(R['H-all_minus_C-wrongq'])} | "
                      f"{fmt_ci(R['C-wrongq_minus_R-native'])} | {R['matches_partner_gold']:.1f} | {R['matches_partner_H-all']:.1f} |")
        md += ["", "Repeat of the recorded confirmatory run on the same items (same prediction %; accuracy now / recorded):", "",
               "| direction | R-native same | R-native acc | H-all same | H-all acc |", "|---|---|---|---|---|"]
        for d, R in r["dirs"].items():
            rp = {c: (v if v["n"] else {"same_prediction": float("nan"), "acc_now": float("nan"), "acc_recorded": float("nan")})
                  for c, v in R["repeat_vs_confirm"].items()}
            md.append(f"| {d} | {rp['R-native']['same_prediction']:.1f} | {rp['R-native']['acc_now']:.1f} / {rp['R-native']['acc_recorded']:.1f} | "
                      f"{rp['H-all']['same_prediction']:.1f} | {rp['H-all']['acc_now']:.1f} / {rp['H-all']['acc_recorded']:.1f} |")
        md.append("")
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(sys.argv[2], "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
