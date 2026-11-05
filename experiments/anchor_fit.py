#!/usr/bin/env python
"""Controlled anchor ablation of the recurrent-state refinement. Same recorded calibration states and split, same mixed-lift
initialization, same B then A update, same lambda grid, same selection by held-out per-head cosine; the only change is the
ridge target alpha * (initialization), alpha in {1, 0.5, 0}. alpha=0 is ridge toward zero from the same initialization."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE  # noqa: E402
import argparse, json, os, sys, time

import torch

from harness import log, ORDER  # noqa: E402
from gdn import lift, state_r2  # noqa: E402
from refine import head_cos, fit_mix, als_step_B_reg, als_step_A_reg  # noqa: E402

BASE = {("4B", "9B"): f"{CACHE}/p4b9b", ("9B", "27B"): CACHE}
ALPHAS, LAMS = (1.0, 0.5, 0.0), (0.1, 0.3, 1.0, 3.0, 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--layers", type=int, default=0, help="development only: first N layers")
    args = ap.parse_args()
    s, t = args.direction.split(">")
    small, large = sorted((s, t), key=ORDER.get)
    role = {small: "small", large: "large"}
    b = BASE[(small, large)]
    src, tgt = role[s], role[t]
    dev = torch.device("cuda:0")
    fs = torch.load(f"{b}/delta_{src}_s1024_c512_e64_st8_p896_t4-64.pt")
    ft = torch.load(f"{b}/delta_{tgt}_s1024_c512_e64_st8_p896_t4-64.pt")
    smaps = torch.load(f"{b}/smaps_{src}_to_{tgt}_s1024_c512_st8_lam0.0001_als2.pt")
    out = {a: {"maps": {}, "report": {}} for a in ALPHAS}
    lts = sorted(smaps["maps"])[: args.layers or None]
    for lt in lts:
        t0 = time.time()
        mp = smaps["maps"][lt]
        ls = mp["src"]
        Pm, Qm = mp["Pm"].to(dev).float(), mp["Qm"].to(dev).float()
        S_s = torch.cat([fs["st_full"][ls], fs["st_half"][ls]], 0).to(dev).float()
        S_t = torch.cat([ft["st_full"][lt], ft["st_half"][lt]], 0).to(dev).float()
        n = S_s.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
        S_s, S_t = S_s[perm], S_t[perm]
        n_ho = max(n // 8, 8)
        Tr_s, Ho_s, Tr_t, Ho_t = S_s[:-n_ho], S_s[-n_ho:], S_t[:-n_ho], S_t[-n_ho:]
        C = fit_mix(Tr_s, Tr_t, Pm, Qm, 1e-3, dev)
        A0, B0 = Pm * C.view(C.shape[0], C.shape[1], 1, 1), Qm
        for al in ALPHAS:
            rep, cand = {}, {}
            for lam in LAMS:
                Bn = als_step_B_reg(Tr_s, Tr_t, A0, al * B0, lam, dev)
                An = als_step_A_reg(Tr_s, Tr_t, Bn, al * A0, lam, dev)
                L = lift(Ho_s, An, Bn)
                rep[f"lam{lam:g}"] = {"cos": head_cos(L, Ho_t), "r2": state_r2(L, Ho_t), "cos_train": head_cos(lift(Tr_s, An, Bn), Tr_t), "r2_train": state_r2(lift(Tr_s, An, Bn), Tr_t)}
                cand[f"lam{lam:g}"] = (An.half().cpu(), Bn.half().cpu())
            best = max(rep, key=lambda k: rep[k]["cos"])
            rep["best"] = best
            out[al]["maps"][lt] = {"src": ls, "A": cand[best][0], "B": cand[best][1]}
            out[al]["report"][str(lt)] = rep
        log(f"{args.direction} L{lt} " + " ".join(f"a{al:g}={out[al]['report'][str(lt)][out[al]['report'][str(lt)]['best']]['cos']:.3f}@{out[al]['report'][str(lt)]['best']}" for al in ALPHAS) + f" [{time.time() - t0:.0f}s]")
        del S_s, S_t, Tr_s, Tr_t, Ho_s, Ho_t
        torch.cuda.empty_cache()
    d = f"{CACHE}/rev_anchor/{small}_{large}"
    os.makedirs(d, exist_ok=True)
    if not args.layers:
        for al in ALPHAS:
            torch.save(out[al], f"{d}/anchor{al:g}_{src}_to_{tgt}.pt")
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/anchor"), exist_ok=True)
    json.dump({f"{al:g}": out[al]["report"] for al in ALPHAS}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/anchor/report_{args.direction.replace('>', 'to')}{'_dev' if args.layers else ''}.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
