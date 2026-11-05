#!/usr/bin/env python
"""Baselines from prior work on the next-token-fidelity harness (same data, features, maps and eval as baton/refine.py).

  plainals    LatentPort-style learned bilinear state map: initialized at the direct copy (identity on matched heads,
              head-replicated when value-head counts differ), three alternating ridge updates toward zero
              (B, A, B), ridge strength chosen per layer by held-out state R^2 (reconstruction error).
              Configs: delta_plainals (target attention, mapped conv + plain-ALS state), combined_plainals (full cache).
  attn_only   attention KV mapped as in Heo et al., recurrent and convolution state left empty (zero):
              a KV-only cache translation applied to a hybrid.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE, RAW, model_path  # noqa: E402
import argparse, json, os, sys, time

import torch

import attention as P1  # noqa: E402
import gdn as P2  # noqa: E402
import refine as P3  # noqa: E402
from attention import log, load_model, text_model  # noqa: E402

LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def plain_als_maps(args, dfeats, smaps, models, src, tgt, dev):
    path = os.path.join(args.cache, f"plainals_{src}_to_{tgt}_s{args.seq}_c{args.n_calib}_st{args.stride}.pt")
    if os.path.exists(path):
        return torch.load(path)
    cs, ct = text_model(models[src]).config, text_model(models[tgt]).config
    Hs, Ht = cs.linear_num_value_heads, ct.linear_num_value_heads
    rep_s, rep_t = Hs // cs.linear_num_key_heads, Ht // ct.linear_num_key_heads
    dk, dv = ct.linear_key_head_dim, ct.linear_value_head_dim
    head_idx = [(j // rep_t) * rep_s + (j % rep_t) % rep_s for j in range(Ht)]
    A0 = torch.zeros(Ht, Hs, dk, dk, device=dev); B0 = torch.zeros(Ht, Hs, dv, dv, device=dev)
    for jp, j in enumerate(head_idx):
        A0[jp, j] = torch.eye(dk, device=dev); B0[jp, j] = torch.eye(dv, device=dev)
    out = {"maps": {}, "report": {}}
    for lt, mp in sorted(smaps["maps"].items()):
        t0 = time.time()
        ls = mp["src"]
        S_src = torch.cat([dfeats[src]["st_full"][ls], dfeats[src]["st_half"][ls]], 0).to(dev).float()
        S_tgt = torch.cat([dfeats[tgt]["st_full"][lt], dfeats[tgt]["st_half"][lt]], 0).to(dev).float()
        n = S_src.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)  # same split as refine_state_maps
        S_src, S_tgt = S_src[perm], S_tgt[perm]
        n_ho = max(n // 8, 8)
        Str_s, Sho_s, Str_t, Sho_t = S_src[:-n_ho], S_src[-n_ho:], S_tgt[:-n_ho], S_tgt[-n_ho:]
        rep, cands = {}, {"copy": (A0, B0)}
        for lam in LAMS:
            B = P2.als_step_B(Str_s, Str_t, A0, lam, dev)
            A = P2.als_step_A(Str_s, Str_t, B, lam, dev)
            B = P2.als_step_B(Str_s, Str_t, A, lam, dev)
            cands[f"lam{lam:g}"] = (A, B)
        for name, (A, B) in cands.items():
            L = P2.lift(Sho_s, A, B)
            rep[name] = {"r2": P2.state_r2(L, Sho_t), "cos": P3.head_cos(L, Sho_t), "r2_train": P2.state_r2(P2.lift(Str_s[:64], A, B), Str_t[:64])}
        best = max((k_ for k_ in rep if k_ != "copy"), key=lambda k_: rep[k_]["r2"])
        rep["best"] = best
        out["maps"][lt] = (cands[best][0].half().cpu(), cands[best][1].half().cpu())
        out["report"][str(lt)] = rep
        log(f"plainALS {src}->{tgt} L{lt:2d}<-L{ls:2d} held-out R2/cos: " + " ".join(f"{k_}={v['r2']:.3f}/{v['cos']:.3f}" for k_, v in rep.items() if isinstance(v, dict))
            + f" | best={best} [{time.time() - t0:.0f}s]")
        del S_src, S_tgt
        torch.cuda.empty_cache()
    torch.save(out, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=model_path("9B"))
    ap.add_argument("--large", default=model_path("27B"))
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--tag", default="_9b27b")
    ap.add_argument("--directions", nargs="+", default=["small>large", "large>small"])
    a = ap.parse_args()
    # the original phase-3 settings (results/phase3_20261027_073055_copy.json, ..._080709_4b9b.json)
    args = argparse.Namespace(small=a.small, large=a.large, small_dev="cuda:0", large_dev="cuda:0", fit_dev="cuda:0", cache=a.cache,
                              out=RAW, seq=1024, prefix=896, n_calib=512, n_calib_half=512, n_eval=64,
                              stride=8, stride_pre=16, n_match=16384, bs=4, lam=1e-4, lam_pre=1e-3, als_lam=1e-3, als_iters=2,
                              reg_lams=[0.3, 3.0], stream_ns=[64, 256], tails=[4, 64], attn_topk=[1, 2, 4, 99], directions=a.directions, tag=a.tag)
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.small)
    p1args = argparse.Namespace(**vars(args)); p1args.topk = args.attn_topk; p1args.stride = 4
    data = P1.build_data(p1args, tok)
    extra = argparse.Namespace(**vars(args)); extra.n_calib = args.n_calib + args.n_calib_half
    data["calib_half"] = P1.build_data(extra, tok)["calib"][args.n_calib:, : args.seq // 2]
    models = {"small": load_model(args.small, args.small_dev), "large": load_model(args.large, args.large_dev)}
    feats = P1.extract(p1args, models, data)
    dfeats = P2.extract_delta(args, models, data)
    sfeats = P3.extract_stream(args, models, data)
    all_res = {}
    for d in args.directions:
        src, tgt = d.split(">")
        dev = torch.device("cuda:0")
        amaps = P1.fit_maps(p1args, feats, src, tgt, dev)
        smaps = P2.fit_state_maps(args, dfeats, src, tgt, dev)
        rmaps = P3.refine_state_maps(args, dfeats, smaps, src, tgt, dev)
        tmaps = P3.fit_token_maps(args, dfeats, sfeats, smaps, src, tgt, dev)
        pmaps = plain_als_maps(args, dfeats, smaps, models, src, tgt, dev)
        res = P3.eval_phase3(args, models, feats, dfeats, sfeats, amaps, rmaps, tmaps, src, tgt, data, dev,
                             extra_rec={"plainals": pmaps["maps"]}, only=["delta_best", "combined_best", "delta_copy", "combined_copy", "attn_only"])
        res["plainals_report"] = pmaps["report"]
        all_res[d] = res
        print(P3.fmt(res), flush=True)
    path = os.path.join(args.out, f"baselines_{time.strftime('%Y%m%d_%H%M%S')}{args.tag}.json")
    json.dump({"args": vars(args), "results": all_res}, open(path, "w"), indent=1)
    log("saved", path)


if __name__ == "__main__":
    main()
