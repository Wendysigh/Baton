#!/usr/bin/env python
"""Calibration entry point. Fits the attention maps, token maps and lift, then the mixed and anchored state candidates
(one B solve and one A solve per candidate), selects by held-out per-head cosine and fits the convolution-buffer map.
The run ends with the component ablations on held-out documents."""
from paths import CACHE, RAW, model_path  # noqa: E402
import argparse, json, math, os, sys, time
from collections import defaultdict

import torch
import torch.nn.functional as F

import attention as P1  # noqa: E402
import gdn as P2  # noqa: E402
from attention import log, load_model, text_model, attn_layers, KVTap, rope_cos_sin, apply_rope_k, forward_text, r2  # noqa: E402
from gdn import delta_layers, DeltaTap, lift, sufficient_stats, ridge_from_stats, head_blocks  # noqa: E402



# ----------------------------------------------------------------------------- extraction
@torch.no_grad()
def extract_stream(args, models, data):
    """per model: {'pre': {layer: [N, conv_dim] half} (calib, stride_pre),
                   'eval': {layer: {k,v,g,beta: [Ne,P,.] half}},
                   'eval_pre': {L: {layer: [Ne,4,conv_dim] float}} for L in [P]+[P-r]}"""
    feats = {}
    Ls = [args.prefix] + [args.prefix - r for r in [4, 64] + list(args.stream_ns)]
    for name, m in models.items():
        path = os.path.join(args.cache, f"stream_{name}_s{args.seq}_c{args.n_calib}_e{args.n_eval}_sp{args.stride_pre}_p{args.prefix}_n{'-'.join(map(str, args.stream_ns))}.pt")
        if os.path.exists(path):
            feats[name] = torch.load(path)
            log(f"loaded stream features {name}")
            continue
        dev = next(m.parameters()).device
        tm = text_model(m)
        dl = delta_layers(m)
        tap = DeltaTap(m)
        pre_store = {}
        handles = []
        for i in dl:
            def mk(i):
                def hook(mod, inp, out):
                    pre_store[i] = out
                return hook
            handles.append(tm.layers[i].linear_attn.in_proj_qkv.register_forward_hook(mk(i)))
        pre = defaultdict(list)
        ev = defaultdict(lambda: defaultdict(list))
        ev_pre = {L: defaultdict(list) for L in Ls}
        t0 = time.time()
        for s in range(0, args.n_calib, args.bs):
            ids = data["calib"][s: s + args.bs].to(dev)
            forward_text(m, ids)
            for i in dl:
                x = pre_store[i][:, args.stride_pre - 1:: args.stride_pre]
                pre[i].append(x.reshape(-1, x.shape[-1]).half().cpu())
        log(f"{name} calib pre-conv done ({time.time() - t0:.0f}s)")
        for s in range(0, len(data["eval"]), args.bs):
            ids = data["eval"][s: s + args.bs, : args.prefix].to(dev)
            tap.clear(); tap.enabled = True
            forward_text(m, ids)
            tap.enabled = False
            for i in dl:
                for key in ("k", "v", "g", "beta"):
                    ev[i][key].append(tap.store[i][key].half().cpu())
                for L in Ls:
                    ev_pre[L][i].append(pre_store[i][:, L - 4: L].float().cpu())
        log(f"{name} eval stream done ({time.time() - t0:.0f}s)")
        tap.remove()
        for h in handles:
            h.remove()
        res = {"pre": {i: torch.cat(v, 0) for i, v in pre.items()},
               "eval": {i: {k_: torch.cat(v, 0) for k_, v in d.items()} for i, d in ev.items()},
               "eval_pre": {L: {i: torch.cat(v, 0) for i, v in d.items()} for L, d in ev_pre.items()}}
        torch.save(res, path)
        feats[name] = res
        log(f"extracted stream {name} in {time.time() - t0:.0f}s")
    return feats


# ----------------------------------------------------------------------------- state metrics / refinement
@torch.no_grad()
def head_cos(S_pred, S_true):
    """mean over states and heads of cosine between flattened per-head states."""
    a = S_pred.float().flatten(2); b = S_true.float().flatten(2)
    return F.cosine_similarity(a, b, dim=-1).mean().item()


@torch.no_grad()
def fit_scale(S_src, S_tgt, A, B):
    """per-target-head scalar alpha minimizing ||alpha*lift - S_tgt||."""
    Lf = lift(S_src, A, B)
    num = (Lf * S_tgt).flatten(2).sum((0, 2))
    den = (Lf * Lf).flatten(2).sum((0, 2)).clamp_min(1e-12)
    return num / den  # [Ht]


@torch.no_grad()
def fit_mix(S_src, S_tgt, A, B, lam_rel, dev):
    """per (target head, source head) scalar c so that S_tgt[j'] = sum_j c[j',j] A[j',j] S_src[j] B[j',j]^T."""
    N, Hs = S_src.shape[:2]
    Ht = S_tgt.shape[1]
    C = torch.zeros(Ht, Hs, device=dev)
    for jp in range(Ht):
        T = torch.einsum("jab,njbc,jdc->njad", A[jp], S_src, B[jp])  # [N,Hs,dk,dv]
        X = T.permute(0, 2, 3, 1).reshape(-1, Hs)  # [N*dk*dv, Hs]
        Y = S_tgt[:, jp].reshape(-1, 1)
        XtX = X.T @ X
        lam = lam_rel * XtX.diagonal().mean()
        C[jp] = torch.linalg.solve(XtX + lam * torch.eye(Hs, device=dev), X.T @ Y).squeeze(1)
    return C


@torch.no_grad()
def als_step_B_reg(S_src, S_tgt, A, B0, lam_rel, dev):
    """ridge toward init: W = (XtX + lam I)^-1 (XtY + lam W0)."""
    N, Hs, dk, dv = S_src.shape
    Ht = S_tgt.shape[1]
    B_new = torch.zeros_like(B0)
    for jp in range(Ht):
        X = torch.einsum("jab,njbc->njac", A[jp], S_src).permute(0, 2, 1, 3).reshape(N * dk, Hs * dv)
        Y = S_tgt[:, jp].reshape(N * dk, dv)
        XtX = (X.T @ X).double()
        XtY = (X.T @ Y).double()
        W0 = B0[jp].transpose(1, 2).reshape(Hs * dv, dv).double()
        lam = lam_rel * XtX.diagonal().mean()
        W = torch.linalg.solve(XtX + lam * torch.eye(Hs * dv, device=dev, dtype=torch.float64), XtY + lam * W0).float()
        B_new[jp] = W.view(Hs, dv, dv).transpose(1, 2)
    return B_new


@torch.no_grad()
def als_step_A_reg(S_src, S_tgt, B, A0, lam_rel, dev):
    return als_step_B_reg(S_src.transpose(-1, -2).contiguous(), S_tgt.transpose(-1, -2).contiguous(), B, A0, lam_rel, dev)


def refine_state_maps(args, dfeats, smaps, src, tgt, dev):
    path = os.path.join(args.cache, f"refine_{src}_to_{tgt}_s{args.seq}_c{args.n_calib}_st{args.stride}.pt")
    if os.path.exists(path):
        return torch.load(path)
    fs, ft = dfeats[src]["tok"], dfeats[tgt]["tok"]
    out = {"maps": {}, "report": {}}
    for lt, mp in sorted(smaps["maps"].items()):
        t0 = time.time()
        ls = mp["src"]
        Pm, Qm = mp["Pm"].to(dev).float(), mp["Qm"].to(dev).float()
        S_src = torch.cat([dfeats[src]["st_full"][ls], dfeats[src]["st_half"][ls]], 0).to(dev).float()
        S_tgt = torch.cat([dfeats[tgt]["st_full"][lt], dfeats[tgt]["st_half"][lt]], 0).to(dev).float()
        n = S_src.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
        S_src, S_tgt = S_src[perm], S_tgt[perm]
        n_ho = max(n // 8, 8)
        Str_s, Sho_s, Str_t, Sho_t = S_src[:-n_ho], S_src[-n_ho:], S_tgt[:-n_ho], S_tgt[-n_ho:]
        cands = {}
        # 0) plain lift
        cands["lift"] = (Pm, Qm)
        # 1) per-head scale
        alpha = fit_scale(Str_s, Str_t, Pm, Qm)
        cands["scale"] = (Pm * alpha.view(-1, 1, 1, 1), Qm)
        # 2) per-pair mixing scalars
        C = fit_mix(Str_s, Str_t, Pm, Qm, 1e-3, dev)
        cands["mix"] = (Pm * C.view(C.shape[0], C.shape[1], 1, 1), Qm)
        # 3) ridge-toward-init ALS, from the mix init, a few lambdas
        A0, B0 = cands["mix"]
        for lam in args.reg_lams:
            Bn = als_step_B_reg(Str_s, Str_t, A0, B0, lam, dev)
            An = als_step_A_reg(Str_s, Str_t, Bn, A0, lam, dev)
            cands[f"alsreg{lam:g}"] = (An, Bn)
        rep = {}
        for name, (A, B) in cands.items():
            Lho = lift(Sho_s, A, B)
            rep[name] = {"cos": head_cos(Lho, Sho_t), "r2": P2.state_r2(Lho, Sho_t), "cos_train": head_cos(lift(Str_s, A, B), Str_t)}
        best = max(rep, key=lambda k_: rep[k_]["cos"])
        rep["best"] = best
        rep["zero_cos"] = 0.0
        out["maps"][lt] = {"src": ls, "best": best, "A": cands[best][0].half().cpu(), "B": cands[best][1].half().cpu(),
                           "Pm": mp["Pm"], "Qm": mp["Qm"]}
        out["report"][str(lt)] = rep
        log(f"{src}->{tgt} L{lt:2d}<-L{ls:2d} held-out head-cos: " + " ".join(f"{k_}={v['cos']:.3f}" for k_, v in rep.items() if isinstance(v, dict))
            + f" | best={best} [{time.time() - t0:.0f}s]")
        del S_src, S_tgt
        torch.cuda.empty_cache()
    torch.save(out, path)
    return out


# ----------------------------------------------------------------------------- per-token maps (stream + conv)
def fit_token_maps(args, dfeats, sfeats, smaps, src, tgt, dev):
    """Per target DeltaNet layer (matched src layer): ridge maps with bias for k, v, g, beta and pre-conv qkv."""
    path = os.path.join(args.cache, f"tokmaps_{src}_to_{tgt}_s{args.seq}_c{args.n_calib}_st{args.stride}_sp{args.stride_pre}.pt")
    if os.path.exists(path):
        return torch.load(path)
    fs, ft = dfeats[src]["tok"], dfeats[tgt]["tok"]
    ps, pt = sfeats[src]["pre"], sfeats[tgt]["pre"]
    out = {"maps": {}, "report": {}}
    for lt, mp in sorted(smaps["maps"].items()):
        t0 = time.time()
        ls = mp["src"]
        maps, rep = {}, {}
        for key in ("k", "v", "g", "beta"):
            X, Y = fs[ls][key], ft[lt][key]
            N = X.shape[0]; tr, ho = slice(0, N - N // 10), slice(N - N // 10, N)
            xm, XtX = sufficient_stats(X[tr], dev)
            W, b = ridge_from_stats(xm, XtX, X[tr], Y[tr], args.lam, dev)
            rep[key] = r2(X[ho], Y[ho], W, b, dev)
            maps[key] = (W.half().cpu(), b.cpu())
        X, Y = ps[ls], pt[lt]
        N = X.shape[0]; tr, ho = slice(0, N - N // 10), slice(N - N // 10, N)
        xm, XtX = sufficient_stats(X[tr], dev)
        W, b = ridge_from_stats(xm, XtX, X[tr], Y[tr], args.lam_pre, dev)
        rep["pre"] = r2(X[ho], Y[ho], W, b, dev)
        maps["pre"] = (W.half().cpu(), b.cpu())
        out["maps"][lt] = {"src": ls, **maps}
        out["report"][str(lt)] = rep
        log(f"{src}->{tgt} L{lt:2d}<-L{ls:2d} token-map held-out R2: " + " ".join(f"{k_}={v:.3f}" for k_, v in rep.items()) + f" [{time.time() - t0:.0f}s]")
        del XtX
        torch.cuda.empty_cache()
    torch.save(out, path)
    return out


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def eval_phase3(args, models, feats, dfeats, sfeats, amaps, rmaps, tmaps, src, tgt, data, dev, extra_rec=None, only=None):
    """extra_rec: {rec_mode: {lt: (A, B)}} extra bilinear state maps; only: config names to keep (plus any extra configs)."""
    m = models[tgt]
    tm = text_model(m)
    cfg = tm.config
    Hkv, hd = cfg.num_key_value_heads, cfg.head_dim
    tl, dl = attn_layers(m), delta_layers(m)
    P, S = args.prefix, args.seq
    Ht, Hk_t = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    rep_t = Ht // Hk_t
    dk = cfg.linear_key_head_dim
    scfg = text_model(models[src]).config
    Hs, Hk_s = scfg.linear_num_value_heads, scfg.linear_num_key_heads
    rep_s = Hs // Hk_s
    ids_all = data["eval"]
    cos, sin = rope_cos_sin(m, P, dev)
    ktap = KVTap(m, capture_attn_out=True); ktap.enabled = False
    dtap = DeltaTap(m, capture_out=True); dtap.enabled = False
    kernel = dtap.orig
    lm_head = m.lm_head
    amap_k = max(amaps["maps"])
    rdev = {lt: {k_: (v.to(dev).float() if torch.is_tensor(v) else v) for k_, v in d.items()} for lt, d in rmaps["maps"].items()}
    tdev = {lt: {k_: ((v[0].to(dev).float(), v[1].to(dev).float()) if isinstance(v, tuple) else v) for k_, v in d.items()} for lt, d in tmaps["maps"].items()}
    xdev = {md: {lt: (A.to(dev).float(), B.to(dev).float()) for lt, (A, B) in mp.items()} for md, mp in (extra_rec or {}).items()}

    def snapshot(c):
        snap = {}
        for i, layer in enumerate(c.layers):
            if cfg.layer_types[i] == "full_attention":
                snap[i] = (layer.keys.clone(), layer.values.clone())
            else:
                snap[i] = (layer.conv_states[0].clone(), layer.recurrent_states[0].clone())
        return snap

    def restore(c, snap):
        for i, layer in enumerate(c.layers):
            if cfg.layer_types[i] == "full_attention":
                layer.keys, layer.values = snap[i][0].clone(), snap[i][1].clone()
            else:
                layer.conv_states[0].copy_(snap[i][0]); layer.recurrent_states[0].copy_(snap[i][1])

    def run_cont(c, cont):
        ktap.enabled = dtap.enabled = True; ktap.clear(); dtap.clear()
        o = forward_text(m, cont, past=c, use_cache=True)
        logits = lm_head(o.last_hidden_state)[:, :-1].float()
        aout = {i: ktap.attn_out[i][:, :-1].float() for i in tl}
        dout = {i: dtap.out[i][:, :-1].float() for i in dl}
        ktap.enabled = dtap.enabled = False
        return logits, aout, dout

    def src_stream(ls, seq_idx):
        d = sfeats[src]["eval"][ls]
        return {k_: d[k_][seq_idx].to(dev).float() for k_ in ("k", "v", "g", "beta")}

    def run_kernel(k, v, g, beta, init):
        """k [B,T,H,dk] (any H, pre-normed), v [B,T,H,dv], g/beta [B,T,H] -> final state [B,H,dk,dv]"""
        _, st = kernel(k, k, v, g, beta, initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True)
        return st

    def src_state_at(stream, L):
        B, _, D = stream["k"].shape
        k = stream["k"][:, :L].view(B, L, Hk_s, dk).repeat_interleave(rep_s, dim=2)
        v = stream["v"][:, :L].view(B, L, Hs, -1)
        return run_kernel(k, v, stream["g"][:, :L], stream["beta"][:, :L], None)

    def mapped_stream(lt, stream, lo, hi):
        """map source per-token inputs in [lo,hi) to the target layer's spaces."""
        tmp = tdev[lt]
        B = stream["k"].shape[0]
        T = hi - lo
        k = (stream["k"][:, lo:hi] @ tmp["k"][0] + tmp["k"][1]).view(B, T, Hk_t, dk).repeat_interleave(rep_t, dim=2)
        v = (stream["v"][:, lo:hi] @ tmp["v"][0] + tmp["v"][1]).view(B, T, Ht, -1)
        g = (stream["g"][:, lo:hi] @ tmp["g"][0] + tmp["g"][1]).clamp(max=0.0)
        beta = (stream["beta"][:, lo:hi] @ tmp["beta"][0] + tmp["beta"][1]).clamp(1e-4, 1 - 1e-4)
        return k, v, g, beta

    def mapped_conv(lt, L, seq_idx):
        ls = tdev[lt]["src"]
        x = sfeats[src]["eval_pre"][L][ls][seq_idx].to(dev).float()  # [B,4,conv_dim_s]
        W, b = tdev[lt]["pre"]
        return (x @ W + b).transpose(1, 2)  # [B,conv_dim_t,4]

    # ---- training-free copy baselines (LatentPort-style direct reuse; head-replicated when geometry differs)
    assert Hk_s == Hk_t, "copy baseline needs equal key-head counts"
    head_idx = torch.tensor([(j // rep_t) * rep_s + (j % rep_t) % rep_s for j in range(Ht)], device=dev)
    dv_ = scfg.linear_value_head_dim
    qk_ch = 2 * Hk_s * dk
    conv_idx = torch.cat([torch.arange(qk_ch, device=dev),
                          (qk_ch + head_idx[:, None] * dv_ + torch.arange(dv_, device=dev)[None]).reshape(-1)])
    sl_d, sl_a = delta_layers(models[src]), attn_layers(models[src])

    def depth_pick(lst_s, lst_t, lt):
        a = lst_t.index(lt)
        return lst_s[round(a * (len(lst_s) - 1) / max(len(lst_t) - 1, 1))]

    copy_src = {"copy": {lt: rdev[lt]["src"] for lt in dl}, "copyd": {lt: depth_pick(sl_d, dl, lt) for lt in dl}}
    attn_copy_src = {lt: depth_pick(sl_a, tl, lt) for lt in tl}

    def copy_conv(lt, L, seq_idx, mode):
        ls = copy_src[mode][lt]
        x = sfeats[src]["eval_pre"][L][ls][seq_idx].to(dev).float()  # [B,4,conv_dim_s]
        return x[..., conv_idx].transpose(1, 2)

    # configs: (name, attn, rec, conv, tail)   rec in: oracle|zero|lift|best|stream_full|stream_<N> ; conv in: oracle|zero|mapped
    configs = [("native", "oracle", "oracle", "oracle", 0),
               ("no_prefix", None, None, None, 0),
               ("rec_zero", "oracle", "zero", "oracle", 0),
               ("conv_zero", "oracle", "oracle", "zero", 0),
               ("conv_mapped", "oracle", "oracle", "mapped", 0),
               ("rec_lift", "oracle", "lift", "oracle", 0),
               ("rec_best", "oracle", "best", "oracle", 0),
               ("delta_best", "oracle", "best", "mapped", 0),
               ("rec_stream_full", "oracle", "stream_full", "oracle", 0),
               ("delta_stream_full", "oracle", "stream_full", "mapped", 0)]
    for N in args.stream_ns:
        configs += [(f"delta_stream{N}", "oracle", f"stream_{N}", "mapped", 0)]
    configs += [("attn_mapped", "mapped", "oracle", "oracle", 0),
                ("combined_best", "mapped", "best", "mapped", 0),
                (f"combined_stream{args.stream_ns[-1]}", "mapped", f"stream_{args.stream_ns[-1]}", "mapped", 0),
                ("combined_stream_full", "mapped", "stream_full", "mapped", 0),
                ("delta_best_tail64", "oracle", "best", "mapped", 64),
                ("combined_best_tail64", "mapped", "best", "mapped", 64),
                ("rec_copy", "oracle", "copy", "oracle", 0),
                ("delta_copy", "oracle", "copy", "copy", 0),
                ("delta_copyd", "oracle", "copyd", "copyd", 0),
                ("combined_copy", "mapped", "copy", "copy", 0),
                ("combined_copyd", "mapped", "copyd", "copyd", 0),
                ("all_copyd", "copyd", "copyd", "copyd", 0),
                ("combined_best_bf16", "mapped_bf16", "best_bf16", "mapped_bf16", 0)]
    for md in xdev:
        configs += [(f"delta_{md}", "oracle", md, "mapped", 0), (f"combined_{md}", "mapped", md, "mapped", 0)]
    # attention KV mapped, recurrent and convolution state left empty: a KV-only method applied to a hybrid
    configs += [("attn_only", "mapped", "zero", "zero", 0)]
    if only:
        configs = [c for c in configs if c[0] in only or c[0] in ("native", "no_prefix") or any(c[0].endswith("_" + md) for md in xdev)]
    need = {c[2] for c in configs if c[2]}
    acc = {c[0]: defaultdict(list) for c in configs}
    per_doc = {c[0]: {"nll": [], "kl": []} for c in configs}
    checks = defaultdict(list)

    for s in range(0, len(ids_all), args.bs):
        ids = ids_all[s: s + args.bs].to(dev)
        B = ids.shape[0]
        pre, cont = ids[:, :P], ids[:, P:]
        tgt_cont = cont[:, 1:]
        seq_idx = slice(s, s + B)
        out = forward_text(m, pre, use_cache=True); cache = out.past_key_values; snap = snapshot(cache)
        o2 = forward_text(m, pre[:, : P - 64], use_cache=True); cache64 = o2.past_key_values; snap64 = snapshot(o2.past_key_values)
        nat_logits, nat_aout, nat_dout = run_cont(cache, cont)
        nat_logp = F.log_softmax(nat_logits, -1)
        mkv = P1.mapped_prefix_kv(amaps["maps"][amap_k], feats[src]["eval"], seq_idx, dev, B)
        mkv16 = {}
        for kv in ("K", "V"):
            for lt_, (sel, W, b_) in amaps["maps"][amap_k][kv].items():
                X = torch.cat([feats[src]["eval"][l][kv][seq_idx] for l in sel], -1).to(dev).bfloat16()
                mkv16.setdefault(lt_, {})[kv] = (X @ W.to(dev).bfloat16() + b_.to(dev).bfloat16()).float()
        # precompute per-layer states for all rec modes at L in {P, P-64}
        streams = {ls: src_stream(ls, seq_idx) for ls in sorted({rdev[lt]["src"] for lt in dl})}
        if s == 0:
            ls0 = rdev[dl[0]]["src"]
            S_true = dfeats[src]["eval"][P][ls0][seq_idx].to(dev)
            S_rec = src_state_at(streams[ls0], P)
            checks["src_state_recompute_relerr"] = ((S_rec - S_true).norm() / S_true.norm()).item()
        rec = {L: {} for L in (P, P - 64)}
        for lt in dl:
            ls = rdev[lt]["src"]; st = streams[ls]
            for L in (P, P - 64):
                S_src = dfeats[src]["eval"][L][ls][seq_idx].to(dev).float() if L in dfeats[src]["eval"] else src_state_at(st, L)
                d = {"lift": lift(S_src, rdev[lt]["Pm"], rdev[lt]["Qm"]), "best": lift(S_src, rdev[lt]["A"], rdev[lt]["B"]),
                     "best_bf16": lift(S_src.bfloat16(), rdev[lt]["A"].bfloat16(), rdev[lt]["B"].bfloat16()).float()}
                for md, mp in xdev.items():
                    d[md] = lift(S_src, mp[lt][0], mp[lt][1])
                if L == P:
                    for md in ("copy", "copyd"):
                        d[md] = dfeats[src]["eval"][P][copy_src[md][lt]][seq_idx].to(dev).float()[:, head_idx]
                    if "stream_full" in need:
                        d["stream_full"] = run_kernel(*mapped_stream(lt, st, 0, P), None)
                    for N in (args.stream_ns if any(n_.startswith("stream_") and n_ != "stream_full" for n_ in need) else []):
                        S_ck = src_state_at(st, P - N)
                        init = lift(S_ck, rdev[lt]["A"], rdev[lt]["B"])
                        d[f"stream_{N}"] = run_kernel(*mapped_stream(lt, st, P - N, P), init)
                rec[L][lt] = d
            if s == 0:
                tgt_true = snap[lt][1].float()
                for k_, v_ in rec[P][lt].items():
                    checks[f"evalstate_cos_{k_}_L{lt}"] = head_cos(v_, tgt_true)

        for name, attn_mode, rec_mode, conv_mode, r in configs:
            if name == "native":
                logits, aout, dout = nat_logits, nat_aout, nat_dout
            elif name == "no_prefix":
                logits, aout, dout = run_cont(None, cont)
            else:
                L = P - r
                c = cache64 if r else cache
                restore(c, snap64 if r else snap)
                if attn_mode in ("mapped", "mapped_bf16"):
                    src_kv = mkv if attn_mode == "mapped" else mkv16
                    for lt in tl:
                        K = apply_rope_k(src_kv[lt]["K"].view(B, P, Hkv, hd).transpose(1, 2), cos, sin).to(torch.bfloat16)
                        V = src_kv[lt]["V"].view(B, P, Hkv, hd).transpose(1, 2).to(torch.bfloat16)
                        c.layers[lt].keys[:, :, :L] = K[:, :, :L]; c.layers[lt].values[:, :, :L] = V[:, :, :L]
                elif attn_mode == "copyd":
                    for lt in tl:
                        fe = feats[src]["eval"][attn_copy_src[lt]]
                        K = apply_rope_k(fe["K"][seq_idx].to(dev).float().view(B, P, Hkv, hd).transpose(1, 2), cos, sin).to(torch.bfloat16)
                        V = fe["V"][seq_idx].to(dev).float().view(B, P, Hkv, hd).transpose(1, 2).to(torch.bfloat16)
                        c.layers[lt].keys[:, :, :L] = K[:, :, :L]; c.layers[lt].values[:, :, :L] = V[:, :, :L]
                for lt in dl:
                    layer = c.layers[lt]
                    if rec_mode == "zero":
                        layer.recurrent_states[0].zero_()
                    elif rec_mode != "oracle":
                        layer.recurrent_states[0].copy_(rec[L][lt][rec_mode].to(layer.recurrent_states[0].dtype))
                    if conv_mode == "zero":
                        layer.conv_states[0].zero_()
                    elif conv_mode == "mapped":
                        layer.conv_states[0].copy_(mapped_conv(lt, L, seq_idx).to(layer.conv_states[0].dtype))
                    elif conv_mode == "mapped_bf16":
                        ls_ = tdev[lt]["src"]
                        x16 = sfeats[src]["eval_pre"][L][ls_][seq_idx].to(dev).bfloat16()
                        W16, b16 = tdev[lt]["pre"][0].bfloat16(), tdev[lt]["pre"][1].bfloat16()
                        layer.conv_states[0].copy_((x16 @ W16 + b16).transpose(1, 2).to(layer.conv_states[0].dtype))
                    elif conv_mode in ("copy", "copyd"):
                        layer.conv_states[0].copy_(copy_conv(lt, L, seq_idx, conv_mode).to(layer.conv_states[0].dtype))
                if r:
                    forward_text(m, pre[:, P - r:], past=c, use_cache=True)
                logits, aout, dout = run_cont(c, cont)
            logp = F.log_softmax(logits, -1)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_cont.reshape(-1), reduction="none")
            acc[name]["nll"].append(nll.cpu())
            per_doc[name]["nll"] += nll.view(B, -1).mean(1).tolist()
            per_doc[name]["kl"] += (nat_logp.exp() * (nat_logp - logp)).sum(-1).view(B, -1).mean(1).tolist()
            acc[name]["kl"].append((nat_logp.exp() * (nat_logp - logp)).sum(-1).reshape(-1).cpu())
            acc[name]["top1"].append((logits.argmax(-1) == nat_logits.argmax(-1)).float().reshape(-1).cpu())
            if name not in ("native", "no_prefix"):
                for i in tl:
                    acc[name][f"acos_L{i}"].append(F.cosine_similarity(aout[i], nat_aout[i], dim=-1).reshape(-1).cpu())
                for i in dl:
                    acc[name][f"dcos_L{i}"].append(F.cosine_similarity(dout[i], nat_dout[i], dim=-1).reshape(-1).cpu())
        log(f"eval {src}->{tgt} {s + B}/{len(ids_all)}")
    ktap.remove(); dtap.remove()

    res = {"src": src, "tgt": tgt, "prefix": P, "cont": S - P, "n_eval": len(ids_all), "checks": dict(checks), "configs": {},
           "per_doc": per_doc, "copy_src": {k_: {str(a): b for a, b in v.items()} for k_, v in copy_src.items()},
           "attn_copy_src": {str(a): b for a, b in attn_copy_src.items()}}
    for name, d in acc.items():
        row = {k_: torch.cat(v).mean().item() for k_, v in d.items()}
        row["ppl"] = math.exp(row["nll"])
        for pref in ("acos", "dcos"):
            ks = [k_ for k_ in row if k_.startswith(pref + "_L")]
            if ks:
                row[pref + "_mean"] = sum(row[k_] for k_ in ks) / len(ks)
        res["configs"][name] = row
    return res


def fmt(res):
    nat, flo = res["configs"]["native"], res["configs"]["no_prefix"]
    ck = {k_: v for k_, v in res["checks"].items() if not k_.startswith("evalstate")}
    lines = [f"\n=== phase3 {res['src']} -> {res['tgt']}  prefix={res['prefix']} cont={res['cont']} n={res['n_eval']} checks={ck}",
             f"{'config':<24} {'NLL':>7} {'PPL':>8} {'KL':>8} {'top1':>7} {'attn-cos':>9} {'delta-cos':>10} {'NLL-retained':>13}"]
    for name, r in res["configs"].items():
        ret = (flo["nll"] - r["nll"]) / max(flo["nll"] - nat["nll"], 1e-9)
        lines.append(f"{name:<24} {r['nll']:7.4f} {r['ppl']:8.3f} {r['kl']:8.4f} {r['top1']:7.3f} "
                     f"{r.get('acos_mean', float('nan')):9.4f} {r.get('dcos_mean', float('nan')):10.4f} {ret:12.1%}")
    # eval-state cosine summary (first batch) per rec mode, averaged over layers
    modes = sorted({k_.split("_L")[0].replace("evalstate_cos_", "") for k_ in res["checks"] if k_.startswith("evalstate")})
    for md in modes:
        vals = [v for k_, v in res["checks"].items() if k_.startswith(f"evalstate_cos_{md}_L")]
        if vals:
            lines.append(f"eval-state head-cos vs target native, {md:<12}: mean {sum(vals) / len(vals):.3f}  min {min(vals):.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=model_path("9B"))
    ap.add_argument("--large", default=model_path("27B"))
    ap.add_argument("--small_dev", default="cuda:0")
    ap.add_argument("--large_dev", default="cuda:1")
    ap.add_argument("--fit_dev", default="cuda:0")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=RAW)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--prefix", type=int, default=896)
    ap.add_argument("--n_calib", type=int, default=512)
    ap.add_argument("--n_calib_half", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--stride_pre", type=int, default=16)
    ap.add_argument("--n_match", type=int, default=16384)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--lam_pre", type=float, default=1e-3)
    ap.add_argument("--als_lam", type=float, default=1e-3)
    ap.add_argument("--als_iters", type=int, default=2)
    ap.add_argument("--reg_lams", type=float, nargs="+", default=[0.3, 3.0])
    ap.add_argument("--stream_ns", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--tails", type=int, nargs="+", default=[4, 64])
    ap.add_argument("--attn_topk", type=int, nargs="+", default=[1, 2, 4, 99])
    ap.add_argument("--directions", nargs="+", default=["small>large", "large>small"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True); os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.small)
    p1args = argparse.Namespace(**vars(args)); p1args.topk = args.attn_topk; p1args.stride = 4
    data = P1.build_data(p1args, tok)
    extra = argparse.Namespace(**vars(args)); extra.n_calib = args.n_calib + args.n_calib_half
    data_more = P1.build_data(extra, tok)
    data["calib_half"] = data_more["calib"][args.n_calib:, : args.seq // 2]

    models = {}
    log("loading small"); models["small"] = load_model(args.small, args.small_dev)
    log("loading large"); models["large"] = load_model(args.large, args.large_dev)
    feats = P1.extract(p1args, models, data)
    dfeats = P2.extract_delta(args, models, data)
    sfeats = extract_stream(args, models, data)
    all_res = {}
    for d in args.directions:
        src, tgt = d.split(">")
        fit_dev = torch.device(args.fit_dev)
        amaps = P1.fit_maps(p1args, feats, src, tgt, fit_dev)
        smaps = P2.fit_state_maps(args, dfeats, src, tgt, fit_dev)
        rmaps = refine_state_maps(args, dfeats, smaps, src, tgt, fit_dev)
        tmaps = fit_token_maps(args, dfeats, sfeats, smaps, src, tgt, fit_dev)
        dev = next(models[tgt].parameters()).device
        res = eval_phase3(args, models, feats, dfeats, sfeats, amaps, rmaps, tmaps, src, tgt, data, dev)
        res["refine_report"] = rmaps["report"]; res["token_map_report"] = tmaps["report"]
        all_res[d] = res
        print(fmt(res), flush=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, f"phase3_{stamp}{args.tag}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "results": all_res}, f, indent=1)
    log("saved", path)


if __name__ == "__main__":
    main()
