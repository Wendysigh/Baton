#!/usr/bin/env python
"""GDN-cache calibration: paired token features and recurrent states from both models, layer matching by held-out R^2,
and the token-level bilinear lift of Baton's recurrent-state map."""
import argparse, json, math, os, sys, time
from collections import defaultdict

import torch
import torch.nn.functional as F

import attention as P1  # noqa: E402
from attention import log, load_model, text_model, attn_layers, KVTap, rope_cos_sin, apply_rope_k, forward_text, ridge_fit, r2  # noqa: E402


def delta_layers(m):
    cfg = text_model(m).config
    return [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]


# ----------------------------------------------------------------------------- tap
class DeltaTap:
    """Captures per-token recurrence inputs (l2-normed k, v, g, beta) per DeltaNet layer, and layer outputs."""

    def __init__(self, m, capture_out=False):
        import transformers.models.qwen3_5.modeling_qwen3_5 as M
        self.M = M
        self.tm = text_model(m)
        cfg = self.tm.config
        self.layers = delta_layers(m)
        self.rep = cfg.linear_num_value_heads // cfg.linear_num_key_heads
        self.cur = None
        self.enabled = False
        self.store = {}
        self.out = {}
        self.handles = []
        self.orig = M.torch_chunk_gated_delta_rule
        M.torch_chunk_gated_delta_rule = self._wrapped
        for i in self.layers:
            la = self.tm.layers[i].linear_attn
            self.handles.append(la.register_forward_pre_hook(self._mk_pre(i), with_kwargs=True))
            if capture_out:
                self.handles.append(la.out_proj.register_forward_hook(self._mk_out(i)))

    def _mk_pre(self, i):
        def hook(mod, args, kwargs):
            self.cur = i
        return hook

    def _mk_out(self, i):
        def hook(mod, inp, out):
            if self.enabled:
                self.out[i] = out
        return hook

    def _wrapped(self, query, key, value, g, beta, **kw):
        if self.enabled:
            k = self.M.l2norm(key.float(), dim=-1, eps=1e-6)[:, :, :: self.rep]  # [B,T,Hk,dk]
            B, T = k.shape[:2]
            self.store[self.cur] = {"k": k.reshape(B, T, -1), "v": value.float().reshape(B, T, -1),
                                    "g": g.float(), "beta": beta.float()}
        return self.orig(query, key, value, g, beta, **kw)

    def clear(self):
        self.store = {}
        self.out = {}

    def remove(self):
        self.M.torch_chunk_gated_delta_rule = self.orig
        for h in self.handles:
            h.remove()


def get_states(cache, layers):
    return {i: {"rec": cache.layers[i].recurrent_states[0].clone(), "conv": cache.layers[i].conv_states[0].clone()}
            for i in layers}


# ----------------------------------------------------------------------------- extract
@torch.no_grad()
def extract_delta(args, models, data):
    """per model: {'tok': {layer: {k,v,g,beta: [N,D] half}}, 'st1024': {layer: rec [N,Hv,dk,dv] half},
    'st512': {...}, 'eval': {P: {layer: rec}, P-r: {layer: rec}}, 'eval_conv': {P: {layer: conv}}}"""
    feats = {}
    for name, m in models.items():
        tails_tag = "-".join(str(t) for t in args.tails)
        path = os.path.join(args.cache, f"delta_{name}_s{args.seq}_c{args.n_calib}_e{args.n_eval}_st{args.stride}_p{args.prefix}_t{tails_tag}.pt")
        if os.path.exists(path):
            feats[name] = torch.load(path)
            log(f"loaded delta features {name}")
            continue
        dev = next(m.parameters()).device
        dl = delta_layers(m)
        tap = DeltaTap(m)
        tok = defaultdict(lambda: defaultdict(list))
        st = {"st_full": defaultdict(list), "st_half": defaultdict(list)}
        ev = {L: defaultdict(list) for L in [args.prefix] + [args.prefix - t for t in args.tails]}
        ev_conv = {args.prefix: defaultdict(list)}
        t0 = time.time()
        calib = data["calib"]
        n_full = args.n_calib
        # (a) full-length calib docs: per-token features + final state
        for s in range(0, n_full, args.bs):
            ids = calib[s: s + args.bs].to(dev)
            tap.clear(); tap.enabled = True
            out = forward_text(m, ids, use_cache=True)
            tap.enabled = False
            for i in dl:
                d = tap.store[i]
                for key in ("k", "v", "g", "beta"):
                    x = d[key][:, args.stride - 1:: args.stride]
                    tok[i][key].append(x.reshape(-1, x.shape[-1]).half().cpu())
                st["st_full"][i].append(out.past_key_values.layers[i].recurrent_states[0].half().cpu())
            if s % (args.bs * 16) == 0:
                log(f"{name} calib-full {s}/{n_full} ({time.time() - t0:.0f}s)")
        # (b) extra calib docs at half length: states only
        half = data["calib_half"]
        for s in range(0, len(half), args.bs):
            ids = half[s: s + args.bs].to(dev)
            out = forward_text(m, ids, use_cache=True)
            for i in dl:
                st["st_half"][i].append(out.past_key_values.layers[i].recurrent_states[0].half().cpu())
        log(f"{name} calib-half done ({time.time() - t0:.0f}s)")
        # (c) eval docs: states at P and P-tail (source role)
        for s in range(0, len(data["eval"]), args.bs):
            ids = data["eval"][s: s + args.bs].to(dev)
            for L in ev:
                out = forward_text(m, ids[:, :L], use_cache=True)
                for i in dl:
                    ev[L][i].append(out.past_key_values.layers[i].recurrent_states[0].float().cpu())
                    if L == args.prefix:
                        ev_conv[L][i].append(out.past_key_values.layers[i].conv_states[0].float().cpu())
        log(f"{name} eval states done ({time.time() - t0:.0f}s)")
        tap.remove()
        res = {"tok": {i: {k_: torch.cat(v, 0) for k_, v in d.items()} for i, d in tok.items()},
               "st_full": {i: torch.cat(v, 0) for i, v in st["st_full"].items()},
               "st_half": {i: torch.cat(v, 0) for i, v in st["st_half"].items()},
               "eval": {L: {i: torch.cat(v, 0) for i, v in d.items()} for L, d in ev.items()},
               "eval_conv": {L: {i: torch.cat(v, 0) for i, v in d.items()} for L, d in ev_conv.items()}}
        torch.save(res, path)
        feats[name] = res
        log(f"extracted delta {name} in {time.time() - t0:.0f}s")
    return feats


# ----------------------------------------------------------------------------- fitting
@torch.no_grad()
def sufficient_stats(X, dev, chunk=8192):
    """Returns xm [d], XtX centered [d,d] fp64 for X [N,d] half cpu."""
    xm = X.double().mean(0).to(dev)
    d = X.shape[1]
    XtX = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for s in range(0, X.shape[0], chunk):
        xb = X[s: s + chunk].to(dev).float() - xm.float()
        XtX += (xb.T @ xb).double()
    return xm, XtX


@torch.no_grad()
def ridge_from_stats(xm, XtX, X, Y, lam_rel, dev, chunk=8192):
    ym = Y.double().mean(0).to(dev)
    d, o = X.shape[1], Y.shape[1]
    XtY = torch.zeros(d, o, dtype=torch.float64, device=dev)
    for s in range(0, X.shape[0], chunk):
        xb = X[s: s + chunk].to(dev).float() - xm.float()
        yb = Y[s: s + chunk].to(dev).float() - ym.float()
        XtY += (xb.T @ yb).double()
    lam = lam_rel * XtX.diagonal().mean()
    W = torch.linalg.solve(XtX + lam * torch.eye(d, dtype=torch.float64, device=dev), XtY)
    b = ym - xm @ W
    return W.float(), b.float()


def layer_match(args, fs, ft, src_layers, tgt_layers, dev, n_sub):
    """Held-out R2 matrices for k-maps and v-maps on a subsample. Returns R2k, R2v [T,S]."""
    N = fs[src_layers[0]]["k"].shape[0]
    idx_tr = torch.arange(0, min(n_sub, N - N // 10))
    idx_ho = torch.arange(N - N // 10, N)
    R = {"k": torch.zeros(len(tgt_layers), len(src_layers)), "v": torch.zeros(len(tgt_layers), len(src_layers))}
    for key in ("k", "v"):
        stats = {ls: sufficient_stats(fs[ls][key][idx_tr], dev) for ls in src_layers}
        for c, ls in enumerate(src_layers):
            xm, XtX = stats[ls]
            for a, lt in enumerate(tgt_layers):
                W, b = ridge_from_stats(xm, XtX, fs[ls][key][idx_tr], ft[lt][key][idx_tr], args.lam, dev)
                R[key][a, c] = r2(fs[ls][key][idx_ho], ft[lt][key][idx_ho], W, b, dev)
        log(f"layer-match {key} R2 (rows=tgt, cols=src):\n" + "\n".join(
            f"  L{lt:2d}: " + " ".join(f"{v:4.2f}" for v in row) + f"  best=L{src_layers[int(row.argmax())]}"
            for lt, row in zip(tgt_layers, R[key])))
    return R


@torch.no_grad()
def lift(S_src, Pm, Qm):
    """S_src [N,Hs,dk,dv], Pm [Ht,Hs,dk,dk], Qm [Ht,Hs,dv,dv] -> [N,Ht,dk,dv]"""
    outs = []
    for s in range(0, S_src.shape[0], 8):
        T1 = torch.einsum("xjab,njbc->nxjac", Pm, S_src[s: s + 8])
        outs.append(torch.einsum("nxjac,xjdc->nxad", T1, Qm))
    return torch.cat(outs, 0)


def head_blocks(W, Hs, Ht, d):
    """W [Hs*d, Ht*d] (x @ W) -> blocks M[t,s] = W[s*d:(s+1)*d, t*d:(t+1)*d]^T so that y_t = sum_s M[t,s] x_s."""
    return W.view(Hs, d, Ht, d).permute(2, 0, 3, 1).contiguous()  # [Ht,Hs,d_out,d_in]


@torch.no_grad()
def state_r2(S_pred, S_true):
    S_pred, S_true = S_pred.float(), S_true.float()
    sse = ((S_pred - S_true) ** 2).sum()
    sst = ((S_true - S_true.mean(0, keepdim=True)) ** 2).sum()
    return (1 - sse / sst).item()


@torch.no_grad()
def als_step_B(S_src, S_tgt, A, lam_rel, dev):
    """Given A [Ht,Hs,dk,dk], solve B [Ht,Hs,dv,dv] by ridge.  S_tgt[j'] = sum_j A[j',j] S_src[j] B[j',j]^T."""
    N, Hs, dk, dv = S_src.shape
    Ht = S_tgt.shape[1]
    B_new = torch.zeros(Ht, Hs, dv, dv, device=dev)
    for jp in range(Ht):
        # features for row r: concat_j (A[jp,j] S_src[j])[r,:]  -> X [N*dk, Hs*dv]
        X = torch.einsum("jab,njbc->njac", A[jp], S_src).permute(0, 2, 1, 3).reshape(N * dk, Hs * dv)
        Y = S_tgt[:, jp].reshape(N * dk, dv)
        XtX = X.T @ X
        lam = lam_rel * XtX.diagonal().mean()
        W = torch.linalg.solve(XtX.double() + lam.double() * torch.eye(Hs * dv, device=dev, dtype=torch.float64),
                               (X.T @ Y).double()).float()  # [Hs*dv, dv]; row block j = B[jp,j]^T
        B_new[jp] = W.view(Hs, dv, dv).transpose(1, 2)
    return B_new


@torch.no_grad()
def als_step_A(S_src, S_tgt, B, lam_rel, dev):
    """Given B, solve A.  Transpose the problem: S_tgt[j']^T = sum_j B[j',j] S_src[j]^T A[j',j]^T."""
    A_T = als_step_B(S_src.transpose(-1, -2).contiguous(), S_tgt.transpose(-1, -2).contiguous(), B, lam_rel, dev)
    return A_T


def fit_state_maps(args, feats, src, tgt, dev):
    path = os.path.join(args.cache, f"smaps_{src}_to_{tgt}_s{args.seq}_c{args.n_calib}_st{args.stride}_lam{args.lam}_als{args.als_iters}.pt")
    if os.path.exists(path):
        return torch.load(path)
    fs, ft = feats[src]["tok"], feats[tgt]["tok"]
    src_layers, tgt_layers = sorted(fs), sorted(ft)
    Hs = feats[src]["st_full"][src_layers[0]].shape[1]
    Ht = feats[tgt]["st_full"][tgt_layers[0]].shape[1]
    dk = dv = feats[src]["st_full"][src_layers[0]].shape[-1]
    Hk_s, Hk_t = fs[src_layers[0]]["k"].shape[1] // dk, ft[tgt_layers[0]]["k"].shape[1] // dk
    rep_s, rep_t = Hs // Hk_s, Ht // Hk_t
    R = layer_match(args, fs, ft, src_layers, tgt_layers, dev, args.n_match)
    score = (R["k"] + R["v"]) / 2
    report = {"match_r2_k": R["k"].tolist(), "match_r2_v": R["v"].tolist(), "src_layers": src_layers,
              "tgt_layers": tgt_layers, "per_layer": {}}
    maps = {}
    N = fs[src_layers[0]]["k"].shape[0]
    tr, ho = slice(0, N - N // 10), slice(N - N // 10, N)
    for a, lt in enumerate(tgt_layers):
        ls = src_layers[int(score[a].argmax())]
        t0 = time.time()
        # per-token maps
        xm, XtX = sufficient_stats(fs[ls]["k"][tr], dev)
        Pw, Pb = ridge_from_stats(xm, XtX, fs[ls]["k"][tr], ft[lt]["k"][tr], args.lam, dev)
        r2k = r2(fs[ls]["k"][ho], ft[lt]["k"][ho], Pw, Pb, dev)
        xm, XtX = sufficient_stats(fs[ls]["v"][tr], dev)
        Qw, Qb = ridge_from_stats(xm, XtX, fs[ls]["v"][tr], ft[lt]["v"][tr], args.lam, dev)
        r2v = r2(fs[ls]["v"][ho], ft[lt]["v"][ho], Qw, Qb, dev)
        # lift: Pm[j',j] = P block (kh_t(j'), kh_s(j)), Qm[j',j] = Q block (j', j)
        Pk = head_blocks(Pw, Hk_s, Hk_t, dk)  # [Hk_t,Hk_s,dk,dk]
        Qm = head_blocks(Qw, Hs, Ht, dv)  # [Ht,Hs,dv,dv]
        kt = torch.arange(Ht, device=dev) // rep_t
        ks = torch.arange(Hs, device=dev) // rep_s
        Pm = Pk[kt][:, ks]  # [Ht,Hs,dk,dk]
        # states
        S_src = torch.cat([feats[src]["st_full"][ls], feats[src]["st_half"][ls]], 0).to(dev).float()
        S_tgt = torch.cat([feats[tgt]["st_full"][lt], feats[tgt]["st_half"][lt]], 0).to(dev).float()
        n = S_src.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
        S_src, S_tgt = S_src[perm], S_tgt[perm]
        n_ho = max(n // 8, 8)
        Str_s, Sho_s = S_src[:-n_ho], S_src[-n_ho:]
        Str_t, Sho_t = S_tgt[:-n_ho], S_tgt[-n_ho:]
        r2_lift = state_r2(lift(Sho_s, Pm, Qm), Sho_t)
        r2_lift_tr = state_r2(lift(Str_s, Pm, Qm), Str_t)
        # scale-only correction of the lift (one scalar per target head), cheap sanity
        A, Bm = Pm.clone(), Qm.clone()
        hist = [r2_lift]
        for it in range(args.als_iters):
            Bm = als_step_B(Str_s, Str_t, A, args.als_lam, dev)
            A = als_step_A(Str_s, Str_t, Bm, args.als_lam, dev)
            hist.append(state_r2(lift(Sho_s, A, Bm), Sho_t))
        r2_als_tr = state_r2(lift(Str_s, A, Bm), Str_t)
        maps[lt] = {"src": ls, "Pm": Pm.half().cpu(), "Qm": Qm.half().cpu(), "A": A.half().cpu(), "B": Bm.half().cpu()}
        report["per_layer"][str(lt)] = {"src": ls, "r2_tok_k": r2k, "r2_tok_v": r2v, "r2_state_lift": r2_lift,
                                        "r2_state_lift_train": r2_lift_tr, "r2_state_als": hist[-1],
                                        "r2_state_als_train": r2_als_tr, "als_hist": hist}
        log(f"{src}->{tgt} L{lt:2d} <- L{ls:2d}: tok R2 k={r2k:.3f} v={r2v:.3f} | state R2 lift={r2_lift:.3f} "
            f"(train {r2_lift_tr:.3f}) ALS={' -> '.join(f'{h:.3f}' for h in hist)} (train {r2_als_tr:.3f}) [{time.time() - t0:.0f}s]")
        del S_src, S_tgt
        torch.cuda.empty_cache()
    out = {"maps": maps, "report": report}
    torch.save(out, path)
    return out


# ----------------------------------------------------------------------------- eval


# ----------------------------------------------------------------------------- main
