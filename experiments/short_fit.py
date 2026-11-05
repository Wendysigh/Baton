#!/usr/bin/env python
"""Prefix-length check and post hoc refits of the recurrent-state and conv-buffer maps.

The recorded state maps were refined on final states of 512- and 1,024-token sequences and judged by the plain
per-head cosine of the whole state. This script
  (1) measures, at prefix lengths 16..1024 on held-out documents, the plain and the readout-weighted per-head cosine
      of the copy, the lift and the recorded refined map, where readout weighting means comparing W S instead of S
      with W = (Sigma_q + eps I)^{1/2} and Sigma_q the covariance of the target head's l2-normalized queries: the
      target only reads the state through q^T S;
  (2) refits the bilinear map on states pooled over those lengths (each pair weighted by the inverse norm of the
      target state so every length counts the same) with one ALS round of ridge toward an anchor, the mix-scaled
      lift (baton/refine.py) or the copy, under the plain or the readout-weighted objective;
  (3) refits the conv-buffer map as a ridge toward zero (recorded) or toward the copy, selected by held-out R^2.
Variants written as map directories (common.map_dir), each a refine_*.pt plus symlinks or a tokmaps_*.pt:
  mll / mlc / ml     plain objective, lift / copy / better anchor per layer
  mlq / mlqc / mlqs  readout-weighted objective, lift / copy / better anchor per layer
  mlo / mlol         orthogonal (Procrustes) bilinear map, no shrinkage: target head i reads the stacked source
                     heads through a semi-orthogonal key-side map and an orthogonal value-side map, fit by
                     alternating Procrustes from the copy (mlo) or from the lift (mlol)
  mld<lam>           head-diagonal map: target head i reads only its matched source head (as the copy does) through
                     A_i S B_i^T, ridge toward the copy with strength lam, one ALS round; no cross-head mixing
  lift               the per-token lift alone
  cur                the recorded maps
  <variant>N         energy matching: every target head's state map and every channel of the conv-buffer map are
                     rescaled so that the mapped quantity has the second moment of the true one (a least-squares
                     map shrinks its prediction to cos * the true norm)
  <variant>C         conv-buffer map selected among zero- and copy-anchored ridges (N applies to it too)
usage: short_fit.py --direction 4B>9B [--write curN curNC ...]
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE, RAW  # noqa: E402
import argparse, glob, json, os, sys, time

import torch

from common import log, load_model, forward_text, MODELS, map_dir, ORDER  # noqa: E402
from gdn import delta_layers, lift  # noqa: E402
from refine import head_cos, fit_mix, als_step_B_reg, als_step_A_reg  # noqa: E402
from attention import text_model  # noqa: E402

BASE = {("4B", "9B"): f"{CACHE}/p4b9b", ("9B", "27B"): CACHE}
REC_VARIANTS = ("ml", "mlc", "mll", "mlq", "mlqc", "mlqs")


class QTap:
    """Accumulates per GDN layer the covariance of the l2-normalized queries, per value head (queries are repeated
    over the value heads of a key head, so heads of one group share it)."""

    def __init__(self, m):
        import transformers.models.qwen3_5.modeling_qwen3_5 as M
        self.M, self.tm, self.cur, self.enabled = M, text_model(m), None, False
        self.acc, self.n = {}, {}
        self.orig = M.torch_chunk_gated_delta_rule
        M.torch_chunk_gated_delta_rule = self._wrapped
        self.handles = [self.tm.layers[i].linear_attn.register_forward_pre_hook(self._mk(i), with_kwargs=True) for i in delta_layers(m)]

    def _mk(self, i):
        def hook(mod, args, kwargs):
            self.cur = i
        return hook

    def _wrapped(self, query, key, value, g, beta, **kw):
        if self.enabled:
            q = self.M.l2norm(query.float(), dim=-1, eps=1e-6)  # [B,T,Hv,dk]
            q = q.reshape(-1, q.shape[2], q.shape[3])
            self.acc[self.cur] = self.acc.get(self.cur, 0) + torch.einsum("nha,nhb->hab", q, q)
            self.n[self.cur] = self.n.get(self.cur, 0) + q.shape[0]
        return self.orig(query, key, value, g, beta, **kw)

    def cov(self):
        return {i: self.acc[i] / self.n[i] for i in self.acc}

    def remove(self):
        self.M.torch_chunk_gated_delta_rule = self.orig
        for h in self.handles:
            h.remove()


@torch.no_grad()
def collect(m, docs, L, layers, bs):
    out = {i: [] for i in layers}
    for s in range(0, len(docs), bs):
        o = forward_text(m, docs[s:s + bs, :L].to(next(m.parameters()).device), use_cache=True)
        for i in layers:
            out[i].append(o.past_key_values.layers[i].recurrent_states[0].half().cpu())
    return {i: torch.cat(v) for i, v in out.items()}


def eye_maps(Ht, Hs, head_idx, d, dev):
    """Copy as a bilinear map: target head j' reads source head head_idx[j'] through identity matrices."""
    A = torch.zeros(Ht, Hs, d, d, device=dev)
    for jp in range(Ht):
        A[jp, head_idx[jp]] = torch.eye(d, device=dev)
    return A, A.clone()


def sqrtm_psd(C, eps_rel):
    """(C + eps I)^{1/2} and its inverse for a batch of PSD matrices [H,d,d]; eps relative to the mean eigenvalue."""
    w, V = torch.linalg.eigh(C)
    w = w.clamp_min(0) + eps_rel * C.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None]
    return (V * w.sqrt()[:, None, :]) @ V.transpose(-1, -2), (V * w.rsqrt()[:, None, :]) @ V.transpose(-1, -2)


@torch.no_grad()
def wcos(S_pred, S_true, W):
    """per-head cosine between W S_pred and W S_true, W [Ht,dk,dk]."""
    return head_cos(torch.einsum("hab,nhbc->nhac", W, S_pred.float()), torch.einsum("hab,nhbc->nhac", W, S_true.float()))


def als_round(S_s, S_t, A0, B0, lam, dev, W=None, Winv=None):
    """One ALS round (B then A) of ridge toward (A0, B0); with W the residual is weighted on the key side, W (A S B^T - S_t)."""
    if W is None:
        Bn = als_step_B_reg(S_s, S_t, A0, B0, lam, dev)
        return als_step_A_reg(S_s, S_t, Bn, A0, lam, dev), Bn
    S_tw = torch.einsum("hab,nhbc->nhac", W, S_t)
    Bn = als_step_B_reg(S_s, S_tw, torch.einsum("hab,hjbc->hjac", W, A0), B0, lam, dev)
    At = als_step_A_reg(S_s, S_tw, Bn, torch.einsum("hab,hjbc->hjac", W, A0), lam, dev)
    return torch.einsum("hab,hjbc->hjac", Winv, At), Bn


@torch.no_grad()
def chunk_mean(X, dev):
    acc = torch.zeros(X.shape[1], device=dev, dtype=torch.float64)
    for s in range(0, X.shape[0], 4096):
        acc += X[s:s + 4096].to(dev).double().sum(0)
    return (acc / X.shape[0]).float()


def polar(M):
    """nearest (semi-)orthogonal matrix to M: U V^T from its SVD."""
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


@torch.no_grad()
def procrustes_bilinear(S_s, S_t, A0, B0, iters):
    """S_t[i] ~ Ahat_i S_stack B_i^T with Ahat_i [dk, Hs*dk] row-orthonormal and B_i [dv, dv] orthogonal, by alternating
    orthogonal Procrustes; A0 [Ht,Hs,dk,dk] (any init, projected), B0 [Ht,dv,dv]. Returns maps in the (A, B) block format."""
    N, Hs, dk, dv = S_s.shape
    Ht = S_t.shape[1]
    Ss = S_s.reshape(N, Hs * dk, dv)
    A = torch.stack([polar(A0[i].permute(1, 0, 2).reshape(dk, Hs * dk)) for i in range(Ht)])
    B = torch.stack([polar(B0[i]) for i in range(Ht)])
    for _ in range(iters):
        for i in range(Ht):
            X = Ss @ B[i].T  # [N, Hs*dk, dv]
            A[i] = polar(torch.einsum("nab,ncb->ac", S_t[:, i], X))
            Z = A[i] @ Ss  # [N, dk, dv]
            B[i] = polar(torch.einsum("nab,nac->bc", S_t[:, i], Z))
    return A.view(Ht, dk, Hs, dk).permute(0, 2, 1, 3).contiguous(), B[:, None].expand(Ht, Hs, dv, dv).contiguous()


@torch.no_grad()
def diag_round(S_s, S_t, head_idx, A, B, lam_rel, dev):
    """One ALS round of the head-diagonal map S_t[i] ~ A_i S_s[head_idx[i]] B_i^T, ridge toward (A, B) [Ht,d,d]."""
    N, Hs, dk, dv = S_s.shape
    Ht = S_t.shape[1]
    Ss = S_s[:, head_idx]  # [N,Ht,dk,dv]
    def step(X, Y, W0):  # Y ~ X W, X [N*r, c] rows, ridge toward W0 [c, c']
        XtX, XtY = (X.T @ X).double(), (X.T @ Y).double()
        lam = lam_rel * XtX.diagonal().mean()
        return torch.linalg.solve(XtX + lam * torch.eye(XtX.shape[0], device=dev, dtype=torch.float64), XtY + lam * W0.double()).float()
    Bn, An = torch.zeros_like(B), torch.zeros_like(A)
    for i in range(Ht):
        X = (A[i] @ Ss[:, i]).reshape(-1, dv)  # rows (n, a): (A S)[a, :] -> target row = X B^T
        Bn[i] = step(X, S_t[:, i].reshape(-1, dv), B[i].T).T
    for i in range(Ht):
        X = (Ss[:, i] @ Bn[i].T).transpose(1, 2).reshape(-1, dk)  # rows (n, c): (S B^T)[:, c]^T -> target col = X A^T
        An[i] = step(X, S_t[:, i].transpose(1, 2).reshape(-1, dk), A[i].T).T
    Ab = torch.zeros(Ht, Hs, dk, dk, device=dev); Bb = torch.zeros(Ht, Hs, dv, dv, device=dev)
    for i in range(Ht):
        Ab[i, head_idx[i]], Bb[i, head_idx[i]] = An[i], Bn[i]
    return Ab, Bb


@torch.no_grad()
def ridge_anchor(X, Y, W0, lam_rel, dev):
    """Ridge with bias toward W0: W = (Xc'Xc + lam I)^-1 (Xc'Yc + lam W0), lam relative to the mean diagonal."""
    xm, ym = chunk_mean(X, dev), chunk_mean(Y, dev)
    XtX = torch.zeros(X.shape[1], X.shape[1], device=dev, dtype=torch.float64)
    XtY = torch.zeros(X.shape[1], Y.shape[1], device=dev, dtype=torch.float64)
    for s in range(0, X.shape[0], 4096):
        xc, yc = X[s:s + 4096].to(dev).float() - xm, Y[s:s + 4096].to(dev).float() - ym
        XtX += (xc.T @ xc).double(); XtY += (xc.T @ yc).double()
    lam = lam_rel * XtX.diagonal().mean()
    W = torch.linalg.solve(XtX + lam * torch.eye(XtX.shape[0], device=dev, dtype=torch.float64), XtY + lam * W0.double()).float()
    return W, ym - xm @ W


@torch.no_grad()
def var_match(X, Y, W, b, dev):
    """Rescale a linear map per output channel so that its prediction has the variance of the target."""
    xm, ym = chunk_mean(X, dev), chunk_mean(Y, dev)
    vy = vp = 0.0
    for s in range(0, X.shape[0], 4096):
        x, y = X[s:s + 4096].to(dev).float(), Y[s:s + 4096].to(dev).float()
        vp = vp + ((x - xm) @ W).pow(2).sum(0); vy = vy + (y - ym).pow(2).sum(0)
    g = (vy / vp.clamp_min(1e-12)).sqrt()
    return W * g[None, :], ym - g * (xm @ W)


@torch.no_grad()
def r2_of(X, Y, W, b, dev):
    sse = sst = 0.0
    ym = chunk_mean(Y, dev)
    for s in range(0, X.shape[0], 4096):
        x, y = X[s:s + 4096].to(dev).float(), Y[s:s + 4096].to(dev).float()
        sse += float(((x @ W + b - y) ** 2).sum()); sst += float(((y - ym) ** 2).sum())
    return 1 - sse / sst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--n_train", type=int, default=128, help="calibration docs per length (disjoint across lengths)")
    ap.add_argument("--lams_lift", type=float, nargs="*", default=[0.3, 3.0])
    ap.add_argument("--lams_copy", type=float, nargs="*", default=[0.3, 3.0, 30.0])
    ap.add_argument("--qweight", action="store_true", help="also fit the readout-weighted candidates (mlq*)")
    ap.add_argument("--ortho", action="store_true", help="also fit the orthogonal Procrustes candidates (mlo, mlol)")
    ap.add_argument("--lams_diag", type=float, nargs="*", default=[], help="head-diagonal copy-anchored candidates (mld*)")
    ap.add_argument("--ortho_iters", type=int, default=8)
    ap.add_argument("--lams_conv", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--q_eps", type=float, default=0.1, help="floor of the readout weighting, relative to the mean eigenvalue")
    ap.add_argument("--out", default=f"{RAW}/extra/short_fit")
    ap.add_argument("--write", nargs="+", default=list(REC_VARIANTS) + [v + "C" for v in REC_VARIANTS])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")
    s, t = args.direction.split(">")
    small, large = sorted((s, t), key=ORDER.get)
    base = BASE[(small, large)]
    role = {small: "small", large: "large"}
    rs, rt = role[s], role[t]
    data = torch.load(os.path.join(base, "data_s1024_c1024_e64.pt"))
    assert len(args.lengths) * args.n_train <= len(data["calib"])
    log("loading", s, t)
    src_m, tgt_m = load_model(MODELS[s], dev), load_model(MODELS[t], dev)
    dl_s, dl_t = delta_layers(src_m), delta_layers(tgt_m)
    scfg, tcfg = text_model(src_m).config, text_model(tgt_m).config
    Hs, Hk_s, Ht, Hk_t = scfg.linear_num_value_heads, scfg.linear_num_key_heads, tcfg.linear_num_value_heads, tcfg.linear_num_key_heads
    reps, rept, d = Hs // Hk_s, Ht // Hk_t, tcfg.linear_key_head_dim
    head_idx = [(j // rept) * reps + (j % rept) % reps for j in range(Ht)]
    # states per length: training docs disjoint across lengths, held-out = the 64 eval docs at every length;
    # the target's query covariance is accumulated over the training passes
    t0 = time.time()
    qtap = QTap(tgt_m)
    tr, ho = {}, {}
    for k, L in enumerate(args.lengths):
        docs = data["calib"][k * args.n_train:(k + 1) * args.n_train]
        bs = max(1, min(32, 8192 // L))
        qtap.enabled = True
        tr[L] = {"s": collect(src_m, docs, L, dl_s, bs), "t": collect(tgt_m, docs, L, dl_t, bs)}
        qtap.enabled = False
        ho[L] = {"s": collect(src_m, data["eval"], L, dl_s, bs), "t": collect(tgt_m, data["eval"], L, dl_t, bs)}
        log(f"states at L={L} ({time.time() - t0:.0f}s)")
    qcov = qtap.cov()
    qtap.remove()
    del src_m, tgt_m
    torch.cuda.empty_cache()
    rec = torch.load(glob.glob(os.path.join(base, f"refine_{rs}_to_{rt}_s1024_c512_st8.pt"))[0])
    report, cands_all = {}, {}
    for lt in dl_t:
        t1 = time.time()
        mp = rec["maps"][lt]
        ls = mp["src"]
        Pm, Qm = mp["Pm"].to(dev).float(), mp["Qm"].to(dev).float()
        W, Winv = sqrtm_psd(qcov[lt], args.q_eps)
        Wd, _ = sqrtm_psd(qcov[lt], 1e-3)  # near-pure readout weighting, diagnostic metric only
        S_s = torch.cat([tr[L]["s"][ls] for L in args.lengths]).to(dev).float()
        S_t = torch.cat([tr[L]["t"][lt] for L in args.lengths]).to(dev).float()
        norms = S_t.flatten(1).norm(dim=1)
        w = (1.0 / norms.clamp_min(1e-6)); w = (w / w.mean()).sqrt().view(-1, 1, 1, 1)  # inverse-norm sample weights
        S_s, S_t = S_s * w, S_t * w
        cands = {"copy": eye_maps(Ht, Hs, head_idx, d, dev), "lift": (Pm, Qm), "cur": (mp["A"].to(dev).float(), mp["B"].to(dev).float())}
        C = fit_mix(S_s, S_t, Pm, Qm, 1e-3, dev)
        A0, B0 = Pm * C.view(Ht, Hs, 1, 1), Qm
        for lam in args.lams_lift:
            cands[f"mll{lam:g}"] = als_round(S_s, S_t, A0, B0, lam, dev)
            if args.qweight:
                cands[f"mlq{lam:g}"] = als_round(S_s, S_t, A0, B0, lam, dev, W, Winv)
        A0, B0 = cands["copy"]
        for lam in args.lams_copy:
            cands[f"mlc{lam:g}"] = als_round(S_s, S_t, A0, B0, lam, dev)
            if args.qweight:
                cands[f"mlqc{lam:g}"] = als_round(S_s, S_t, A0, B0, lam, dev, W, Winv)
        for lam in args.lams_diag:
            I = torch.eye(d, device=dev).expand(Ht, d, d).contiguous()
            cands[f"mld{lam:g}"] = diag_round(S_s, S_t, head_idx, I, I, lam, dev)
        if args.ortho:
            Ac, _ = cands["copy"]
            cands["mlo"] = procrustes_bilinear(S_s, S_t, Ac, torch.eye(d, device=dev).expand(Ht, d, d), args.ortho_iters)
            jbest = C.abs().argmax(1)  # value-side init: the lift block of the most-mixed source head
            cands["mlol"] = procrustes_bilinear(S_s, S_t, Pm * C.view(Ht, Hs, 1, 1), Qm[torch.arange(Ht), jbest], args.ortho_iters)
        # energy matching: per target head, sqrt of the ratio of mean squared norms, true over mapped, averaged over
        # prefix lengths on the (unweighted) training states
        S_s0, S_t0 = S_s / w, S_t / w
        gam = {}
        for name, (A, B) in list(cands.items()):
            P_ = lift(S_s0, A, B)
            g = []
            for k in range(len(args.lengths)):
                sl = slice(k * args.n_train, (k + 1) * args.n_train)
                g.append((S_t0[sl].flatten(2).pow(2).sum((0, 2)) / lift(S_s0[sl], A, B).flatten(2).pow(2).sum((0, 2)).clamp_min(1e-12)).sqrt())
            g = torch.stack(g).mean(0)  # [Ht]
            gam[name] = g
            cands[name + "N"] = (A * g.view(-1, 1, 1, 1), B)
        del S_s0, S_t0, P_
        rep = {}
        for name, (A, B) in cands.items():
            per, perw, perd = {}, {}, {}
            for L in args.lengths:
                P_, T_ = lift(ho[L]["s"][ls].to(dev).float(), A, B), ho[L]["t"][lt].to(dev).float()
                per[str(L)], perw[str(L)], perd[str(L)] = head_cos(P_, T_), wcos(P_, T_, W), wcos(P_, T_, Wd)
            rep[name] = {"cos": per, "mean": sum(per.values()) / len(per), "wcos": perw, "wmean": sum(perw.values()) / len(perw),
                         "dcos": perd, "dmean": sum(perd.values()) / len(perd)}
        rep["norm_by_len"] = {str(L): float(norms[k * args.n_train:(k + 1) * args.n_train].mean()) for k, L in enumerate(args.lengths)}
        rep["energy_gain"] = {k_: float(g.mean()) for k_, g in gam.items()}
        pool = lambda *pres: [k_ for k_ in cands if not k_.endswith("N") and any(k_.startswith(p) and not (p == "mlq" and k_.startswith("mlqc")) for p in pres)]
        for var, pre, key in (("mll", ("mll",), "mean"), ("mlc", ("mlc", "copy"), "mean"), ("ml", ("mll", "mlc", "copy"), "mean"),
                              ("mlq", ("mlq",), "wmean"), ("mlqc", ("mlqc", "copy"), "wmean"), ("mlqs", ("mlq", "mlqc", "copy"), "wmean")):
            if pool(*pre) and (var in ("mll", "mlc", "ml") or args.qweight):
                rep[f"best_{var}"] = max(pool(*pre), key=lambda k_: rep[k_][key])
        rep["best_cur"] = "cur"
        report[str(lt)] = rep
        cands_all[lt] = {k_: (A.half().cpu(), B.half().cpu()) for k_, (A, B) in cands.items()}
        log(f"{args.direction} L{lt:2d}<-L{ls:2d} cos/wcos: " + " ".join(f"{k_}={v['mean']:.3f}/{v['wmean']:.3f}" for k_, v in rep.items() if isinstance(v, dict) and "mean" in v and not k_.endswith("N"))
            + " | energy gain " + " ".join(f"{k_}={v:.2f}" for k_, v in rep["energy_gain"].items() if k_ in ("copy", "cur", rep.get("best_ml", "cur")))
            + f" | best ml={rep.get('best_ml')} [{time.time() - t1:.0f}s]")
        del S_s, S_t, cands
        torch.cuda.empty_cache()
    # conv-buffer map: pre-conv projections of the stream features (stride-16 tokens of the 512 calibration docs)
    st_s = torch.load(glob.glob(os.path.join(base, f"stream_{rs}_s1024_c512_e64_sp16_p896_n*.pt"))[0], weights_only=False)["pre"]
    st_t = torch.load(glob.glob(os.path.join(base, f"stream_{rt}_s1024_c512_e64_sp16_p896_n*.pt"))[0], weights_only=False)["pre"]
    tok_base = torch.load(glob.glob(os.path.join(base, f"tokmaps_{rs}_to_{rt}_s1024_c512_st8_sp16.pt"))[0], weights_only=False)
    dv = tcfg.linear_value_head_dim
    qk = 2 * Hk_s * d
    conv_idx = torch.cat([torch.arange(qk), (qk + torch.tensor(head_idx)[:, None] * dv + torch.arange(dv)[None]).reshape(-1)]).to(dev)
    conv_rep, conv_maps = {}, {}
    for lt in dl_t:
        t1 = time.time()
        ls = rec["maps"][lt]["src"]
        X, Y = st_s[ls], st_t[lt]
        N = X.shape[0]; trs, hos = slice(0, N - N // 10), slice(N - N // 10, N)
        W0 = torch.zeros(X.shape[1], Y.shape[1], device=dev)
        W0[conv_idx, torch.arange(Y.shape[1])] = 1.0
        cc = {"copy": (W0, torch.zeros(Y.shape[1], device=dev)), "cur": (tok_base["maps"][lt]["pre"][0].to(dev).float(), tok_base["maps"][lt]["pre"][1].to(dev).float())}
        for lam in args.lams_conv:
            cc[f"cc{lam:g}"] = ridge_anchor(X[trs], Y[trs], W0, lam, dev)
        # energy matching per output channel around the mean: Y' = ym + gamma (Y_hat - mean Y_hat)
        for k_, (Wm, bm) in list(cc.items()):
            cc[k_ + "N"] = var_match(X[trs], Y[trs], Wm, bm, dev)
        r2 = {k_: r2_of(X[hos], Y[hos], Wm, bm, dev) for k_, (Wm, bm) in cc.items()}
        best = max((k_ for k_ in cc if k_ not in ("cur", "curN") and not k_.endswith("N")), key=r2.get)
        conv_rep[str(lt)] = {"r2": r2, "best": best}
        conv_maps[lt] = {k_: (Wm.half().cpu(), bm.cpu()) for k_, (Wm, bm) in cc.items()}
        log(f"{args.direction} conv L{lt:2d}<-L{ls:2d} held-out R2: " + " ".join(f"{k_}={v:.3f}" for k_, v in r2.items()) + f" | best={best} [{time.time() - t1:.0f}s]")
        del cc
        torch.cuda.empty_cache()
    json.dump({"args": vars(args), "report": report, "conv": conv_rep}, open(os.path.join(args.out, f"{args.direction.replace('>', 'to')}.json"), "w"), indent=1)
    # summary: mean over layers per length
    first = report[str(dl_t[0])]
    names = [k_ for k_ in first if isinstance(first[k_], dict) and "mean" in first[k_] and not k_.endswith("N")]
    for key, title in (("cos", "plain"), ("wcos", f"readout-weighted (eps {args.q_eps})"), ("dcos", "readout-weighted (eps 1e-3, diagnostic)")):
        log(f"mean over layers of held-out {title} cosine, per prefix length")
        log("cand       " + " ".join(f"{L:>6d}" for L in args.lengths) + "   mean")
        for k_ in names:
            vals = [sum(report[str(lt)][k_][key][str(L)] for lt in dl_t) / len(dl_t) for L in args.lengths]
            log(f"{k_:10s} " + " ".join(f"{v:6.3f}" for v in vals) + f"  {sum(vals) / len(vals):6.3f}")
        for var in REC_VARIANTS:
            if f"best_{var}" not in first:
                continue
            vals = [sum(report[str(lt)][report[str(lt)][f'best_{var}']][key][str(L)] for lt in dl_t) / len(dl_t) for L in args.lengths]
            log(f"{'sel_' + var:10s} " + " ".join(f"{v:6.3f}" for v in vals) + f"  {sum(vals) / len(vals):6.3f}")
    log("state-map energy gain (mean over layers and heads): " + " ".join(f"{k_}={sum(report[str(lt)]['energy_gain'][k_] for lt in dl_t) / len(dl_t):.3f}" for k_ in first["energy_gain"]))
    log("conv-buffer map, mean held-out R2 over layers: " + " ".join(f"{k_}={sum(conv_rep[str(lt)]['r2'][k_] for lt in dl_t) / len(dl_t):.3f}" for k_ in conv_rep[str(dl_t[0])]["r2"])
        + " | selected " + " ".join(conv_rep[str(lt)]["best"] for lt in dl_t))
    # write the variants: refine file per state-map variant, symlinked attention maps, token maps symlinked or rewritten
    for var in args.write:
        rvar, conv = (var[:-1], True) if var.endswith("C") else (var, False)
        rvar, nm = (rvar[:-1], True) if rvar.endswith("N") else (rvar, False)
        if f"best_{rvar}" not in report[str(dl_t[0])] and rvar not in cands_all[dl_t[0]]:
            log("skip", var, "(no candidates)"); continue
        out_dir = map_dir(small, large, var)
        os.makedirs(out_dir, exist_ok=True)
        pats = [f"maps_{rs}_to_{rt}_s1024_c512_st4_*.pt"] + ([] if conv or nm else [f"tokmaps_{rs}_to_{rt}_s1024_c512_st8_sp16.pt"])
        for pat in pats:
            for f in glob.glob(os.path.join(base, pat)):
                dst = os.path.join(out_dir, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(f, dst)
        maps = {}
        for lt in dl_t:
            best = report[str(lt)].get(f"best_{rvar}", rvar) + ("N" if nm else "")  # a selected variant or a fixed candidate
            A, B = cands_all[lt][best]
            maps[lt] = {"src": rec["maps"][lt]["src"], "best": best, "A": A, "B": B, "Pm": rec["maps"][lt]["Pm"], "Qm": rec["maps"][lt]["Qm"]}
        torch.save({"maps": maps, "report": {lt: report[str(lt)] for lt in dl_t}, "variant": var, "args": vars(args)},
                   os.path.join(out_dir, f"refine_{rs}_to_{rt}_s1024_c512_st8.pt"))
        if conv or nm:
            pick = lambda lt: (conv_rep[str(lt)]["best"] if conv else "cur") + ("N" if nm else "")
            tm = {"maps": {lt: {**tok_base["maps"][lt], "pre": conv_maps[lt][pick(lt)]} for lt in dl_t}, "report": {**tok_base["report"], "conv_refit": conv_rep}}
            torch.save(tm, os.path.join(out_dir, f"tokmaps_{rs}_to_{rt}_s1024_c512_st8_sp16.pt"))
        log("wrote", out_dir, {lt: maps[lt]["best"] for lt in dl_t})
    log("done")


if __name__ == "__main__":
    main()
