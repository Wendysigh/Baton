#!/usr/bin/env python
"""Attention-cache calibration: token data, paired KV extraction (post k_norm, pre-RoPE), and ridge maps from
source attention layers to each target attention layer, with layer selection by held-out R^2."""
from paths import FINEWEB_GLOB  # noqa: E402
import argparse, glob, json, math, os, time
from collections import defaultdict

import torch
import torch.nn.functional as F

PARQUET_GLOB = FINEWEB_GLOB


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data
def build_data(args, tok):
    path = os.path.join(args.cache, f"data_s{args.seq}_c{args.n_calib}_e{args.n_eval}.pt")
    if os.path.exists(path):
        return torch.load(path)
    import pyarrow.parquet as pq
    files = sorted(glob.glob(PARQUET_GLOB))
    assert len(files) >= 2, files

    def take(file, n, min_chars):
        out = []
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=512, columns=["text", "token_count"]):
            for text, tc in zip(batch.column("text").to_pylist(), batch.column("token_count").to_pylist()):
                if len(text) < min_chars:
                    continue
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) >= args.seq:
                    out.append(torch.tensor(ids[: args.seq], dtype=torch.long))
                if len(out) >= n:
                    return torch.stack(out)
        raise RuntimeError("not enough long docs")

    calib = take(files[0], args.n_calib, min_chars=4 * args.seq)
    evals = take(files[-1], args.n_eval, min_chars=4 * args.seq)
    data = {"calib": calib, "eval": evals}
    torch.save(data, path)
    return data


# ----------------------------------------------------------------------------- model utils
def load_model(path, dev):
    from transformers import Qwen3_5ForConditionalGeneration
    m = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(dev)
    m.eval()
    return m


def text_model(m):
    return m.model.language_model


def attn_layers(m):
    cfg = text_model(m).config
    return [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]


class KVTap:
    """Forward hooks that capture K (post k_norm, pre-RoPE) and V per attention layer, plus attention output."""

    def __init__(self, m, capture_attn_out=False):
        self.tm = text_model(m)
        self.layers = attn_layers(m)
        self.hd = self.tm.config.head_dim
        self.store = {}
        self.attn_out = {}
        self.handles = []
        self.enabled = True
        for i in self.layers:
            attn = self.tm.layers[i].self_attn
            self.handles.append(attn.k_proj.register_forward_hook(self._mk_k(i, attn)))
            self.handles.append(attn.v_proj.register_forward_hook(self._mk_v(i)))
            if capture_attn_out:
                self.handles.append(attn.o_proj.register_forward_hook(self._mk_o(i)))

    def _mk_k(self, i, attn):
        def hook(mod, inp, out):
            if not self.enabled:
                return
            B, T, _ = out.shape
            k = attn.k_norm(out.view(B, T, -1, self.hd))  # [B,T,Hkv,hd] pre-RoPE, post-norm
            self.store.setdefault(i, {})["K"] = k.reshape(B, T, -1)
        return hook

    def _mk_v(self, i):
        def hook(mod, inp, out):
            if not self.enabled:
                return
            self.store.setdefault(i, {})["V"] = out  # [B,T,Hkv*hd]
        return hook

    def _mk_o(self, i):
        def hook(mod, inp, out):
            if not self.enabled:
                return
            self.attn_out[i] = out  # [B,T,hidden]
        return hook

    def clear(self):
        self.store = {}
        self.attn_out = {}

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def rope_cos_sin(m, P, device):
    tm = text_model(m)
    pos = torch.arange(P, device=device).view(1, 1, -1).expand(3, 1, -1)
    dummy = torch.zeros(1, device=device, dtype=torch.bfloat16)
    cos, sin = tm.rotary_emb(dummy, pos)  # [1,P,rot]
    return cos.float(), sin.float()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_k(k, cos, sin):
    """k: [B,Hkv,P,hd] float; cos/sin: [1,P,rot]."""
    rot = cos.shape[-1]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([k_rot, k_pass], dim=-1)


@torch.no_grad()
def forward_text(m, ids, past=None, use_cache=False):
    tm = text_model(m)
    out = tm(input_ids=ids, past_key_values=past, use_cache=use_cache)
    return out


# ----------------------------------------------------------------------------- extract
@torch.no_grad()
def extract(args, models, data):
    """Returns dict name -> {'calib': {layer: {'K': [N,D], 'V': [N,D]}}, 'eval': {layer: {'K': [Ne,P,D], 'V': ...}}}"""
    feats = {}
    for name, m in models.items():
        path = os.path.join(args.cache, f"feat_{name}_s{args.seq}_c{args.n_calib}_e{args.n_eval}_st{args.stride}_p{args.prefix}.pt")
        if os.path.exists(path):
            feats[name] = torch.load(path)
            log(f"loaded features {name}")
            continue
        dev = next(m.parameters()).device
        tap = KVTap(m)
        res = {"calib": defaultdict(lambda: {"K": [], "V": []}), "eval": defaultdict(lambda: {"K": [], "V": []})}
        t0 = time.time()
        for split, ids_all in (("calib", data["calib"]), ("eval", data["eval"])):
            for s in range(0, len(ids_all), args.bs):
                ids = ids_all[s: s + args.bs].to(dev)
                if split == "eval":
                    ids = ids[:, : args.prefix]
                tap.clear()
                forward_text(m, ids)
                for i, d in tap.store.items():
                    for kv in ("K", "V"):
                        x = d[kv].float()
                        if split == "calib":
                            x = x[:, args.stride - 1:: args.stride].reshape(-1, x.shape[-1])
                        res[split][i][kv].append(x.half().cpu())
                if s % (args.bs * 16) == 0:
                    log(f"{name} {split} {s}/{len(ids_all)} ({time.time() - t0:.0f}s)")
        out = {split: {i: {kv: torch.cat(v, 0) for kv, v in d.items()} for i, d in res[split].items()} for split in res}
        torch.save(out, path)
        feats[name] = out
        tap.remove()
        log(f"extracted {name} in {time.time() - t0:.0f}s")
    return feats


# ----------------------------------------------------------------------------- fit
@torch.no_grad()
def ridge_fit(X, Y, lam_rel, dev):
    """X: [N,d] half cpu, Y: [N,o]. Centered ridge in fp64 on GPU. Returns W [d,o] fp32, b [o] fp32, xm, ym."""
    N = X.shape[0]
    xm = X.double().mean(0)
    ym = Y.double().mean(0)
    d, o = X.shape[1], Y.shape[1]
    XtX = torch.zeros(d, d, dtype=torch.float64, device=dev)
    XtY = torch.zeros(d, o, dtype=torch.float64, device=dev)
    xm_d, ym_d = xm.to(dev), ym.to(dev)
    for s in range(0, N, 16384):
        xb = X[s: s + 16384].to(dev).double() - xm_d
        yb = Y[s: s + 16384].to(dev).double() - ym_d
        XtX += xb.T @ xb
        XtY += xb.T @ yb
    lam = lam_rel * XtX.diagonal().mean()
    W = torch.linalg.solve(XtX + lam * torch.eye(d, dtype=torch.float64, device=dev), XtY)
    b = ym_d - xm_d @ W
    return W.float(), b.float()


@torch.no_grad()
def r2(X, Y, W, b, dev):
    N = X.shape[0]
    sse = 0.0
    ym = Y.double().mean(0).to(dev)
    sst = 0.0
    for s in range(0, N, 16384):
        xb = X[s: s + 16384].to(dev).float()
        yb = Y[s: s + 16384].to(dev).float()
        pred = xb @ W + b
        sse += ((pred - yb) ** 2).sum().item()
        sst += ((yb.double() - ym) ** 2).sum().item()
    return 1.0 - sse / sst


def fit_maps(args, feats, src, tgt, dev):
    """Returns maps[topk][kv][tgt_layer] = (src_layers, W, b) and an R^2 report."""
    path = os.path.join(args.cache, f"maps_{src}_to_{tgt}_s{args.seq}_c{args.n_calib}_st{args.stride}_lam{args.lam}.pt")
    if os.path.exists(path):
        return torch.load(path)
    fs, ft = feats[src]["calib"], feats[tgt]["calib"]
    src_layers, tgt_layers = sorted(fs), sorted(ft)
    N = next(iter(fs.values()))["K"].shape[0]
    n_ho = N // 10
    tr, ho = slice(0, N - n_ho), slice(N - n_ho, N)
    report = {"single_r2": {}, "topk_r2": {}}
    maps = {}
    # 1) single-source R^2 matrix
    single = {}
    for kv in ("K", "V"):
        R = torch.zeros(len(tgt_layers), len(src_layers))
        for a, lt in enumerate(tgt_layers):
            for c, ls in enumerate(src_layers):
                W, b = ridge_fit(fs[ls][kv][tr], ft[lt][kv][tr], args.lam, dev)
                R[a, c] = r2(fs[ls][kv][ho], ft[lt][kv][ho], W, b, dev)
                single[(kv, lt, ls)] = (W.cpu(), b.cpu())
        report["single_r2"][kv] = {"tgt_layers": tgt_layers, "src_layers": src_layers, "r2": R.tolist()}
        log(f"{src}->{tgt} {kv} single-source held-out R2 (rows=tgt, cols=src):\n" +
            "\n".join(f"  L{lt:2d}: " + " ".join(f"{v:5.2f}" for v in row) + f"  best=L{src_layers[int(row.argmax())]}"
                      for lt, row in zip(tgt_layers, R)))
    # 2) top-k concatenation
    for k in args.topk:
        k_eff = min(k, len(src_layers))
        maps[k_eff] = {}
        report["topk_r2"][k_eff] = {}
        for kv in ("K", "V"):
            R = torch.tensor(report["single_r2"][kv]["r2"])
            maps[k_eff][kv] = {}
            r2s = []
            for a, lt in enumerate(tgt_layers):
                sel = [src_layers[j] for j in R[a].topk(k_eff).indices.tolist()]
                sel = sorted(sel)
                if k_eff == 1:
                    W, b = single[(kv, lt, sel[0])]
                    W, b = W.to(dev), b.to(dev)
                else:
                    Xtr = torch.cat([fs[l][kv][tr] for l in sel], 1)
                    W, b = ridge_fit(Xtr, ft[lt][kv][tr], args.lam, dev)
                Xho = torch.cat([fs[l][kv][ho] for l in sel], 1)
                rr = r2(Xho, ft[lt][kv][ho], W, b, dev)
                r2s.append(rr)
                maps[k_eff][kv][lt] = (sel, W.cpu(), b.cpu())
            report["topk_r2"][k_eff][kv] = dict(zip(map(str, tgt_layers), r2s))
            log(f"{src}->{tgt} top{k_eff} {kv} held-out R2 per tgt layer: " + " ".join(f"{v:.3f}" for v in r2s) +
                f" | mean {sum(r2s) / len(r2s):.3f}")
    out = {"maps": maps, "report": report}
    torch.save(out, path)
    return out


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def mapped_prefix_kv(maps_k, feats_src_eval, seq_idx, dev, B):
    """Return dict tgt_layer -> (K [B,Hkv,P,hd] pre-RoPE fp32, V [B,Hkv,P,hd])"""
    out = {}
    for kv in ("K", "V"):
        for lt, (sel, W, b) in maps_k[kv].items():
            X = torch.cat([feats_src_eval[l][kv][seq_idx] for l in sel], -1).to(dev).float()  # [B,P,D*k]
            Y = X @ W.to(dev) + b.to(dev)  # [B,P,1024]
            out.setdefault(lt, {})[kv] = Y
    return out


# ----------------------------------------------------------------------------- main
