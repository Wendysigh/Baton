#!/usr/bin/env python
"""Explicit map registry, verified map loading, segmented prefill with taps, and poisoned cache installation."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import CACHE, model_path  # noqa: E402
import copy, glob, hashlib, json, os, sys

import torch

from attention import text_model, attn_layers, KVTap, rope_cos_sin, apply_rope_k, forward_text  # noqa: E402
from gdn import delta_layers, lift  # noqa: E402

MODELS = {s: model_path(s) for s in ("4B", "9B", "27B")}
SIZE_ORDER = {"4B": 0, "9B": 1, "27B": 2}
CALIB = {
    "chat": {("9B", "27B"): f"{CACHE}/chatcal", ("4B", "9B"): f"{CACHE}/p4b9b_chat",
             ("4B", "27B"): f"{CACHE}/p4b27b_chat"},
    "web": {("9B", "27B"): CACHE, ("4B", "9B"): f"{CACHE}/p4b9b"},
}
REGISTRY_PATH = os.path.join(CACHE, "map_registry.json")


def pair_key(a, b):
    return tuple(sorted((a, b), key=SIZE_ORDER.get))


def map_paths(calib, src, tgt):
    small, large = pair_key(src, tgt)
    d = CALIB[calib][(small, large)]
    s, t = ("small", "large") if src == small else ("large", "small")
    paths = {"attn": os.path.join(d, f"maps_{s}_to_{t}_s1024_c512_st4_lam0.0001.pt"),
             "state": os.path.join(d, f"refine_{s}_to_{t}_s1024_c512_st8.pt"),
             "tok": os.path.join(d, f"tokmaps_{s}_to_{t}_s1024_c512_st8_sp16.pt")}
    # refuse ambiguity: exactly one attention map file of this direction may exist
    cands = glob.glob(os.path.join(d, f"maps_{s}_to_{t}_s1024_c512_st4_*.pt"))
    assert cands == [paths["attn"]], f"ambiguous or missing attention maps: {cands}"
    for p in paths.values():
        assert os.path.isfile(p), p
    return paths


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def registry_entry(path):
    """sha256 cached by (path, size, mtime) in map_registry.json"""
    reg = json.load(open(REGISTRY_PATH)) if os.path.exists(REGISTRY_PATH) else {}
    st = os.stat(path)
    key = os.path.realpath(path)
    e = reg.get(key)
    if e and e["size"] == st.st_size and e["mtime"] == st.st_mtime:
        return e
    e = {"size": st.st_size, "mtime": st.st_mtime, "sha256": file_sha256(path)}
    reg = json.load(open(REGISTRY_PATH)) if os.path.exists(REGISTRY_PATH) else {}
    reg[key] = e
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    tmp = REGISTRY_PATH + f".{os.getpid()}"
    json.dump(reg, open(tmp, "w"), indent=1)
    os.replace(tmp, REGISTRY_PATH)
    return e


class DirectedMap:
    """Loads and validates the three map files of one direction against the two live models."""

    def __init__(self, calib, src, tgt, src_m, tgt_m, dev):
        self.calib, self.src, self.tgt, self.dev = calib, src, tgt, dev
        self.paths = map_paths(calib, src, tgt)
        self.hashes = {k: registry_entry(p)["sha256"] for k, p in self.paths.items()}
        scfg, tcfg = text_model(src_m).config, text_model(tgt_m).config
        self.scfg, self.tcfg = scfg, tcfg
        s_attn, t_attn = attn_layers(src_m), attn_layers(tgt_m)
        s_delta, t_delta = delta_layers(src_m), delta_layers(tgt_m)
        kvd = tcfg.num_key_value_heads * tcfg.head_dim
        am = torch.load(self.paths["attn"], mmap=True)
        self.k_used = max(am["maps"])
        assert self.k_used >= len(s_attn), (self.k_used, len(s_attn))  # all-source-layer concatenation (paper method)
        self.attn = {}
        for kv in ("K", "V"):
            d = am["maps"][self.k_used][kv]
            assert sorted(d) == t_attn, (kv, sorted(d), t_attn)
            for lt, (sel, W, b) in d.items():
                sel = list(sel)
                assert sorted(sel) == s_attn, ("attention map must use all source layers", sel)
                assert W.shape == (len(sel) * scfg.num_key_value_heads * scfg.head_dim, kvd) and b.shape[-1] == kvd, (W.shape, b.shape)
                self.attn.setdefault(kv, {})[lt] = (sel, W.to(dev).float(), b.to(dev).float())
        del am
        sm = torch.load(self.paths["state"], mmap=True)
        assert sorted(sm["maps"]) == t_delta
        Ht, Hs = tcfg.linear_num_value_heads, scfg.linear_num_value_heads
        dk, dv = tcfg.linear_key_head_dim, tcfg.linear_value_head_dim
        self.state = {}
        for lt, d in sm["maps"].items():
            assert d["src"] in s_delta
            A, B = d["A"], d["B"]
            assert A.shape == (Ht, Hs, dk, dk) and B.shape == (Ht, Hs, dv, dv), (A.shape, B.shape)
            self.state[lt] = {"src": d["src"], "A": A.to(dev).float(), "B": B.to(dev).float(), "best": d.get("best")}
        self.state_report = sm.get("report")
        del sm
        tm = torch.load(self.paths["tok"], mmap=True)
        cd_s = 2 * scfg.linear_num_key_heads * scfg.linear_key_head_dim + scfg.linear_num_value_heads * scfg.linear_value_head_dim
        cd_t = 2 * tcfg.linear_num_key_heads * tcfg.linear_key_head_dim + Ht * dv
        self.conv = {}
        for lt, d in tm["maps"].items():
            W, b = d["pre"]
            assert W.shape == (cd_s, cd_t), (W.shape, cd_s, cd_t)
            assert d.get("src", self.state[lt]["src"]) == self.state[lt]["src"]
            self.conv[lt] = (W.to(dev).float(), b.to(dev).float())
        assert sorted(self.conv) == t_delta
        del tm
        # copy baseline only where the recurrent geometry matches exactly
        self.copy_ok = (Hs == Ht and scfg.linear_num_key_heads == tcfg.linear_num_key_heads and cd_s == cd_t)
        self.t_attn, self.t_delta = t_attn, t_delta
        self.kd = tcfg.linear_num_key_heads * tcfg.linear_key_head_dim

    def describe(self):
        return {"calib": self.calib, "src": self.src, "tgt": self.tgt, "paths": self.paths, "sha256": self.hashes,
                "attn_k_used": self.k_used, "copy_ok": self.copy_ok}

    @torch.no_grad()
    def map_prefix(self, feats, snap_src, L):
        """feats: {src_layer: {'K': [1,T,D], 'V': [1,T,D]}} (T >= L), snap_src: source snapshot at L.
        Returns target tensors {'attn': {lt: (K_rot bf16 [1,H,L,hd], V bf16)}, 'rec': {lt: S}, 'conv': {lt: C}}."""
        tcfg = self.tcfg
        H, hd = tcfg.num_key_value_heads, tcfg.head_dim
        cos, sin = self.rope
        out = {"attn": {}, "rec": {}, "conv": {}}
        for lt in self.t_attn:
            kvs = {}
            for kv in ("K", "V"):
                sel, W, b = self.attn[kv][lt]
                X = torch.cat([feats[l][kv][:, :L] for l in sel], -1).to(self.dev).float()
                kvs[kv] = (X @ W + b).view(1, L, H, hd).transpose(1, 2)
            out["attn"][lt] = (apply_rope_k(kvs["K"], cos[:, :L], sin[:, :L]).to(torch.bfloat16), kvs["V"].to(torch.bfloat16))
        for lt in self.t_delta:
            mp = self.state[lt]
            S = snap_src["rec"][mp["src"]].to(self.dev).float()
            out["rec"][lt] = lift(S, mp["A"], mp["B"])
            Wc, bc = self.conv[lt]
            x = snap_src["conv"][mp["src"]].to(self.dev).float().transpose(1, 2)
            out["conv"][lt] = (x @ Wc + bc).transpose(1, 2)
        return out

    @torch.no_grad()
    def copy_delta(self, snap_src):
        assert self.copy_ok
        return {lt: snap_src["rec"][self.state[lt]["src"]] for lt in self.t_delta}, \
               {lt: snap_src["conv"][self.state[lt]["src"]] for lt in self.t_delta}


class ModelIO:
    """One live model: tap, template cache, segmented prefill, snapshots, install."""

    def __init__(self, name, m):
        self.name, self.m = name, m
        self.tm = text_model(m)
        self.dev = next(m.parameters()).device
        self.attn, self.delta = attn_layers(m), delta_layers(m)
        self.tap = KVTap(m)
        self.tap.enabled = False
        with torch.no_grad():
            out = forward_text(m, torch.tensor([[11]], device=self.dev), use_cache=True)
        self.template = out.past_key_values
        cfg = self.tm.config
        self.cfg = cfg

    def rope(self, P):
        return rope_cos_sin(self.m, P, self.dev)

    @torch.no_grad()
    def segmented(self, ids, cuts, taps=True, keep_live=False):
        """Prefill ids[:, :cuts[-1]] in segments ending at each cut. Returns feats (concatenated taps),
        snapshots {cut: {'attn': {l: (K,V)}, 'rec': {l: S}, 'conv': {l: C}}}, last hidden at every cut, live cache."""
        assert all(b > a for a, b in zip([0] + cuts[:-1], cuts)), cuts
        cache, prev = None, 0
        feats = {l: {"K": [], "V": []} for l in self.attn} if taps else None
        snaps, hid = {}, {}
        self.tap.enabled = taps
        for c in cuts:
            self.tap.clear()
            out = forward_text(self.m, ids[:, prev:c], past=cache, use_cache=True)
            cache = out.past_key_values
            if taps:
                for l in self.attn:
                    feats[l]["K"].append(self.tap.store[l]["K"]); feats[l]["V"].append(self.tap.store[l]["V"])
            snaps[c] = self.snapshot(cache)
            hid[c] = out.last_hidden_state[:, -1:]
            prev = c
        self.tap.enabled = False
        if taps:
            feats = {l: {kv: torch.cat(v, 1) for kv, v in d.items()} for l, d in feats.items()}
        return feats, snaps, hid, (cache if keep_live else None)

    def snapshot(self, cache):
        s = {"attn": {}, "rec": {}, "conv": {}}
        for l in self.attn:
            L = cache.layers[l]
            s["attn"][l] = (L.keys, L.values)  # DynamicLayer.update concatenates into new tensors: references stay valid
        for l in self.delta:
            L = cache.layers[l]
            s["rec"][l] = L.recurrent_states[0].clone()
            s["conv"][l] = L.conv_states[0].clone()
        return s

    @torch.no_grad()
    def install(self, attn, rec, conv):
        """NaN-poisoned template copy, then overwrite every attention K/V, recurrent and conv state."""
        c = copy.deepcopy(self.template)
        nan = float("nan")
        Ls = set()
        for l in self.attn:
            layer = c.layers[l]
            layer.keys.fill_(nan); layer.values.fill_(nan)
            K, V = attn[l]
            assert K.shape[0] == 1 and K.shape[1] == self.cfg.num_key_value_heads and K.shape[-1] == self.cfg.head_dim
            layer.keys = K.to(layer.keys.dtype).contiguous()
            layer.values = V.to(layer.values.dtype).contiguous()
            Ls.add(K.shape[2]); assert V.shape == K.shape
        assert len(Ls) == 1, Ls
        for l in self.delta:
            layer = c.layers[l]
            r, cv = layer.recurrent_states[0], layer.conv_states[0]
            r.fill_(nan); cv.fill_(nan)
            assert rec[l].shape == r.shape and conv[l].shape == cv.shape, (rec[l].shape, r.shape, conv[l].shape, cv.shape)
            r.copy_(rec[l].to(r.dtype)); cv.copy_(conv[l].to(cv.dtype))
        for l in self.attn:
            assert torch.isfinite(c.layers[l].keys).all() and torch.isfinite(c.layers[l].values).all(), f"non-finite attn {l}"
        for l in self.delta:
            assert torch.isfinite(c.layers[l].recurrent_states[0]).all() and torch.isfinite(c.layers[l].conv_states[0]).all(), f"non-finite delta {l}"
        assert c.get_seq_length() == Ls.pop()
        return c

    def install_snapshot(self, s):
        return self.install(s["attn"], s["rec"], s["conv"])

    def logits_last(self, hidden):
        """fp32 logits at the last position (output head upcast in vocabulary chunks; avoids bf16 logit rounding)"""
        W = self.m.lm_head.weight
        h = hidden[:, -1].float()
        return torch.cat([h @ W[i: i + 32768].float().T for i in range(0, W.shape[0], 32768)], -1)


def rms_per(x, dims):
    return x.float().pow(2).mean(dim=dims, keepdim=True).sqrt()


@torch.no_grad()
def noise_like_error(native, mapped, gen, kd):
    """C-noise: native + iid Gaussian noise with the RMS of (mapped - native), per head (attention, recurrent)
    and per q/k/v channel block (conv). Returns (attn, rec, conv, rel_errors)."""
    attn, rec, conv, rel = {}, {}, {}, {"attn_K": [], "attn_V": [], "rec": [], "conv": []}
    for l, (Kn, Vn) in native["attn"].items():
        Km, Vm = mapped["attn"][l]
        outs = []
        for nat, mp, name in ((Kn, Km, "attn_K"), (Vn, Vm, "attn_V")):
            err = mp.float() - nat.float()
            s = rms_per(err, (2, 3))
            noise = torch.randn(nat.shape, generator=gen, device=nat.device) * s
            outs.append((nat.float() + noise).to(nat.dtype))
            rel[name].append((err.norm() / nat.float().norm()).item())
        attn[l] = tuple(outs)
    for l, Sn in native["rec"].items():
        err = mapped["rec"][l].float() - Sn.float()
        s = rms_per(err, (2, 3))
        rec[l] = (Sn.float() + torch.randn(Sn.shape, generator=gen, device=Sn.device) * s).to(Sn.dtype)
        rel["rec"].append((err.norm() / Sn.float().norm()).item())
    for l, Cn in native["conv"].items():
        err = mapped["conv"][l].float() - Cn.float()
        out = Cn.float().clone()
        for a, b in ((0, kd), (kd, 2 * kd), (2 * kd, Cn.shape[1])):
            s = err[:, a:b].pow(2).mean().sqrt()
            out[:, a:b] += torch.randn(out[:, a:b].shape, generator=gen, device=Cn.device) * s
        conv[l] = out.to(Cn.dtype)
        rel["conv"].append((err.norm() / Cn.float().norm()).item())
    return attn, rec, conv, {k: sum(v) / len(v) for k, v in rel.items()}


@torch.no_grad()
def mapping_error(native, mapped):
    """mapped - native per component (float32), used by the C-swap control."""
    return {"attn": {l: (mapped["attn"][l][0].float() - K.float(), mapped["attn"][l][1].float() - V.float()) for l, (K, V) in native["attn"].items()},
            "rec": {l: mapped["rec"][l].float() - S.float() for l, S in native["rec"].items()},
            "conv": {l: mapped["conv"][l].float() - C.float() for l, C in native["conv"].items()}}


def _right_align(e, L):
    Lj = e.shape[2]
    if Lj >= L:
        return e[:, :, Lj - L:]
    reps = -(-L // Lj)
    return e.repeat(1, 1, reps, 1)[:, :, -L:]


@torch.no_grad()
def apply_error(native, err):
    """C-swap: this item's native receiver cache plus another item's mapping error (attention right-aligned by position)."""
    attn = {}
    for l, (K, V) in native["attn"].items():
        dK, dV = err["attn"][l]
        L = K.shape[2]
        attn[l] = ((K.float() + _right_align(dK, L)).to(K.dtype), (V.float() + _right_align(dV, L)).to(V.dtype))
    rec = {l: (S.float() + err["rec"][l]).to(S.dtype) for l, S in native["rec"].items()}
    conv = {l: (C.float() + err["conv"][l]).to(C.dtype) for l, C in native["conv"].items()}
    return attn, rec, conv
