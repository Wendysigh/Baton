#!/usr/bin/env python
"""Fitted map loading and target-cache assembly used by the downstream evaluations."""
import glob, os

import torch

from attention import log, load_model, text_model, attn_layers, KVTap, rope_cos_sin, apply_rope_k, forward_text  # noqa: E402
from gdn import delta_layers, lift  # noqa: E402



def load_maps(cache, src, tgt):
    def one(pat):
        fs = sorted(glob.glob(os.path.join(cache, pat)))
        assert fs, pat
        return torch.load(fs[-1])
    amaps = one(f"maps_{src}_to_{tgt}_s1024_c512_st4_*.pt")
    rmaps = one(f"refine_{src}_to_{tgt}_s1024_c512_st8.pt")
    tmaps = one(f"tokmaps_{src}_to_{tgt}_s1024_c512_st8_sp16.pt")
    return amaps, rmaps, tmaps


class Handoff:
    def __init__(self, src_m, tgt_m, amaps, rmaps, tmaps, dev):
        self.src, self.tgt, self.dev = src_m, tgt_m, dev
        self.tcfg = text_model(tgt_m).config
        self.tl, self.dl = attn_layers(tgt_m), delta_layers(tgt_m)
        self.tap = KVTap(src_m)
        self.tap.enabled = False
        k = max(amaps["maps"])
        self.amaps = {kv: {lt: (sel, W.to(dev), b.to(dev)) for lt, (sel, W, b) in d.items()} for kv, d in amaps["maps"][k].items()}
        self.rmaps = {lt: {"src": d["src"], "A": d["A"].to(dev).float(), "B": d["B"].to(dev).float()} for lt, d in rmaps["maps"].items()}
        self.cmaps = {lt: (d["pre"][0].to(dev).float(), d["pre"][1].to(dev).float()) for lt, d in tmaps["maps"].items()}
        self.sdev = next(src_m.parameters()).device
        # copy baseline (LatentPort-style direct GDN reuse, head-replicated when value-head counts differ)
        scfg, t = text_model(src_m).config, self.tcfg
        Hs, Hk_s, Ht, Hk_t = scfg.linear_num_value_heads, scfg.linear_num_key_heads, t.linear_num_value_heads, t.linear_num_key_heads
        rs, rt, dk, dv = Hs // Hk_s, Ht // Hk_t, t.linear_key_head_dim, t.linear_value_head_dim
        self.head_idx = torch.tensor([(j // rt) * rs + (j % rt) % rs for j in range(Ht)], device=dev)
        qk = 2 * Hk_s * dk
        self.conv_idx = torch.cat([torch.arange(qk, device=dev), (qk + self.head_idx[:, None] * dv + torch.arange(dv, device=dev)[None]).reshape(-1)])

    @torch.no_grad()
    def build_cache(self, ids_mapped, first_tgt_token, gdn="map", copy_layers=()):
        """Run source on ids_mapped [1,P]; return a target cache holding the mapped prefix of P tokens.
        copy_layers: target GDN layers whose recurrent state and conv buffer are copied instead of mapped."""
        P = ids_mapped.shape[1]
        self.tap.clear(); self.tap.enabled = True
        sout = forward_text(self.src, ids_mapped.to(self.sdev), use_cache=True)
        self.tap.enabled = False
        scache = sout.past_key_values
        # init a target cache object with a 1-token forward, then overwrite everything
        tout = forward_text(self.tgt, first_tgt_token, use_cache=True)
        tcache = tout.past_key_values
        cos, sin = rope_cos_sin(self.tgt, P, self.dev)
        Hkv, hd = self.tcfg.num_key_value_heads, self.tcfg.head_dim
        for lt in self.tl:
            out = {}
            for kv in ("K", "V"):
                sel, W, b = self.amaps[kv][lt]
                X = torch.cat([self.tap.store[l][kv] for l in sel], -1).to(self.dev).float()
                out[kv] = (X @ W + b).view(1, P, Hkv, hd).transpose(1, 2)
            layer = tcache.layers[lt]
            layer.keys = apply_rope_k(out["K"], cos, sin).to(torch.bfloat16)
            layer.values = out["V"].to(torch.bfloat16)
        for lt in self.dl:
            mp = self.rmaps[lt]
            sl = scache.layers[mp["src"]]
            layer = tcache.layers[lt]
            # gdn: map | copy (recurrent state and conv buffer copied) | copy_rec (state copied, buffer mapped) |
            # copy_conv (state mapped, buffer copied)
            if lt in copy_layers:
                gdn_l = "copy"
            else:
                gdn_l = gdn
            if gdn_l in ("copy", "copy_rec"):
                layer.recurrent_states[0].copy_(sl.recurrent_states[0].to(self.dev)[:, self.head_idx].to(layer.recurrent_states[0].dtype))
            else:
                S = lift(sl.recurrent_states[0].to(self.dev).float(), mp["A"], mp["B"])
                layer.recurrent_states[0].copy_(S.to(layer.recurrent_states[0].dtype))
            if gdn_l in ("copy", "copy_conv"):
                layer.conv_states[0].copy_(sl.conv_states[0].to(self.dev)[:, self.conv_idx].to(layer.conv_states[0].dtype))
            else:
                Wc, bc = self.cmaps[lt]
                x = sl.conv_states[0].to(self.dev).float().transpose(1, 2)  # [1,4,cd_s]
                layer.conv_states[0].copy_((x @ Wc + bc).transpose(1, 2).to(layer.conv_states[0].dtype))
        del scache
        return tcache


BAN_IDS = []


