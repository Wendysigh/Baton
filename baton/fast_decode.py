"""CUDA-graph greedy decoding for Qwen3.5 in transformers.

Eager single-token decode is bound by kernel-launch overhead (~1 ms per layer per step on H200, independent of batch
size), so it runs ~4-5x above the weight-read floor. Here the prefilled DynamicCache is copied into a StaticCache (fixed
addresses: attention K/V preallocated to max_len, DeltaNet conv/recurrent states updated in place), one decode step
(text model + lm_head + banned-token mask + argmax) is captured as a CUDA graph, and each new token is a graph replay.
"""
import torch
from transformers import StaticCache


@torch.no_grad()
def to_static(model, cache, max_len):
    """Copy a prefilled DynamicCache into a new StaticCache that can hold max_len tokens in total."""
    cfg = model.config.get_text_config(decoder=True)
    sc = StaticCache(config=cfg, max_cache_len=max_len)
    for src, dst in zip(cache.layers, sc.layers):
        if getattr(src, "keys", None) is not None and src.keys.numel():
            dst.update(src.keys, src.values)  # writes positions [0, T) and sets the length counter to T
        for i, st in getattr(src, "conv_states", {}).items():
            if st is not None:
                dst.lazy_initialization(conv_states=st, state_idx=i, conv_kernel_size=st.shape[-1])
                dst.conv_states[i].copy_(st)
                dst.has_previous_state[i] = True
        for i, st in getattr(src, "recurrent_states", {}).items():
            if st is not None:
                dst.lazy_initialization(recurrent_states=st, state_idx=i)
                dst.recurrent_states[i].copy_(st)
    return sc


def _cache_tensors(sc):
    ts = []
    for l in sc.layers:
        for name in ("keys", "values", "cumulative_length"):
            t = getattr(l, name, None)
            if isinstance(t, torch.Tensor) and t.device.type == "cuda":
                ts.append(t)
        for d in (getattr(l, "conv_states", {}), getattr(l, "recurrent_states", {})):
            ts += [t for t in d.values() if t is not None]
    return ts


class GraphDecoder:
    """step(tokens [B]) -> next greedy tokens [B] and fp32 logits [B, V], advancing the static cache by one token."""

    def __init__(self, model, static_cache, batch, ban_ids=(), warmup=2):
        self.m, self.c = model, static_cache
        tm = model.model.language_model
        dev = model.lm_head.weight.device
        # every tensor the graph reads must outlive it: replay uses raw addresses, not Python references
        self.ids = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.ban = torch.tensor(sorted(ban_ids), dtype=torch.long, device=dev)

        def fwd():
            h = tm(input_ids=self.ids, past_key_values=self.c, use_cache=True).last_hidden_state[:, -1]
            lg = self.m.lm_head(h).float()
            if self.ban.numel():
                lg.index_fill_(1, self.ban, float("-inf"))
            return lg, lg.argmax(-1)

        # warm up (Triton autotune, cuBLAS workspaces) on a side stream, then restore the cache it advanced
        snap = [t.clone() for t in _cache_tensors(self.c)]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fwd()
        torch.cuda.current_stream().wait_stream(s)
        for t, v in zip(_cache_tensors(self.c), snap):
            t.copy_(v)
        del snap
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.logits, self.next = fwd()
        # capture records but does not run; the cache is still at the prefill state

    def step(self, tokens):
        self.ids.copy_(tokens.view(-1, 1))
        self.g.replay()
        return self.next, self.logits
