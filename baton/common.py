"""Shared setup for the downstream evaluations."""
from paths import CACHE, model_path  # noqa: E402
import copy, os, re, sys

import torch

from attention import log, load_model, forward_text  # noqa: E402
from handoff import Handoff, load_maps, BAN_IDS  # noqa: E402

MODELS = {s: model_path(s) for s in ("4B", "9B", "27B")}
# map directories per unordered pair and calibration set; maps inside are named by the pair's small/large roles
MAP_DIRS = {("9B", "27B", "web"): CACHE, ("9B", "27B", "chat"): f"{CACHE}/chatcal",
            ("4B", "9B", "web"): f"{CACHE}/p4b9b", ("4B", "9B", "chat"): f"{CACHE}/p4b9b_chat"}
ORDER = {"4B": 0, "9B": 1, "27B": 2}


def map_dir(small, large, calib):
    """Map directory of a pair and calibration/variant name; refit variants (experiments/short_fit.py) live in sibling
    directories of the pair's web-map directory, named <web dir>_<variant>."""
    if (small, large, calib) in MAP_DIRS:
        return MAP_DIRS[(small, large, calib)]
    return MAP_DIRS[(small, large, "web")].rstrip("/") + "_" + calib


def setup(direction, calibs, dev="cuda:0"):
    """direction '9B>27B'. Returns tok, eos_ids, src_m, tgt_m, {calib: Handoff}."""
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer
    s, t = direction.split(">")
    small, large = sorted((s, t), key=ORDER.get)
    tok = AutoTokenizer.from_pretrained(MODELS["9B"])
    eos_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|endoftext|>") if x in tok.get_vocab()}
    if not BAN_IDS:
        BAN_IDS.extend(tok.convert_tokens_to_ids(x) for x in ("<think>",) if x in tok.get_vocab())
    log("loading", s, t)
    src_m, tgt_m = load_model(MODELS[s], dev), load_model(MODELS[t], dev)
    role = {small: "small", large: "large"}
    hos = {}
    for c in calibs:
        amaps, rmaps, tmaps = load_maps(map_dir(small, large, c), role[s], role[t])
        hos[c] = Handoff(src_m, tgt_m, amaps, rmaps, tmaps, dev)
        del amaps, rmaps, tmaps
    return tok, eos_ids, src_m, tgt_m, hos


@torch.no_grad()
def prefix_cache(cfg, tgt_m, hos, ids_prefix):
    """Target cache for ids_prefix [1,P] under a config: 'native', 'none' (None), '<calib>' (translation),
    'copy_<calib>' (attention KV mapped, recurrent and conv state copied), 'copyrec_<calib>' (only the recurrent state
    copied) or 'copyconv_<calib>' (only the conv buffer copied)."""
    if cfg == "none" or ids_prefix.shape[1] == 0:
        return None
    if cfg == "native":
        return forward_text(tgt_m, ids_prefix, use_cache=True).past_key_values
    if cfg.startswith("cl"):  # 'cl<a>-<b>_<calib>': GDN layers a..b (inclusive, target indices) copied, the rest mapped
        spec, calib = cfg[2:].split("_", 1)
        a, b = map(int, spec.split("-"))
        return hos[calib].build_cache(ids_prefix, ids_prefix[:, :1], copy_layers=set(range(a, b + 1)))
    for pre, gdn in (("copy_", "copy"), ("copyrec_", "copy_rec"), ("copyconv_", "copy_conv")):
        if cfg.startswith(pre):
            return hos[cfg[len(pre):]].build_cache(ids_prefix, ids_prefix[:, :1], gdn=gdn)
    return hos[cfg].build_cache(ids_prefix, ids_prefix[:, :1])


def expand_cache(cache, B):
    """Repeat a batch-1 cache B times (attention K/V, GDN recurrent and conv states)."""
    c = copy.copy(cache)
    c.layers = []
    for l in cache.layers:
        n = copy.copy(l)
        if getattr(l, "keys", None) is not None and isinstance(l.keys, torch.Tensor) and l.keys.numel():
            n.keys, n.values = l.keys.expand(B, *l.keys.shape[1:]).contiguous(), l.values.expand(B, *l.values.shape[1:]).contiguous()
        for name in ("recurrent_states", "conv_states"):
            d = getattr(l, name, None)
            if isinstance(d, dict):
                setattr(n, name, {i: (v.expand(B, *v.shape[1:]).contiguous() if v is not None else None) for i, v in d.items()})
        c.layers.append(n)
    return c


@torch.no_grad()
def generate_graph(m, tok, cache, ids_rest, max_new, eos_ids, stops):
    """Greedy decoding with RULER-style stop and cut rules, each step a CUDA-graph replay."""
    from fast_decode import to_static, GraphDecoder
    out = forward_text(m, ids_rest, past=cache, use_cache=True)
    cache = out.past_key_values
    lg = m.lm_head(out.last_hidden_state[:, -1]).float()
    if BAN_IDS:
        lg[:, BAN_IDS] = float("-inf")
    nxt = lg.argmax(-1)
    gd = GraphDecoder(m, to_static(m, cache, cache.get_seq_length() + max_new + 1), 1, BAN_IDS)
    del cache
    toks = []
    for _ in range(max_new):
        t = int(nxt)
        if t in eos_ids:
            break
        toks.append(t)
        text = tok.decode(toks).lstrip()
        m0 = re.search(r"[A-Za-z0-9]", text)
        if m0 and any(s_ in text[m0.start():] for s_ in stops):
            break
        nxt = gd.step(nxt)[0]
    del gd
    text = tok.decode(toks).strip()
    m0 = re.search(r"[A-Za-z0-9]", text)
    start = m0.start() if m0 else 0
    cut = min([text.find(s_, start) for s_ in stops if s_ in text[start:]] + [len(text)])
    return text[:cut].strip(), len(toks)
