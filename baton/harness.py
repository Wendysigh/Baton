"""Shared code of the fidelity, RULER, LongBench and timing harnesses: one source pass shared by the map sets, a KV-only
(zero GDN) cache, full decoding records, frozen manifests and strict resume."""
from paths import RAW, MAN  # noqa: E402
import hashlib, json, os, re, sys

import torch

from common import setup, log, forward_text, MODELS, ORDER, map_dir, BAN_IDS, load_model  # noqa: E402,F401
from attention import rope_cos_sin, apply_rope_k  # noqa: E402
from gdn import lift  # noqa: E402
from fast_decode import to_static, GraphDecoder  # noqa: E402

RES = RAW


def sha(ids):
    return hashlib.sha1(torch.as_tensor(ids, dtype=torch.long).numpy().tobytes()).hexdigest()


def chat_wrap_user(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def tokenize_split(tok, prompt, cut, snap=False):
    """Tokenize the whole prompt once; return ids and the token index of character position cut. A cut inside a token
    raises, or with snap moves back to the previous token boundary (the straddling token joins the suffix)."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    E = sum(1 for a, b in off if b <= cut)
    if not (0 < E < len(ids)) or (off[E][0] < cut and not snap):
        raise ValueError(f"cut {cut} inside a token near index {E}")
    return ids, E


def source_pass(hos, src_m, ids):
    """One source forward over ids [1,P]; every map set reads the same keys and values."""
    first = next(iter(hos.values()))
    first.tap.clear(); first.tap.enabled = True
    out = forward_text(src_m, ids.to(first.sdev), use_cache=True)
    first.tap.enabled = False
    return out.past_key_values, first.tap.store


@torch.no_grad()
def assemble(ho, scache, store, P, first_tok, gdn="map", rmaps=None):
    """Target cache from one source pass. gdn: map | copy | zero (initialized all-zero recurrent state and conv buffer).
    rmaps: {lt: {src, A, B}} replaces the recurrent-state maps (ablations)."""
    tcache = forward_text(ho.tgt, first_tok, use_cache=True).past_key_values
    cos, sin = rope_cos_sin(ho.tgt, P, ho.dev)
    Hkv, hd = ho.tcfg.num_key_value_heads, ho.tcfg.head_dim
    for lt in ho.tl:
        o = {}
        for kv in ("K", "V"):
            sel, W, b = ho.amaps[kv][lt]
            X = torch.cat([store[l][kv] for l in sel], -1).to(ho.dev).float()
            o[kv] = (X @ W + b).view(1, P, Hkv, hd).transpose(1, 2)
        layer = tcache.layers[lt]
        layer.keys = apply_rope_k(o["K"], cos, sin).to(torch.bfloat16)
        layer.values = o["V"].to(torch.bfloat16)
    for lt in ho.dl:
        layer = tcache.layers[lt]
        rs, cs = layer.recurrent_states[0], layer.conv_states[0]
        if gdn == "zero":
            rs.zero_(); cs.zero_()
            continue
        mp = (rmaps or ho.rmaps)[lt]
        sl = scache.layers[ho.rmaps[lt]["src"]]
        if gdn == "copy":
            rs.copy_(sl.recurrent_states[0].to(ho.dev)[:, ho.head_idx].to(rs.dtype))
            cs.copy_(sl.conv_states[0].to(ho.dev)[:, ho.conv_idx].to(cs.dtype))
            continue
        rs.copy_(lift(sl.recurrent_states[0].to(ho.dev).float(), mp["A"].float(), mp["B"].float()).to(rs.dtype))
        Wc, bc = ho.cmaps[lt]
        x = sl.conv_states[0].to(ho.dev).float().transpose(1, 2)
        cs.copy_((x @ Wc + bc).transpose(1, 2).to(cs.dtype))
    return tcache


def _cut(text, stops):
    m0 = re.search(r"[A-Za-z0-9]", text)
    start = m0.start() if m0 else 0
    cut = min([text.find(s, start) for s in stops if s in text[start:]] + [len(text)])
    return text[:cut].strip()


@torch.no_grad()
def decode(m, tok, cache, rest, max_new, eos_ids, stops=(), graph=True):
    """Greedy decoding with RULER-style stop and cut rules. Returns text, output ids and stop reason."""
    out = forward_text(m, rest, past=cache, use_cache=True)
    cache = out.past_key_values
    lg = m.lm_head(out.last_hidden_state[:, -1]).float()
    if BAN_IDS:
        lg[:, BAN_IDS] = float("-inf")
    nxt = lg.argmax(-1)
    gd = None
    toks, reason = [], "max_new"
    for i in range(max_new):
        t = int(nxt)
        if t in eos_ids:
            reason = "eos"
            break
        toks.append(t)
        text = tok.decode(toks).lstrip()
        m0 = re.search(r"[A-Za-z0-9]", text)
        if m0 and any(s in text[m0.start():] for s in stops):
            reason = "stop"
            break
        if i == max_new - 1:
            break
        if graph:
            if gd is None:
                gd = GraphDecoder(m, to_static(m, cache, cache.get_seq_length() + max_new + 1), 1, BAN_IDS)
                cache = None
            nxt = gd.step(nxt)[0]
        else:
            o = forward_text(m, nxt.view(1, 1), past=cache, use_cache=True)
            cache = o.past_key_values
            lg = m.lm_head(o.last_hidden_state[:, -1]).float()
            if BAN_IDS:
                lg[:, BAN_IDS] = float("-inf")
            nxt = lg.argmax(-1)
    del gd, cache
    return {"text": _cut(tok.decode(toks).strip(), stops), "n_tokens": len(toks), "stop": reason, "output_ids": toks}


def read_manifest(name):
    rows = [json.loads(l) for l in open(os.path.join(MAN, name))]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
    return rows


def shard_rows(rows, shard):
    k, n = map(int, shard.split("/"))
    rows = sorted(rows, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
    return rows[k::n]


def open_resume(path, allowed):
    """Previously written ids; rejects duplicates and ids outside this shard's manifest."""
    done = set()
    if os.path.exists(path):
        for l in open(path):
            i = json.loads(l)["id"]
            assert i in allowed, f"{path}: id {i} not in manifest"
            assert i not in done, f"{path}: duplicate id {i}"
            done.add(i)
    return done, open(path, "a")
