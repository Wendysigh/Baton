#!/usr/bin/env python
"""Handoff timing with fitted maps, distinct layer weights, the same operations and precision as the evaluated caches, and
the same PyTorch backend and first-token boundary for native prefill and handoff. Time to first token is measured with
CUDA synchronized at the start and end of each timed path; a second handoff timing also synchronizes between components.
Scenarios: (a) resident source state, warm maps = handoff; (b) cold maps = load time reported once per process;
(c) request from scratch = source prefill + handoff. Requests alternate in a seeded random order after warm-up."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import MAN  # noqa: E402
import argparse, gc, glob, json, os, platform, random, subprocess, sys, time

import torch

from harness import log, forward_text, MODELS, ORDER, map_dir, load_model, BAN_IDS, RES, assemble  # noqa: E402
from common import load_maps, Handoff  # noqa: E402
from attention import rope_cos_sin, apply_rope_k  # noqa: E402
from gdn import lift  # noqa: E402


def devs(ho):
    return {ho.dev, ho.sdev}


def sync(ds):
    for d in ds:
        torch.cuda.synchronize(d)


class Clock:
    def __init__(self, ds, parts):
        self.ds, self.parts, self.t = ds, parts, {}
        sync(ds)
        self.start = self.last = time.perf_counter()

    def mark(self, name):
        if self.parts:
            sync(self.ds)
            now = time.perf_counter()
            self.t[name] = self.t.get(name, 0.0) + now - self.last
            self.last = now

    def stop(self):
        sync(self.ds)
        self.t["ttft"] = time.perf_counter() - self.start
        return self.t


@torch.no_grad()
def first_token(m, out):
    lg = m.lm_head(out.last_hidden_state[:, -1]).float()
    if BAN_IDS:
        lg[:, BAN_IDS] = float("-inf")
    return int(lg.argmax(-1))


@torch.no_grad()
def native(m, ids, ds):
    c = Clock(ds, False)
    tok = first_token(m, forward_text(m, ids, use_cache=True))
    return c.stop(), tok


@torch.no_grad()
def source(src_m, ho, ids, ds):
    ho.tap.clear(); ho.tap.enabled = True
    c = Clock(ds, False)
    sc = forward_text(src_m, ids.to(ho.sdev), use_cache=True).past_key_values
    t = c.stop()
    ho.tap.enabled = False
    return t, sc, ho.tap.store


@torch.no_grad()
def handoff(ho, tgt_m, sc, store, P, first, rest, parts):
    """The operations of harness.assemble in component order, followed by the bridge or tail and the first token."""
    c = Clock(devs(ho), parts)
    moved = {l: {k: store[l][k].to(ho.dev, non_blocking=True) for k in ("K", "V")} for l in store}
    states = {lt: (sc.layers[ho.rmaps[lt]["src"]].recurrent_states[0].to(ho.dev, non_blocking=True), sc.layers[ho.rmaps[lt]["src"]].conv_states[0].to(ho.dev, non_blocking=True)) for lt in ho.dl}
    c.mark("move")
    tcache = forward_text(ho.tgt, first, use_cache=True).past_key_values
    c.mark("cache_setup")
    cos, sin = rope_cos_sin(ho.tgt, P, ho.dev)
    Hkv, hd = ho.tcfg.num_key_value_heads, ho.tcfg.head_dim
    for lt in ho.tl:
        o = {}
        for kv in ("K", "V"):
            sel, W, b = ho.amaps[kv][lt]
            X = torch.cat([moved[l][kv] for l in sel], -1).float()
            o[kv] = (X @ W + b).view(1, P, Hkv, hd).transpose(1, 2)
        layer = tcache.layers[lt]
        layer.keys = apply_rope_k(o["K"], cos, sin).to(torch.bfloat16)
        layer.values = o["V"].to(torch.bfloat16)
    c.mark("attn_map")
    for lt in ho.dl:
        layer = tcache.layers[lt]
        rs, cs = layer.recurrent_states[0], layer.conv_states[0]
        mp = ho.rmaps[lt]
        rs.copy_(lift(states[lt][0].float(), mp["A"], mp["B"]).to(rs.dtype))
        Wc, bc = ho.cmaps[lt]
        cs.copy_((states[lt][1].float().transpose(1, 2) @ Wc + bc).transpose(1, 2).to(cs.dtype))
    c.mark("gdn_map")
    tok = first_token(tgt_m, forward_text(tgt_m, rest, past=tcache, use_cache=True))
    c.mark("bridge")
    return c.stop(), tok


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def foreign(gpus):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if str(os.getpid()) not in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--placement", choices=["same", "two"], required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 32768])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cold_reps", type=int, default=3)
    ap.add_argument("--mem_fraction", type=float, default=0)
    ap.add_argument("--out", default=os.path.join(RES, "timing"))
    ap.add_argument("--docs", default=f"{MAN}/timing_docs.pt")
    args = ap.parse_args()
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["9B"])
    if not BAN_IDS:
        BAN_IDS.extend(tok.convert_tokens_to_ids(x) for x in ("<think>",) if x in tok.get_vocab())
    s, t = args.direction.split(">")
    small, large = sorted((s, t), key=ORDER.get)
    role = {small: "small", large: "large"}
    sdev, tdev = ("cuda:0", "cuda:1") if args.placement == "two" else ("cuda:0", "cuda:0")
    src_m, tgt_m = load_model(MODELS[s], sdev), load_model(MODELS[t], tdev)
    dev = next(tgt_m.parameters()).device
    tag = f"fla_{args.placement}_{args.direction.replace('>', 'to')}"
    D = map_dir(small, large, "web")
    # (b) cold maps: evict the map files from the page cache (these files only), then load and place them
    cold = []
    files = [f for pat in (f"maps_{role[s]}_to_{role[t]}_s1024_c512_st4_*.pt", f"refine_{role[s]}_to_{role[t]}_s1024_c512_st8.pt", f"tokmaps_{role[s]}_to_{role[t]}_s1024_c512_st8_sp16.pt") for f in glob.glob(os.path.join(D, pat))]
    for _ in range(args.cold_reps):
        for f in files:
            fd = os.open(os.path.realpath(f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        t0 = time.perf_counter()
        am, rm, tm = load_maps(D, role[s], role[t])
        t1 = time.perf_counter()
        ho = Handoff(src_m, tgt_m, am, rm, tm, dev)
        sync({ho.dev, ho.sdev})
        cold.append({"disk_read_s": t1 - t0, "to_gpu_s": time.perf_counter() - t1})
        ho.tap.remove()
        del am, rm, tm, ho
        clean()
    am, rm, tm = load_maps(D, role[s], role[t])
    ho = Handoff(src_m, tgt_m, am, rm, tm, dev)
    del am, rm, tm
    tensors = [W for d in ho.amaps.values() for (_, W, b) in d.values()] + [v for d in ho.rmaps.values() for v in (d["A"], d["B"])] + [v for pr in ho.cmaps.values() for v in pr]
    meta = {"args": vars(args), "cold_maps": cold, "map_files_gib": {os.path.basename(f): os.path.getsize(f) / 2**30 for f in files}, "maps_on_gpu_gib": sum(x.numel() * x.element_size() for x in tensors) / 2**30,
            "map_dtypes": sorted({str(x.dtype) for x in tensors}), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "host": platform.node(),
            "kernels_wrapped_by_accelerated_impl": hasattr(__import__("transformers.models.qwen3_5.modeling_qwen3_5", fromlist=["x"]).torch_chunk_gated_delta_rule, "__wrapped__"),
            "topology": subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout[:2000], "started": time.strftime("%F %T")}
    json.dump(meta, open(os.path.join(args.out, f"meta_{tag}.json"), "w"), indent=1)
    log("cold", cold, "maps on GPU GiB", meta["maps_on_gpu_gib"], meta["map_dtypes"], meta["kernels_wrapped_by_accelerated_impl"])
    docs = torch.load(args.docs)
    rng = random.Random(20261104)
    f = open(os.path.join(args.out, f"timing_{tag}.jsonl"), "a")
    ds = devs(ho)
    for L in args.lengths:
        for i in range(args.warmup + args.n):
            ids = docs[i, :L][None].to(dev)
            first = ids[:, :1]
            rec = {"length": L, "doc": i, "warmup": i < args.warmup, "foreign_apps": foreign(None)}
            order = ["native", "handoff"]
            rng.shuffle(order)
            rec["order"] = order
            for what in order:
                clean()
                if what == "native":
                    for d in ds:
                        torch.cuda.reset_peak_memory_stats(d)
                    a0 = {str(d): torch.cuda.memory_allocated(d) for d in ds}
                    tn, ntok = native(tgt_m, ids, ds)
                    rec["native"] = tn
                    rec["native_peak_gib"] = {str(d): (torch.cuda.max_memory_allocated(d) - a0[str(d)]) / 2**30 for d in ds}
                    rec["native_tok"] = ntok
                    continue
                ts, sc, store = source(src_m, ho, ids[:, :-1], ds)
                rec["source_prefill"] = ts
                for d in ds:
                    torch.cuda.reset_peak_memory_stats(d)
                a0 = {str(d): torch.cuda.memory_allocated(d) for d in ds}
                he, htok = handoff(ho, tgt_m, sc, store, L - 1, first, ids[:, -1:], False)
                rec["handoff_e2e"] = he
                rec["handoff_peak_gib"] = {str(d): (torch.cuda.max_memory_allocated(d) - a0[str(d)]) / 2**30 for d in ds}
                rec["handoff_tok"] = htok
                hp, _ = handoff(ho, tgt_m, sc, store, L - 1, first, ids[:, -1:], True)
                rec["handoff_parts"] = hp
                if i == args.warmup:  # the timed path must build the same cache as the evaluated one
                    ref = first_token(tgt_m, forward_text(tgt_m, ids[:, -1:], past=assemble(ho, sc, store, L - 1, first), use_cache=True))
                    assert ref == htok, (L, ref, htok)
                del sc, store
                ho.tap.clear(); clean()
                ts, sc, store = source(src_m, ho, ids[:, :-12], ds)
                rec["source_prefill_tail"] = ts
                ht, _ = handoff(ho, tgt_m, sc, store, L - 12, first, ids[:, -12:], False)
                rec["handoff_tail_e2e"] = ht
                del sc, store
                ho.tap.clear(); clean()
            f.write(json.dumps(rec) + "\n"); f.flush()
        log(f"{tag} L={L}: last native {rec['native']['ttft'] * 1e3:.0f} ms, handoff {rec['handoff_e2e']['ttft'] * 1e3:.0f} ms, source {rec['source_prefill']['ttft'] * 1e3:.0f} ms, parts " + " ".join(f"{k}={v * 1e3:.1f}" for k, v in rec["handoff_parts"].items()))
    log("done")


if __name__ == "__main__":
    main()
