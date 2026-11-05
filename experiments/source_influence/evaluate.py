#!/usr/bin/env python
"""Source-influence evaluation worker. One process = one model group (two models, both directions), one task,
one shard of a frozen manifest. Writes one JSON line per item, resumes only under an identical run manifest.

Tasks: mmlu (raw 5-shot, original + shift2 ordering), mmlu_chat, gsm8k (raw 8-shot)."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
import argparse, hashlib, importlib.metadata as md, inspect, json, os, platform, socket, sys, time, traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from attention import load_model, forward_text  # noqa: E402
from fast_decode import GraphDecoder, to_static  # noqa: E402
from cache_adapter import MODELS, DirectedMap, ModelIO, noise_like_error, mapping_error, apply_error, file_sha256  # noqa: E402
import scoring  # noqa: E402

T_TAIL = 12


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def kernel_report():
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    rep = {}
    for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        f = getattr(M, n)
        impl = inspect.getclosurevars(f).nonlocals.get("implementation", f)
        rep[n] = f"{impl.__module__}.{impl.__qualname__}"
    assert rep["torch_chunk_gated_delta_rule"].startswith("fla."), rep
    assert rep["torch_recurrent_gated_delta_rule"].startswith("fla."), rep
    return rep


def model_identity(path):
    out = {}
    for f in ("config.json", "tokenizer.json", "model.safetensors.index.json", "chat_template.jinja"):
        p = os.path.join(path, f)
        if os.path.exists(p):
            out[f] = file_sha256(p)
    out["weight_bytes"] = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors"))
    return out


def run_manifest(args, maps, ios):
    code = {f: file_sha256(os.path.join(d, f)) for d, fs in ((HERE, ["evaluate.py", "cache_adapter.py", "scoring.py"]),
                                                            (os.path.join(HERE, "..", "..", "baton"), ["attention.py", "gdn.py", "fast_decode.py"])) for f in fs}
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "code_sha256": code,
        "manifest_sha256": {m: file_sha256(m) for m in args.manifests},
        "maps": {f"{k[0]}>{k[1]}": v.describe() for k, v in maps.items()},
        "models": {n: {"path": MODELS[n], **model_identity(MODELS[n])} for n in ios},
        "packages": {p: md.version(p) for p in ("torch", "transformers", "flash-linear-attention", "causal-conv1d")},
        "kernels": kernel_report(),
        "gpu": {"name": props.name, "uuid": str(getattr(props, "uuid", "")), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "attn_impl": {n: getattr(io.tm.config, "_attn_implementation", None) for n, io in ios.items()},
        "args": {k: v for k, v in vars(args).items() if k not in ("shard",)},
        "host": socket.gethostname(), "python": platform.python_version(),
    }


def manifest_key(man):
    """the part of the run manifest that must match for resumption (gpu identity may differ)"""
    return json.dumps({k: man[k] for k in ("protocol_sha256", "code_sha256", "manifest_sha256", "maps", "models", "packages", "kernels", "args")}, sort_keys=True)


def item_seed(*parts):
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:12], 16)


@torch.no_grad()
def partner_maps(prec, ios, maps):
    """C-wrongq: each donor reads the partner item's full prompt (up to its bridge token); returns the mapped
    receiver caches {(src, tgt): mapped}. The receiver then continues with its own item's bridge token, so the donor
    never saw the current question. The prefill uses the same segment cuts as the main conditions, so the mapped cache
    equals the one H-all used on the partner item."""
    Pj, Ej = prec["P"], prec["E"]
    cuts = sorted({Ej, Pj - T_TAIL, Pj - 1})
    store = {}
    for name, io in ios.items():
        if any(s == name for s, _ in maps):
            ids = torch.tensor([prec["ids"]], device=io.dev)
            feats, snaps, _, _ = io.segmented(ids, cuts)
            store[name] = (feats, snaps[Pj - 1])
    out = {k: dm.map_prefix(store[k[0]][0], store[k[0]][1], Pj - 1) for k, dm in maps.items()}
    del store
    return out


# ----------------------------------------------------------------------------------------------- MMLU
class LetterScorer:
    def __init__(self, tok, bare):
        toks = ["A", "B", "C", "D"] if bare else [" A", " B", " C", " D"]
        self.ids = []
        for t in toks:
            e = tok(t, add_special_tokens=False)["input_ids"]
            assert len(e) == 1, (t, e)
            self.ids.append(e[0])

    def __call__(self, logits):
        lp = logits.float().log_softmax(-1)[0, self.ids]
        return {"lp": [round(x, 4) for x in lp.tolist()], "pred": int(lp.argmax().item()), "mass": round(lp.exp().sum().item(), 5)}


@torch.no_grad()
def cond_forward(io, attn, rec, conv, rest):
    cache = io.install(attn, rec, conv)
    out = forward_text(io.m, rest, past=cache, use_cache=True)
    lg = io.logits_last(out.last_hidden_state)
    assert torch.isfinite(lg).all()
    return lg


@torch.no_grad()
def mmlu_variant(rec, ios, maps, scorer, conds, diag, swap, partner=None):
    dev0 = next(iter(ios.values())).dev
    P, E = rec["P"], rec["E"]
    assert E < P - T_TAIL, (E, P)
    cuts = sorted({E, P - T_TAIL, P - 1, P})
    res = {"native": {}, "split": {}, "self": {}, "dirs": {}}
    wq = None
    if "C-wrongq" in conds and partner is not None:
        assert partner["item_id"] != rec["item_id"] and partner["ids"][-1] == rec["ids"][-1]
        wq = partner_maps(partner, ios, maps)
    store = {}
    for name, io in ios.items():
        ids = torch.tensor([rec["ids"]], device=io.dev)
        out = forward_text(io.m, ids)
        res["native"][name] = scorer(io.logits_last(out.last_hidden_state))
        feats, snaps, hid, _ = io.segmented(ids, cuts)
        store[name] = (ids, feats, snaps)
        if diag:
            res["split"][name] = scorer(io.logits_last(hid[P]))
            res["self"][name] = scorer(cond_forward(io, snaps[P - 1]["attn"], snaps[P - 1]["rec"], snaps[P - 1]["conv"], ids[:, P - 1:]))
    for (src, tgt), dm in maps.items():
        _, sfe, ssn = store[src]
        ids, _, tsn = store[tgt]
        tio = ios[tgt]
        d = {}
        if wq is not None:
            w = wq.pop((src, tgt))
            d["C-wrongq"] = scorer(cond_forward(tio, w["attn"], w["rec"], w["conv"], ids[:, P - 1:]))
            d["C-wrongq_from"] = partner["item_id"]
            del w
        m_all = dm.map_prefix(sfe, ssn[P - 1], P - 1)
        if "H-all" in conds:
            d["H-all"] = scorer(cond_forward(tio, m_all["attn"], m_all["rec"], m_all["conv"], ids[:, P - 1:]))
        if "C-noise" in conds:
            g = torch.Generator(device=tio.dev); g.manual_seed(item_seed(rec["item_id"], rec.get("perm", ""), src, tgt))
            na, nr, nc, rel = noise_like_error(tsn[P - 1], m_all, g, dm.kd)
            d["C-noise"] = scorer(cond_forward(tio, na, nr, nc, ids[:, P - 1:]))
            d["map_relerr"] = {k: round(v, 4) for k, v in rel.items()}
        if "C-swap" in conds:
            key = (src, tgt, rec.get("perm", "orig"))
            if key in swap:
                sa, sr, sc_ = apply_error(tsn[P - 1], swap[key][1])
                d["C-swap"] = scorer(cond_forward(tio, sa, sr, sc_, ids[:, P - 1:]))
                d["C-swap_from"] = swap[key][0]
            swap[key] = (rec["item_id"], mapping_error(tsn[P - 1], m_all))
        if "H-attn" in conds:
            d["H-attn"] = scorer(cond_forward(tio, m_all["attn"], tsn[P - 1]["rec"], tsn[P - 1]["conv"], ids[:, P - 1:]))
        if "H-delta" in conds:
            d["H-delta"] = scorer(cond_forward(tio, tsn[P - 1]["attn"], m_all["rec"], m_all["conv"], ids[:, P - 1:]))
        if "H-copy" in conds and dm.copy_ok:
            cr, cc = dm.copy_delta(ssn[P - 1])
            d["H-copy"] = scorer(cond_forward(tio, m_all["attn"], cr, cc, ids[:, P - 1:]))
        del m_all
        if "H-tail" in conds:
            mt = dm.map_prefix(sfe, ssn[P - T_TAIL], P - T_TAIL)
            d["H-tail"] = scorer(cond_forward(tio, mt["attn"], mt["rec"], mt["conv"], ids[:, P - T_TAIL:]))
        if "H-ctx" in conds:
            mc = dm.map_prefix(sfe, ssn[E], E)
            d["H-ctx"] = scorer(cond_forward(tio, mc["attn"], mc["rec"], mc["conv"], ids[:, E:]))
        res["dirs"][f"{src}>{tgt}"] = d
    return res


# ----------------------------------------------------------------------------------------------- GSM8K
@torch.no_grad()
def merge_caches(io, caches):
    import copy
    c = copy.deepcopy(io.template)
    for l in io.attn:
        c.layers[l].keys = torch.cat([x.layers[l].keys for x in caches], 0)
        c.layers[l].values = torch.cat([x.layers[l].values for x in caches], 0)
    for l in io.delta:
        c.layers[l].recurrent_states[0] = torch.cat([x.layers[l].recurrent_states[0] for x in caches], 0)
        c.layers[l].conv_states[0] = torch.cat([x.layers[l].conv_states[0] for x in caches], 0)
    lens = {x.get_seq_length() for x in caches}
    assert len(lens) == 1, lens
    return c


@torch.no_grad()
def decode_rows(io, tok, rows, max_new, eos_ids, ban_ids, decode="eager"):
    """rows: list of (label, cache_after_prefill, last_logits [1,V]). All caches have the same length; decoded as one
    unpadded batch with a fixed batch size until every row has stopped. decode="graph" replays a captured CUDA graph
    per step on a static copy of the cache (same kernels; ~3-5x faster, see fast_decode.py)."""
    labels = [r[0] for r in rows]
    cache = merge_caches(io, [r[1] for r in rows])
    lg = torch.cat([r[2] for r in rows], 0)
    B = len(rows)
    gd = None
    if decode == "graph":
        gd = GraphDecoder(io.m, to_static(io.m, cache, cache.get_seq_length() + max_new + 1), B, ban_ids)
        del cache
    toks = [[] for _ in range(B)]
    reason = [None] * B
    for step in range(max_new + 1):
        if ban_ids:
            lg[:, ban_ids] = float("-inf")
        nxt = lg.argmax(-1)
        for b in range(B):
            if reason[b] is not None:
                continue
            t = int(nxt[b])
            if t in eos_ids:
                reason[b] = "eos"; continue
            if step == max_new:
                reason[b] = "length"; continue
            toks[b].append(t)
            tail = tok.decode(toks[b][-8:])
            if any(s in tail for s in scoring.STOPS) and any(s in tok.decode(toks[b]) for s in scoring.STOPS):
                reason[b] = "stop"
        if all(r is not None for r in reason):
            break
        if gd is not None:
            lg = gd.step(nxt)[1]
            continue
        out = forward_text(io.m, nxt.view(B, 1), past=cache, use_cache=True)
        cache = out.past_key_values
        lg = io.m.lm_head(out.last_hidden_state[:, -1]).float()
    del gd
    res = {}
    for b, lab in enumerate(labels):
        text = scoring.cut_at_stop(tok.decode(toks[b]))
        res[lab] = {"text": text, "n_tokens": len(toks[b]), "stop": reason[b], "token_ids": toks[b]}
    return res


@torch.no_grad()
def gsm_item(rec, ios, maps, tok, args, eos_ids, ban_ids, swap, partner=None):
    P, E = rec["P"], rec["E"]
    cuts = [E, P - T_TAIL, P - 1]
    wq = None
    if "C-wrongq" in args.conditions and partner is not None:
        assert partner["item_id"] != rec["item_id"] and partner["ids"][-1] == rec["ids"][-1]
        wq = partner_maps(partner, ios, maps)
    store = {}
    for name, io in ios.items():
        ids = torch.tensor([rec["ids"]], device=io.dev)
        feats, snaps, hid, live = io.segmented(ids, cuts, keep_live=args.dev_checks)
        store[name] = (ids, feats, snaps, live)
    out = {"recv": {}}
    for tgt, tio in ios.items():
        ids, _, tsn, live = store[tgt]
        rows = []

        def add(label, cache, rest):
            o = forward_text(tio.m, rest, past=cache, use_cache=True)
            lg = tio.logits_last(o.last_hidden_state)
            assert torch.isfinite(lg).all()
            rows.append((label, o.past_key_values, lg))

        add("R-native", None, ids)
        if args.dev_checks:
            add("R-split", live, ids[:, P - 1:])
            add("R-self", tio.install(tsn[P - 1]["attn"], tsn[P - 1]["rec"], tsn[P - 1]["conv"]), ids[:, P - 1:])
        rel_all = {}
        for (src, t2), dm in maps.items():
            if t2 != tgt:
                continue
            _, sfe, ssn, _ = store[src]
            if wq is not None:
                w = wq.pop((src, tgt))
                add(f"C-wrongq|{src}", tio.install(w["attn"], w["rec"], w["conv"]), ids[:, P - 1:])
                rel_all.setdefault("wrongq_from", {})[src] = partner["item_id"]
                del w
            m_all = dm.map_prefix(sfe, ssn[P - 1], P - 1)
            if "H-all" in args.conditions:
                add(f"H-all|{src}", tio.install(m_all["attn"], m_all["rec"], m_all["conv"]), ids[:, P - 1:])
            if "C-noise" in args.conditions:
                g = torch.Generator(device=tio.dev); g.manual_seed(item_seed(rec["item_id"], src, tgt))
                na, nr, nc, rel = noise_like_error(tsn[P - 1], m_all, g, dm.kd)
                add(f"C-noise|{src}", tio.install(na, nr, nc), ids[:, P - 1:])
                rel_all[src] = {k: round(v, 4) for k, v in rel.items()}
            if "C-swap" in args.conditions:
                key = (src, tgt)
                if key in swap:
                    sa, sr, sc_ = apply_error(tsn[P - 1], swap[key][1])
                    add(f"C-swap|{src}", tio.install(sa, sr, sc_), ids[:, P - 1:])
                    rel_all.setdefault("swap_from", {})[src] = swap[key][0]
                swap[key] = (rec["item_id"], mapping_error(tsn[P - 1], m_all))
            del m_all
            if "H-tail" in args.conditions:
                mt = dm.map_prefix(sfe, ssn[P - T_TAIL], P - T_TAIL)
                add(f"H-tail|{src}", tio.install(mt["attn"], mt["rec"], mt["conv"]), ids[:, P - T_TAIL:])
            if "H-ctx" in args.conditions:
                mc = dm.map_prefix(sfe, ssn[E], E)
                add(f"H-ctx|{src}", tio.install(mc["attn"], mc["rec"], mc["conv"]), ids[:, E:])
        gen = {}
        by_len = {}
        for r in rows:  # rows decode as one unpadded batch per cache length (C-wrongq rows have the partner's length)
            by_len.setdefault(r[1].get_seq_length(), []).append(r)
        for L_ in sorted(by_len):
            gen.update(decode_rows(tio, tok, by_len[L_], args.max_new, eos_ids, ban_ids, args.decode))
        for lab, g in gen.items():
            g.update(scoring.score(g["text"], rec["gold"]))
        out["recv"][tgt] = {"rows": gen, "map_relerr": rel_all, "batch": [r[0] for r in rows]}
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="e.g. 9B,27B")
    ap.add_argument("--task", required=True, choices=["mmlu", "mmlu_chat", "gsm8k"])
    ap.add_argument("--manifests", nargs="+", required=True, help="mmlu: base [perm]; others: one file")
    ap.add_argument("--calib", default="chat", choices=["chat", "web"])
    ap.add_argument("--conditions", nargs="+", default=["H-all", "H-tail", "H-ctx", "C-noise", "C-swap", "H-attn", "H-delta", "H-copy"])
    ap.add_argument("--swap_prime", default="", help="manifest whose first item primes the C-swap error buffer")
    ap.add_argument("--diag", type=int, default=1, help="mmlu: run R-split and R-self on the original ordering")
    ap.add_argument("--dev_checks", type=int, default=0, help="gsm8k: add R-split and R-self rows")
    ap.add_argument("--max_new", type=int, default=1024)
    ap.add_argument("--decode", default="graph", choices=["graph", "eager"],
                    help="greedy decode loop; the recorded study runs used eager (pass it to resume or reproduce them)")
    ap.add_argument("--conv_fallback", action="store_true", help="PyTorch convolution kernels instead of causal-conv1d, as in the confirmatory runs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.conv_fallback:
        import transformers.models.qwen3_5.modeling_qwen3_5 as M
        for n in ("causal_conv1d_fn", "causal_conv1d_update"):
            setattr(M, n, getattr(M, n).__wrapped__)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    names = args.group.split(",")
    assert len(names) == 2 and all(n in MODELS for n in names)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[names[0]])
    ios = {}
    for n in names:
        log("loading", n); ios[n] = ModelIO(n, load_model(MODELS[n], "cuda:0"))
    maps = {}
    for src, tgt in ((names[0], names[1]), (names[1], names[0])):
        log("maps", src, ">", tgt)
        dm = DirectedMap(args.calib, src, tgt, ios[src].m, ios[tgt].m, ios[tgt].dev)
        dm.rope = ios[tgt].rope(8192)
        maps[(src, tgt)] = dm
    man = run_manifest(args, maps, ios)
    base = f"{args.task}__{args.calib}__{names[0]}-{names[1]}{args.tag}__s{args.shard}of{args.nshards}"
    out_path, man_path = os.path.join(args.out, base + ".jsonl"), os.path.join(args.out, base + ".manifest.json")
    done = set()
    if os.path.exists(man_path):
        old = json.load(open(man_path))
        if manifest_key(old) != manifest_key(man):
            raise SystemExit(f"run manifest mismatch for {man_path}; refusing to resume (use a new --tag or --out)")
        if os.path.exists(out_path):
            for line in open(out_path):
                try:
                    done.add(json.loads(line)["item_id"])
                except Exception:
                    pass
        man["resumed_runs"] = old.get("resumed_runs", []) + [{"time": time.strftime("%F %T"), "gpu": man["gpu"]}]
    json.dump(man, open(man_path, "w"), indent=1)
    rows = [json.loads(l) for l in open(args.manifests[0])]
    partner = {}
    if "C-wrongq" in args.conditions:
        # partner = next item in manifest order with the same exemplar block (same subject for MMLU), cyclic
        byk = {}
        for r in rows:
            byk.setdefault(r.get("subject") if args.task != "gsm8k" else "all", []).append(r)
        for g in byk.values():
            for k, r in enumerate(g):
                if len(g) > 1:
                    partner[r["item_id"]] = g[(k + 1) % len(g)]
                else:  # single-item block (development manifests only): next item in manifest order
                    partner[r["item_id"]] = rows[(rows.index(r) + 1) % len(rows)]
                    log("partner from a different exemplar block for", r["item_id"])
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard:: args.nshards]
    perm = {}
    if args.task == "mmlu" and len(args.manifests) > 1:
        perm = {r["item_id"]: r for r in map(json.loads, open(args.manifests[1]))}
    eos_ids = {tok.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_end|>")} | ({tok.eos_token_id} if tok.eos_token_id is not None else set())
    ban_ids = [tok.convert_tokens_to_ids("<think>")]
    swap = {}
    if args.swap_prime and "C-swap" in args.conditions:
        prime = json.loads(open(args.swap_prime).readline())
        if args.task in ("mmlu", "mmlu_chat"):
            sc0 = LetterScorer(tok, bare=(args.task == "mmlu_chat"))
            mmlu_variant(prime, ios, maps, sc0, {"C-swap"}, diag=False, swap=swap)
            for k in list(swap):
                swap[(k[0], k[1], "shift2")] = swap[k]  # the shift2 buffer is primed with the same development item
        else:
            gsm_item(prime, ios, maps, tok, argparse.Namespace(**{**vars(args), "conditions": ["C-swap"], "dev_checks": 0, "max_new": 1}), eos_ids, ban_ids, swap)
        log("C-swap buffer primed with", prime["item_id"])
    log(f"{len(rows)} items, {len(done)} already done; kernels {man['kernels']}")
    t0, n_new = time.time(), 0
    f = open(out_path, "a")
    for i, rec in enumerate(rows):
        if rec["item_id"] in done:
            continue
        t1 = time.time()
        try:
            if args.task in ("mmlu", "mmlu_chat"):
                sc = LetterScorer(tok, bare=(args.task == "mmlu_chat"))
                pj = partner.get(rec["item_id"])
                res = {"orig": mmlu_variant(rec, ios, maps, sc, set(args.conditions), diag=bool(args.diag), swap=swap, partner=pj)}
                if rec["item_id"] in perm:
                    pc = set(args.conditions) & {"H-all", "H-tail", "H-ctx", "C-noise", "C-swap", "C-wrongq"}
                    res["shift2"] = mmlu_variant(perm[rec["item_id"]], ios, maps, sc, pc, diag=False, swap=swap,
                                                 partner=perm.get(pj["item_id"]) if pj else None)
                    res["shift2_gold"] = perm[rec["item_id"]]["gold"]
            else:
                res = gsm_item(rec, ios, maps, tok, args, eos_ids, ban_ids, swap, partner=partner.get(rec["item_id"]))
            line = {"item_id": rec["item_id"], "gold": rec["gold"], "subject": rec.get("subject"), "P": rec["P"], "E": rec["E"],
                    "sha": rec["sha"], "status": "ok", "sec": round(time.time() - t1, 2), **res}
        except Exception as e:  # persist failures explicitly
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            line = {"item_id": rec["item_id"], "status": "failed", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            log("FAILED", rec["item_id"], line["error"])
        f.write(json.dumps(line) + "\n"); f.flush()
        n_new += 1
        if n_new % 20 == 0:
            log(f"{i + 1}/{len(rows)} items, {(time.time() - t0) / n_new:.2f} s/item")
    log("done", out_path)


if __name__ == "__main__":
    main()
