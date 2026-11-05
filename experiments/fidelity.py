#!/usr/bin/env python
"""Next-token fidelity on fresh web documents and PG19 books at 1K and 4K prefixes. Both prefixes end at the same token
and score the same 128 targets: tokens P+1..P+128 given bridge token P. Saves per-document NLL, KL to native and top-1
agreement, so uncertainty is a paired document bootstrap. Extra rec maps (anchor ablation, plain ALS) and independently
calibrated map directories are extra configurations of the same run."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE, MAN  # noqa: E402
import argparse, glob, json, os, sys, time

import torch
import torch.nn.functional as F

from harness import setup, log, source_pass, assemble, forward_text, RES, ORDER  # noqa: E402

PX = 4096


def rec_variants(direction):
    """{name: file} of refine-format state maps for this direction, if they exist."""
    s, t = direction.split(">")
    small, large = sorted((s, t), key=ORDER.get)
    role = {small: "small", large: "large"}
    base = {("4B", "9B"): f"{CACHE}/p4b9b", ("9B", "27B"): CACHE}[(small, large)]
    out = {}
    p = f"{base}/plainals_{role[s]}_to_{role[t]}_s1024_c512_st8.pt"
    if os.path.exists(p):
        out["plainals"] = p
    for f in sorted(glob.glob(f"{CACHE}/rev_anchor/{small}_{large}/anchor*_{role[s]}_to_{role[t]}.pt")):
        out[os.path.basename(f).split("_")[0]] = f
    return out


@torch.no_grad()
def score(m, cache, cont, ref=None):
    """cont [1,129]; returns NLL over the 128 targets, and KL / top-1 against ref logits."""
    o = forward_text(m, cont, past=cache, use_cache=True)
    lp = F.log_softmax(m.lm_head(o.last_hidden_state)[:, :-1].float(), -1)
    nll = -lp.gather(-1, cont[:, 1:, None]).mean().item()
    if ref is None:
        return nll, None, None, lp
    kl = F.kl_div(lp, ref, log_target=True, reduction="none").sum(-1).mean().item()
    return nll, kl, (lp.argmax(-1) == ref.argmax(-1)).float().mean().item(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--sets", nargs="+", default=["web", "book"])
    ap.add_argument("--prefixes", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_variants", type=int, default=0)
    ap.add_argument("--fit", type=int, default=0, help="use the independent calibration fit of this seed instead of the recorded maps")
    ap.add_argument("--out", default=os.path.join(RES, "fidelity"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tagf = ""
    if args.fit:
        import common
        small, large = sorted(args.direction.split(">"), key=ORDER.get)
        common.MAP_DIRS[(small, large, "web")] = f"{CACHE}/rev_fit/{small}_{large}_s{args.fit}"
        args.no_variants, tagf = 1, f"_fit{args.fit}"
    tok, eos, src_m, tgt_m, hos = setup(args.direction, ["web"])
    ho = hos["web"]
    dev = next(tgt_m.parameters()).device
    docs = torch.load(f"{MAN}/fidelity_docs.pt")
    var = {}
    if not args.no_variants:
        for name, p in rec_variants(args.direction).items():
            d = torch.load(p)
            mp = d["maps"]
            var[name] = {lt: {"src": ho.rmaps[lt]["src"], "A": (v["A"] if isinstance(v, dict) else v[0]).to(dev).half(), "B": (v["B"] if isinstance(v, dict) else v[1]).to(dev).half()} for lt, v in mp.items()}
            del d
            log("variant", name, p)
    k, n = map(int, args.shard.split("/"))
    path = os.path.join(args.out, f"fid_{args.direction.replace('>', 'to')}{tagf}_sh{k}of{n}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            i = json.loads(l)["id"]
            assert i not in done, i
            done.add(i)
    f = open(path, "a")
    t0, c = time.time(), 0
    for st in args.sets:
        for di in range(k, docs[st].shape[0], n):
            for P in args.prefixes:
                did = f"{st}|{di}|{P}"
                if did in done:
                    continue
                w = docs[st][di].to(dev)
                prefix, cont = w[PX - P: PX][None], w[PX: PX + 129][None]
                nat = forward_text(tgt_m, prefix, use_cache=True).past_key_values
                n_nat, _, _, ref = score(tgt_m, nat, cont)
                del nat
                res = {"native": {"nll": n_nat, "kl": 0.0, "top1": 1.0}}
                sn, sk, st1 = None, None, None
                nll, kl, t1, _ = score(tgt_m, None, cont, ref)
                res["no_prefix"] = {"nll": nll, "kl": kl, "top1": t1}
                sc = forward_text(src_m, prefix, use_cache=True)
                s_n = -F.log_softmax(src_m.lm_head(forward_text(src_m, cont, past=sc.past_key_values, use_cache=True).last_hidden_state)[:, :-1].float(), -1).gather(-1, cont[:, 1:, None]).mean().item()
                res["source"] = {"nll": s_n}
                del sc
                scache, store = source_pass(hos, src_m, prefix)
                for name, kw in [("kv_only", {"gdn": "zero"}), ("copy", {"gdn": "copy"}), ("baton", {})] + [(v, {"rmaps": var[v]}) for v in var]:
                    nll, kl, t1, _ = score(tgt_m, assemble(ho, scache, store, P, prefix[:, :1], **kw), cont, ref)
                    res[name] = {"nll": nll, "kl": kl, "top1": t1}
                del scache, store, ref
                ho.tap.clear()
                f.write(json.dumps({"id": did, "set": st, "doc": di, "prefix": P, "configs": res}) + "\n"); f.flush()
                c += 1
                if c % 10 == 0:
                    log(f"{args.direction} sh{args.shard} {c} {(time.time() - t0) / c:.1f}s/doc-prefix, peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
                if args.limit and c >= args.limit:
                    log("done"); return
    log("done")


if __name__ == "__main__":
    main()
