#!/usr/bin/env python
"""Zero-shot log-likelihood tasks (ARC-Challenge, HellaSwag, WinoGrande) under cross-model cache translation.

Requests, tokenization (lm-eval's _encode_pair) and metrics (task.process_results: acc, acc_norm) come from lm-eval
0.4.13 itself. For every context the target's cache over context[:-1] is either its own prefill (native), translated
from the source (web-calibrated maps), or the copy baseline; the target then processes the last context token and the
continuation, whose log-likelihood is scored. No chat template, as in lm-eval.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW  # noqa: E402
import argparse, glob, json, math, os, sys, time
from collections import defaultdict

import torch

from common import log, setup, prefix_cache, expand_cache, forward_text  # noqa: E402


def encode_pair(tok, context, continuation):
    n = len(context) - len(context.rstrip())
    if n > 0:
        continuation, context = context[-n:] + continuation, context[:-n]
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]


@torch.no_grad()
def score_conts(tgt_m, cache, last_tok, conts, dev):
    """cache over context[:-1] (or None); returns [(ll, is_greedy)] per continuation."""
    B = len(conts)
    L = max(len(c) for c in conts)
    inp = torch.full((B, 1 + L), last_tok, dtype=torch.long, device=dev)
    for b, c in enumerate(conts):  # right padding: padded positions come after every scored one
        inp[b, 1:1 + len(c)] = torch.tensor(c, device=dev)
    inp = inp[:, :L]  # the last continuation token is only predicted, never fed
    past = expand_cache(cache, B) if cache is not None else None
    h = forward_text(tgt_m, inp, past=past, use_cache=past is not None).last_hidden_state
    out = []
    for b, c in enumerate(conts):
        lp = torch.log_softmax(tgt_m.lm_head(h[b, :len(c)]).float(), -1)
        tgt = torch.tensor(c, device=dev)
        out.append((float(lp.gather(1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--tasks", nargs="+", default=["arc_challenge", "hellaswag", "winogrande"])
    ap.add_argument("--configs", nargs="+", default=["native", "web", "copy_web"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=f"{RAW}/extra/ll")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    from lm_eval.tasks import TaskManager, get_task_dict
    tdict = get_task_dict(args.tasks, TaskManager())
    calibs = sorted({c.split("_", 1)[-1] for c in args.configs if c not in ("native", "none")})
    tok, eos_ids, src_m, tgt_m, hos = setup(args.direction, calibs)
    dev = next(tgt_m.parameters()).device
    tag = args.direction.replace(">", "to")
    for name in args.tasks:
        task = tdict[name]
        task.build_all_requests(limit=args.limit or None, rank=0, world_size=1)
        by_doc = defaultdict(list)
        for inst in task.instances:
            by_doc[inst.doc_id].append(inst)
        doc_ids = sorted(by_doc)[k::n]
        path = os.path.join(args.out, f"{name}_{tag}_sh{k}of{n}.jsonl")
        done = set()  # docs scored by any shard layout of this task and direction
        for q in glob.glob(os.path.join(args.out, f"{name}_{tag}_sh*.jsonl")):
            done |= {json.loads(l)["doc_id"] for l in open(q)}
        f = open(path, "a")
        t0 = time.time()
        for i, d in enumerate(doc_ids):
            if d in done:
                continue
            insts = sorted(by_doc[d], key=lambda x: x.idx)
            doc = insts[0].doc
            pairs = [encode_pair(tok, *x.arguments) for x in insts]
            rec = {"doc_id": d, "n_ctx_tokens": [len(p[0]) for p in pairs]}
            for c in args.configs:
                res = [None] * len(pairs)
                groups = defaultdict(list)
                for j, (ctx, cont) in enumerate(pairs):
                    groups[tuple(ctx)].append(j)
                for ctx, js in groups.items():
                    ids = torch.tensor([list(ctx)], device=dev)
                    cache = prefix_cache(c, tgt_m, hos, ids[:, :-1])
                    for j, r in zip(js, score_conts(tgt_m, cache, ctx[-1], [pairs[j][1] for j in js], dev)):
                        res[j] = r
                    del cache
                rec[c] = {"ll": [r[0] for r in res], **task.process_results(doc, res)}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (i + 1) % 200 == 0:
                log(f"{name} {args.direction} sh{k} {i + 1}/{len(doc_ids)} ({time.time() - t0:.0f}s)")
        f.close()
    log("done")


if __name__ == "__main__":
    main()
