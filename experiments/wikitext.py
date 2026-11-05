#!/usr/bin/env python
"""WikiText-2 perplexity under cross-model cache translation, following Heo et al. (2026, App. D): the raw test split
is cut into non-overlapping 2,048-token chunks; the second 1,024 tokens of each chunk are scored conditioned on the
first 1,024.

The target's cache covers prefix tokens [0, 1023) (native prefill, translated from the source, copy baseline, or
nothing); the target itself processes token 1023 and the continuation, so every config scores the same 1,024 tokens.
'none' (only token 1023 as context) is the no-prefix reference of NLL retained.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW  # noqa: E402
import argparse, json, math, os, sys, time

import torch

from common import log, setup, prefix_cache, forward_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--configs", nargs="+", default=["native", "none", "web", "chat", "copy_web"])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--detok", action="store_true", help="lm-eval's wikitext_detokenizer on the raw test text")
    ap.add_argument("--out", default=f"{RAW}/extra/ppl")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    from datasets import load_dataset
    calibs = sorted({c.split("_", 1)[-1] for c in args.configs if c not in ("native", "none")})
    tok, eos_ids, src_m, tgt_m, hos = setup(args.direction, calibs)
    dev = next(tgt_m.parameters()).device
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    if args.detok:
        from lm_eval.tasks.wikitext.preprocess_wikitext import wikitext_detokenizer
        text = wikitext_detokenizer({"page": text})
    ids_all = tok(text, add_special_tokens=False)["input_ids"]
    C, P = args.chunk, args.chunk // 2
    chunks = list(range(len(ids_all) // C))
    log(f"{len(ids_all)} tokens, {len(chunks)} chunks")
    path = os.path.join(args.out, f"wikitext2_{args.direction.replace('>', 'to')}_sh{k}of{n}.jsonl")
    done = {json.loads(l)["chunk"] for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    t0 = time.time()
    for i in chunks[k::n]:
        if i in done:
            continue
        ids = torch.tensor([ids_all[i * C:(i + 1) * C]], device=dev)
        rec = {"chunk": i}
        for c in args.configs:
            with torch.no_grad():
                cache = prefix_cache(c, tgt_m, hos, ids[:, :P - 1])
                h = forward_text(tgt_m, ids[:, P - 1:C - 1], past=cache, use_cache=cache is not None).last_hidden_state[0]
                tok_nll = []
                for s in range(0, h.shape[0], 256):
                    lp = torch.log_softmax(tgt_m.lm_head(h[s:s + 256]).float(), -1)
                    tok_nll.append(-lp.gather(1, ids[0, P + s:P + s + 256, None])[:, 0])
                tok_nll = torch.cat(tok_nll)
            rec[c] = float(tok_nll.mean())
            # scored-token segments [0,128), [128,512), [512,1024): does the prefix's value hold over a long continuation?
            rec[c + "@seg"] = [float(tok_nll[a:b].mean()) for a, b in ((0, 128), (128, 512), (512, P))]
            del cache, h
        f.write(json.dumps(rec) + "\n"); f.flush()
        log(f"chunk {i} ({time.time() - t0:.0f}s) " + " ".join(f"{c}={math.exp(rec[c]):.2f}" for c in args.configs))
    log("done")


if __name__ == "__main__":
    main()
