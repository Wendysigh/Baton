#!/usr/bin/env python
"""LongBench Qasper, HotpotQA, MultiFieldQA-en, official prompts and output limits, in the Qwen chat template with thinking
disabled. The source reads the ordered document prefix; the target natively processes the question suffix."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE  # noqa: E402
import argparse, json, os, sys, time

import torch

from harness import setup, log, decode, source_pass, assemble, read_manifest, shard_rows, open_resume, sha, chat_wrap_user, RES, MODELS  # noqa: E402

D = f"{CACHE}/longbench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--manifest", default="longbench_test.jsonl")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RES, "longbench"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok, eos, src_m, tgt_m, hos = setup(args.direction, ["web", "chat"])
    dev = next(tgt_m.parameters()).device
    P = json.load(open(f"{D}/dataset2prompt.json"))
    data = {ds: [json.loads(l) for l in open(f"{D}/data/data/{ds}.jsonl")] for ds in ("qasper", "hotpotqa", "multifieldqa_en")}
    rows = shard_rows(read_manifest(args.manifest), args.shard)
    k, n = map(int, args.shard.split("/"))
    path = os.path.join(args.out, f"{args.manifest[:-6]}_{args.direction.replace('>', 'to')}_sh{k}of{n}.jsonl")
    done, f = open_resume(path, {r["id"] for r in rows})
    t0, c = time.time(), 0
    for r in rows:
        if r["id"] in done:
            continue
        s = data[r["dataset"]][r["row"]]
        head, tail = P[r["dataset"]].split("{context}")
        ctx = s["context"]
        if r["truncated"]:
            ctx = tok.decode(tok(ctx, add_special_tokens=False)["input_ids"][:31000])
        tail = tail.format(input=s["input"])
        ids = tok(chat_wrap_user(tok, head + ctx + tail), add_special_tokens=False)["input_ids"]
        assert sha(ids) == r["token_sha1"], r["id"]
        ids = torch.tensor([ids], device=dev)
        E, m = r["E"], r["max_new"]
        q = torch.tensor([tok(chat_wrap_user(tok, head + tail), add_special_tokens=False)["input_ids"]], device=dev)
        go = lambda model, cache, rest: decode(model, tok, cache, rest, m, eos, (), graph=False)
        rec = {k_: r[k_] for k_ in ("id", "dataset", "row", "n_tokens", "E", "token_sha1", "doc", "answers", "all_classes")}
        rec["configs"] = {"native": go(tgt_m, None, ids), "no_document": go(tgt_m, None, q), "source": go(src_m, None, ids)}
        scache, store = source_pass(hos, src_m, ids[:, :E])
        for cal in ("web", "chat"):
            rec["configs"][cal] = go(tgt_m, assemble(hos[cal], scache, store, E, ids[:, :1]), ids[:, E:])
        rec["configs"]["copy_web"] = go(tgt_m, assemble(hos["web"], scache, store, E, ids[:, :1], gdn="copy"), ids[:, E:])
        del scache, store
        hos["web"].tap.clear()
        f.write(json.dumps(rec) + "\n"); f.flush()
        c += 1
        if c % 5 == 0:
            log(f"{args.direction} sh{args.shard} {c}/{len(rows) - len(done)} {(time.time() - t0) / c:.1f}s/q, peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
        if args.limit and c >= args.limit:
            break
    log("done")


if __name__ == "__main__":
    main()
