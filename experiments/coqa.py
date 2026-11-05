#!/usr/bin/env python
"""CoQA multi-turn handoff (lm-eval's CoQA prompt and F1), turns 1-10 of 100 validation conversations
(20 per source domain, seed 0). Each turn's prompt is the story, the previous turns with gold answers, and the current
question (lm-eval's doc_to_text). Greedy decoding until a newline (lm-eval keeps the first line).

Configs per turn:
  native    : the target prefills the whole prompt.
  once_web  : one switch at the start of the conversation: the story is translated, and the target processes all
              later turns (questions and gold answers) and the current question itself, on top of the translated cache.
  each_web  : a switch at this turn: everything but the last prompt token is translated.
  each_copy_web : as each_web with the copy baseline.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import RAW  # noqa: E402
import argparse, json, os, random, sys, time

import torch

from common import log, setup, prefix_cache, generate_graph  # noqa: E402


def f1_turn(doc, t, pred):
    import transformers.data.metrics.squad_metrics as sq
    golds = [doc["answers"]["input_text"][t]]
    for key in doc.get("additional_answers") or {}:
        a = doc["additional_answers"][key]["input_text"][t]
        if a.lower() not in map(str.lower, golds):
            golds.append(a)
    if len(golds) > 1:  # lm-eval: leave-one-out max, averaged
        return sum(max(sq.compute_f1(a, pred) for a in golds[:i] + golds[i + 1:]) for i in range(len(golds))) / len(golds)
    return sq.compute_f1(golds[0], pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--configs", nargs="+", default=["native", "once_web", "each_web", "each_copy_web"])
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--per_domain", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=f"{RAW}/extra/coqa")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/coqa", split="validation")
    rng = random.Random(0)
    by_src = {}
    for i, d in enumerate(ds):
        if len(d["questions"]["input_text"]) >= args.turns:
            by_src.setdefault(d["source"], []).append(i)
    picked = sorted(j for s in sorted(by_src) for j in rng.sample(by_src[s], args.per_domain))
    calibs = sorted({c.split("_")[-1] for c in args.configs if c != "native"})
    tok, eos_ids, src_m, tgt_m, hos = setup(args.direction, calibs)
    dev = next(tgt_m.parameters()).device
    path = os.path.join(args.out, f"coqa_{args.direction.replace('>', 'to')}_sh{k}of{n}.jsonl")
    done = {json.loads(l)["doc"] for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    t0 = time.time()
    for di in picked[k::n]:
        if di in done:
            continue
        doc = ds[di]
        story = doc["story"] + "\n\n"
        rec = {"doc": di, "source": doc["source"], "turns": []}
        for t in range(args.turns):
            hist = "".join(f"Q: {q}\n\nA: {a}\n\n" for q, a in zip(doc["questions"]["input_text"][:t], doc["answers"]["input_text"][:t]))
            s_ids, r_ids = enc(story), enc(hist + f"Q: {doc['questions']['input_text'][t]}\n\nA:")
            ids = torch.tensor([s_ids + r_ids], device=dev)
            E = len(s_ids)
            row = {"len": ids.shape[1]}
            for c in args.configs:
                if c == "native":
                    cache, rest = None, ids
                elif c.startswith("once_"):
                    cache, rest = prefix_cache(c[5:], tgt_m, hos, ids[:, :E]), ids[:, E:]
                else:
                    cache, rest = prefix_cache(c[5:], tgt_m, hos, ids[:, :-1]), ids[:, -1:]
                text, _ = generate_graph(tgt_m, tok, cache, rest, args.max_new, eos_ids, stops=("\n",))
                pred = text.strip().split("\n")[0]
                row[c] = {"pred": pred, "f1": f1_turn(doc, t, pred)}
                del cache
            rec["turns"].append(row)
        f.write(json.dumps(rec) + "\n"); f.flush()
        log(f"doc {di} ({time.time() - t0:.0f}s) " + " ".join(f"{c}={sum(r[c]['f1'] for r in rec['turns']) / args.turns:.2f}" for c in args.configs))
    log("done")


if __name__ == "__main__":
    main()
