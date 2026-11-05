#!/usr/bin/env python
"""Build a mixed calibration set: 50% raw FineWeb-Edu docs (reused from the phase-1 data file) and 50% synthetic
multi-turn chat-template sequences built from FineWeb text, dense in the special tokens and answer cues that
the task prompts use (<|im_start|>, <|im_end|>, assistant, <think>, </think>, "Q:/A:", "Question:/Answer:",
"The answer is N.", "A./B./C./D. ... Answer: X").  Writes data files in the layout pilot.build_data expects."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import CACHE, FINEWEB_GLOB, model_path  # noqa: E402
import argparse, glob, os, random, re

import pyarrow.parquet as pq
import torch

PARQUET_GLOB = FINEWEB_GLOB
U = "<|im_start|>user\n{u}<|im_end|>\n"
A = "<|im_start|>assistant\n<think>\n\n</think>\n\n{a}<|im_end|>\n"


def sentences(text):
    s = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return [x for x in s if 20 < len(x) < 400]


def make_chat(sents, rng):
    """Return one multi-turn chat text from a list of sentences."""
    out = []
    i = 0
    while i < len(sents) - 4:
        kind = rng.choice(["qa", "question", "fewshot", "mc", "plain", "math"])
        if kind == "qa":
            out.append(U.format(u=f"Q: {sents[i]}") + A.format(a=f"A: {sents[i + 1]} {sents[i + 2]}"))
            i += 3
        elif kind == "question":
            out.append(U.format(u=f"{sents[i]} {sents[i + 1]}\nQuestion: {sents[i + 2]}") + A.format(a=f"Answer: {sents[i + 3]}"))
            i += 4
        elif kind == "fewshot":
            n = rng.randint(2, 4)
            shots = "".join(f"Q: {sents[i + 2 * k]}\nA: {sents[i + 2 * k + 1]}\n\n" for k in range(n) if i + 2 * k + 1 < len(sents))
            q = sents[min(i + 2 * n, len(sents) - 1)]
            num = rng.randint(1, 5000)
            out.append(U.format(u=f"{shots}Q: {q}\nA:") + A.format(a=f"{sents[min(i + 2 * n + 1, len(sents) - 1)]} The answer is {num}."))
            i += 2 * n + 2
        elif kind == "mc":
            words = [w for w in re.findall(r"[A-Za-z]{4,}", " ".join(sents[i: i + 3]))][:12] or ["alpha", "beta", "gamma", "delta"]
            ch = rng.sample(words, min(4, len(words)))
            while len(ch) < 4:
                ch.append(rng.choice(["none", "all", "both", "neither"]))
            letters = "ABCD"
            ans = rng.choice(letters)
            body = f"The following are multiple choice questions (with answers).\n\n{sents[i]}\n" + "\n".join(f"{L}. {c}" for L, c in zip(letters, ch)) + "\nAnswer:"
            out.append(U.format(u=body) + A.format(a=f"{ans}"))
            i += 2
        elif kind == "math":
            a_, b_ = rng.randint(2, 99), rng.randint(2, 99)
            out.append(U.format(u=f"Q: {sents[i]} If there are {a_} items and {b_} more arrive, how many are there?\nA:") +
                       A.format(a=f"There are {a_} originally. {b_} more arrive. {a_} + {b_} = {a_ + b_}. The answer is {a_ + b_}."))
            i += 1
        else:
            out.append(U.format(u=sents[i]) + A.format(a=f"{sents[i + 1]} {sents[i + 2]}"))
            i += 3
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cache", default=CACHE)
    ap.add_argument("--dst_cache", default=f"{CACHE}/chatcal")
    ap.add_argument("--tok", default=model_path("9B"))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--n_chat", type=int, default=512)
    ap.add_argument("--doc_offset", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(args.dst_cache, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok)
    rng = random.Random(1)
    files = sorted(glob.glob(PARQUET_GLOB))
    pf = pq.ParquetFile(files[0])
    chats = []
    seen = 0
    for batch in pf.iter_batches(batch_size=512, columns=["text"]):
        for text in batch.column("text").to_pylist():
            seen += 1
            if seen < args.doc_offset:
                continue
            sents = sentences(text)
            if len(sents) < 12:
                continue
            ids = tok(make_chat(sents, rng), add_special_tokens=False)["input_ids"]
            if len(ids) >= args.seq:
                chats.append(torch.tensor(ids[: args.seq], dtype=torch.long))
            if len(chats) >= args.n_chat:
                break
        if len(chats) >= args.n_chat:
            break
    chats = torch.stack(chats)
    old = torch.load(os.path.join(args.src_cache, f"data_s{args.seq}_c1024_e64.pt"))
    raw_full, raw_half, evals = old["calib"][:256], old["calib"][512:768], old["eval"]
    calib_1024 = torch.cat([raw_full, chats[:256], raw_half, chats[256:512]], 0)
    special = {tok.convert_tokens_to_ids(t) for t in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]}
    frac = sum(int(x in special) for x in chats.flatten().tolist()) / chats.numel()
    print(f"chat sequences {tuple(chats.shape)}, special-token fraction {frac:.3%}; example:\n{tok.decode(chats[0][:300])}")
    torch.save({"calib": calib_1024[:512], "eval": evals}, os.path.join(args.dst_cache, f"data_s{args.seq}_c512_e64.pt"))
    torch.save({"calib": calib_1024, "eval": evals}, os.path.join(args.dst_cache, f"data_s{args.seq}_c1024_e64.pt"))
    print("wrote", args.dst_cache)


if __name__ == "__main__":
    main()
