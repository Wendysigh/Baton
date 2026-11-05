#!/usr/bin/env python
"""GSM8K on the full test set (1,319 problems) with the original chat prompt and scoring
(8-shot CoT, greedy, up to 512 new tokens, thinking disabled); decoding replays a CUDA graph per step.

Configs: native; web / chat (all prompt tokens but the last translated, FineWeb- or chat-calibrated maps);
tail_web (the last 12 prompt tokens prefilled by the target); copy_chat (chat-calibrated attention maps, copied state).
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE, RAW  # noqa: E402
import argparse, json, os, random, re, sys, time

import pyarrow.parquet as pq
import torch

from common import log, setup, prefix_cache, generate_graph  # noqa: E402


GSM_SHOTS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]


GSM_INSTR = "Answer the final question. Reason step by step as in the examples and end with \"The answer is N.\"\n\n"


def gsm_number(text):
    m = re.search(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def chat_wrap(tok, user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def ids_split(tok, pre_text, post_text, dev):
    a = tok(pre_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    b = tok(post_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return torch.cat([a, b], 1).to(dev), a.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--configs", nargs="+", default=["native", "web", "tail_web", "chat", "copy_chat"])
    ap.add_argument("--tail", type=int, default=12)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=f"{RAW}/extra/gsm8k")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    calibs = sorted({c.split("_")[-1] for c in args.configs if c != "native"})
    tok, eos_ids, src_m, tgt_m, hos = setup(args.direction, calibs)
    dev = next(tgt_m.parameters()).device
    test = pq.read_table(f"{CACHE}/data/gsm8k_test.parquet").to_pylist()
    order = random.Random(0).sample(range(len(test)), len(test))
    shots = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in GSM_SHOTS)
    path = os.path.join(args.out, f"gsm8k_{args.direction.replace('>', 'to')}_sh{k}of{n}.jsonl")
    done = {json.loads(l)["idx"] for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    t0, cnt, corr = time.time(), 0, {c: 0 for c in args.configs}
    for idx in order[k::n]:
        if idx in done:
            continue
        r = test[idx]
        q = f"Q: {r['question'].strip()}\nA:"
        full = chat_wrap(tok, GSM_INSTR + shots + q)
        ids, E = ids_split(tok, full[:full.rfind(q)], full[full.rfind(q):], dev)
        ref = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        rec = {"idx": idx, "ref": ref, "len_tokens": ids.shape[1]}
        for c in args.configs:
            if c == "native":
                cache, rest = None, ids
            elif c.startswith("tail_"):
                cache, rest = prefix_cache(c[5:], tgt_m, hos, ids[:, :-args.tail]), ids[:, -args.tail:]
            else:
                cache, rest = prefix_cache(c, tgt_m, hos, ids[:, :-1]), ids[:, -1:]
            text, nt = generate_graph(tgt_m, tok, cache, rest, args.max_new, eos_ids, stops=("\nQ:", "\n\nQ:"))
            num = gsm_number(text)
            rec[c] = {"text": text, "num": num, "n_tokens": nt, "ok": int(num is not None and abs(num - ref) < 1e-6)}
            corr[c] += rec[c]["ok"]
            del cache
        cnt += 1
        f.write(json.dumps(rec) + "\n"); f.flush()
        if cnt % 10 == 0:
            log(f"{args.direction} sh{k} {cnt} ({time.time() - t0:.0f}s) " + " ".join(f"{c}={100 * corr[c] / cnt:.1f}" for c in args.configs))
    log("done")


if __name__ == "__main__":
    main()
