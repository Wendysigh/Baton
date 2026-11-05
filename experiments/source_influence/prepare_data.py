#!/usr/bin/env python
"""Freeze item manifests (ids, splits, gold, full prompt token ids, segment boundaries) for the source-influence study.

Outputs (under --out): mmlu_base.jsonl, mmlu_base_perm.jsonl, mmlu_chat.jsonl, gsm8k.jsonl,
mmlu_dev.jsonl, mmlu_dev_perm... (dev variants), and manifest_index.json with SHA-256 of every file.
Each line: {item_id, split, role, orig_index, subject, gold, ids, P, E, sha}.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import CACHE, MAN, MMLU_SNAPSHOTS, GSM8K_ARROW, model_path  # noqa: E402
import argparse, hashlib, json, os, random, sys

import pyarrow.parquet as pq
import pyarrow as pa

LETTERS = ["A", "B", "C", "D"]
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
CHAT_INSTR = "\n\nAnswer with only the letter (A, B, C, or D) of the correct option."
MMLU_PARQ = MMLU_SNAPSHOTS + "/{}/all/{}-00000-of-00001.parquet"
GSM_ARROW = GSM8K_ARROW


def sha1(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256_ids(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mmlu_q(r, choices=None):
    ch = r["choices"] if choices is None else choices
    return r["question"].strip() + "\n" + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, ch)) + "\nAnswer:"


def mmlu_ctx(subject, shots):
    head = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    return head + "".join(mmlu_q(d) + f" {LETTERS[d['answer']]}\n\n" for d in shots)


def perm2(r):
    """cyclic shift by two: option at old position i moves to position (i+2)%4"""
    ch = [None] * 4
    for i, c in enumerate(r["choices"]):
        ch[(i + 2) % 4] = c
    return ch, (r["answer"] + 2) % 4


def encode_split(tok, ctx_text, q_text):
    full = tok(ctx_text + q_text, add_special_tokens=False)["input_ids"]
    ctx = tok(ctx_text, add_special_tokens=False)["input_ids"]
    if full[: len(ctx)] != ctx:
        raise AssertionError("exemplar boundary is not a token boundary")
    return full, len(ctx)


def chat_split(tok, ctx_text, q_text):
    """user turn = ctx + q + instruction; returns ids and the token index where q starts"""
    msg = ctx_text + q_text + CHAT_INSTR
    full_text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    pre = full_text[: full_text.index(q_text)]
    full = tok(full_text, add_special_tokens=False)["input_ids"]
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    if full[: len(pre_ids)] != pre_ids:
        raise AssertionError("chat boundary is not a token boundary")
    return full, len(pre_ids)


def stratified(rows, n, seed):
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
    tot = len(rows)
    quota = {s: n * len(v) / tot for s, v in by.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    rest = n - sum(alloc.values())
    for s in sorted(quota, key=lambda s: (-(quota[s] - alloc[s]), s))[:rest]:
        alloc[s] += 1
    rng = random.Random(seed)
    out = []
    for s in sorted(by):
        out += rng.sample(by[s], alloc[s])
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default=model_path("9B"))
    ap.add_argument("--out", default=f"{MAN}/source_influence")
    ap.add_argument("--n_mmlu", type=int, default=3000)
    ap.add_argument("--n_mmlu_chat", type=int, default=1000)
    ap.add_argument("--n_mmlu_dev", type=int, default=200)
    ap.add_argument("--n_gsm_dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok)

    snap = os.listdir(MMLU_SNAPSHOTS)[0]
    test = pq.read_table(MMLU_PARQ.format(snap, "test")).to_pylist()
    devsplit = pq.read_table(MMLU_PARQ.format(snap, "dev")).to_pylist()
    val = pq.read_table(MMLU_PARQ.format(snap, "validation")).to_pylist()
    legacy_test = pq.read_table(f"{CACHE}/data/mmlu_test.parquet").to_pylist()
    assert len(test) == len(legacy_test) == 14042
    assert all(a["question"] == b["question"] and list(a["choices"]) == list(b["choices"]) for a, b in zip(test, legacy_test)), "legacy MMLU order differs"
    shots_by = {}
    for r in devsplit:
        shots_by.setdefault(r["subject"], []).append(r)
    for i, r in enumerate(test):
        r["orig_index"] = i
        r["item_id"] = "mmlu-" + sha1({"s": r["subject"], "q": r["question"], "c": list(r["choices"]), "a": r["answer"]})[:16]
    inspected = random.Random(0).sample(test, 500)
    insp_ids = {r["item_id"] for r in inspected}
    fresh = [r for r in test if r["item_id"] not in insp_ids]
    conf = stratified(fresh, args.n_mmlu, args.seed)
    for i, r in enumerate(val):
        r["orig_index"] = i
        r["item_id"] = "mmluval-" + sha1({"s": r["subject"], "q": r["question"], "c": list(r["choices"]), "a": r["answer"]})[:16]
    mdev = stratified(val, args.n_mmlu_dev, args.seed + 1)

    def mmlu_rows(rows, split, role):
        base, perm, chat = [], [], []
        for r in rows:
            ctx = mmlu_ctx(r["subject"], shots_by[r["subject"]][:5])
            ids, E = encode_split(tok, ctx, mmlu_q(r))
            rec = {"item_id": r["item_id"], "split": split, "role": role, "orig_index": r["orig_index"], "subject": r["subject"],
                   "gold": r["answer"], "ids": ids, "P": len(ids), "E": E, "sha": sha256_ids(ids), "perm": "orig"}
            base.append(rec)
            ch, g = perm2(r)
            ids2, E2 = encode_split(tok, ctx, mmlu_q(r, ch))
            assert E2 == E and ids2[:E] == ids[:E]
            perm.append({**rec, "gold": g, "ids": ids2, "P": len(ids2), "sha": sha256_ids(ids2), "perm": "shift2"})
            cids, cE = chat_split(tok, ctx, mmlu_q(r))
            chat.append({**rec, "ids": cids, "P": len(cids), "E": cE, "sha": sha256_ids(cids), "perm": "orig", "protocol": "chat"})
        return base, perm, chat

    files = {}
    b, p, c = mmlu_rows(conf, "test", "confirmatory")
    files["mmlu_base.jsonl"], files["mmlu_base_perm.jsonl"], files["mmlu_chat.jsonl"] = b, p, c[: args.n_mmlu_chat]
    b, p, c = mmlu_rows(mdev, "validation", "dev")
    files["mmlu_dev_base.jsonl"], files["mmlu_dev_base_perm.jsonl"], files["mmlu_dev_chat.jsonl"] = b, p, c
    b, p, c = mmlu_rows(inspected, "test", "legacy_inspected")
    files["mmlu_legacy_base.jsonl"] = b

    # ---------------------------------------------------------------- GSM8K
    gtest = pa.ipc.open_stream(GSM_ARROW.format("test")).read_all().to_pylist()
    gtrain = pa.ipc.open_stream(GSM_ARROW.format("train")).read_all().to_pylist()
    legacy_g = pq.read_table(f"{CACHE}/data/gsm8k_test.parquet").to_pylist()
    assert len(gtest) == len(legacy_g) == 1319 and all(a["question"] == b["question"] for a, b in zip(gtest, legacy_g)), "legacy GSM8K order differs"
    for i, r in enumerate(gtest):
        r["orig_index"] = i; r["item_id"] = "gsm-" + sha1({"q": r["question"], "a": r["answer"]})[:16]
    for i, r in enumerate(gtrain):
        r["orig_index"] = i; r["item_id"] = "gsmtrain-" + sha1({"q": r["question"], "a": r["answer"]})[:16]
    insp = {r["item_id"] for r in random.Random(0).sample(gtest, 100)} | {r["item_id"] for r in random.Random(0).sample(gtest, 50)} | {r["item_id"] for r in gtest[:4]}
    gfresh = [r for r in gtest if r["item_id"] not in insp]
    random.Random(args.seed).shuffle(gfresh)
    gdev = random.Random(args.seed + 2).sample(gtrain, args.n_gsm_dev)
    gctx = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in GSM_SHOTS)

    def gsm_rows(rows, split, role):
        out = []
        for r in rows:
            ids, E = encode_split(tok, gctx, f"Q: {r['question']}\nA:")
            ref = r["answer"].split("####")[-1].strip()
            out.append({"item_id": r["item_id"], "split": split, "role": role, "orig_index": r["orig_index"], "gold": ref,
                        "ids": ids, "P": len(ids), "E": E, "sha": sha256_ids(ids)})
        return out

    files["gsm8k.jsonl"] = gsm_rows(gfresh, "test", "confirmatory")
    files["gsm8k_dev.jsonl"] = gsm_rows(gdev, "train", "dev")
    files["gsm8k_legacy.jsonl"] = gsm_rows([r for r in gtest if r["item_id"] in insp], "test", "legacy_inspected")
    ctx_ids = set(tuple(r["ids"][: r["E"]]) for r in files["gsm8k.jsonl"] + files["gsm8k_dev.jsonl"])
    assert len(ctx_ids) == 1, "GSM8K exemplar block differs across items"

    index = {"tokenizer_sha256": {s: file_sha256(f"{model_path(s)}/tokenizer.json") for s in ("4B", "9B", "27B")},
             "mmlu_snapshot": snap, "gsm8k_arrow": GSM_ARROW, "seed": args.seed, "files": {}}
    assert len(set(index["tokenizer_sha256"].values())) == 1
    for name, rows in files.items():
        path = os.path.join(args.out, name)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        index["files"][name] = {"n": len(rows), "sha256": file_sha256(path),
                                "mean_P": sum(r["P"] for r in rows) / len(rows), "max_P": max(r["P"] for r in rows)}
    with open(os.path.join(args.out, "manifest_index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
