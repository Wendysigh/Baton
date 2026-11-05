#!/usr/bin/env python
"""Long timing prompts: 35 non-overlapping 131,072-token PG19 windows (5 warm-up, 30 timed), after each book's first 2,048 tokens.
Only 25 local books are long enough, so the longest books give extra windows. Timing only."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import CACHE, MAN, model_path  # noqa: E402
import glob, json, os, random
import torch
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path("9B"))
W = 131072
names = sorted(glob.glob(f"{CACHE}/pg19/books/*.txt"))
random.Random(20261105).shuffle(names)
books = {}
for p in names:
    ids = tok(open(p, errors="ignore").read(), add_special_tokens=False)["input_ids"]
    if len(ids) >= 2048 + W:
        books[os.path.basename(p)] = ids
out, used = [], []
for name, ids in books.items():  # first window of every book, in shuffled order
    out.append(ids[2048: 2048 + W]); used.append([name, 0])
k = 1
while len(out) < 35:  # then later non-overlapping windows from the longest books
    more = [n for n in sorted(books, key=lambda n: -len(books[n])) if len(books[n]) >= 2048 + (k + 1) * W]
    assert more
    for n in more[: 35 - len(out)]:
        out.append(books[n][2048 + k * W: 2048 + (k + 1) * W]); used.append([n, k])
    k += 1
print("books", len(books), "windows", len(out))
torch.save(torch.tensor(out, dtype=torch.long), os.path.join(MAN, "timing_docs_long.pt"))
json.dump(used, open(os.path.join(MAN, "timing_docs_long_books.json"), "w"))
print("ok")
