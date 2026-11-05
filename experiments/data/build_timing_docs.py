#!/usr/bin/env python
"""Timing prompts: 35 distinct PG19 books (5 warm-up, 30 timed), 32,768 tokens each after the first 2,048. Timing only."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import CACHE, MAN, model_path  # noqa: E402
import glob, os, random
import torch
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path("9B"))
names = sorted(glob.glob(f"{CACHE}/pg19/books/*.txt"))
random.Random(20261105).shuffle(names)
out = []
for p in names:
    ids = tok(open(p, errors="ignore").read(), add_special_tokens=False)["input_ids"]
    if len(ids) >= 2048 + 32768:
        out.append(torch.tensor(ids[2048: 2048 + 32768], dtype=torch.long))
    if len(out) == 35:
        break
assert len(out) == 35
torch.save(torch.stack(out), os.path.join(MAN, "timing_docs.pt"))
print("ok")
