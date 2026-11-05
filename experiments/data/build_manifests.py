#!/usr/bin/env python
"""Freeze the item sets of the revision before any inference on them.
  ruler_{dev,test}.jsonl       one row per prompt: task, length, seed, row, full-token hash, context boundary
  longbench_{dev,test}.jsonl   qasper, hotpotqa, multifieldqa_en: official prompts, verified document/question boundary
  fidelity_docs.pt             128 web + 128 book token windows (4,096 prefix + 129 continuation) (and 8 of each for dev)
  calib_fit_s{1,2,3}.pt        document-disjoint FineWeb calibration sets for the fit-variability study
Boundaries come from one tokenization of the whole prompt; a row where tokenizing the two parts separately gives other
tokens is an error for RULER and is counted in the report (see manifests/report.json)."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
from paths import CACHE, MAN, FINEWEB_GLOB, model_path  # noqa: E402
import glob, hashlib, json, os, random, sys

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from harness import sha, chat_wrap_user, tokenize_split  # noqa: E402

tok = AutoTokenizer.from_pretrained(model_path("9B"))
os.makedirs(MAN, exist_ok=True)
what = sys.argv[1:] or ["ruler", "longbench", "fidelity", "calib"]
report = json.load(open(f"{MAN}/report.json")) if os.path.exists(f"{MAN}/report.json") else {}
MARK = {"vt": "\nQuestion:", "fwe": "\nQuestion:", "cwe": "\nQuestion:", "niah": "\nWhat "}


def ruler(seeds, name):
    D = f"{CACHE}/rev_ruler"
    rows, sep_diff = [], 0
    for s, n_want in seeds:
        for L in (4096, 16384):
            for T in ("vt", "vt_hard", "fwe", "cwe", "niah_single_1", "niah_multikey_2", "niah_multikey_3"):
                fs = f"{D}/s{s}/len{L}/{T}/validation.jsonl"
                sd = s
                if not os.path.exists(fs):  # the upstream generator crashed on this seed: the task moves to the next seed
                    sd = s + 1
                    fs = f"{D}/s{sd}/len{L}/{T}/validation.jsonl"
                    report.setdefault("ruler_seed_moves", []).append([s, sd, L, T])
                src = [json.loads(l) for l in open(fs)]
                assert len(src) >= n_want, (s, L, T, len(src))
                for i, r in enumerate(src[:n_want]):
                    p = chat_wrap_user(tok, r["input"]) + r["answer_prefix"].lstrip()
                    cut = p.rfind(MARK[T.split("_")[0]])
                    ids, E = tokenize_split(tok, p, cut)
                    a, b = tok(p[:cut], add_special_tokens=False)["input_ids"], tok(p[cut:], add_special_tokens=False)["input_ids"]
                    sep_diff += int(a + b != ids)
                    rows.append({"id": f"{T}|{L}|{s}|{i}|{sha(ids)[:16]}", "task": T, "length": L, "seed": sd, "row": i, "file": fs,
                                 "n_tokens": len(ids), "E": E, "token_sha1": sha(ids), "refs": r["outputs"]})
    assert len({r["token_sha1"] for r in rows}) == len(rows), "duplicate prompts"
    with open(f"{MAN}/{name}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report[name] = {"rows": len(rows), "separate_tokenization_differs": sep_diff,
                    "median_tokens": sorted(r["n_tokens"] for r in rows)[len(rows) // 2], "max_tokens": max(r["n_tokens"] for r in rows)}


def longbench():
    D = f"{CACHE}/longbench"
    P = json.load(open(f"{D}/dataset2prompt.json")); ML = json.load(open(f"{D}/dataset2maxlen.json"))
    CAP = 31000  # frozen context cap in tokens: fits the 32K mapped range; longer contexts keep their first CAP tokens
    rows, dev = [], []
    for ds in ("qasper", "hotpotqa", "multifieldqa_en"):
        trunc = 0
        src = [json.loads(l) for l in open(f"{D}/data/data/{ds}.jsonl")]
        for i, r in enumerate(src):
            ctx = r["context"]
            c = tok(ctx, add_special_tokens=False)["input_ids"]
            if len(c) > CAP:
                ctx = tok.decode(c[:CAP]); trunc += 1
            head, tail = P[ds].split("{context}")
            user = head + ctx + tail.format(input=r["input"])
            wrapped = chat_wrap_user(tok, user)
            cut = wrapped.index(user) + len(head) + len(ctx)
            ids, E = tokenize_split(tok, wrapped, cut, snap=True)
            rows.append({"id": f"{ds}|{i}|{sha(ids)[:16]}", "dataset": ds, "row": i, "n_tokens": len(ids), "E": E, "token_sha1": sha(ids), "max_new": ML[ds],
                         "doc": hashlib.sha1(r["context"].encode()).hexdigest()[:16], "answers": r["answers"], "all_classes": r.get("all_classes"), "input": r["input"], "truncated": len(c) > CAP})
        s = [x for x in rows if x["dataset"] == ds]
        report[f"longbench_{ds}"] = {"questions": len(s), "documents": len({x["doc"] for x in s}), "median_tokens": sorted(x["n_tokens"] for x in s)[len(s) // 2],
                                     "max_tokens": max(x["n_tokens"] for x in s), "truncated": trunc}
    with open(f"{MAN}/longbench_test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


FW = sorted(glob.glob(FINEWEB_GLOB))
USED = {FW[0], FW[-1]}  # the recorded calibration and evaluation files


def web_docs(file, n, length, seen, offset=0):
    out, hs = [], []
    pf = pq.ParquetFile(file)
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if len(text) < 3 * length:
                continue
            h = hashlib.sha1(text.encode()).hexdigest()
            if h in seen:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= length + offset:
                seen.add(h); out.append(torch.tensor(ids[offset: offset + length], dtype=torch.long)); hs.append(h)
                if len(out) >= n:
                    return torch.stack(out), hs
    raise RuntimeError(f"{file}: only {len(out)}")


def fidelity():
    LEN = 4096 + 129
    seen = set()
    web, webh = [], []
    for f in FW[1:-1]:
        need = 136 - len(webh)
        if need <= 0:
            break
        try:
            a, h = web_docs(f, need, LEN, seen)
            web.append(a); webh += h
        except RuntimeError:
            pass
    assert len(webh) >= 136, len(webh)
    web = torch.cat(web)
    names = sorted(glob.glob(f"{CACHE}/pg19/books/*.txt"))
    assert len(names) >= 136, len(names)
    random.Random(20261103).shuffle(names)
    books, booknames = [], []
    for p in names:
        ids = tok(open(p, errors="ignore").read(), add_special_tokens=False)["input_ids"]
        if len(ids) >= 2048 + LEN:  # skip the first 2,048 tokens (title pages, contents)
            books.append(torch.tensor(ids[2048: 2048 + LEN], dtype=torch.long)); booknames.append(os.path.basename(p))
        if len(books) == 136:
            break
    assert len(books) == 136
    torch.save({"web_dev": web[:8], "web": web[8:136], "web_sha1": webh, "book_dev": torch.stack(books[:8]), "book": torch.stack(books[8:136]), "book_names": booknames}, f"{MAN}/fidelity_docs.pt")
    report["fidelity"] = {"web": 128, "book": 128, "dev_each": 8, "web_files": "fineweb-edu sample/100BT, excluding the recorded calibration and evaluation files", "book_source": "PG19 validation+test"}


def calib():
    """Each fit: 1,024 documents; the recorded first-file/last-file sets and the fidelity documents are excluded by hash."""
    seen = set(torch.load(f"{MAN}/fidelity_docs.pt")["web_sha1"])
    files = FW[-40:-1]
    for s, f in zip((1, 2, 3), (files[0], files[12], files[24])):
        a, _ = web_docs(f, 1024, 1024, seen)
        torch.save({"calib": a, "file": os.path.basename(f)}, f"{MAN}/calib_fit_s{s}.pt")
    report["calib_fit"] = {"docs_per_fit": 1024, "disjoint": True}


if "ruler" in what:
    ruler([(7777, 8)], "ruler_dev.jsonl")
    ruler([(20261103, 167), (20261104, 167), (20261105, 166)], "ruler_test.jsonl")
if "longbench" in what:
    longbench()
if "fidelity" in what:
    fidelity()
if "calib" in what:
    calib()
json.dump(report, open(f"{MAN}/report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
