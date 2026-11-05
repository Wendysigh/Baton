#!/usr/bin/env python
"""RULER 500 per task and length, 2x2 calibration x cut plus one native reference, on frozen prompts.
Panel rows (first 100 of each panel task and length in manifest order) add a no-context reference and a copied state with
the web attention maps. Web and chat maps read the same source pass, so All and Ctx each cost one source prefill."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
import argparse, json, os, sys, time

import torch

from harness import setup, log, decode, source_pass, assemble, read_manifest, shard_rows, open_resume, sha, chat_wrap_user, RES, forward_text  # noqa: E402

GEN = {"vt": 30, "fwe": 50, "cwe": 120, "niah": 128}
STOPS = ("\n\n", "Question:", "Below is", "Memorize", "Read the following")
PANEL = ("vt", "vt_hard", "niah_single_1", "niah_multikey_3")
MARK = {"vt": "\nQuestion:", "fwe": "\nQuestion:", "cwe": "\nQuestion:", "niah": "\nWhat "}
N_PANEL = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--manifest", default="ruler_test.jsonl")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RES, "ruler"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok, eos, src_m, tgt_m, hos = setup(args.direction, ["web", "chat"])
    dev = next(tgt_m.parameters()).device
    all_rows = read_manifest(args.manifest)
    panel, cnt = set(), {}
    for r in all_rows:
        k = (r["task"], r["length"])
        cnt[k] = cnt.get(k, 0) + 1
        if r["task"] in PANEL and cnt[k] <= N_PANEL:
            panel.add(r["id"])
    rows = shard_rows(all_rows, args.shard)
    k, n = map(int, args.shard.split("/"))
    path = os.path.join(args.out, f"{args.manifest[:-6]}_{args.direction.replace('>', 'to')}_sh{k}of{n}.jsonl")
    done, f = open_resume(path, {r["id"] for r in rows})
    t0, c = time.time(), 0
    for r in rows:
        if r["id"] in done:
            continue
        src = None
        with open(r["file"]) as fh:
            for i, l in enumerate(fh):
                if i == r["row"]:
                    src = json.loads(l); break
        p = chat_wrap_user(tok, src["input"]) + src["answer_prefix"].lstrip()
        ids = tok(p, add_special_tokens=False)["input_ids"]
        assert sha(ids) == r["token_sha1"], r["id"]
        ids = torch.tensor([ids], device=dev)
        E, kind = r["E"], r["task"].split("_")[0]
        m = 60 if r["task"] == "vt_hard" else GEN[kind]
        rec = {k_: r[k_] for k_ in ("id", "task", "length", "seed", "row", "n_tokens", "E", "token_sha1", "refs")}
        rec["configs"] = {}
        go = lambda cache, rest: decode(tgt_m, tok, cache, rest, m, eos, STOPS, graph=bool(args.graph))
        rec["configs"]["native"] = go(None, ids)
        scache, store = source_pass(hos, src_m, ids[:, :-1])
        for cal in ("web", "chat"):
            rec["configs"][f"all_{cal}"] = go(assemble(hos[cal], scache, store, ids.shape[1] - 1, ids[:, :1]), ids[:, -1:])
        if r["id"] in panel:
            rec["configs"]["all_copy_web"] = go(assemble(hos["web"], scache, store, ids.shape[1] - 1, ids[:, :1], gdn="copy"), ids[:, -1:])
        del scache, store
        scache, store = source_pass(hos, src_m, ids[:, :E])
        for cal in ("web", "chat"):
            rec["configs"][f"ctx_{cal}"] = go(assemble(hos[cal], scache, store, E, ids[:, :1]), ids[:, E:])
        del scache, store
        if r["id"] in panel:
            inp = src["input"]
            q = inp[inp.rfind(MARK[kind]):].lstrip()
            qi = torch.tensor([tok(chat_wrap_user(tok, q) + src["answer_prefix"].lstrip(), add_special_tokens=False)["input_ids"]], device=dev)
            rec["configs"]["no_context"] = go(None, qi)
        hos["web"].tap.clear()
        f.write(json.dumps(rec) + "\n"); f.flush()
        c += 1
        if c % 5 == 0:
            log(f"{args.direction} sh{args.shard} {c}/{len(rows) - len(done)} {(time.time() - t0) / c:.1f}s/prompt, peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB")
        if args.limit and c >= args.limit:
            break
    log("done")


if __name__ == "__main__":
    main()
