#!/usr/bin/env python
"""One independent calibration fit of a pair on document-disjoint FineWeb documents (manifests/calib_fit_s*.pt), with the
recorded pipeline (baton/refine.py), minus its evaluation. The held-out states of the selection step are
all half-length documents, which no per-token map sees; intermediate features are deleted afterwards."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baton"))
from paths import CACHE, MAN  # noqa: E402
import argparse, glob, os, sys

import torch

from harness import log, MODELS, RES  # noqa: E402
import refine as phase3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)  # e.g. 4B,9B
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    small, large = args.pair.split(",")
    cache = f"{CACHE}/rev_fit/{small}_{large}_s{args.seed}"
    os.makedirs(cache, exist_ok=True)
    d = torch.load(f"{MAN}/calib_fit_s{args.seed}.pt")["calib"]
    ev = torch.load(f"{CACHE}/data_s1024_c512_e64.pt")["eval"]
    assert d.shape[0] == 1024
    torch.save({"calib": d[:512], "eval": ev}, f"{cache}/data_s1024_c512_e64.pt")
    torch.save({"calib": d, "eval": ev}, f"{cache}/data_s1024_c1024_e64.pt")
    torch.backends.cuda.matmul.allow_tf32 = False
    orig = torch.randperm

    def rp(n, *a, **kw):
        if n != 1024:
            return orig(n, *a, **kw)
        g = torch.Generator().manual_seed(args.seed)
        half = 512 + orig(512, generator=g)
        hold, rest = half[:128], torch.cat([orig(512, generator=g), half[128:]])
        return torch.cat([rest[orig(len(rest), generator=g)], hold])
    torch.randperm = rp
    phase3.eval_phase3 = lambda *a, **k: {}
    phase3.fmt = lambda r: ""
    os.makedirs(f"{RES}/calib_fit", exist_ok=True)
    sys.argv = ["phase3", "--small", MODELS[small], "--large", MODELS[large], "--small_dev", "cuda:0", "--large_dev", "cuda:0", "--fit_dev", "cuda:0",
                "--cache", cache, "--out", f"{RES}/calib_fit", "--tag", f"_{small}_{large}_s{args.seed}"]
    phase3.main()
    for pat in ("feat_*", "delta_*", "stream_*"):
        for f in glob.glob(f"{cache}/{pat}"):
            os.remove(f)
    log("done")


if __name__ == "__main__":
    main()
