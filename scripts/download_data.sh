#!/bin/bash
# Download the models and datasets used in the paper into $BATON_MODELS, $BATON_CACHE and the Hugging Face cache.
# Finished parts are skipped on a rerun. The models take about 80 GiB.
# usage: download_data.sh [--only models|fineweb|mmlu|gsm8k|pg19|ruler|longbench]
# WikiText-2, CoQA and the zero-shot tasks are fetched on first use.
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
: "${BATON_CACHE:?source env.sh first}" "${BATON_MODELS:?source env.sh first}"
ONLY=${2:-all}
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

FINEWEB_REV=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
MMLU_REV=c30699e8356da336a370243923dbaf21066bb9fe
RULER_COMMIT=c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a

hub() { # repo revision pattern
  $PY -c 'import sys; from huggingface_hub import snapshot_download as d; print(d(sys.argv[1], repo_type="dataset", revision=sys.argv[2], allow_patterns=sys.argv[3]))' "$@"
}

if want models; then
  while read -r size rev; do
    $PY -c 'import sys; from huggingface_hub import snapshot_download as d; print(d(f"Qwen/Qwen3.5-{sys.argv[1]}", revision=sys.argv[2], local_dir=f"{sys.argv[3]}/Qwen3.5-{sys.argv[1]}"))' "$size" "$rev" "$BATON_MODELS" < /dev/null
  done <<REV
4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
27B fc05daec18b0a78c049392ed2e771dde82bdf654
REV
fi

if want fineweb; then
  hub HuggingFaceFW/fineweb-edu $FINEWEB_REV "sample/100BT/*"
fi

if want mmlu; then
  hub cais/mmlu $MMLU_REV "all/*"
  mkdir -p "$BATON_CACHE/data"
  [ -f "$BATON_CACHE/data/mmlu_test.parquet" ] || $PY - <<PY
from datasets import load_dataset
load_dataset("cais/mmlu", "all", split="test", revision="$MMLU_REV").to_parquet("$BATON_CACHE/data/mmlu_test.parquet")
PY
fi

if want gsm8k; then
  mkdir -p "$BATON_CACHE/data"
  $PY - <<PY
import os
from datasets import load_dataset
d = load_dataset("openai/gsm8k", "main")
print(d["test"].cache_files[0]["filename"])
p = "$BATON_CACHE/data/gsm8k_test.parquet"
if not os.path.exists(p):
    d["test"].to_parquet(p)
PY
fi

if want pg19; then
  mkdir -p "$BATON_CACHE/pg19/books"
  while read -r f; do
    [ -s "$BATON_CACHE/pg19/books/$f" ] && continue
    for split in test validation; do
      curl -fsS -o "$BATON_CACHE/pg19/books/$f" "https://storage.googleapis.com/deepmind-gutenberg/$split/$f" 2>/dev/null && break
    done
    [ -s "$BATON_CACHE/pg19/books/$f" ] || { echo "missing book $f" >&2; exit 1; }
  done < experiments/data/pg19_books.txt
fi

if want ruler; then
  R=$BATON_CACHE/ruler_repo/RULER
  if [ ! -d "$R" ]; then
    git clone https://github.com/NVIDIA/RULER "$R"
    git -C "$R" checkout -q $RULER_COMMIT
    git -C "$R" apply "$PWD/experiments/data/ruler_vt_hard.diff"
  fi
fi

if want longbench; then
  L=$BATON_CACHE/longbench
  mkdir -p "$L/data"
  B=https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config
  for f in dataset2prompt.json dataset2maxlen.json; do
    [ -f "$L/$f" ] || curl -fsSL -o "$L/$f" "$B/$f"
  done
  [ -f "$L/data.zip" ] || curl -fsSL -o "$L/data.zip" https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
  [ -d "$L/data/data" ] || $PY -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$L/data.zip" "$L/data"
fi
echo done
