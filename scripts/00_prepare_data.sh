#!/bin/bash
# Freeze the item sets. Requires the datasets described in the README under $BATON_CACHE and the Hugging Face cache.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
mkdir -p "$BATON_MANIFESTS" "$BATON_RESULTS"
bash experiments/data/gen_ruler.sh
$PY experiments/data/build_manifests.py ruler longbench fidelity calib
$PY experiments/data/build_timing_docs.py
$PY experiments/data/build_timing_docs_long.py
$PY experiments/source_influence/prepare_data.py
