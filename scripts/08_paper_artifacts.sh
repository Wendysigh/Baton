#!/bin/bash
# results/summary/*.json -> results/tables/ and results/figures/. Needs only numpy and matplotlib; no GPU, models or datasets.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
$PY analysis/make_tables.py > /dev/null
$PY analysis/make_figures.py
