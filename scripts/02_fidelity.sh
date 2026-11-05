#!/bin/bash
# Next-token fidelity, its baselines, the controlled anchor ablation and the independent refits.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
M=$BATON_MODELS; C=$BATON_CACHE
$PY experiments/plain_als.py --small $M/Qwen3.5-9B --large $M/Qwen3.5-27B --cache "$C" --tag _9b27b
$PY experiments/plain_als.py --small $M/Qwen3.5-4B --large $M/Qwen3.5-9B --cache "$C/p4b9b" --tag _4b9b
for d in "4B>9B" "9B>4B" "9B>27B" "27B>9B"; do
  $PY experiments/anchor_fit.py --direction "$d"
  $PY experiments/fidelity.py --direction "$d"
done
for s in 1 2 3; do
  $PY experiments/calib_fit.py --pair 4B,9B --seed $s
  $PY experiments/fidelity.py --direction "4B>9B" --fit $s
  $PY experiments/calib_fit.py --pair 9B,27B --seed $s
  $PY experiments/fidelity.py --direction "9B>27B" --fit $s
  $PY experiments/fidelity.py --direction "27B>9B" --fit $s
done
