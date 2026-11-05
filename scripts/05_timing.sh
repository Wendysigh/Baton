#!/bin/bash
# Latency. 1K-32K run one process at a time on otherwise idle GPUs; 64K/128K run one direction per GPU pair.
set -u
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
A=${GPU_A:-0}; B=${GPU_B:-1}
for D in "4B>9B" "9B>4B" "9B>27B" "27B>9B"; do
  CUDA_VISIBLE_DEVICES=$A $PY experiments/timing.py --direction "$D" --placement same
  CUDA_VISIBLE_DEVICES=$A,$B $PY experiments/timing.py --direction "$D" --placement two --cold_reps 1
done
COMMON="--cold_reps 0 --docs $BATON_MANIFESTS/timing_docs_long.pt --out $BATON_RESULTS/timing_long"
stream() { # direction gpu_a gpu_b
  CUDA_VISIBLE_DEVICES=$2 $PY experiments/timing.py --direction "$1" --placement same $COMMON --lengths 32768 65536 131072
  CUDA_VISIBLE_DEVICES=$2,$3 $PY experiments/timing.py --direction "$1" --placement two $COMMON --lengths 65536 131072
}
stream "4B>9B" 0 1 & stream "9B>4B" 2 3 & stream "9B>27B" 4 5 & stream "27B>9B" 6 7 & wait
