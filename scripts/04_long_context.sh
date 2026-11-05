#!/bin/bash
# LongBench in all directions and RULER between 9B and 27B. Both accept --shard k/n; rows resume after interruption.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
for d in "4B>9B" "9B>4B" "9B>27B" "27B>9B"; do $PY experiments/longbench.py --direction "$d"; done
for d in "9B>27B" "27B>9B"; do $PY experiments/ruler.py --direction "$d"; done
