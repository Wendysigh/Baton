#!/bin/bash
# Fit the web- and chat-calibrated maps (GPUs 0 and 1 hold the two models). The run also writes the component ablations.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
M=$BATON_MODELS; C=$BATON_CACHE
fit() { # small large cache tag
  $PY baton/refine.py --small $M/Qwen3.5-$1 --large $M/Qwen3.5-$2 --cache "$3" --tag "$4"
}
fit 9B 27B "$C" _9b27b
fit 4B 9B "$C/p4b9b" _4b9b
# chat-mixed calibration: same documents for every pair
$PY experiments/data/calib_chat.py --src_cache "$C" --dst_cache "$C/chatcal"
for d in p4b9b_chat p4b27b_chat; do mkdir -p "$C/$d"; cp "$C"/chatcal/data_s1024_c*_e64.pt "$C/$d/"; done
fit 9B 27B "$C/chatcal" _9b27b_chatcal
fit 4B 9B "$C/p4b9b_chat" _4b9b_chatcal
fit 4B 27B "$C/p4b27b_chat" _4b27b_chatcal
