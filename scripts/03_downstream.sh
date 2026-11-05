#!/bin/bash
# WikiText-2, zero-shot tasks, CoQA, GSM8K and the post hoc 4B->9B copy comparison. All evaluations accept --shard k/n.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
R=$BATON_RESULTS/extra
for d in "4B>9B" "9B>4B" "9B>27B" "27B>9B"; do
  $PY experiments/wikitext.py --direction "$d" --detok --configs native none web copy_web --out "$R/ppl_detok"
  $PY experiments/wikitext.py --direction "$d"
  $PY experiments/zero_shot.py --direction "$d"
  $PY experiments/coqa.py --direction "$d"
  $PY experiments/gsm8k.py --direction "$d"
done
# copied-state comparison: refit variants, then ARC-Challenge on each
D="4B>9B"
$PY experiments/short_fit.py --direction "$D" --lams_lift --lams_copy --write copyN
$PY experiments/short_fit.py --direction "$D" --lams_lift --lams_copy --lams_diag 0.01 0.1 1 10 --write mld10
$PY experiments/short_fit.py --direction "$D" --lams_lift 3 --lams_copy 0.3 3 30 --write mlc3
$PY experiments/zero_shot.py --direction "$D" --tasks arc_challenge --configs copyN --out "$R/ll_v6"
$PY experiments/zero_shot.py --direction "$D" --tasks arc_challenge --configs mld10 --out "$R/ll_v5"
$PY experiments/zero_shot.py --direction "$D" --tasks arc_challenge --configs mlc3 --out "$R/ll_v2"
