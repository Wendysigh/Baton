#!/bin/bash
# Raw outputs -> results/summary/*.json. Then scripts/08_paper_artifacts.sh rebuilds the tables and figures.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
mkdir -p results/summary
$PY analysis/aggregate_ablation.py
$PY analysis/aggregate_tasks.py
$PY analysis/aggregate_rev.py
$PY analysis/summarize_rev.py > /dev/null
$PY analysis/aggregate_timing.py timing > /dev/null
$PY analysis/aggregate_timing.py timing_long > /dev/null
$PY analysis/rescore_gsm8k.py
$PY analysis/coqa_paired.py
$PY analysis/zero_shot_compare.py "4B>9B" arc_challenge --dirs ll ll_v2 ll_v5 ll_v6 --json results/summary/copy_ablation_arc_4Bto9B.json
$PY experiments/source_influence/build_results.py > /dev/null
$PY experiments/source_influence/lmeval_compare.py
$PY experiments/data/check_data_overlap.py
