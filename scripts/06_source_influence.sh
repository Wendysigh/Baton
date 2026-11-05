#!/bin/bash
# Six-direction source-influence study. Each group holds two models on one GPU (up to about 120 GiB with 27B).
# Set NSHARDS and SHARD to split a job across GPUs; results from all shards are combined by the analysis.
set -eu
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
E=experiments/source_influence/evaluate.py; M=$BATON_MANIFESTS/source_influence; R=$BATON_RESULTS/source_influence
SH="--shard ${SHARD:-0} --nshards ${NSHARDS:-1}"
for g in 4B,9B 9B,27B 4B,27B; do
  $PY $E --group $g --task mmlu --manifests $M/mmlu_base.jsonl $M/mmlu_base_perm.jsonl --swap_prime $M/mmlu_dev_base.jsonl --out $R/confirm --decode eager --conv_fallback $SH
  $PY $E --group $g --task gsm8k --manifests $M/gsm8k.jsonl --conditions H-all H-tail H-ctx C-swap --swap_prime $M/gsm8k_dev.jsonl --out $R/confirm --decode eager --conv_fallback $SH
  $PY $E --group $g --task mmlu_chat --manifests $M/mmlu_chat.jsonl --diag 0 --conditions H-all H-tail C-noise C-swap --swap_prime $M/mmlu_dev_chat.jsonl --out $R/confirm --decode eager --conv_fallback $SH
  # post hoc wrong-question control and treatment repeat, with graph decoding and the causal-conv1d kernels
  $PY $E --group $g --task mmlu --manifests $M/mmlu_base.jsonl $M/mmlu_base_perm.jsonl --conditions H-all C-wrongq --diag 0 --out $R/wrongq $SH
  $PY $E --group $g --task gsm8k --manifests $M/gsm8k.jsonl --conditions H-all C-wrongq --out $R/wrongq $SH
  $PY $E --group $g --task mmlu_chat --manifests $M/mmlu_chat.jsonl --conditions H-all C-wrongq --diag 0 --out $R/wrongq $SH
done
# native lm-eval check
for s in 4B 9B 27B; do
  bash experiments/source_influence/run_lmeval.sh $s mmlu 5
  bash experiments/source_influence/run_lmeval.sh $s gsm8k_cot 8
done
bash experiments/source_influence/run_lmeval.sh 27B gsm8k_cot_nothink 8
