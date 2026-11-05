#!/bin/bash
# lm-eval (0.4.13, vLLM backend) native scores for one model, to validate the study's reimplemented formats.
# usage: run_lmeval.sh <model size: 4B|9B|27B> <tasks> <fewshot>
set -u
M=$1; TASKS=$2; K=$3
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-python}
INC=$(mktemp -d)
LM=$($PY -c "import lm_eval, os; print(os.path.join(os.path.dirname(lm_eval.__file__), 'tasks'))")
sed "s#__LM_EVAL_TASKS__#$LM#" "$HERE/gsm8k_cot_nothink.yaml" > "$INC/gsm8k_cot_nothink.yaml"
HF_ALLOW_CODE_EVAL=0 TOKENIZERS_PARALLELISM=false $PY -m lm_eval --model vllm \
  --model_args pretrained=$BATON_MODELS/Qwen3.5-$M,dtype=bfloat16,gpu_memory_utilization=${UTIL:-0.45},max_model_len=4096,enable_prefix_caching=True,seed=0 \
  --tasks $TASKS --include_path "$INC" --num_fewshot $K --batch_size auto --log_samples \
  --output_path "$BATON_RESULTS/source_influence/lmeval/$M" --seed 0
