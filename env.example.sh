# Copy to env.sh, edit, and source it before running anything in scripts/.
export BATON_MODELS=/path/to/models          # contains Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B
export BATON_CACHE=/path/to/cache            # datasets, features and fitted maps; fits need several hundred GiB
export BATON_RESULTS=/path/to/results        # per-run outputs
export BATON_MANIFESTS=/path/to/manifests    # frozen item sets
# Optional overrides of the Hugging Face cache locations (see baton/paths.py):
# export BATON_FINEWEB_GLOB=".../sample/100BT/*.parquet"
# export BATON_MMLU_SNAPSHOTS=".../datasets--cais--mmlu/snapshots"
# export BATON_GSM8K_ARROW=".../gsm8k-{}.arrow"
# export BATON_FONTS=/path/to/dir/with/times-new-roman/ttf
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
