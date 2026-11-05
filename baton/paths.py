"""Locations of models, datasets, fitted maps and outputs. Set them through the environment; see env.example.sh."""
import glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

MODEL_ROOT = os.environ.get("BATON_MODELS", os.path.join(ROOT, "models"))  # holds Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B
CACHE = os.environ.get("BATON_CACHE", os.path.join(ROOT, "cache"))  # datasets, features and fitted maps
RAW = os.environ.get("BATON_RESULTS", os.path.join(ROOT, "results", "raw"))  # per-run outputs
MAN = os.environ.get("BATON_MANIFESTS", os.path.join(ROOT, "manifests"))  # frozen item sets
SUMMARY = os.path.join(ROOT, "results", "summary")  # aggregated numbers behind the paper

FINEWEB_GLOB = os.environ.get("BATON_FINEWEB_GLOB", f"{HF_HOME}/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/100BT/*.parquet")
MMLU_SNAPSHOTS = os.environ.get("BATON_MMLU_SNAPSHOTS", f"{HF_HOME}/hub/datasets--cais--mmlu/snapshots")
# the datasets cache names this directory by a hash, which varies between downloads
_GSM8K = sorted(glob.glob(f"{HF_HOME}/datasets/openai___gsm8k/main/0.0.0/*/gsm8k-test.arrow"))
GSM8K_ARROW = os.environ.get("BATON_GSM8K_ARROW", _GSM8K[-1].replace("gsm8k-test", "gsm8k-{}") if _GSM8K else f"{HF_HOME}/datasets/openai___gsm8k/main/0.0.0/*/gsm8k-{{}}.arrow")


def model_path(size):
    return os.path.join(MODEL_ROOT, f"Qwen3.5-{size}")
