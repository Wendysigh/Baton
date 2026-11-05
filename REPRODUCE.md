# Reproducing the paper

## Rebuild the tables and figures

`results/summary/` holds the aggregated numbers behind the paper. Rebuilding the tables and figures from them takes seconds and needs only numpy and matplotlib:

```bash
bash scripts/08_paper_artifacts.sh
```

Output goes to `results/tables/` and `results/figures/`. Figures use Times New Roman; if it isn't installed, set `BATON_FONTS` to a directory with the `.ttf` files.

## Where each result comes from

| Paper | Experiment | Summary file |
|---|---|---|
| Table 1, Table A1 (fidelity) | `experiments/fidelity.py`, `anchor_fit.py`, `plain_als.py`, `calib_fit.py` | `rev_aggregate.json` |
| Table A2 (GDN-cache ablations) | `baton/refine.py`, `experiments/plain_als.py` | `ablation.json` |
| Figure 2a (GSM8K) | `experiments/gsm8k.py`, `analysis/rescore_gsm8k.py` | `gsm8k_rescored.json` |
| Figure 2b (CoQA) | `experiments/coqa.py` | `tasks.json`, `coqa_paired.json` |
| Table 2 (WikiText-2, zero-shot) | `experiments/wikitext.py`, `zero_shot.py` | `tasks.json` |
| 4B→9B copy comparison | `experiments/short_fit.py`, `zero_shot.py`, `analysis/zero_shot_compare.py` | `copy_ablation_arc_4Bto9B.json` |
| Table 3, Table A3 (LongBench) | `experiments/longbench.py` | `rev_aggregate.json`, `rev_summary.json` |
| Table 4 (RULER) | `experiments/ruler.py` | `rev_aggregate.json`, `rev_summary.json` |
| Table 5 and post hoc checks (source influence) | `experiments/source_influence/` | `source_influence.json`, `lmeval_compare.json` |
| Figure 3 (latency) | `experiments/timing.py` | `timing_summary.json`, `timing_long_summary.json` |

## Full reproduction

### Requirements

The paper's runs used NVIDIA H200 GPUs. A 9B/27B pair with its maps and caches needs up to about 120 GiB on one GPU. Disk use during a fit peaks at about 220 GiB for 4B/9B and 450 GiB for 9B/27B.

```bash
pip install -r requirements.txt
cp env.example.sh env.sh   # edit the paths
source env.sh
```

Install the torch build that matches your CUDA driver. The model code comes from `transformers`; versions other than those in `requirements.txt` may change cache internals.

### Models and data

```bash
bash scripts/download_data.sh
```

This downloads the three post-trained checkpoints (about 80 GiB) to `$BATON_MODELS`, FineWeb-Edu, MMLU and GSM8K to the Hugging Face cache, and PG-19, RULER and LongBench to `$BATON_CACHE`. Models, FineWeb-Edu, MMLU and RULER are pinned to the revisions used in the paper. Use `--only <part>` to fetch one part; `experiments/data/pg19_books.txt` lists the 150 books used. WikiText-2, CoQA and the zero-shot tasks are fetched on first use.

### Run

Each step is a plain shell script, run serially. Evaluations accept `--shard k/n` (`NSHARDS` and `SHARD` for the source-influence study) if you want to split them across GPUs, and resume from saved rows. Schedule them however suits your cluster.

```bash
bash scripts/00_prepare_data.sh        # manifests
bash scripts/01_fit_maps.sh            # web and chat maps, component ablations
bash scripts/02_fidelity.sh            # fidelity, baselines, anchor ablation, refits
bash scripts/03_downstream.sh          # WikiText-2, zero-shot, CoQA, GSM8K
bash scripts/04_long_context.sh        # LongBench, RULER
bash scripts/05_timing.sh              # latency
bash scripts/06_source_influence.sh    # source influence
bash scripts/07_aggregate.sh           # raw outputs -> results/summary/
bash scripts/08_paper_artifacts.sh     # tables and figures
```

`07_aggregate.sh` overwrites `results/summary/`. The `native` lm-eval check in step 6 also needs vLLM, and `analysis/rescore_gsm8k.py` needs lm-eval.

### What to expect

- **Kernels.** GDN kernels come from FLA. The source-influence study's confirmatory runs use a PyTorch convolution fallback and eager decoding (`--conv_fallback --decode eager`); its post hoc repeat uses causal-conv1d and graph decoding.
- **Precision.** Models and attention caches are bf16, recurrent states are fp32, and maps and translation arithmetic are fp32.
- **Repeatability.** Exact bf16 logit ties make greedy decoding not fully repeatable, even natively, so generated outputs can differ between runs. Accuracies moved by at most 0.4 points on MMLU and 1.4 on GSM8K between repeats. Kernel and library versions also move results, so a rerun won't match `results/summary/` exactly.
- **Timing.** Latency depends on hardware. The 64K and 128K runs use one process per direction on separate GPU pairs; their 32K controls agreed with the sequential runs to within 2.2%.
- **Bootstrap intervals.** `03_downstream.sh` fits only the three copy-comparison variants reported in the paper, so the bootstrap draws in `copy_ablation_arc_4Bto9B.json` differ slightly when it is regenerated. Accuracies and differences are unaffected.
