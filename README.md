# Baton: Cross-Model Cache Reuse for Hybrid Language Models

Baton lets one hybrid language model continue from another model's cache, without prefilling the shared prefix again. It translates the complete cache (attention keys and values, GDN recurrent states and convolution buffers), including between models with different depths and head counts. The maps are fitted on unlabeled text with both models frozen and no gradient descent.

To rerun the experiments, see [REPRODUCE.md](REPRODUCE.md).

## Key findings

Across four directions among Qwen3.5-4B, 9B and 27B:

- **Context carries over.** Baton retains 70–96% of the native prefix benefit in next-token loss and recovers 73–89% of the F1 gained by reading a document.
- **Handoff pays off on long prefixes.** With source caches and maps already in memory, it takes 9–29% of target prefill time at 128K tokens on H200 GPUs. At 1K tokens it is slower than prefill (1.6–1.8 times native).
- **Direct copying depends on direction.** Copying recurrent states stays within 1.6 points of Baton on next-token loss for 4B→9B and can beat it on short-prompt tasks, but retains only 19.1% for 9B→4B on web text and is at or below no prefix between 9B and 27B.
- **Answers transfer, and so do mistakes.** On MMLU, a source that has read the question passes on its answer, including wrong ones. GSM8K shows no consistent reasoning gain.
- **Fidelity is not exact recall.** High next-token fidelity does not guarantee exact recall or stable continuation.

## Repository

| Path | Contents |
|---|---|
| `baton/` | The method: attention maps (`attention.py`), GDN lift (`gdn.py`), anchored refinement and calibration entry point (`refine.py`), cache assembly (`handoff.py`, `harness.py`, `common.py`), graph decoding (`fast_decode.py`) |
| `experiments/` | One script per evaluation, plus calibration variants (`plain_als.py`, `anchor_fit.py`, `calib_fit.py`, `short_fit.py`) |
| `experiments/data/` | Manifest builders and the RULER generator setup |
| `experiments/source_influence/` | The six-direction source-influence study and its analysis |
| `analysis/` | Aggregation, table and figure code |
| `scripts/` | The full pipeline, in order |
| `results/summary/` | Aggregated numbers behind the paper |

No fitted maps are included. Fitting needs both models on GPU, and a 9B/27B pair uses up to about 450 GiB of disk while fitting. See [REPRODUCE.md](REPRODUCE.md).
