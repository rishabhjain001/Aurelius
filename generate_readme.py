"""
generate_readme.py — README generator for Aurelius CLIP-VLM

Produces README.md at the project root from pre-computed experiment result
JSON files. Does NOT load PyTorch models or fabricate any values.

Public interface:
    check_preconditions() -> tuple[bool, list[str]]
    load_all_results() -> dict
    generate_readme(results: dict) -> str
    run() -> None

Requirements: 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 26.7, 26.8, 28.8
"""

import json
from datetime import datetime, timezone
from pathlib import Path


_RESULTS_DIR = Path("results")
_EXPERIMENT_FILES = [_RESULTS_DIR / f"experiment{n}.json" for n in range(1, 7)]
_BENCHMARK_FILE = _RESULTS_DIR / "benchmark.json"
_README_PATH = Path("README.md")


def check_preconditions() -> tuple[bool, list[str]]:
    """Return (all_ok, list_of_missing_or_incomplete_files).

    Mirrors visualize.check_preconditions exactly.
    """
    issues: list[str] = []

    if not _BENCHMARK_FILE.exists():
        issues.append(str(_BENCHMARK_FILE))

    for path in _EXPERIMENT_FILES:
        if not path.exists():
            issues.append(str(path))
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            issues.append(f"{path} (cannot parse JSON)")
            continue

        status = data.get("status", "")
        failure_reason = data.get("failure_reason")

        if status in ("failed", "partial"):
            if not failure_reason:
                issues.append(
                    f"{path} (status='{status}' but failure_reason is null/missing)"
                )
        else:
            bookkeeping = {"experiment", "status", "failure_reason"}
            real_keys = {
                k for k, v in data.items()
                if k not in bookkeeping and v is not None
            }
            if not real_keys:
                issues.append(
                    f"{path} (no real measured data and no valid failed/partial status)"
                )

    return len(issues) == 0, issues


def load_all_results() -> dict:
    """Load all result and metadata files.

    Returns:
        {
            1: {...}, 2: {...}, ..., 6: {...},
            "benchmark": {...},
            "metadata": {1: {...}, 2: {...}, ..., 6: {...}},
        }
    """
    results: dict = {}

    for n in range(1, 7):
        path = _RESULTS_DIR / f"experiment{n}.json"
        try:
            results[n] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            results[n] = {"experiment": n, "status": "failed", "failure_reason": str(exc)}

    try:
        results["benchmark"] = json.loads(_BENCHMARK_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        results["benchmark"] = {"error": str(exc)}

    metadata: dict = {}
    for n in range(1, 7):
        meta_path = _RESULTS_DIR / f"experiment{n}_metadata.json"
        if meta_path.exists():
            try:
                metadata[n] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata[n] = "unavailable"
        else:
            metadata[n] = "unavailable"
    results["metadata"] = metadata

    return results


def _fmt(value, precision: int = 4) -> str:
    """Format a numeric value to fixed precision, or return 'N/A'."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _retrieval_table(retrieval_dict: dict | None, label: str) -> str:
    """Render a retrieval metrics block as a markdown table."""
    if not retrieval_dict:
        return f"*{label}: retrieval data unavailable.*\n"

    skipped = retrieval_dict.get("skipped", False)
    if skipped:
        reason = retrieval_dict.get("skip_reason", "unknown reason")
        return f"*{label}: retrieval skipped — {reason}.*\n"

    i2t = retrieval_dict.get("i2t") or {}
    t2i = retrieval_dict.get("t2i") or {}

    lines = [
        f"**{label}**\n",
        "| Direction | R@1 | R@5 | R@10 | MRR | Mean Rank | Median Rank |",
        "|---|---|---|---|---|---|---|",
        (
            f"| Image → Text "
            f"| {_fmt(i2t.get('r1'))} "
            f"| {_fmt(i2t.get('r5'))} "
            f"| {_fmt(i2t.get('r10'))} "
            f"| {_fmt(i2t.get('mrr'))} "
            f"| {_fmt(i2t.get('mean_rank'))} "
            f"| {_fmt(i2t.get('median_rank'))} |"
        ),
        (
            f"| Text → Image "
            f"| {_fmt(t2i.get('r1'))} "
            f"| {_fmt(t2i.get('r5'))} "
            f"| {_fmt(t2i.get('r10'))} "
            f"| {_fmt(t2i.get('mrr'))} "
            f"| {_fmt(t2i.get('mean_rank'))} "
            f"| {_fmt(t2i.get('median_rank'))} |"
        ),
    ]
    return "\n".join(lines) + "\n"


def _meta_summary(meta) -> str:
    """Extract seed and git commit from metadata dict, or note unavailability."""
    if meta == "unavailable" or not isinstance(meta, dict):
        return "- **Seed**: unavailable\n- **Git commit**: unavailable\n"

    seeds = meta.get("seeds", {})
    if isinstance(seeds, dict):
        seed_val = seeds.get("torch", "unavailable")
    else:
        seed_val = "unavailable"

    git_commit = meta.get("git_commit", "unavailable") or "unavailable"

    return f"- **Seed** (torch/numpy/random): `{seed_val}`\n- **Git commit**: `{git_commit}`\n"


def _exp1_section(data: dict, meta) -> str:
    status = data.get("status", "failed")
    failure_reason = data.get("failure_reason")
    lines = ["### Experiment 1 — Training Dynamics\n"]
    lines.append(f"**Status**: `{status}`\n")
    lines.append(_meta_summary(meta))

    if status == "failed":
        lines.append(f"**Failure reason**: {failure_reason or 'unknown'}\n")
    else:
        steps = data.get("steps") or []
        if steps:
            last = steps[-1]
            first = steps[0]
            lines.append(f"- **Total steps recorded**: {len(steps)}")
            lines.append(f"- **Initial loss** (step {first.get('step', 0)}): {_fmt(first.get('loss'))}")
            lines.append(f"- **Final loss** (step {last.get('step', '?')}): {_fmt(last.get('loss'))}")
            lines.append(f"- **Final temperature**: {_fmt(last.get('temperature'))}")
            lines.append(f"- **Final i2t acc@1**: {_fmt(last.get('i2t_acc1'))}")
            lines.append(f"- **Final t2i acc@1**: {_fmt(last.get('t2i_acc1'))}")
        else:
            lines.append("*No per-step data recorded.*")

        if failure_reason:
            lines.append(f"\n**Partial failure reason**: {failure_reason}\n")

    lines.append("\n#### Lessons Learned\n")
    lines.append(
        "Tracking per-step InfoNCE loss and temperature revealed whether the "
        "contrastive objective converged smoothly. Large initial loss values "
        "are expected; a steady decline across steps indicates that the "
        "shared embedding space is being shaped by the image-caption pairs. "
        "Monitoring temperature separately confirmed the log-parameterized "
        "scalar remained positive throughout training.\n"
    )
    return "\n".join(lines) + "\n"


def _exp2_section(data: dict, meta) -> str:
    status = data.get("status", "failed")
    failure_reason = data.get("failure_reason")
    lines = ["### Experiment 2 — Retrieval Evaluation vs. Random Baseline\n"]
    lines.append(f"**Status**: `{status}`\n")
    lines.append(_meta_summary(meta))

    if status == "failed":
        lines.append(f"**Failure reason**: {failure_reason or 'unknown'}\n")
    else:
        retrieval = data.get("retrieval") or {}
        trained = retrieval.get("trained")
        baseline = retrieval.get("random_baseline")
        lines.append(_retrieval_table(trained, "Trained Model"))
        lines.append(_retrieval_table(baseline, "Random Baseline"))
        if failure_reason:
            lines.append(f"\n**Partial failure reason**: {failure_reason}\n")

    lines.append("\n#### Lessons Learned\n")
    lines.append(
        "Comparing the trained model against a randomly initialized encoder "
        "on identical evaluation code isolates the contribution of contrastive "
        "training. A random encoder's retrieval metrics serve as a lower bound; "
        "any gap above chance demonstrates that InfoNCE loss successfully "
        "aligned cross-modal embeddings in the shared 256-dimensional space.\n"
    )
    return "\n".join(lines) + "\n"


def _exp3_section(data: dict, meta) -> str:
    status = data.get("status", "failed")
    failure_reason = data.get("failure_reason")
    lines = ["### Experiment 3 — Embedding Dimension Ablation\n"]
    lines.append(f"**Status**: `{status}`\n")
    lines.append(_meta_summary(meta))

    if status == "failed":
        lines.append(f"**Failure reason**: {failure_reason or 'unknown'}\n")
    else:
        runs = data.get("runs") or []
        for run in runs:
            dim = run.get("embed_dim", "?")
            run_status = run.get("status", "failed")
            run_fail = run.get("failure_reason")
            lines.append(f"#### embed_dim = {dim} | status: `{run_status}`\n")
            if run_status == "failed":
                lines.append(f"Failure reason: {run_fail or 'unknown'}\n")
            else:
                lines.append(_retrieval_table(run.get("retrieval"), f"embed_dim={dim}"))
                convergence = run.get("convergence") or []
                if convergence:
                    last_c = convergence[-1]
                    lines.append(
                        f"- Convergence (final step {last_c.get('step', '?')}): "
                        f"loss={_fmt(last_c.get('loss'))}, "
                        f"i2t_acc1={_fmt(last_c.get('i2t_acc1'))}, "
                        f"t2i_acc1={_fmt(last_c.get('t2i_acc1'))}\n"
                    )
                if run_fail:
                    lines.append(f"Partial failure reason: {run_fail}\n")

        if failure_reason and status != "failed":
            lines.append(f"\n**Overall partial failure reason**: {failure_reason}\n")

    lines.append("\n#### Lessons Learned\n")
    lines.append(
        "Varying the embedding dimension across 128, 256, and 512 reveals "
        "the capacity-vs-efficiency trade-off in the shared representation "
        "space. Smaller dimensions compress the representation aggressively; "
        "larger dimensions give the model more expressive freedom but require "
        "more parameters in the projection heads and may overfit with limited data.\n"
    )
    return "\n".join(lines) + "\n"


def _exp4_section(data: dict, meta) -> str:
    status = data.get("status", "failed")
    failure_reason = data.get("failure_reason")
    lines = ["### Experiment 4 — Projection Head Ablation\n"]
    lines.append(f"**Status**: `{status}`\n")
    lines.append(_meta_summary(meta))

    if status == "failed":
        lines.append(f"**Failure reason**: {failure_reason or 'unknown'}\n")
    else:
        runs = data.get("runs") or []
        for run in runs:
            head = run.get("projection_head", "?")
            run_status = run.get("status", "failed")
            run_fail = run.get("failure_reason")
            lines.append(f"#### projection_head = {head} | status: `{run_status}`\n")
            if run_status == "failed":
                lines.append(f"Failure reason: {run_fail or 'unknown'}\n")
            else:
                lines.append(_retrieval_table(run.get("retrieval"), f"projection_head={head}"))
                sep = run.get("embedding_separation")
                if sep and isinstance(sep, dict):
                    lines.append("**Embedding separation** (cosine similarity on validation split):\n")
                    lines.append(f"- Matched pairs: mean={_fmt(sep.get('matched_mean'))}, std={_fmt(sep.get('matched_std'))}")
                    lines.append(f"- Unmatched pairs: mean={_fmt(sep.get('unmatched_mean'))}, std={_fmt(sep.get('unmatched_std'))}\n")
                if run_fail:
                    lines.append(f"Partial failure reason: {run_fail}\n")

        if failure_reason and status != "failed":
            lines.append(f"\n**Overall partial failure reason**: {failure_reason}\n")

    lines.append("\n#### Lessons Learned\n")
    lines.append(
        "The projection head is the sole mapping between the encoder's raw "
        "feature space and the shared embedding space. A linear head is the "
        "simplest possible option; adding a single hidden layer with ReLU "
        "(MLP head) can increase capacity. Embedding-separation metrics "
        "quantify whether matched pairs cluster closer together than "
        "unmatched ones, providing a complementary signal beyond retrieval rank.\n"
    )
    return "\n".join(lines) + "\n"


def _exp5_section(data: dict, meta) -> str:
    status = data.get("status", "failed")
    failure_reason = data.get("failure_reason")
    lines = ["### Experiment 5 — Temperature Ablation\n"]
    lines.append(f"**Status**: `{status}`\n")
    lines.append(_meta_summary(meta))

    if status == "failed":
        lines.append(f"**Failure reason**: {failure_reason or 'unknown'}\n")
    else:
        runs = data.get("runs") or []
        for run in runs:
            temp = run.get("temperature_init", "?")
            run_status = run.get("status", "failed")
            run_fail = run.get("failure_reason")
            lines.append(f"#### temperature_init = {temp} | status: `{run_status}`\n")
            if run_status == "failed":
                lines.append(f"Failure reason: {run_fail or 'unknown'}\n")
            else:
                lines.append(_retrieval_table(run.get("retrieval"), f"temperature_init={temp}"))
                convergence = run.get("convergence") or []
                if convergence:
                    last_c = convergence[-1]
                    lines.append(
                        f"- Convergence (final step {last_c.get('step', '?')}): "
                        f"loss={_fmt(last_c.get('loss'))}, "
                        f"i2t_acc1={_fmt(last_c.get('i2t_acc1'))}, "
                        f"t2i_acc1={_fmt(last_c.get('t2i_acc1'))}\n"
                    )
                if run_fail:
                    lines.append(f"Partial failure reason: {run_fail}\n")

        if failure_reason and status != "failed":
            lines.append(f"\n**Overall partial failure reason**: {failure_reason}\n")

    lines.append("\n#### Lessons Learned\n")
    lines.append(
        "Temperature controls the sharpness of the softmax distribution "
        "in InfoNCE loss. A smaller temperature creates a peakier distribution "
        "that emphasizes the correct pair more strongly but can make "
        "optimization brittle early in training. A larger temperature smooths "
        "gradients and can improve stability at the cost of weaker discrimination.\n"
    )
    return "\n".join(lines) + "\n"


def _exp6_section(data: dict, meta) -> str:
    status = data.get("status", "failed")
    failure_reason = data.get("failure_reason")
    lines = ["### Experiment 6 — Data Augmentation Ablation\n"]
    lines.append(f"**Status**: `{status}`\n")
    lines.append(_meta_summary(meta))

    if status == "failed":
        lines.append(f"**Failure reason**: {failure_reason or 'unknown'}\n")
    else:
        runs = data.get("runs") or []
        for run in runs:
            aug = run.get("augmentation_mode", "?")
            run_status = run.get("status", "failed")
            run_fail = run.get("failure_reason")
            lines.append(f"#### augmentation = {aug} | status: `{run_status}`\n")
            if run_status == "failed":
                lines.append(f"Failure reason: {run_fail or 'unknown'}\n")
            else:
                lines.append(_retrieval_table(run.get("retrieval"), f"augmentation={aug}"))
                stability = run.get("training_stability")
                if stability and isinstance(stability, dict):
                    lines.append("**Training stability** (loss variance across sliding windows):\n")
                    lines.append(f"- Window size: {stability.get('window_size', 'N/A')}")
                    lines.append(f"- Mean variance: {_fmt(stability.get('mean_variance'))}")
                    lines.append(f"- Max variance: {_fmt(stability.get('max_variance'))}\n")
                if run_fail:
                    lines.append(f"Partial failure reason: {run_fail}\n")

        if failure_reason and status != "failed":
            lines.append(f"\n**Overall partial failure reason**: {failure_reason}\n")

    lines.append("\n#### Lessons Learned\n")
    lines.append(
        "Data augmentation regularizes the image encoder by presenting "
        "different views of the same image. The weak setting applies only "
        "mild geometric and color distortions; the strong setting adds "
        "random grayscale, aggressive crops, and Gaussian blur. Training "
        "stability metrics reveal whether strong augmentation introduces "
        "gradient noise that slows convergence or causes loss spikes.\n"
    )
    return "\n".join(lines) + "\n"


def _benchmark_section(bench: dict) -> str:
    lines = ["## Benchmark Results\n"]
    if not bench or "error" in bench:
        err = bench.get("error", "unknown error") if bench else "benchmark.json not loaded"
        lines.append(f"*Benchmark data unavailable: {err}*\n")
        return "\n".join(lines) + "\n"

    def _latency_row(key: str, label: str) -> str:
        d = bench.get(key) or {}
        return (
            f"| {label} "
            f"| {_fmt(d.get('mean'))} "
            f"| {_fmt(d.get('median'))} "
            f"| {_fmt(d.get('p95'))} "
            f"| {_fmt(d.get('std'))} |"
        )

    lines.append("| Measurement | Mean (ms) | Median (ms) | p95 (ms) | Std (ms) |")
    lines.append("|---|---|---|---|---|")
    lines.append(_latency_row("image_encode_latency_ms", "Image encode latency"))
    lines.append(_latency_row("text_encode_latency_ms", "Text encode latency"))
    lines.append(_latency_row("retrieval_query_latency_ms", "Retrieval query latency"))
    lines.append(_latency_row("zero_shot_latency_ms", "Zero-shot inference latency"))
    lines.append("")

    tput = bench.get("throughput_items_per_sec") or {}
    lines.append("| Throughput | Mean | Median | p95 | Std |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Items / sec "
        f"| {_fmt(tput.get('mean'))} "
        f"| {_fmt(tput.get('median'))} "
        f"| {_fmt(tput.get('p95'))} "
        f"| {_fmt(tput.get('std'))} |"
    )
    lines.append("")

    peak_ram = bench.get("peak_ram_mb")
    avg_ram = bench.get("avg_ram_mb")
    iters = bench.get("iterations")
    lines.append(f"- **Peak RAM**: {_fmt(peak_ram, 2)} MB")
    lines.append(f"- **Average RAM**: {_fmt(avg_ram, 2)} MB")
    lines.append(f"- **Measured iterations**: {iters if iters is not None else 'N/A'}\n")

    return "\n".join(lines) + "\n"


def _qualitative_section(results: dict) -> str:
    lines = ["## Qualitative Retrieval Examples\n"]
    qualitative = None
    for n in range(1, 7):
        data = results.get(n, {})
        q = data.get("qualitative_retrieval")
        if q:
            qualitative = q
            break

    if not qualitative:
        lines.append(
            "No pre-computed qualitative retrieval examples were found in the "
            "experiment result files. To generate examples, run the trained "
            "model with `compare.py` on the validation split.\n"
        )
        return "\n".join(lines) + "\n"

    n_show = min(10, len(qualitative))
    for i, entry in enumerate(qualitative[:n_show]):
        query = entry.get("query", "")
        matches = entry.get("matches", [])[:5]
        lines.append(f"**Query {i + 1}**: {query}\n")
        for rank, m in enumerate(matches, start=1):
            caption = m.get("caption", m.get("text", ""))
            score = m.get("score", m.get("similarity"))
            score_str = _fmt(score) if score is not None else "N/A"
            lines.append(f"  {rank}. {caption} (similarity: {score_str})")
        lines.append("")

    return "\n".join(lines) + "\n"


def _header_section() -> str:
    return """\
# Aurelius — CLIP-Style Vision-Language Model

*Generated on {}*

---
""".format(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))


def _motivation_section() -> str:
    return """\
## Motivation

Aurelius is a from-scratch implementation of CLIP-style contrastive vision-language
learning. The core goal is to understand how joint training of image and text encoders
with a symmetric InfoNCE contrastive objective causes semantically related image-caption
pairs to cluster together in a shared embedding space, without relying on any pretrained
weights or external contrastive-learning libraries.

Every model component — the ResNet-style CNN image encoder, the Transformer text
encoder, the projection heads, and the InfoNCE loss — is built from primitive PyTorch
layers and tensor operations. This constraint ensures the implementation demonstrates
genuine understanding of the underlying mechanisms rather than wrapping existing code.

Training uses the Flickr30K dataset (31,783 images, five captions each) with a fixed
validation split for all evaluation, making every result in this document reproducible
from a single `python experiments.py` invocation.

---
"""


def _architecture_section() -> str:
    return """\
## Architecture Overview

### Image Encoder (ResNet-style CNN)

The image encoder follows the standard residual network pattern:

1. **Stem**: Conv2d(3→64, 7×7, stride 2) → BatchNorm2d → ReLU → MaxPool2d(3×3, stride 2)
2. **Stage 1**: 2× ResidualBlock(64→64)
3. **Stage 2**: 2× ResidualBlock(64→128, stride 2)
4. **Stage 3**: 2× ResidualBlock(128→256, stride 2)
5. **Stage 4**: 2× ResidualBlock(256→512, stride 2)
6. **Head**: AdaptiveAvgPool2d(1) → flatten → ProjectionHead → L2 normalize

Each `ResidualBlock` contains two 3×3 conv layers with BatchNorm and ReLU, plus a skip
connection. When spatial dimensions or channel widths differ, the skip uses a 1×1 conv
projection. Input images must be 224×224.

### Text Encoder (Transformer)

The text encoder is a Transformer encoder stack built from scratch:

- **Tokenizer**: GPT-2 BPE via tiktoken; sequences truncated or padded to 77 tokens
- **Token embedding**: `nn.Embedding(vocab_size, d_model)`, randomly initialized
- **Positional embedding**: learned `nn.Parameter` of shape `[77, d_model]`
- **Transformer blocks** (6 layers, pre-norm): multi-head self-attention (8 heads,
  Q/K/V from scratch) + feed-forward (Linear → GELU → Linear) + residual connections
- **Pooling**: mean over non-padded token positions
- **Head**: ProjectionHead → L2 normalize

No `torch.nn.MultiheadAttention` or `torch.nn.Transformer` is used anywhere.

### Projection Heads

Two variants are supported (selected per run by `configs/default.yaml`):

- **LinearProjectionHead**: exactly one `nn.Linear`, no activation, no normalization
- **MLPProjectionHead**: two `nn.Linear` layers with one `ReLU` between them

Both image and text encoders use the same variant within a single run. The output
dimension always equals `embed_dim` (default: 256).

### InfoNCE Loss

The `InfoNCEModule` computes symmetric contrastive loss:

1. Compute N×N cosine similarity matrix between image and text embeddings
2. Scale by learnable temperature τ = exp(log_τ) > 0
3. Image→text cross-entropy: each row's target is its diagonal index
4. Text→image cross-entropy: each column's target is its diagonal index
5. Final loss = (image→text loss + text→image loss) / 2

---
"""


def _infonce_section() -> str:
    return """\
## InfoNCE Loss Explanation

InfoNCE (Noise Contrastive Estimation) trains the encoders by treating each
image-caption pair in a batch as a positive example and all other pairings in
the same batch as negatives.

For a batch of N image-caption pairs:

- Let **I** ∈ ℝ^(N×d) be the L2-normalized image embeddings
- Let **T** ∈ ℝ^(N×d) be the L2-normalized text embeddings
- The similarity matrix **S** = **I** · **T**ᵀ / τ, where τ > 0 is the learned temperature

The image-to-text loss treats each row of **S** as class scores over N text candidates,
with target label equal to the diagonal index (the matching caption). The text-to-image
loss does the same with columns. The final loss averages both directions:

```
L = (CrossEntropy(S, targets) + CrossEntropy(Sᵀ, targets)) / 2
```

**Temperature parameterization**: τ is represented as exp(log_τ) where log_τ is the
learnable parameter. This ensures τ > 0 at all times while allowing unconstrained
gradient updates. Smaller τ sharpens the distribution (stronger contrastive signal);
larger τ smooths it (weaker discrimination but more stable gradients early in training).

**Symmetric loss**: Computing loss in both directions forces the image encoder and text
encoder to jointly agree on the similarity ranking — neither modality is treated as the
anchor. This symmetry is the key property that aligns the two embedding spaces.

---
"""


def _limitations_section() -> str:
    return """\
## Limitations

- **Dataset scale**: Flickr30K contains ~31K images. CLIP was trained on 400M image-text
  pairs; the absolute retrieval numbers here are not comparable to published CLIP results.
- **Architecture scale**: The ResNet backbone and 6-layer Transformer are intentionally
  small for a research/educational codebase. A larger backbone would improve metrics.
- **Single GPU / CPU training**: No distributed training support; runtime on CPU is long.
- **No pretrained initialization**: Both encoders start from random weights. Transfer
  learning from pretrained vision or language backbones is excluded by design.
- **Fixed dataset**: Only Flickr30K is supported. Other image-text datasets require
  code changes to `data.py`.
- **Qualitative examples**: Rendered only if the model has been run through `compare.py`
  and the outputs stored in experiment result files.

---
"""


def _future_work_section() -> str:
    return """\
## Future Work

- **Larger backbone**: Replace the 4-stage ResNet with a deeper variant or a ViT encoder
  trained from scratch to improve representational capacity.
- **Larger dataset**: Add support for CC3M or LAION subsets to study scaling behaviour.
- **Hard negative mining**: Sample difficult negatives within each batch rather than
  random in-batch negatives to improve training signal at higher batch sizes.
- **Multi-caption aggregation**: Average five-caption embeddings at evaluation time
  instead of selecting one randomly, to reduce variance in retrieval metrics.
- **Distributed training**: Support multi-GPU training via `torch.nn.parallel.DistributedDataParallel`.
- **Checkpoint averaging**: Average weights from the last K checkpoints for improved
  generalization without additional training cost.

---
"""


def _citations_section() -> str:
    return """\
## Citations

- **CLIP**: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S.,
  Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021).
  Learning Transferable Visual Models From Natural Language Supervision.
  *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*.

- **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for
  Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern
  Recognition (CVPR 2016)*.

- **Flickr30K**: Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). From Image
  Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference Over
  Event Descriptions. *Transactions of the Association for Computational Linguistics, 2*,
  67–78.

- **InfoNCE**: Oord, A. van den, Li, Y., & Vinyals, O. (2018). Representation Learning
  with Contrastive Predictive Coding. *arXiv preprint arXiv:1807.03748*.
"""


def generate_readme(results: dict) -> str:
    """Build and return the full README.md markdown string."""
    metadata = results.get("metadata", {})

    _exp_builders = [
        (1, _exp1_section),
        (2, _exp2_section),
        (3, _exp3_section),
        (4, _exp4_section),
        (5, _exp5_section),
        (6, _exp6_section),
    ]

    sections = [
        _header_section(),
        _motivation_section(),
        _architecture_section(),
        _infonce_section(),
        "## Experiment Results\n\n",
    ]

    for n, builder in _exp_builders:
        sections.append(builder(results.get(n, {}), metadata.get(n, "unavailable")))
        sections.append("\n---\n\n")

    sections.append(_benchmark_section(results.get("benchmark", {})))
    sections.append("\n---\n\n")
    sections.append(_qualitative_section(results))
    sections.append("\n---\n\n")
    sections.append(_limitations_section())
    sections.append(_future_work_section())
    sections.append(_citations_section())

    return "".join(sections)


def run() -> None:
    """Precondition check → load results → generate → write README.md."""
    ok, issues = check_preconditions()
    if not ok:
        print("README generation rejected — the following files are missing or incomplete:")
        for issue in issues:
            print(f"  • {issue}")
        return

    results = load_all_results()
    content = generate_readme(results)
    _README_PATH.write_text(content, encoding="utf-8")
    print("README.md written.")


if __name__ == "__main__":
    run()
