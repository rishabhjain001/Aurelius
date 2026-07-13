"""
visualize.py — Visualization tool for Aurelius CLIP-VLM

Generates all plots from pre-computed experiment result JSON files.
Does NOT load PyTorch models or run encoders; all data is read from
results/*.json files only.

Public interface:
    check_preconditions() -> tuple[bool, list[str]]
    run_visualization() -> None

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9,
              21.10, 25.3, 28.8
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path("results")
_PLOTS_DIR = Path("plots")

_EXPERIMENT_FILES = [
    _RESULTS_DIR / f"experiment{n}.json" for n in range(1, 7)
]
_BENCHMARK_FILE = _RESULTS_DIR / "benchmark.json"


# ---------------------------------------------------------------------------
# Precondition check (Requirement 21.10)
# ---------------------------------------------------------------------------

def check_preconditions() -> tuple[bool, list[str]]:
    """Return (all_ok, list_of_missing_or_incomplete_files).

    A file is considered valid when it either contains real measured data
    (non-null metrics) OR has status 'failed'/'partial' with a non-null
    failure_reason.
    """
    issues: list[str] = []

    # 1. Benchmark file must exist.
    if not _BENCHMARK_FILE.exists():
        issues.append(str(_BENCHMARK_FILE))

    # 2. Each experiment result file must exist and be valid.
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
            # Valid only when failure_reason is provided.
            if not failure_reason:
                issues.append(
                    f"{path} (status='{status}' but failure_reason is null/missing)"
                )
        else:
            # Must contain at least one non-null metric field beyond bookkeeping keys.
            _bookkeeping = {"experiment", "status", "failure_reason"}
            real_keys = {
                k for k, v in data.items()
                if k not in _bookkeeping and v is not None
            }
            if not real_keys:
                issues.append(f"{path} (no real measured data and no valid failed/partial status)")

    all_ok = len(issues) == 0
    return all_ok, issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _placeholder_plot(path: str, message: str) -> None:
    """Save a 'not available' placeholder figure at path."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(
        0.5, 0.5,
        f"Data not available — {message}",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=12, color="gray",
        wrap=True,
    )
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    _save(fig, path)


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _load_result(n: int) -> dict:
    """Load results/experimentN.json as a dict."""
    path = _RESULTS_DIR / f"experiment{n}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_available(data: dict) -> bool:
    """True when the result file has real data (not failed/partial with no data)."""
    status = data.get("status", "")
    return status not in ("failed", "partial") or bool(
        {k for k, v in data.items()
         if k not in {"experiment", "status", "failure_reason"} and v is not None}
    )


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_exp1_loss(results: dict, output_path: str) -> None:
    """Training loss curve from Experiment 1 per-step data (Requirement 21.2)."""
    steps_data = results.get("steps")
    if not steps_data:
        _placeholder_plot(output_path, "Experiment 1 produced no per-step data")
        return

    xs = [r["step"] for r in steps_data if r.get("step") is not None]
    ys = [r["loss"] for r in steps_data if r.get("loss") is not None]

    if not xs or not ys:
        _placeholder_plot(output_path, "Experiment 1 loss values missing")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, linewidth=1.2, color="#2878b5")
    ax.set_xlabel("Step")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Experiment 1 — Training Loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_exp1_temperature(results: dict, output_path: str) -> None:
    """Temperature curve from Experiment 1 per-step data (Requirement 21.3)."""
    steps_data = results.get("steps")
    if not steps_data:
        _placeholder_plot(output_path, "Experiment 1 produced no per-step data")
        return

    xs = [r["step"] for r in steps_data if r.get("step") is not None]
    ys = [r["temperature"] for r in steps_data if r.get("temperature") is not None]

    if not xs or not ys:
        _placeholder_plot(output_path, "Experiment 1 temperature values missing")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, linewidth=1.2, color="#e05c22")
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature")
    ax.set_title("Experiment 1 — Temperature Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_exp2_retrieval_bars(results: dict, output_path: str) -> None:
    """R@1, R@5, R@10 grouped bar chart for Experiment 2 (Requirement 21.4)."""
    retrieval = results.get("retrieval")
    if not retrieval:
        _placeholder_plot(output_path, "Experiment 2 retrieval metrics unavailable")
        return

    trained = retrieval.get("trained", {})
    baseline = retrieval.get("random_baseline", {})

    metrics = ["r1", "r5", "r10"]
    labels = ["R@1", "R@5", "R@10"]
    directions = ["i2t", "t2i"]
    dir_labels = {"i2t": "Image → Text", "t2i": "Text → Image"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bar_width = 0.35
    x = np.arange(len(metrics))
    colors_trained = "#2878b5"
    colors_baseline = "#f8ac8c"

    for ax, direction in zip(axes, directions):
        t_vals = [trained.get(direction, {}).get(m, 0.0) or 0.0 for m in metrics]
        b_vals = [baseline.get(direction, {}).get(m, 0.0) or 0.0 for m in metrics]

        bars_t = ax.bar(x - bar_width / 2, t_vals, bar_width,
                        label="Trained", color=colors_trained, alpha=0.85)
        bars_b = ax.bar(x + bar_width / 2, b_vals, bar_width,
                        label="Random Baseline", color=colors_baseline, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(dir_labels[direction])
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate bar values
        for bar in bars_t:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars_b:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Experiment 2 — Retrieval Metrics: Trained vs Random Baseline")
    fig.tight_layout()
    _save(fig, output_path)


def plot_zero_shot_confusion(results: dict, output_path: str) -> None:
    """Confusion matrix heatmap from zero-shot evaluation (Requirement 21.5)."""
    # Search all results for zero-shot data.
    zero_shot = results.get("zero_shot")
    if not zero_shot:
        _placeholder_plot(output_path, "No zero-shot evaluation data in results")
        return

    confusion = zero_shot.get("confusion_matrix")
    class_labels = zero_shot.get("class_labels")

    if not confusion or not class_labels:
        _placeholder_plot(output_path, "Zero-shot confusion matrix data missing")
        return

    cm = np.array(confusion, dtype=float)
    # Normalize rows to fractions so colors show relative performance.
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Fraction of true class")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Zero-Shot Confusion Matrix")

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.tight_layout()
    _save(fig, output_path)


def plot_umap(output_path: str) -> None:
    """UMAP projection of image + text embeddings (Requirement 21.6).

    Reads embedding vectors from experiment result files if available.
    Generates a placeholder if no embedding data is stored.
    Does NOT load PyTorch models or run encoders.
    """
    # Check all experiment files for stored embedding data.
    embedding_data = None
    for n in range(1, 7):
        data = _load_result(n)
        emb = data.get("embeddings")
        if emb and emb.get("image_embeddings") and emb.get("text_embeddings"):
            embedding_data = emb
            break

    if embedding_data is None:
        _placeholder_plot(
            output_path,
            "Requires running the model — no pre-computed embeddings found in result files"
        )
        return

    image_embs = np.array(embedding_data["image_embeddings"], dtype=np.float32)
    text_embs = np.array(embedding_data["text_embeddings"], dtype=np.float32)
    all_embs = np.concatenate([image_embs, text_embs], axis=0)
    labels = (["image"] * len(image_embs)) + (["text"] * len(text_embs))

    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        proj = reducer.fit_transform(all_embs)
    except ImportError:
        _placeholder_plot(output_path, "umap-learn not installed")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    img_mask = np.array(labels) == "image"
    ax.scatter(proj[img_mask, 0], proj[img_mask, 1],
               s=8, alpha=0.5, color="#2878b5", label="Image")
    ax.scatter(proj[~img_mask, 0], proj[~img_mask, 1],
               s=8, alpha=0.5, color="#e05c22", label="Text")
    ax.set_title("UMAP Projection — Image & Text Embeddings")
    ax.legend(markerscale=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    _save(fig, output_path)


def plot_tsne(output_path: str) -> None:
    """t-SNE projection of image + text embeddings (Requirement 21.7).

    Same logic as UMAP: reads from result files or generates a placeholder.
    Does NOT load PyTorch models or run encoders.
    """
    embedding_data = None
    for n in range(1, 7):
        data = _load_result(n)
        emb = data.get("embeddings")
        if emb and emb.get("image_embeddings") and emb.get("text_embeddings"):
            embedding_data = emb
            break

    if embedding_data is None:
        _placeholder_plot(
            output_path,
            "Requires running the model — no pre-computed embeddings found in result files"
        )
        return

    image_embs = np.array(embedding_data["image_embeddings"], dtype=np.float32)
    text_embs = np.array(embedding_data["text_embeddings"], dtype=np.float32)
    all_embs = np.concatenate([image_embs, text_embs], axis=0)
    labels = (["image"] * len(image_embs)) + (["text"] * len(text_embs))

    try:
        from sklearn.manifold import TSNE  # type: ignore
    except ImportError:
        _placeholder_plot(output_path, "scikit-learn not installed")
        return

    perplexity = min(30, max(5, len(all_embs) // 5))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    proj = tsne.fit_transform(all_embs)

    fig, ax = plt.subplots(figsize=(9, 7))
    img_mask = np.array(labels) == "image"
    ax.scatter(proj[img_mask, 0], proj[img_mask, 1],
               s=8, alpha=0.5, color="#2878b5", label="Image")
    ax.scatter(proj[~img_mask, 0], proj[~img_mask, 1],
               s=8, alpha=0.5, color="#e05c22", label="Text")
    ax.set_title("t-SNE Projection — Image & Text Embeddings")
    ax.legend(markerscale=2)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    fig.tight_layout()
    _save(fig, output_path)


def plot_cosine_histogram(results: dict, output_path: str) -> None:
    """Histogram of cosine similarities from the validation matrix (Requirement 21.8)."""
    sim_data = results.get("similarity_matrix")
    if sim_data is None:
        # Also try nested under retrieval
        retrieval = results.get("retrieval", {})
        sim_data = retrieval.get("similarity_values")

    if not sim_data:
        _placeholder_plot(output_path, "No cosine similarity data in Experiment 1 results")
        return

    values = np.array(sim_data, dtype=np.float32).ravel()
    if values.size == 0:
        _placeholder_plot(output_path, "Cosine similarity array is empty")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=80, color="#2878b5", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Validation Cosine Similarity Distribution")
    ax.axvline(float(values.mean()), color="#e05c22", linewidth=1.5,
               linestyle="--", label=f"Mean = {values.mean():.3f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_qualitative_retrieval(output_path: str) -> None:
    """Qualitative retrieval examples: 10 queries × top-5 matches (Requirement 21.9).

    Reads qualitative retrieval examples stored in result files.
    Generates a placeholder if no such data is found.
    Does NOT load PyTorch models or run encoders.
    """
    qualitative = None
    for n in range(1, 7):
        data = _load_result(n)
        q = data.get("qualitative_retrieval")
        if q:
            qualitative = q
            break

    if not qualitative:
        _placeholder_plot(
            output_path,
            "Requires model and validation data — no pre-computed retrieval examples in result files"
        )
        return

    # qualitative is expected to be a list of dicts:
    # [{"query": str, "matches": [{"caption": str, "score": float}, ...]}]
    n_queries = min(10, len(qualitative))
    n_matches = 5

    fig, axes = plt.subplots(
        n_queries, n_matches + 1,
        figsize=(2.5 * (n_matches + 1), 2.2 * n_queries)
    )

    if n_queries == 1:
        axes = [axes]

    for row_idx, entry in enumerate(qualitative[:n_queries]):
        query = entry.get("query", "")
        matches = entry.get("matches", [])[:n_matches]

        # Query cell
        ax_q = axes[row_idx][0]
        ax_q.text(0.5, 0.5, f"Query:\n{query[:60]}",
                  transform=ax_q.transAxes,
                  ha="center", va="center", fontsize=7, wrap=True)
        ax_q.set_axis_off()
        ax_q.set_facecolor("#f0f4fa")

        for col_idx in range(n_matches):
            ax = axes[row_idx][col_idx + 1]
            ax.set_axis_off()
            if col_idx < len(matches):
                m = matches[col_idx]
                caption = m.get("caption", m.get("text", ""))[:60]
                score = m.get("score", m.get("similarity", None))
                label = f"#{col_idx + 1}\n{caption}"
                if score is not None:
                    label += f"\n({score:.3f})"
                ax.text(0.5, 0.5, label,
                        transform=ax.transAxes,
                        ha="center", va="center", fontsize=7)

    fig.suptitle("Qualitative Retrieval Examples (10 queries × top-5 matches)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_visualization() -> None:
    """Check preconditions, create plots/, generate all 8 plots (Requirements 21.1–21.10)."""
    ok, issues = check_preconditions()
    if not ok:
        print("Visualization rejected — the following files are missing or incomplete:")
        for issue in issues:
            print(f"  • {issue}")
        return

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    exp1 = _load_result(1)
    exp2 = _load_result(2)

    # 1. Training loss curve
    print("Generating plots/exp1_loss_curve.png …")
    plot_exp1_loss(exp1, str(_PLOTS_DIR / "exp1_loss_curve.png"))

    # 2. Temperature curve
    print("Generating plots/exp1_temperature_curve.png …")
    plot_exp1_temperature(exp1, str(_PLOTS_DIR / "exp1_temperature_curve.png"))

    # 3. Retrieval bar chart
    print("Generating plots/exp2_retrieval_bars.png …")
    plot_exp2_retrieval_bars(exp2, str(_PLOTS_DIR / "exp2_retrieval_bars.png"))

    # 4. Confusion matrix
    print("Generating plots/zero_shot_confusion.png …")
    # Search all experiments for zero-shot data.
    zs_data: dict = {}
    for n in range(1, 7):
        d = _load_result(n)
        if d.get("zero_shot"):
            zs_data = d
            break
    plot_zero_shot_confusion(zs_data, str(_PLOTS_DIR / "zero_shot_confusion.png"))

    # 5. UMAP
    print("Generating plots/umap_embeddings.png …")
    plot_umap(str(_PLOTS_DIR / "umap_embeddings.png"))

    # 6. t-SNE
    print("Generating plots/tsne_embeddings.png …")
    plot_tsne(str(_PLOTS_DIR / "tsne_embeddings.png"))

    # 7. Cosine similarity histogram
    print("Generating plots/cosine_similarity_hist.png …")
    plot_cosine_histogram(exp1, str(_PLOTS_DIR / "cosine_similarity_hist.png"))

    # 8. Qualitative retrieval
    print("Generating plots/qualitative_retrieval.png …")
    plot_qualitative_retrieval(str(_PLOTS_DIR / "qualitative_retrieval.png"))

    print(f"Done. All plots written to '{_PLOTS_DIR}/'.")


if __name__ == "__main__":
    run_visualization()
