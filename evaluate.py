"""
evaluate.py — Retrieval and zero-shot evaluation for Aurelius CLIP-VLM

Retrieval_Evaluator: encodes validation images and captions, computes
cosine-similarity ranks under Requirement 8 tie-breaking rules (identical
scores share the best-numbered rank), and reports R@1/R@5/R@10, MRR, mean
rank, and median rank for each direction.

Zero_Shot_Evaluator: uses the frozen PROMPT_SET (locked once Experiment 1
begins per Requirement 9.3); reports overall accuracy, per-class accuracy,
and confusion matrix.

Public interface:
    - DirectionalMetrics, RetrievalResult, ZeroShotResult
    - retrieval_metrics(image_embeds, text_embeds, caption_to_image)
    - retrieval_metrics_from_model(model, val_dataset)
    - PROMPT_SET, lock_prompt_set, is_prompt_set_locked, modify_prompt_set
    - zero_shot_metrics(model, images, true_labels, prompt_set=None)

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.1, 9.2, 9.3, 9.4, 25.3
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DirectionalMetrics:
    """Retrieval metrics for a single direction (i2t or t2i)."""
    r1: float
    r5: float
    r10: float
    mrr: float
    mean_rank: float
    median_rank: float


@dataclass
class RetrievalResult:
    """Bidirectional retrieval result; skipped when validation split is empty."""
    i2t: Optional[DirectionalMetrics]
    t2i: Optional[DirectionalMetrics]
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class ZeroShotResult:
    """Zero-shot classification result over a class list."""
    overall_accuracy: float
    per_class_accuracy: dict   # {class_name: accuracy_float}
    confusion_matrix: list     # list of lists, rows=true, cols=pred
    class_labels: list         # ordered class names matching matrix axes


# ---------------------------------------------------------------------------
# Ranking utilities
# ---------------------------------------------------------------------------

def _best_rank(scores: np.ndarray, correct_indices) -> int:
    """1-indexed best rank of any correct match under Req 8.5 tie-breaking.

    A candidate's rank equals 1 + the number of candidates with strictly
    higher score, so identical scores share the smallest-numbered position.
    """
    correct = np.asarray(list(correct_indices), dtype=np.int64)
    best_correct_score = scores[correct].max()
    return 1 + int((scores > best_correct_score).sum())


def _directional_metrics(
    sim: torch.Tensor,
    correct_per_query: list,
) -> DirectionalMetrics:
    """Aggregate rank-based metrics for one retrieval direction."""
    sim_np = sim.detach().cpu().numpy()
    ranks: list = []
    for q in range(sim_np.shape[0]):
        correct = correct_per_query[q]
        if not correct:
            continue
        ranks.append(_best_rank(sim_np[q], correct))
    if not ranks:
        return DirectionalMetrics(
            r1=0.0, r5=0.0, r10=0.0,
            mrr=0.0, mean_rank=0.0, median_rank=0.0,
        )
    arr = np.asarray(ranks, dtype=np.float64)
    return DirectionalMetrics(
        r1=float((arr <= 1).mean()),
        r5=float((arr <= 5).mean()),
        r10=float((arr <= 10).mean()),
        mrr=float((1.0 / arr).mean()),
        mean_rank=float(arr.mean()),
        median_rank=float(np.median(arr)),
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    caption_to_image: list,
) -> RetrievalResult:
    """Compute bidirectional retrieval metrics.

    Args:
        image_embeds:      [N_images, D] L2-normalized image embeddings.
        text_embeds:       [N_texts,  D] L2-normalized text embeddings.
        caption_to_image:  length N_texts; each entry is the image index that
                           caption belongs to (multiple captions per image).

    Returns:
        RetrievalResult. If either N is zero, returns a skipped result
        (Req 8.6). Otherwise contains i2t and t2i DirectionalMetrics.
    """
    n_images = int(image_embeds.shape[0])
    n_texts = int(text_embeds.shape[0])
    if n_images == 0 or n_texts == 0:
        return RetrievalResult(
            i2t=None, t2i=None,
            skipped=True,
            skip_reason="empty validation split",
        )
    if len(caption_to_image) != n_texts:
        raise ValueError(
            f"caption_to_image length {len(caption_to_image)} "
            f"does not match number of text embeddings {n_texts}."
        )

    # Cosine similarity; embeddings are assumed L2-normalized by encoders.
    sim = image_embeds @ text_embeds.T   # [N_images, N_texts]

    # i2t: each image row → any caption pointing to it counts as correct.
    correct_captions: list = [[] for _ in range(n_images)]
    for j, img_idx in enumerate(caption_to_image):
        if 0 <= img_idx < n_images:
            correct_captions[img_idx].append(j)
    i2t = _directional_metrics(sim, correct_captions)

    # t2i: each caption column → exactly one correct image.
    correct_images: list = [[int(caption_to_image[j])] for j in range(n_texts)]
    t2i = _directional_metrics(sim.T, correct_images)

    return RetrievalResult(i2t=i2t, t2i=t2i, skipped=False, skip_reason=None)


def retrieval_metrics_from_model(model, val_dataset) -> RetrievalResult:
    """Encode a validation dataset with `model`, then call retrieval_metrics.

    `val_dataset` items may be either a dict {"image": Tensor, "captions":
    list[str]} or a tuple (image_tensor, captions). All captions per image
    are used so image→text retrieval respects the five-correct-matches rule
    (Req 8.1).
    """
    if len(val_dataset) == 0:
        return RetrievalResult(
            i2t=None, t2i=None,
            skipped=True,
            skip_reason="empty validation split",
        )

    image_chunks: list = []
    text_chunks: list = []
    caption_to_image: list = []

    model.eval()
    with torch.no_grad():
        for image_idx in range(len(val_dataset)):
            item = val_dataset[image_idx]
            if isinstance(item, tuple):
                image_tensor, captions = item[0], item[1]
            else:
                image_tensor = item["image"]
                captions = item["captions"]
            if not isinstance(captions, list):
                captions = [captions]

            img_embed = model.encode_image(image_tensor.unsqueeze(0))  # [1, D]
            image_chunks.append(img_embed)

            text_embed = model.encode_text(captions)                   # [K, D]
            text_chunks.append(text_embed)
            caption_to_image.extend([image_idx] * len(captions))

    image_embeds = torch.cat(image_chunks, dim=0)
    text_embeds = torch.cat(text_chunks, dim=0)
    return retrieval_metrics(image_embeds, text_embeds, caption_to_image)


# ---------------------------------------------------------------------------
# Prompt set (frozen before Experiment 1)
# ---------------------------------------------------------------------------

# The Prompt_Set is defined ONCE before Experiment 1 begins (Req 9.1).
# Once locked, any mutation is rejected and the original set is preserved
# (Req 9.3).
PROMPT_SET: dict = {
    "dog":      ["a photo of a dog", "a picture of a dog"],
    "cat":      ["a photo of a cat", "a picture of a cat"],
    "person":   ["a photo of a person", "a picture of a person"],
    "car":      ["a photo of a car", "a picture of a car"],
    "building": ["a photo of a building", "a picture of a building"],
    "food":     ["a photo of food", "a picture of food"],
    "animal":   ["a photo of an animal", "a picture of an animal"],
    "vehicle":  ["a photo of a vehicle", "a picture of a vehicle"],
    "child":    ["a photo of a child", "a picture of a child"],
    "sport":    ["a photo of a sport", "a picture of a sport"],
}

_prompt_set_locked = False


def lock_prompt_set() -> None:
    """Freeze PROMPT_SET; called by Experiment_Runner before Experiment 1."""
    global _prompt_set_locked
    _prompt_set_locked = True


def is_prompt_set_locked() -> bool:
    """True once lock_prompt_set() has been called."""
    return _prompt_set_locked


def modify_prompt_set(new_prompts: dict) -> None:
    """Update PROMPT_SET; rejected once locked (Req 9.3)."""
    if _prompt_set_locked:
        raise RuntimeError(
            "PROMPT_SET is locked (Experiment 1 has begun). "
            "Modification rejected; original set is preserved."
        )
    PROMPT_SET.update(new_prompts)


# ---------------------------------------------------------------------------
# Zero-shot classification
# ---------------------------------------------------------------------------

def zero_shot_metrics(
    model,
    images: torch.Tensor,
    true_labels: list,
    prompt_set: Optional[dict] = None,
) -> ZeroShotResult:
    """Zero-shot classify images against class prompt embeddings.

    Args:
        model:        Object exposing `encode_image` and `encode_text`.
        images:       [N, 3, H, W] float tensor.
        true_labels:  Length-N list of class names; each must be a key in
                      `prompt_set`. Labels outside the set are ignored.
        prompt_set:   Class-label → list of prompt strings. Defaults to
                      PROMPT_SET.

    Returns:
        ZeroShotResult with overall accuracy, per-class accuracy, confusion
        matrix, and the ordered class labels (Req 9.4).
    """
    if prompt_set is None:
        prompt_set = PROMPT_SET

    class_labels = list(prompt_set.keys())
    n_classes = len(class_labels)

    model.eval()
    with torch.no_grad():
        # Each class embedding: mean of prompt embeddings, re-normalized.
        class_vecs = []
        for cls in class_labels:
            prompt_embeds = model.encode_text(prompt_set[cls])       # [P, D]
            mean = prompt_embeds.mean(dim=0, keepdim=True)           # [1, D]
            class_vecs.append(F.normalize(mean, dim=-1).squeeze(0))  # [D]
        class_embeds = torch.stack(class_vecs, dim=0)                # [C, D]

        image_embeds = model.encode_image(images)                    # [N, D]
        sim = image_embeds @ class_embeds.T                          # [N, C]
        preds = sim.argmax(dim=1).detach().cpu().tolist()

    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    confusion = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    per_class_correct = {c: 0 for c in class_labels}
    per_class_total = {c: 0 for c in class_labels}
    correct_total = 0

    for pred_idx, true_label in zip(preds, true_labels):
        if true_label not in label_to_idx:
            continue
        true_idx = label_to_idx[true_label]
        confusion[true_idx][pred_idx] += 1
        per_class_total[true_label] += 1
        if pred_idx == true_idx:
            correct_total += 1
            per_class_correct[true_label] += 1

    total = sum(per_class_total.values())
    overall = correct_total / total if total > 0 else 0.0
    per_class = {
        c: (per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0)
        for c in class_labels
    }
    return ZeroShotResult(
        overall_accuracy=overall,
        per_class_accuracy=per_class,
        confusion_matrix=confusion,
        class_labels=class_labels,
    )
