"""
compare.py — Interactive retrieval demo for Aurelius CLIP-VLM

Supports two query directions against the validation split:
  - Text → top-K images ranked by descending cosine similarity
  - Image → top-K captions ranked by descending cosine similarity

Each result includes its cosine similarity score. K is clamped to the pool
size when K exceeds available candidates; K < 1 raises ValueError.

Optional grid save writes top-K retrieved images via matplotlib. On save
failure the error is raised so the caller can surface it; any previously
saved grid is left untouched.

Public interface:
    - CompareTool
    - build_val_embeddings(model, val_dataset) -> (image_embeds, text_embeds, captions, images)

Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 25.3
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CompareTool:
    """Retrieval demo against pre-encoded validation embeddings.

    val_image_embeds: [N_images, D] L2-normalized image embeddings.
    val_text_embeds:  [N_texts,  D] L2-normalized text embeddings.
    val_captions:     list of N_texts strings matching val_text_embeds rows.
    val_images:       optional list of PIL Images or tensors for grid display.
    """

    def __init__(
        self,
        model,
        val_image_embeds: torch.Tensor,
        val_text_embeds: torch.Tensor,
        val_captions: list,
        val_images: Optional[list] = None,
    ) -> None:
        self.model = model
        self.val_image_embeds = val_image_embeds   # [N_images, D]
        self.val_text_embeds = val_text_embeds     # [N_texts, D]
        self.val_captions = val_captions
        self.val_images = val_images

    def text_to_images(self, query: str, k: int) -> list:
        """Return top-K images ranked by descending cosine similarity to query.

        Each result: {"rank": int, "image_idx": int, "score": float}.
        Raises ValueError when k < 1. Clamps k to pool size when k exceeds it.
        """
        if k < 1:
            raise ValueError("K must be at least 1")

        n_images = self.val_image_embeds.shape[0]
        k = min(k, n_images)

        self.model.eval()
        with torch.no_grad():
            text_embed = self.model.encode_text([query])        # [1, D]

        # cosine similarity: embeddings are L2-normalized so dot product suffices
        scores = (self.val_image_embeds @ text_embed.T).squeeze(1)  # [N_images]
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist()), start=1
        ):
            results.append({"rank": rank, "image_idx": idx, "score": score})
        return results

    def image_to_captions(self, image_tensor: torch.Tensor, k: int) -> list:
        """Return top-K captions ranked by descending cosine similarity to image.

        image_tensor: [3, 224, 224] or [1, 3, 224, 224].
        Each result: {"rank": int, "caption_idx": int, "caption": str, "score": float}.
        Raises ValueError when k < 1. Clamps k to pool size when k exceeds it.
        """
        if k < 1:
            raise ValueError("K must be at least 1")

        n_texts = self.val_text_embeds.shape[0]
        k = min(k, n_texts)

        # Ensure batch dimension.
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)   # [1, 3, 224, 224]

        self.model.eval()
        with torch.no_grad():
            image_embed = self.model.encode_image(image_tensor)  # [1, D]

        scores = (self.val_text_embeds @ image_embed.T).squeeze(1)  # [N_texts]
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist()), start=1
        ):
            results.append({
                "rank": rank,
                "caption_idx": idx,
                "caption": self.val_captions[idx],
                "score": score,
            })
        return results

    def save_image_grid(self, results: list, save_path: str) -> None:
        """Save top retrieved images as a matplotlib grid to save_path.

        On failure: raises the exception so the caller can surface it.
        Previously saved grids are never deleted or overwritten on failure.
        """
        import matplotlib.pyplot as plt  # local import: optional dependency path

        if self.val_images is None:
            logger.info("val_images not provided; skipping grid render.")
            return

        images_to_show = []
        scores = []
        for entry in results:
            idx = entry.get("image_idx")
            if idx is None or idx >= len(self.val_images):
                logger.info("Image index %s out of range; skipping.", idx)
                continue
            images_to_show.append(self.val_images[idx])
            scores.append(entry.get("score", float("nan")))

        if not images_to_show:
            logger.info("No valid images to render; grid not saved.")
            return

        n = len(images_to_show)
        cols = min(n, 5)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        # Flatten axes to a 1-D list regardless of shape.
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = list(axes.flatten())
        else:
            axes = [ax for row in axes for ax in row]

        for i, (img, score) in enumerate(zip(images_to_show, scores)):
            ax = axes[i]
            # Accept PIL images or tensors.
            if hasattr(img, "convert"):   # PIL image
                ax.imshow(img)
            else:
                # Tensor: [C, H, W] → [H, W, C], clamp to [0, 1].
                t = img
                if t.dim() == 3:
                    t = t.permute(1, 2, 0)
                ax.imshow(t.clamp(0.0, 1.0).cpu().numpy())
            ax.set_title(f"{score:.3f}", fontsize=8)
            ax.axis("off")

        # Hide any unused axes.
        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        # Any IOError or matplotlib exception propagates to the caller.
        fig.savefig(save_path)
        plt.close(fig)


def build_val_embeddings(model, val_dataset) -> tuple:
    """Encode all validation images and captions; return (image_embeds, text_embeds, captions, images).

    val_dataset items must be dict {"image": Tensor, "captions": list[str]}
    or tuple (image_tensor, captions).
    """
    image_chunks: list = []
    text_chunks: list = []
    all_captions: list = []
    all_images: list = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            item = val_dataset[idx]
            if isinstance(item, tuple):
                image_tensor, captions = item[0], item[1]
            else:
                image_tensor = item["image"]
                captions = item["captions"]
            if not isinstance(captions, list):
                captions = [captions]

            img_embed = model.encode_image(image_tensor.unsqueeze(0))  # [1, D]
            image_chunks.append(img_embed)
            all_images.append(image_tensor)

            text_embed = model.encode_text(captions)                   # [K, D]
            text_chunks.append(text_embed)
            all_captions.extend(captions)

    image_embeds = torch.cat(image_chunks, dim=0)   # [N_images, D]
    text_embeds = torch.cat(text_chunks, dim=0)     # [N_texts, D]
    return image_embeds, text_embeds, all_captions, all_images
