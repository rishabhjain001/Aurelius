"""
model.py — CLIP Model: InfoNCE Loss + Encoder Composition

Combines ImageEncoder and TextEncoder into a single CLIPModel module,
with a learnable log-parameterized temperature for symmetric InfoNCE loss.

Implements:
    - InfoNCEModule
    - CLIPModel

Requirements: 1.1, 1.5, 6.8, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 25.3
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from encoders.image import ImageEncoder
from encoders.text import TextEncoder


class InfoNCEModule(nn.Module):
    """Symmetric InfoNCE (contrastive) loss with learnable temperature.

    Temperature is stored as log(temperature) so exp() always > 0 (Req 7.7).
    """

    def __init__(self, temperature_init: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature_init))
        )

    def forward(
        self, image_embeds: Tensor, text_embeds: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute symmetric InfoNCE loss.

        Args:
            image_embeds: [N, embed_dim] L2-normalized image embeddings.
            text_embeds:  [N, embed_dim] L2-normalized text embeddings.

        Returns:
            loss:               Scalar mean of image→text and text→image CE losses.
            similarity_matrix:  [N, N] raw cosine similarity (before temperature scaling).
        """
        # Cosine similarity (embeddings are L2-normalized, so dot product = cosine sim)
        similarity_matrix = image_embeds @ text_embeds.T  # [N, N]

        # Scale by learned temperature
        scaled = similarity_matrix / self.log_temperature.exp()

        # Targets are diagonal indices: image i matches text i
        N = image_embeds.shape[0]
        targets = torch.arange(N, device=image_embeds.device)

        # Image-to-text: each row of scaled vs. target column index
        loss_i2t = F.cross_entropy(scaled, targets)

        # Text-to-image: each row of scaled.T vs. target column index
        loss_t2i = F.cross_entropy(scaled.T, targets)

        loss = (loss_i2t + loss_t2i) / 2.0

        return loss, similarity_matrix


class CLIPModel(nn.Module):
    """Dual-encoder CLIP model combining image and text encoders.

    Args:
        config: Object with attributes:
            config.projection_head  ("linear" | "mlp")
            config.embed_dim        (int)
            config.temperature_init (float)
            config.image_encoder.*  (image encoder sub-config)
            config.text_encoder.*   (text encoder sub-config)

    Raises:
        ValueError: if config.projection_head is not "linear" or "mlp".
    """

    def __init__(self, config) -> None:
        super().__init__()

        if config.projection_head not in ("linear", "mlp"):
            raise ValueError(
                f"projection_head must be 'linear' or 'mlp', "
                f"got '{config.projection_head}'."
            )

        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.loss_module = InfoNCEModule(
            temperature_init=getattr(config, "temperature_init", 0.07)
        )

    def encode_image(self, x: Tensor) -> Tensor:
        """Encode images to L2-normalized embeddings.

        Args:
            x: [N, 3, 224, 224] float tensor.

        Returns:
            [N, embed_dim] L2-normalized.
        """
        return self.image_encoder(x)

    def encode_text(self, captions: List[str]) -> Tensor:
        """Encode caption strings to L2-normalized embeddings.

        Args:
            captions: List of N caption strings.

        Returns:
            [N, embed_dim] L2-normalized.
        """
        return self.text_encoder(captions)

    def forward(
        self, images: Tensor, captions: List[str]
    ) -> tuple[Tensor, Tensor]:
        """Full forward pass: encode both modalities and compute InfoNCE loss.

        Args:
            images:   [N, 3, 224, 224] float tensor.
            captions: List of N caption strings.

        Returns:
            loss:               Scalar symmetric InfoNCE loss.
            similarity_matrix:  [N, N] cosine similarity matrix (raw, pre-temperature).
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(captions)
        return self.loss_module(image_embeds, text_embeds)
