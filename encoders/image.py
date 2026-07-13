"""
encoders/image.py — ResNet-style Image Encoder for Aurelius CLIP-VLM

Architecture:
    Stem → 4 Residual Stages → AdaptiveAvgPool → flatten → ProjectionHead → L2 normalize

Implements:
    - ResidualBlock
    - LinearProjectionHead
    - MLPProjectionHead
    - ImageEncoder

Requirements: 1.1, 1.3, 1.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 6.1, 6.2, 6.3,
              6.4, 6.5, 6.6, 6.7, 25.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two 3×3 conv layers with BN+ReLU between them.

    Skip connection:
      - Identity when in_channels == out_channels and stride == 1.
      - 1×1 Conv2d (no BN, no activation) otherwise.

    Forward: out = ReLU(main(x) + skip(x))
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # Main path: Conv → BN → ReLU → Conv → BN
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, padding=0, bias=False
            )
        else:
            self.skip = None  # identity

    def forward(self, x: Tensor) -> Tensor:
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip path
        identity = self.skip(x) if self.skip is not None else x

        return F.relu(out + identity)


# ---------------------------------------------------------------------------
# Projection Heads
# ---------------------------------------------------------------------------

class LinearProjectionHead(nn.Module):
    """Single linear layer projection head.

    Exactly one nn.Linear(in_features, embed_dim). No activation, no normalization.
    Raises ValueError in forward if output shape[-1] != embed_dim.
    """

    def __init__(self, in_features: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(in_features, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        if out.shape[-1] != self.embed_dim:
            raise ValueError(
                f"LinearProjectionHead output dim {out.shape[-1]} != "
                f"expected embed_dim {self.embed_dim}"
            )
        return out


class MLPProjectionHead(nn.Module):
    """Two-layer MLP projection head.

    Exactly two nn.Linear layers with exactly one nn.ReLU between them.
    No other activations, no normalization inside the projection head.
    Raises ValueError in forward if output shape[-1] != embed_dim.
    """

    def __init__(self, in_features: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        if out.shape[-1] != self.embed_dim:
            raise ValueError(
                f"MLPProjectionHead output dim {out.shape[-1]} != "
                f"expected embed_dim {self.embed_dim}"
            )
        return out


# ---------------------------------------------------------------------------
# ImageEncoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """ResNet-style image encoder.

    Args:
        config: Object with attributes:
            config.image_encoder.stem_channels  (int, default 64)
            config.image_encoder.stage_channels (list[int], 4 elements)
            config.image_encoder.blocks_per_stage (list[int], 4 elements)
            config.projection_head              ("linear" | "mlp")
            config.embed_dim                    (int)

    Forward:
        x: [N, 3, 224, 224]  →  [N, embed_dim] L2-normalized

    Raises:
        ValueError: if x.shape[-2:] != (224, 224)
    """

    def __init__(self, config) -> None:
        super().__init__()

        ie_cfg = config.image_encoder
        stem_channels: int = ie_cfg.stem_channels
        stage_channels: list = list(ie_cfg.stage_channels)
        blocks_per_stage: list = list(ie_cfg.blocks_per_stage)
        proj_head_type: str = config.projection_head
        embed_dim: int = config.embed_dim

        # --- Stem ---
        # Conv2d(3, stem_channels, 7, stride=2, padding=3) → BN → ReLU → MaxPool2d(3, s=2, p=1)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # --- Residual Stages ---
        # Stage 0: stem_channels → stage_channels[0], no spatial downsampling
        # Stage 1: stage_channels[0] → stage_channels[1], first block stride=2
        # Stage 2: stage_channels[1] → stage_channels[2], first block stride=2
        # Stage 3: stage_channels[2] → stage_channels[3], first block stride=2
        stages = []
        in_ch = stem_channels
        for i, (out_ch, n_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
            first_stride = 2 if i > 0 else 1
            blocks = [ResidualBlock(in_ch, out_ch, stride=first_stride)]
            for _ in range(1, n_blocks):
                blocks.append(ResidualBlock(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)

        # --- Global average pool ---
        self.pool = nn.AdaptiveAvgPool2d(1)

        # --- Projection head ---
        # Input features to projection head = last stage output channels
        proj_in_features = stage_channels[-1]

        if proj_head_type == "linear":
            self.projection_head = LinearProjectionHead(proj_in_features, embed_dim)
        elif proj_head_type == "mlp":
            self.projection_head = MLPProjectionHead(proj_in_features, embed_dim)
        else:
            raise ValueError(
                f"Unknown projection_head type '{proj_head_type}'. "
                "Expected 'linear' or 'mlp'."
            )

        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape [N, 3, 224, 224].

        Returns:
            [N, embed_dim] L2-normalized embeddings.

        Raises:
            ValueError: if spatial dims of x are not (224, 224).
        """
        if x.shape[-2:] != (224, 224):
            raise ValueError(
                f"ImageEncoder expects input spatial size (224, 224), "
                f"but got {tuple(x.shape[-2:])}. Input tensor shape: {tuple(x.shape)}"
            )

        # Stem
        out = self.stem(x)

        # Residual stages
        for stage in self.stages:
            out = stage(out)

        # Global average pool → [N, C, 1, 1] → [N, C]
        out = self.pool(out)
        out = out.flatten(1)

        # Projection head
        out = self.projection_head(out)

        # L2 normalize
        out = F.normalize(out, p=2, dim=-1)

        return out
