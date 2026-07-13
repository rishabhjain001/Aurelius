"""
tests/test_image_examples.py — Unit tests for ImageEncoder edge cases

Validates: Requirements 4.7, 6.7
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from encoders.image import ImageEncoder, LinearProjectionHead, MLPProjectionHead


# ---------------------------------------------------------------------------
# Minimal config fixture shared by both tests
# ---------------------------------------------------------------------------

def _minimal_config():
    return SimpleNamespace(
        image_encoder=SimpleNamespace(
            stem_channels=16,
            stage_channels=[16, 32, 64, 128],
            blocks_per_stage=[1, 1, 1, 1],
        ),
        projection_head="mlp",
        embed_dim=128,
    )


# ---------------------------------------------------------------------------
# Test 1: ImageEncoder.forward raises ValueError for non-224×224 input
# ---------------------------------------------------------------------------

def test_wrong_image_shape():
    """ImageEncoder.forward must raise ValueError when spatial dims != (224, 224).

    Validates: Requirement 4.7
    """
    cfg = _minimal_config()
    encoder = ImageEncoder(cfg)
    encoder.eval()

    wrong_input = torch.randn(1, 3, 128, 128)

    with pytest.raises(ValueError):
        encoder.forward(wrong_input)


# ---------------------------------------------------------------------------
# Test 2: Projection head dim mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_projection_head_dim_mismatch():
    """Projection head forward must raise ValueError when internal linear output
    dim does not match the initialised embed_dim.

    Monkeypatches the internal linear layer so its output channels differ from
    embed_dim, then asserts ValueError is raised.

    Validates: Requirements 4.7, 6.7
    """
    x = torch.randn(2, 512)

    # --- LinearProjectionHead ---
    linear_head = LinearProjectionHead(in_features=512, embed_dim=256)
    # Corrupt: replace linear so it outputs 128 dims instead of the declared 256
    linear_head.linear = nn.Linear(512, 128)

    with pytest.raises(ValueError):
        linear_head.forward(x)

    # --- MLPProjectionHead ---
    mlp_head = MLPProjectionHead(in_features=512, embed_dim=256)
    # Corrupt: replace the second linear so it outputs 128 dims instead of 256
    mlp_head.linear2 = nn.Linear(512, 128)

    with pytest.raises(ValueError):
        mlp_head.forward(x)
