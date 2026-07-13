"""
tests/test_image_properties.py — Property-based tests for image encoder.

Covers:
    Property 1: Image Embedding L2 Norm
    Property 9: Projection Head Output Dimension

Validates: Requirements 4.5, 6.6, 6.7, 19.1, 19.9, 19.10
"""

import sys
import os

# Ensure project root is on the path so `encoders.image` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import torch
import pytest
from hypothesis import given, settings, strategies as st

from encoders.image import ImageEncoder, LinearProjectionHead, MLPProjectionHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_cfg(embed_dim: int = 128) -> SimpleNamespace:
    """Return a minimal ImageEncoder config (small channels for speed)."""
    return SimpleNamespace(
        image_encoder=SimpleNamespace(
            stem_channels=16,
            stage_channels=[16, 32, 64, 128],
            blocks_per_stage=[1, 1, 1, 1],
        ),
        projection_head="mlp",
        embed_dim=embed_dim,
    )


# ---------------------------------------------------------------------------
# Property 1: Image Embedding L2 Norm
# Feature: aurelius-clip-vlm, Property 1: Image Embedding L2 Norm
# ---------------------------------------------------------------------------

@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_image_embedding_l2_norm(batch_size: int, seed: int) -> None:
    # Feature: aurelius-clip-vlm, Property 1: Image Embedding L2 Norm
    #
    # For any batch of valid 224×224 images, every ImageEncoder output
    # shall have L2 norm within 1e-4 of 1.0.
    #
    # Validates: Requirements 4.5, 19.1
    cfg = _minimal_cfg(embed_dim=128)
    model = ImageEncoder(cfg)
    model.eval()

    rng = torch.Generator()
    rng.manual_seed(seed)
    x = torch.randn(batch_size, 3, 224, 224, generator=rng)

    with torch.no_grad():
        embeddings = model(x)  # [N, embed_dim]

    # Every embedding must be L2-unit-norm.
    norms = torch.linalg.norm(embeddings, dim=-1)  # [N]
    assert embeddings.shape[0] == batch_size
    assert torch.all(torch.abs(norms - 1.0) < 1e-4), (
        f"Expected all L2 norms within 1e-4 of 1.0, got norms: {norms.tolist()}"
    )


# ---------------------------------------------------------------------------
# Property 9: Projection Head Output Dimension
# Feature: aurelius-clip-vlm, Property 9: Projection Head Output Dimension
# ---------------------------------------------------------------------------

_EMBED_DIMS = [128, 256, 512]
_HEAD_CLASSES = [LinearProjectionHead, MLPProjectionHead]


@given(
    embed_dim=st.sampled_from(_EMBED_DIMS),
    head_cls=st.sampled_from(_HEAD_CLASSES),
    batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_projection_head_output_dimension(embed_dim, head_cls, batch_size, seed) -> None:
    # Feature: aurelius-clip-vlm, Property 9: Projection Head Output Dimension
    #
    # For every combination of embed_dim ∈ {128, 256, 512} and head variant
    # ∈ {LinearProjectionHead, MLPProjectionHead}, the projection head output
    # shape shall be [N, embed_dim].
    #
    # Validates: Requirements 6.6, 6.7, 19.9
    in_features = 128  # arbitrary fixed input width

    head = head_cls(in_features=in_features, embed_dim=embed_dim)
    head.eval()

    rng = torch.Generator()
    rng.manual_seed(seed)
    x = torch.randn(batch_size, in_features, generator=rng)

    with torch.no_grad():
        out = head(x)

    assert out.shape == (batch_size, embed_dim), (
        f"{head_cls.__name__}(in={in_features}, embed_dim={embed_dim}) "
        f"with batch_size={batch_size}: expected shape {(batch_size, embed_dim)}, "
        f"got {tuple(out.shape)}"
    )
