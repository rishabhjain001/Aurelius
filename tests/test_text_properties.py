"""
tests/test_text_properties.py — Property-based tests for text encoder.

Covers:
    Property 2: Text Embedding L2 Norm

Validates: Requirements 5.7, 19.2, 19.10
"""

import sys
import os

# Ensure project root is on the path so `encoders.text` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import torch
from hypothesis import given, settings, strategies as st

from encoders.text import TextEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_cfg() -> SimpleNamespace:
    """Return a minimal TextEncoder config (small d_model for speed)."""
    return SimpleNamespace(
        text_encoder=SimpleNamespace(
            vocab_size=50257,
            max_seq_len=77,
            pad_id=0,
            d_model=64,
            num_heads=4,
            num_layers=2,
            ffn_multiplier=4,
        ),
        projection_head="mlp",
        embed_dim=32,
    )


# ---------------------------------------------------------------------------
# Property 2: Text Embedding L2 Norm
# Feature: aurelius-clip-vlm, Property 2: Text Embedding L2 Norm
# ---------------------------------------------------------------------------

@given(captions=st.lists(st.text(), min_size=1, max_size=4))
@settings(max_examples=100, deadline=None)
def test_text_embedding_l2_norm(captions) -> None:
    # Feature: aurelius-clip-vlm, Property 2: Text Embedding L2 Norm
    #
    # For any batch of caption strings (empty, whitespace, very long,
    # non-ASCII), every TextEncoder output shall have L2 norm within
    # 1e-4 of 1.0.
    #
    # Validates: Requirements 5.7, 19.2, 19.10
    cfg = _minimal_cfg()
    model = TextEncoder(cfg)
    model.eval()

    with torch.no_grad():
        embeddings = model(captions)  # [N, embed_dim]

    norms = torch.linalg.norm(embeddings, dim=-1)  # [N]
    assert embeddings.shape[0] == len(captions), (
        f"Expected {len(captions)} embeddings, got shape {tuple(embeddings.shape)}"
    )
    assert torch.all(torch.abs(norms - 1.0) < 1e-4), (
        f"Expected all L2 norms within 1e-4 of 1.0, got norms: {norms.tolist()}"
    )
