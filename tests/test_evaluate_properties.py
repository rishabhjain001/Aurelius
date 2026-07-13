"""
tests/test_evaluate_properties.py — Property-based tests for the retrieval evaluator.

Validates deterministic ranking behavior of `retrieval_metrics` using Hypothesis.

Properties covered:
    8. Retrieval Ranking Determinism
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from evaluate import DirectionalMetrics, retrieval_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeds(n_images: int, captions_per_image: int, embed_dim: int, seed: int):
    """Build L2-normalized image/text embeddings and a caption→image mapping."""
    torch.manual_seed(seed)
    image_embeds = F.normalize(torch.randn(n_images, embed_dim), p=2, dim=-1)
    n_texts = n_images * captions_per_image
    text_embeds = F.normalize(torch.randn(n_texts, embed_dim), p=2, dim=-1)
    caption_to_image = [i // captions_per_image for i in range(n_texts)]
    return image_embeds, text_embeds, caption_to_image


# ---------------------------------------------------------------------------
# Property 8: Retrieval Ranking Determinism
# ---------------------------------------------------------------------------

# Feature: aurelius-clip-vlm, Property 8: Retrieval Ranking Determinism
@settings(max_examples=100, deadline=None)
@given(
    n_images=st.integers(min_value=1, max_value=10),
    captions_per_image=st.integers(min_value=1, max_value=5),
    embed_dim=st.integers(min_value=4, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_retrieval_ranking_determinism(n_images, captions_per_image, embed_dim, seed):
    """For fixed inputs and a fixed seed, retrieval metrics are identical across runs.

    Validates: Requirements 8.5, 19.8, 19.10
    """
    image_embeds, text_embeds, caption_to_image = _make_embeds(
        n_images, captions_per_image, embed_dim, seed
    )

    result_a = retrieval_metrics(image_embeds, text_embeds, caption_to_image)
    result_b = retrieval_metrics(image_embeds, text_embeds, caption_to_image)

    assert result_a.skipped == result_b.skipped
    if result_a.skipped:
        return

    for direction in ("i2t", "t2i"):
        m_a: DirectionalMetrics = getattr(result_a, direction)
        m_b: DirectionalMetrics = getattr(result_b, direction)
        assert m_a.r1 == m_b.r1
        assert m_a.r5 == m_b.r5
        assert m_a.r10 == m_b.r10
        assert m_a.mrr == m_b.mrr
        assert m_a.mean_rank == m_b.mean_rank
        assert m_a.median_rank == m_b.median_rank
