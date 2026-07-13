"""
tests/test_evaluate_examples.py — Unit tests for retrieval and prompt-set logic

Validates: Requirements 8.5, 8.6, 9.3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

import pytest
import torch
import torch.nn.functional as F

import evaluate
from evaluate import retrieval_metrics


# ---------------------------------------------------------------------------
# Retrieval — empty split (Req 8.6)
# ---------------------------------------------------------------------------

def test_retrieval_skips_empty_split():
    """Empty validation split → retrieval is skipped, no metrics reported."""
    result = retrieval_metrics(
        torch.empty(0, 32),
        torch.empty(0, 32),
        [],
    )

    assert result.skipped is True
    assert result.skip_reason is not None
    assert "empty" in result.skip_reason.lower()
    assert result.i2t is None
    assert result.t2i is None


# ---------------------------------------------------------------------------
# Prompt set frozen after lock (Req 9.3)
# ---------------------------------------------------------------------------

def test_prompt_set_frozen():
    """Once lock_prompt_set() is called, mutation is rejected and set preserved."""
    # Reload the module so this test runs against a fresh (unlocked) state,
    # independent of any earlier test that may have locked the module-level flag.
    fresh = importlib.reload(evaluate)

    initial_state = dict(fresh.PROMPT_SET)

    fresh.lock_prompt_set()

    with pytest.raises(RuntimeError):
        fresh.modify_prompt_set({"newkey": ["a photo of newkey"]})

    assert fresh.PROMPT_SET == initial_state


# ---------------------------------------------------------------------------
# Tie-breaking uses best rank (Req 8.5)
# ---------------------------------------------------------------------------

def test_tie_breaking_uses_best_rank():
    """Identical similarities share the smallest-numbered rank → R@1 == 1.0."""
    # Two images and four captions (two per image). All embeddings are the same
    # unit vector, so every pairwise cosine similarity is 1.0 — a complete tie.
    dim = 8
    base = F.normalize(torch.ones(1, dim), dim=-1)  # [1, D]

    image_embeds = base.expand(2, dim).contiguous()   # [2, D]
    text_embeds = base.expand(4, dim).contiguous()    # [4, D]
    caption_to_image = [0, 0, 1, 1]

    result = retrieval_metrics(image_embeds, text_embeds, caption_to_image)

    assert result.skipped is False
    assert result.i2t is not None and result.t2i is not None
    # Under Req 8.5 tie-breaking, every correct match shares rank 1.
    assert result.i2t.r1 == 1.0
    assert result.t2i.r1 == 1.0
