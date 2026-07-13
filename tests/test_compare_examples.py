"""
tests/test_compare_examples.py — Unit tests for CompareTool retrieval logic

Validates: Requirements 20.3, 20.6, 20.8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from compare import CompareTool


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 32
N = 3

val_image_embeds = F.normalize(torch.randn(N, DIM), dim=-1)
val_text_embeds  = F.normalize(torch.randn(N, DIM), dim=-1)
val_captions     = ["caption 0", "caption 1", "caption 2"]


def _make_model():
    """Minimal mock model: encode_text and encode_image return L2-normalized vectors."""
    def encode_text(captions):
        return F.normalize(torch.randn(len(captions), DIM), dim=-1)

    def encode_image(images):
        batch = images.shape[0] if images.dim() == 4 else 1
        return F.normalize(torch.randn(batch, DIM), dim=-1)

    model = SimpleNamespace(
        encode_text=encode_text,
        encode_image=encode_image,
        eval=lambda: None,
    )
    return model


def _make_tool(val_images=None):
    return CompareTool(
        model=_make_model(),
        val_image_embeds=val_image_embeds,
        val_text_embeds=val_text_embeds,
        val_captions=val_captions,
        val_images=val_images,
    )


# ---------------------------------------------------------------------------
# Test 1 — k < 1 is rejected (Req 20.8)
# ---------------------------------------------------------------------------

def test_compare_k_less_than_one():
    """K < 1 raises ValueError with the required message; no results returned.

    Validates: Requirements 20.8
    """
    tool = _make_tool()

    # text_to_images: k=0 and k=-1
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="K must be at least 1"):
            tool.text_to_images("a photo of a dog", k=bad_k)

    # image_to_captions: k=0 and k=-1
    dummy_image = torch.randn(3, 224, 224)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="K must be at least 1"):
            tool.image_to_captions(dummy_image, k=bad_k)


# ---------------------------------------------------------------------------
# Test 2 — k > pool size returns all candidates ranked descending (Req 20.3, 20.6)
# ---------------------------------------------------------------------------

def test_compare_k_exceeds_pool():
    """k > pool size returns all N candidates, scores descending.

    Validates: Requirements 20.3, 20.6
    """
    tool = _make_tool()

    # --- text_to_images ---
    t2i_results = tool.text_to_images("some query", k=100)

    assert len(t2i_results) == N, "Should return all pool images when k > pool"
    scores_t2i = [r["score"] for r in t2i_results]
    assert scores_t2i == sorted(scores_t2i, reverse=True), \
        "text_to_images results must be sorted by score descending"

    # --- image_to_captions ---
    dummy_image = torch.randn(3, 224, 224)
    i2c_results = tool.image_to_captions(dummy_image, k=100)

    assert len(i2c_results) == N, "Should return all pool captions when k > pool"
    scores_i2c = [r["score"] for r in i2c_results]
    assert scores_i2c == sorted(scores_i2c, reverse=True), \
        "image_to_captions results must be sorted by score descending"

    # Verify caption text is populated correctly
    for result in i2c_results:
        assert result["caption"] in val_captions


# ---------------------------------------------------------------------------
# Test 3 — save failure propagates; previously saved results untouched (Req 20.8)
# ---------------------------------------------------------------------------

def test_compare_save_failure(tmp_path, monkeypatch):
    """save_image_grid propagates IOError on failure; a previously saved file is untouched.

    Validates: Requirements 20.8
    """
    import torch

    # Build tiny RGB tensor images [3, H, W] so the grid renderer has something to show.
    dummy_images = [torch.rand(3, 8, 8) for _ in range(N)]
    tool = _make_tool(val_images=dummy_images)

    # Manufacture results that reference valid image indices.
    results = [{"rank": i + 1, "image_idx": i, "score": float(N - i)} for i in range(N)]

    # --- First save: must succeed ---
    first_path = str(tmp_path / "grid_ok.png")
    tool.save_image_grid(results, first_path)
    assert os.path.exists(first_path), "First save should create the file"
    first_mtime = os.path.getmtime(first_path)

    # --- Second save: simulate IOError via monkeypatching fig.savefig ---
    import matplotlib.figure as mpl_figure

    original_savefig = mpl_figure.Figure.savefig

    def failing_savefig(self, *args, **kwargs):
        raise IOError("Simulated disk write failure")

    monkeypatch.setattr(mpl_figure.Figure, "savefig", failing_savefig)

    second_path = str(tmp_path / "grid_fail.png")
    with pytest.raises(IOError, match="Simulated disk write failure"):
        tool.save_image_grid(results, second_path)

    # The failed-target file must not have been created.
    assert not os.path.exists(second_path), "Failed save must not create the target file"

    # The first (already saved) file must be completely untouched.
    assert os.path.getmtime(first_path) == first_mtime, \
        "Previously saved file must not be modified after a failed second save"
