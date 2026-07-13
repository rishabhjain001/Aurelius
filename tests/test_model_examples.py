"""
tests/test_model_examples.py — Unit tests for model composition (InfoNCEModule, CLIPModel)

Validates: Requirements 6.8, 7.5
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model import CLIPModel, InfoNCEModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_config(projection_head: str = "mlp") -> SimpleNamespace:
    """Minimal CLIPModel config for fast construction during tests."""
    return SimpleNamespace(
        image_encoder=SimpleNamespace(
            stem_channels=16,
            stage_channels=[16, 32, 64, 128],
            blocks_per_stage=[1, 1, 1, 1],
        ),
        text_encoder=SimpleNamespace(
            vocab_size=50257,
            max_seq_len=77,
            pad_id=0,
            d_model=64,
            num_heads=4,
            num_layers=1,
            ffn_multiplier=4,
        ),
        projection_head=projection_head,
        embed_dim=32,
        temperature_init=0.07,
    )


# ---------------------------------------------------------------------------
# Test 1: InfoNCE symmetry — combined loss equals (loss_i2t + loss_t2i) / 2
# ---------------------------------------------------------------------------

def test_infonce_symmetry():
    """InfoNCEModule.forward returns the exact average of i2t and t2i CE losses.

    Constructs L2-normalized random embeddings, calls forward() to get the
    combined scalar loss, then manually recomputes both individual losses and
    checks the combined loss matches (loss_i2t + loss_t2i) / 2 within 1e-6.

    Validates: Requirement 7.5
    """
    torch.manual_seed(0)

    N = 8
    embed_dim = 32
    temperature_init = 0.07

    module = InfoNCEModule(temperature_init=temperature_init)
    module.eval()

    # Random L2-normalized embeddings
    image_embeds = F.normalize(torch.randn(N, embed_dim), p=2, dim=-1)
    text_embeds = F.normalize(torch.randn(N, embed_dim), p=2, dim=-1)

    # Forward pass
    actual_loss, similarity_matrix = module(image_embeds, text_embeds)

    # Manually reproduce the computation using the module's own temperature
    temperature = module.log_temperature.exp().detach()
    scaled = (image_embeds @ text_embeds.T) / temperature
    targets = torch.arange(N)

    loss_i2t = F.cross_entropy(scaled, targets)
    loss_t2i = F.cross_entropy(scaled.T, targets)
    expected_loss = (loss_i2t + loss_t2i) / 2.0

    assert torch.allclose(actual_loss, expected_loss, atol=1e-6), (
        f"Combined loss {actual_loss.item():.8f} != "
        f"(loss_i2t + loss_t2i) / 2 = {expected_loss.item():.8f}"
    )


# ---------------------------------------------------------------------------
# Test 2: CLIPModel raises ValueError for invalid projection head variant
# ---------------------------------------------------------------------------

def test_same_head_variant_per_run():
    """CLIPModel raises ValueError for an unsupported projection_head value.

    Confirms that:
      - projection_head="transformer" raises ValueError immediately.
      - projection_head="linear" and projection_head="mlp" construct without error.

    Validates: Requirement 6.8
    """
    # Invalid variant must raise
    invalid_cfg = _clip_config(projection_head="transformer")
    with pytest.raises(ValueError):
        CLIPModel(invalid_cfg)

    # Valid variants must not raise
    linear_model = CLIPModel(_clip_config(projection_head="linear"))
    assert linear_model is not None

    mlp_model = CLIPModel(_clip_config(projection_head="mlp"))
    assert mlp_model is not None
