"""
tests/test_train_integration.py — Integration tests for the training pipeline.

Covers:
    - CheckpointManager round-trip (weights, temperature, RNG state)
    - Single training step reduces loss on a fixed synthetic batch
    - Structural config mismatch on load raises descriptive ValueError
    - Corrupt checkpoint file on load raises descriptive RuntimeError

Validates: Requirements 24.1, 24.2, 24.4, 24.5, 24.6
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import pytest
import torch
from torch.optim import AdamW

from model import CLIPModel
from train import CheckpointManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(embed_dim: int = 32) -> SimpleNamespace:
    """Minimal CLIPModel config sized for fast test construction."""
    return SimpleNamespace(
        image_encoder=SimpleNamespace(
            stem_channels=8,
            stage_channels=[8, 16, 32, 64],
            blocks_per_stage=[1, 1, 1, 1],
        ),
        text_encoder=SimpleNamespace(
            vocab_size=50257,
            max_seq_len=77,
            pad_id=0,
            d_model=32,
            num_heads=2,
            num_layers=1,
            ffn_multiplier=2,
        ),
        projection_head="mlp",
        embed_dim=embed_dim,
        temperature_init=0.07,
    )


# ---------------------------------------------------------------------------
# Test 1: Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    """Save and load a checkpoint; every tensor, the temperature, and the
    torch RNG state must be restored exactly.

    Validates: Requirements 24.1, 24.2
    """
    torch.manual_seed(42)

    config = make_config()
    model = CLIPModel(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(str(ckpt_dir))

    # Snapshot state that the checkpoint should restore.
    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved_temp = model.loss_module.log_temperature.detach().exp().item()
    saved_rng = torch.get_rng_state().clone()

    path = manager.save(
        model, optimizer, step=100, experiment_id="roundtrip", config=config
    )
    assert os.path.exists(path)

    # Perturb the torch RNG so we can prove load restores it.
    _ = torch.randn(16)
    perturbed_rng = torch.get_rng_state().clone()
    assert not torch.equal(perturbed_rng, saved_rng)

    # Fresh model + optimizer; different seed guarantees different weights.
    torch.manual_seed(999)
    fresh_model = CLIPModel(config)
    fresh_optimizer = AdamW(fresh_model.parameters(), lr=1e-3)

    # A separately-constructed manager pointing at the same directory.
    fresh_manager = CheckpointManager(str(ckpt_dir))
    loaded_step = fresh_manager.load(path, fresh_model, fresh_optimizer, config)

    assert loaded_step == 100

    # Every state dict tensor must match exactly.
    loaded_state = fresh_model.state_dict()
    assert set(loaded_state.keys()) == set(saved_state.keys())
    for key, expected in saved_state.items():
        actual = loaded_state[key]
        assert torch.equal(actual, expected), f"state dict mismatch for '{key}'"

    # Temperature (exp of log_temperature) must match.
    loaded_temp = fresh_model.loss_module.log_temperature.detach().exp().item()
    assert loaded_temp == pytest.approx(saved_temp, abs=1e-6)

    # Torch RNG state must be restored to the pre-save snapshot.
    restored_rng = torch.get_rng_state()
    assert torch.equal(restored_rng, saved_rng), "torch RNG state not restored"


# ---------------------------------------------------------------------------
# Test 2: One training step reduces the loss on a fixed batch
# ---------------------------------------------------------------------------

def test_training_step_reduces_loss():
    """With a fixed seed, tiny model, and small synthetic batch, one AdamW
    step should reduce the loss measured on the same batch.
    """
    torch.manual_seed(0)

    config = make_config()
    model = CLIPModel(config)
    model.train()

    # Small LR so the very first AdamW step (which has an effective
    # magnitude of roughly LR on every parameter) does not overshoot.
    optimizer = AdamW(model.parameters(), lr=1e-4)

    images = torch.randn(4, 3, 224, 224)
    captions = ["a cat", "a dog", "a bird", "a car"]

    # Forward + backward + step
    optimizer.zero_grad()
    initial_loss, _ = model(images, captions)
    initial_value = initial_loss.item()
    initial_loss.backward()
    optimizer.step()

    # Recompute loss on the same batch after the parameter update.
    with torch.no_grad():
        new_loss, _ = model(images, captions)
    new_value = new_loss.item()

    assert new_value < initial_value, (
        f"loss did not decrease after one step: "
        f"initial={initial_value:.6f}, new={new_value:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Structural mismatch on load raises ValueError
# ---------------------------------------------------------------------------

def test_checkpoint_mismatch_error(tmp_path):
    """Saving with embed_dim=32 then loading with embed_dim=64 must raise
    a ValueError whose message identifies the mismatching field.

    Validates: Requirement 24.5
    """
    ckpt_dir = tmp_path / "checkpoints"

    save_config = make_config(embed_dim=32)
    save_model = CLIPModel(save_config)
    save_optimizer = AdamW(save_model.parameters(), lr=1e-3)

    manager = CheckpointManager(str(ckpt_dir))
    path = manager.save(
        save_model, save_optimizer, step=0, experiment_id="mismatch", config=save_config
    )

    load_config = make_config(embed_dim=64)
    load_model = CLIPModel(load_config)
    load_optimizer = AdamW(load_model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match=r"embed_dim"):
        manager.load(path, load_model, load_optimizer, load_config)


# ---------------------------------------------------------------------------
# Test 4: Corrupt checkpoint file on load raises RuntimeError
# ---------------------------------------------------------------------------

def test_checkpoint_corrupt_error(tmp_path):
    """Loading a file that is not a valid serialized checkpoint must raise
    a descriptive RuntimeError.

    Validates: Requirement 24.6
    """
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    corrupt_path = ckpt_dir / "corrupt_step00000000.pt"
    corrupt_path.write_bytes(b"this is not a real torch checkpoint" * 64)

    config = make_config()
    model = CLIPModel(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(str(ckpt_dir))

    with pytest.raises(RuntimeError, match=r"[Ff]ailed to load"):
        manager.load(corrupt_path, model, optimizer, config)
