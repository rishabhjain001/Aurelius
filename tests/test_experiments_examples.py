"""
tests/test_experiments_examples.py — Unit tests for the experiment runner.

Covers:
    - RuntimeEstimator confirmation gate (Req 3.4, 3.5)
    - Failed experiment result file schema (Req 11.4)
    - Partial experiment result file schema (Req 11.5)
    - Exp 1 failure => Exp 2 skipped with failed status (Req 22.4, 23.6)
    - ReproducibilityLogger snapshot unavailability handling (Req 22.9)
    - Git commit hash literal "unavailable" fallback (Req 22.4)

Validates: Requirements 3.4, 3.5, 11.4, 11.5, 22.4, 22.9, 23.5, 23.6
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import pytest

from experiments import ExperimentRunner, ReproducibilityLogger, RuntimeEstimator


# ---------------------------------------------------------------------------
# Minimal config
# ---------------------------------------------------------------------------

def _minimal_config() -> SimpleNamespace:
    """Minimal config sized for fast ExperimentRunner construction."""
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
        embed_dim=32,
        temperature_init=0.07,
        learning_rate=1e-3,
        weight_decay=0.01,
        epochs=1,
        batch_size=2,
        dataset="flickr30k",
        val_split_seed=42,
        val_split_fraction=0.1,
        image_size=224,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        augmentation=SimpleNamespace(
            weak=SimpleNamespace(
                random_horizontal_flip=True,
                color_jitter={
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "saturation": 0.1,
                    "hue": 0.05,
                },
            ),
            strong=SimpleNamespace(
                random_horizontal_flip=True,
                color_jitter={
                    "brightness": 0.4,
                    "contrast": 0.4,
                    "saturation": 0.4,
                    "hue": 0.1,
                },
                random_grayscale=0.2,
                random_resized_crop={"scale": [0.08, 1.0]},
                gaussian_blur={"sigma": [0.1, 2.0]},
            ),
        ),
        checkpoint_interval=100,
        runtime_budget_hours=12,
        memory_budget_gb=16,
        experiment_seeds=SimpleNamespace(
            experiment_1=0,
            experiment_2=0,
            experiment_3=0,
            experiment_4=0,
            experiment_5=0,
            experiment_6=0,
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: Runtime confirmation gate
# ---------------------------------------------------------------------------

def test_runtime_gate_aborts_without_confirmation():
    """`RuntimeEstimator.confirm` accepts only responses starting with y/Y.

    Validates: Requirements 3.4, 3.5
    """
    estimator = RuntimeEstimator()

    # Non-affirmative responses must abort.
    assert estimator.confirm(3600.0, input_fn=lambda prompt: "n") is False
    assert estimator.confirm(3600.0, input_fn=lambda prompt: "") is False
    assert estimator.confirm(3600.0, input_fn=lambda prompt: "no") is False
    assert estimator.confirm(3600.0, input_fn=lambda prompt: "N") is False

    # Only y/Y prefixed responses proceed.
    assert estimator.confirm(3600.0, input_fn=lambda prompt: "y") is True
    assert estimator.confirm(3600.0, input_fn=lambda prompt: "Y") is True
    assert estimator.confirm(3600.0, input_fn=lambda prompt: "yes") is True


# ---------------------------------------------------------------------------
# Test 2: Failed experiment writes null metrics
# ---------------------------------------------------------------------------

def test_failed_experiment_writes_null_metrics(tmp_path):
    """A failed experiment records status='failed' with all metrics null.

    Validates: Requirements 11.4
    """
    config = _minimal_config()
    runner = ExperimentRunner(config)

    failed_result = {
        "experiment": 1,
        "status": "failed",
        "failure_reason": "test failure",
        "steps": [],
        "retrieval": None,
        "final_loss": None,
        "final_temperature": None,
    }

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        runner.write_result(1, failed_result)

        result_path = tmp_path / "results" / "experiment1.json"
        assert result_path.exists()

        with result_path.open("r", encoding="utf-8") as handle:
            written = json.load(handle)

        assert written["status"] == "failed"
        assert written["failure_reason"] == "test failure"
        assert written["retrieval"] is None
        assert written["final_loss"] is None
        assert written["final_temperature"] is None
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Test 3: Partial experiment preserves collected measurements
# ---------------------------------------------------------------------------

def test_partial_experiment_preserves_measurements(tmp_path):
    """A partial experiment retains step records but nulls unavailable finals.

    Validates: Requirements 11.5
    """
    config = _minimal_config()
    runner = ExperimentRunner(config)

    partial_steps = [
        {"step": 1, "loss": 2.30, "temperature": 0.07},
        {"step": 2, "loss": 2.10, "temperature": 0.07},
        {"step": 3, "loss": 1.95, "temperature": 0.07},
    ]
    partial_result = {
        "experiment": 1,
        "status": "partial",
        "failure_reason": "budget exceeded",
        "steps": partial_steps,
        "retrieval": None,
        "final_loss": None,
        "final_temperature": None,
    }

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        runner.write_result(1, partial_result)

        result_path = tmp_path / "results" / "experiment1.json"
        assert result_path.exists()

        with result_path.open("r", encoding="utf-8") as handle:
            written = json.load(handle)

        assert written["status"] == "partial"
        assert written["failure_reason"] == "budget exceeded"
        # Collected measurements preserved.
        assert written["steps"] == partial_steps
        # Unavailable final metrics null.
        assert written["retrieval"] is None
        assert written["final_loss"] is None
        assert written["final_temperature"] is None
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Test 4: Exp 1 failure => Exp 2 written as failed
# ---------------------------------------------------------------------------

def test_exp1_failure_skips_exp2(tmp_path):
    """When Exp 1 fails, `run_experiment_2(None)` writes a failed result.

    Validates: Requirements 22.4, 23.6
    """
    config = _minimal_config()
    runner = ExperimentRunner(config)

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        runner.run_experiment_2(None)

        result_path = tmp_path / "results" / "experiment2.json"
        assert result_path.exists()

        with result_path.open("r", encoding="utf-8") as handle:
            written = json.load(handle)

        assert written["experiment"] == 2
        assert written["status"] == "failed"
        assert written["failure_reason"] is not None
        assert "Experiment 1 model unavailable" in written["failure_reason"]

        # All retrieval metrics null.
        assert written["retrieval"]["trained"] is None
        assert written["retrieval"]["random_baseline"] is None

        # Metadata file also written.
        metadata_path = tmp_path / "results" / "experiment2_metadata.json"
        assert metadata_path.exists()
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Test 5: Unavailable reproducibility field
# ---------------------------------------------------------------------------

def test_metadata_unavailable_field(monkeypatch):
    """Unavailable fields are recorded as 'unavailable' with a reason string.

    Validates: Requirements 22.9
    """
    real_check_output = subprocess.check_output

    def fail_git(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and cmd[0] == "git":
            raise FileNotFoundError("git not found")
        return real_check_output(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", fail_git)

    logger = ReproducibilityLogger()
    config = _minimal_config()
    seeds = {"torch": 0, "numpy": 0, "random": 0}
    metadata = logger.snapshot(config, seeds)

    # Unavailable git commit recorded as literal string with a reason.
    assert metadata["git_commit"] == "unavailable"
    assert "git_commit_unavailable_reason" in metadata
    assert isinstance(metadata["git_commit_unavailable_reason"], str)
    assert metadata["git_commit_unavailable_reason"] != ""

    # git_dirty likewise recorded as null with a reason (parallel unavailability).
    assert metadata["git_dirty"] is None
    assert "git_dirty_unavailable_reason" in metadata


# ---------------------------------------------------------------------------
# Test 6: Git unavailable hash literal
# ---------------------------------------------------------------------------

def test_git_unavailable_hash(monkeypatch):
    """When git rev-parse fails, `git_commit` is literally 'unavailable'.

    Validates: Requirements 22.4
    """
    real_check_output = subprocess.check_output

    def fail_git(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and cmd[0] == "git":
            raise FileNotFoundError("git not found")
        return real_check_output(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", fail_git)

    logger = ReproducibilityLogger()
    config = _minimal_config()
    seeds = {"torch": 0, "numpy": 0, "random": 0}
    metadata = logger.snapshot(config, seeds)

    assert metadata["git_commit"] == "unavailable"
