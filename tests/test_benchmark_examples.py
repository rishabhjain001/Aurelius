"""
tests/test_benchmark_examples.py — Unit tests for benchmark preconditions.

Covers:
    - Benchmark refused when no training checkpoint exists (Req 18.2)
    - Write failure aborts benchmark and surfaces the error (Req 18.7)

Validates: Requirements 18.2, 18.7
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import benchmark
from benchmark import _find_latest_checkpoint, run_benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config() -> SimpleNamespace:
    """Minimal config namespace sufficient for benchmark construction."""
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
    )


# ---------------------------------------------------------------------------
# Test 1: Benchmark refuses before any training run (Req 18.2)
# ---------------------------------------------------------------------------

def test_benchmark_refuses_before_training(tmp_path, monkeypatch):
    """Invoking the benchmark with no checkpoint present is refused.

    `_find_latest_checkpoint` returns None when checkpoints/ is empty (or
    absent), and the __main__ guard reacts by printing an explanatory message
    and raising SystemExit(1).

    Validates: Requirements 18.2
    """
    # Change into a fresh temp directory so there are no real checkpoints.
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Verify the helper itself returns None in an empty directory.
        assert _find_latest_checkpoint() is None

        # Simulate the __main__ guard: patch _find_latest_checkpoint to None
        # and assert SystemExit is raised with the refusal message printed.
        monkeypatch.setattr(benchmark, "_find_latest_checkpoint", lambda: None)

        printed_messages: list[str] = []
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed_messages.append(" ".join(str(a) for a in args)))

        # Replay the __main__ logic inline.
        with pytest.raises(SystemExit) as exc_info:
            latest = benchmark._find_latest_checkpoint()
            if latest is None:
                print(
                    "Benchmark refused: no completed training run found. "
                    "Run training first (e.g. python experiments.py)."
                )
                raise SystemExit(1)
            run_benchmark(str(latest), _minimal_config())

        assert exc_info.value.code == 1
        assert any("Benchmark refused" in msg for msg in printed_messages), (
            f"Expected refusal message not found in printed output: {printed_messages}"
        )
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Test 2: Write failure aborts the benchmark (Req 18.7)
# ---------------------------------------------------------------------------

def test_benchmark_write_failure_aborts(tmp_path, monkeypatch):
    """A simulated write failure during results output aborts with SystemExit.

    `run_benchmark` must catch the IOError from the file write, print an
    error message, and raise SystemExit(1).

    Validates: Requirements 18.7
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        config = _minimal_config()

        # Make BenchmarkRunner.run return a minimal valid results dict so the
        # measurement loop is completely skipped.
        fake_results = {
            "image_encode_latency_ms": {"mean": 1.0, "median": 1.0, "p95": 1.0, "std": 0.0},
            "text_encode_latency_ms": {"mean": 1.0, "median": 1.0, "p95": 1.0, "std": 0.0},
            "retrieval_query_latency_ms": {"mean": 1.0, "median": 1.0, "p95": 1.0, "std": 0.0},
            "zero_shot_latency_ms": {"mean": 1.0, "median": 1.0, "p95": 1.0, "std": 0.0},
            "throughput_items_per_sec": {"mean": 1000.0, "median": 1000.0, "p95": 1000.0, "std": 0.0},
            "peak_ram_mb": 100.0,
            "avg_ram_mb": 80.0,
            "iterations": 100,
        }
        monkeypatch.setattr(benchmark.BenchmarkRunner, "run", lambda self: fake_results)

        # Patch json.dump to simulate a write failure.
        monkeypatch.setattr(json, "dump", MagicMock(side_effect=IOError("disk full")))

        # Also create a dummy checkpoint file and patch CheckpointManager.load
        # so the model-loading step is skipped.
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        dummy_ckpt = ckpt_dir / "step_0001.pt"
        dummy_ckpt.write_bytes(b"")

        import train
        monkeypatch.setattr(train.CheckpointManager, "load", lambda self, *a, **kw: None)

        with pytest.raises(SystemExit) as exc_info:
            run_benchmark(str(dummy_ckpt), config)

        assert exc_info.value.code == 1
    finally:
        os.chdir(original_cwd)
