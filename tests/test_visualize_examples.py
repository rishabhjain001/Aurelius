"""
tests/test_visualize_examples.py — Unit tests for visualization preconditions.

Covers:
    - Missing experiment result file is listed in issues (Req 21.1, 21.10)
    - Missing benchmark.json is listed in issues (Req 21.1, 21.10)
    - Partial-status result with failure_reason satisfies precondition (Req 21.1, 21.10)

Validates: Requirements 21.1, 21.10
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from visualize import check_preconditions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_RESULT = {
    "experiment": 1,
    "status": "completed",
    "failure_reason": None,
    "steps": [{"step": 1, "loss": 2.3, "temperature": 0.07}],
}

_MINIMAL_BENCHMARK = {
    "image_encode_latency_ms": {
        "mean": 1.0,
        "median": 1.0,
        "p95": 1.0,
        "std": 0.0,
    },
    "iterations": 100,
}


def _write_result(results_dir, n: int, data: dict) -> None:
    """Write a result JSON for experiment n into results_dir."""
    path = results_dir / f"experiment{n}.json"
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_benchmark(results_dir, data: dict) -> None:
    """Write benchmark.json into results_dir."""
    path = results_dir / "benchmark.json"
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: missing experiment result file
# ---------------------------------------------------------------------------

def test_visualize_missing_result_file(tmp_path):
    """With experiment3.json missing, check_preconditions rejects and reports it.

    Validates: Requirements 21.1, 21.10
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    _write_benchmark(results_dir, _MINIMAL_BENCHMARK)

    # Write all experiment files except experiment3.json
    for n in range(1, 7):
        if n == 3:
            continue
        _write_result(results_dir, n, {**_MINIMAL_RESULT, "experiment": n})

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ok, issues = check_preconditions()
    finally:
        os.chdir(original_cwd)

    assert ok is False
    assert any("experiment3.json" in issue for issue in issues), (
        f"Expected 'experiment3.json' in issues but got: {issues}"
    )


# ---------------------------------------------------------------------------
# Test 2: missing benchmark.json
# ---------------------------------------------------------------------------

def test_visualize_missing_benchmark(tmp_path):
    """With benchmark.json missing, check_preconditions rejects and reports it.

    Validates: Requirements 21.1, 21.10
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Write all 6 experiment files — no benchmark.json
    for n in range(1, 7):
        _write_result(results_dir, n, {**_MINIMAL_RESULT, "experiment": n})

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ok, issues = check_preconditions()
    finally:
        os.chdir(original_cwd)

    assert ok is False
    assert any("benchmark.json" in issue for issue in issues), (
        f"Expected 'benchmark.json' in issues but got: {issues}"
    )


# ---------------------------------------------------------------------------
# Test 3: partial status with failure_reason is accepted
# ---------------------------------------------------------------------------

def test_visualize_accepts_partial_status(tmp_path):
    """A result file with status='partial' and a failure_reason satisfies the precondition.

    Validates: Requirements 21.1, 21.10
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    _write_benchmark(results_dir, _MINIMAL_BENCHMARK)

    # Write experiments 1–6; experiment 4 has partial status
    for n in range(1, 7):
        if n == 4:
            data = {
                "experiment": 4,
                "status": "partial",
                "failure_reason": "budget exceeded",
                "steps": [{"step": 1, "loss": 2.3, "temperature": 0.07}],
            }
        else:
            data = {**_MINIMAL_RESULT, "experiment": n}
        _write_result(results_dir, n, data)

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ok, issues = check_preconditions()
    finally:
        os.chdir(original_cwd)

    assert ok is True, f"Expected ok=True but got issues: {issues}"
