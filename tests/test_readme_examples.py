"""
tests/test_readme_examples.py — Unit tests for the README generator.

Covers:
    - Missing precondition files cause refusal and name the missing file (Req 26.1, 26.2)
    - Every numeric value in the generated README traces to a source file (Req 26.4, 26.8)
    - Failed experiment status and failure reason are surfaced in the README (Req 26.1, 26.4)

Validates: Requirements 26.1, 26.2, 26.4, 26.8
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from generate_readme import check_preconditions, generate_readme, load_all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_experiment(results_dir, n: int, payload: dict) -> None:
    """Write a minimal experiment JSON file into results_dir."""
    path = results_dir / f"experiment{n}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_benchmark(results_dir, payload: dict) -> None:
    path = results_dir / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_experiment(n: int) -> dict:
    """Return a minimal valid completed-experiment payload for experiment n."""
    base = {"experiment": n, "status": "completed", "failure_reason": None}
    if n == 1:
        base["steps"] = [
            {"step": 1, "loss": 1.0, "temperature": 0.07,
             "i2t_acc1": 0.5, "t2i_acc1": 0.5}
        ]
    elif n == 2:
        base["retrieval"] = {
            "trained": {
                "i2t": {"r1": 0.5, "r5": 0.8, "r10": 0.9,
                        "mrr": 0.6, "mean_rank": 3.0, "median_rank": 2.0},
                "t2i": {"r1": 0.5, "r5": 0.8, "r10": 0.9,
                        "mrr": 0.6, "mean_rank": 3.0, "median_rank": 2.0},
            },
            "random_baseline": None,
        }
    else:
        base["runs"] = []
    return base


def _minimal_benchmark() -> dict:
    return {
        "image_encode_latency_ms": {
            "mean": 1.0, "median": 1.0, "p95": 2.0, "std": 0.1
        }
    }


# ---------------------------------------------------------------------------
# Test 1: Missing precondition files cause refusal and name the missing file
# ---------------------------------------------------------------------------

def test_readme_refuses_missing_files(tmp_path):
    """Any missing precondition file causes refusal and that file is named.

    Validates: Requirements 26.1, 26.2
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Write experiments 1-3, 5, 6 — leave experiment4.json missing.
    for n in [1, 2, 3, 5, 6]:
        _write_minimal_experiment(results_dir, n, _minimal_experiment(n))
    _write_benchmark(results_dir, _minimal_benchmark())

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        ok, issues = check_preconditions()

        assert ok is False
        # The missing file must be named in the issues list.
        issues_str = " ".join(issues)
        assert "experiment4.json" in issues_str
    finally:
        os.chdir(original_cwd)

    # --- Second scenario: benchmark.json missing, all experiments present ---
    results_dir2 = tmp_path / "results2"
    results_dir2.mkdir()
    for n in range(1, 7):
        _write_minimal_experiment(results_dir2, n, _minimal_experiment(n))
    # Deliberately omit benchmark.json.

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Temporarily rename results dir so check_preconditions sees results2 as results.
        results_canonical = tmp_path / "results"
        results_canonical.rename(tmp_path / "results_bak")
        results_dir2.rename(results_canonical)

        ok2, issues2 = check_preconditions()

        assert ok2 is False
        issues_str2 = " ".join(issues2)
        assert "benchmark.json" in issues_str2
    finally:
        # Restore directories regardless of test outcome.
        try:
            results_canonical.rename(results_dir2)
            (tmp_path / "results_bak").rename(results_canonical)
        except Exception:
            pass
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Test 2: Every numeric value in the README traces to a source file
# ---------------------------------------------------------------------------

def test_readme_traceability():
    """Known numeric values in results appear formatted correctly in the README.

    Validates: Requirements 26.4, 26.8
    """
    results = {
        1: {
            "experiment": 1,
            "status": "completed",
            "failure_reason": None,
            "steps": [
                {
                    "step": 1,
                    "loss": 2.3456,
                    "temperature": 0.07,
                    "i2t_acc1": 0.1234,
                    "t2i_acc1": 0.1234,
                }
            ],
        },
        2: {
            "experiment": 2,
            "status": "completed",
            "failure_reason": None,
            "retrieval": {
                "trained": {
                    "i2t": {
                        "r1": 0.1234,
                        "r5": 0.4321,
                        "r10": 0.5678,
                        "mrr": 0.2345,
                        "mean_rank": 5.0,
                        "median_rank": 4.0,
                    }
                },
                "random_baseline": None,
            },
        },
        3: {
            "experiment": 3,
            "status": "failed",
            "failure_reason": "test failure message",
        },
        4: {
            "experiment": 4,
            "status": "completed",
            "failure_reason": None,
            "runs": [],
        },
        5: {
            "experiment": 5,
            "status": "completed",
            "failure_reason": None,
            "runs": [],
        },
        6: {
            "experiment": 6,
            "status": "completed",
            "failure_reason": None,
            "runs": [],
        },
        "benchmark": {
            "image_encode_latency_ms": {
                "mean": 1.23,
                "median": 1.0,
                "p95": 2.0,
                "std": 0.1,
            }
        },
        "metadata": {n: "unavailable" for n in range(1, 7)},
    }

    readme = generate_readme(results)

    # --- Experiment 1 numeric values ---
    # loss=2.3456 → formatted as "2.3456"
    assert "2.3456" in readme
    # temperature=0.07 → "0.0700"
    assert "0.0700" in readme
    # i2t_acc1=0.1234 → "0.1234"
    assert "0.1234" in readme

    # --- Experiment 2 retrieval metrics ---
    # r5=0.4321 → "0.4321"
    assert "0.4321" in readme
    # r10=0.5678 → "0.5678"
    assert "0.5678" in readme
    # mrr=0.2345 → "0.2345"
    assert "0.2345" in readme
    # mean_rank=5.0 → "5.0000"
    assert "5.0000" in readme
    # median_rank=4.0 → "4.0000"
    assert "4.0000" in readme

    # --- Benchmark numeric values ---
    # image_encode_latency mean=1.23 → "1.2300"
    assert "1.2300" in readme


# ---------------------------------------------------------------------------
# Test 3: Failed experiment status and failure reason surfaced in the README
# ---------------------------------------------------------------------------

def test_readme_includes_failed_experiment_status():
    """A 'failed' experiment's status and failure reason appear in its README section.

    Validates: Requirements 26.1, 26.4
    """
    results = {
        1: {
            "experiment": 1,
            "status": "completed",
            "failure_reason": None,
            "steps": [
                {
                    "step": 1,
                    "loss": 1.0,
                    "temperature": 0.07,
                    "i2t_acc1": 0.5,
                    "t2i_acc1": 0.5,
                }
            ],
        },
        2: {
            "experiment": 2,
            "status": "completed",
            "failure_reason": None,
            "retrieval": {
                "trained": {
                    "i2t": {
                        "r1": 0.5,
                        "r5": 0.8,
                        "r10": 0.9,
                        "mrr": 0.6,
                        "mean_rank": 3.0,
                        "median_rank": 2.0,
                    }
                },
                "random_baseline": None,
            },
        },
        3: {
            "experiment": 3,
            "status": "failed",
            "failure_reason": "test failure message",
        },
        4: {
            "experiment": 4,
            "status": "completed",
            "failure_reason": None,
            "runs": [],
        },
        5: {
            "experiment": 5,
            "status": "completed",
            "failure_reason": None,
            "runs": [],
        },
        6: {
            "experiment": 6,
            "status": "completed",
            "failure_reason": None,
            "runs": [],
        },
        "benchmark": {
            "image_encode_latency_ms": {
                "mean": 1.0,
                "median": 1.0,
                "p95": 2.0,
                "std": 0.1,
            }
        },
        "metadata": {n: "unavailable" for n in range(1, 7)},
    }

    readme = generate_readme(results)

    # Locate the Experiment 3 section.
    exp3_marker = "### Experiment 3"
    assert exp3_marker in readme

    exp3_start = readme.index(exp3_marker)
    # The section ends at the next "### Experiment" heading (or end of string).
    exp3_end_search = readme.find("### Experiment 4", exp3_start)
    if exp3_end_search == -1:
        exp3_section = readme[exp3_start:]
    else:
        exp3_section = readme[exp3_start:exp3_end_search]

    # The word "failed" must appear in the Experiment 3 section.
    assert "failed" in exp3_section

    # The failure reason must appear verbatim in the Experiment 3 section.
    assert "test failure message" in exp3_section
