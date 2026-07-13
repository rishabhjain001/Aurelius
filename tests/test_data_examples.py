"""
tests/test_data_examples.py — Unit tests for the Flickr30K data pipeline.

Covers deterministic train/val split, disjointness, dataset-name rejection,
and uniform random caption sampling in training mode.

Validates: Requirements 2.1, 2.2, 2.3, 2.7, 2.8
"""

import os
import random
import sys
from collections import Counter

import pytest
import torch
from PIL import Image
from torchvision import transforms

# Ensure project root is importable when tests are run from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import Flickr30KDataset, _check_dataset_name, _split_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockHFDataset:
    """Minimal list-of-dicts wrapper that mimics the HuggingFace indexing API."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# ---------------------------------------------------------------------------
# Test 1: deterministic split
# ---------------------------------------------------------------------------

def test_val_split_determinism():
    """Same (n_total, val_fraction, seed) must yield identical splits.

    Validates: Requirement 2.2
    """
    n_total = 1000
    val_fraction = 0.1
    seed = 42

    train_a, val_a = _split_indices(n_total, val_fraction, seed)
    train_b, val_b = _split_indices(n_total, val_fraction, seed)

    assert train_a == train_b
    assert val_a == val_b


# ---------------------------------------------------------------------------
# Test 2: disjoint and complete split
# ---------------------------------------------------------------------------

def test_val_train_disjoint():
    """Train and val index sets must be disjoint and cover [0, n_total).

    Validates: Requirements 2.2, 2.7
    """
    n_total = 1000
    val_fraction = 0.1
    seed = 42

    train_idx, val_idx = _split_indices(n_total, val_fraction, seed)

    train_set = set(train_idx)
    val_set = set(val_idx)

    assert train_set.intersection(val_set) == set()
    assert len(train_idx) + len(val_idx) == n_total
    assert train_set | val_set == set(range(n_total))


# ---------------------------------------------------------------------------
# Test 3: dataset name rejection
# ---------------------------------------------------------------------------

def test_dataset_rejection():
    """Non-Flickr30K dataset names must be rejected; flickr30k is accepted
    case-insensitively.

    Validates: Requirement 2.8
    """
    with pytest.raises(ValueError):
        _check_dataset_name("mscoco")

    # Case-insensitive acceptance: neither call should raise.
    _check_dataset_name("flickr30k")
    _check_dataset_name("FLICKR30K")


# ---------------------------------------------------------------------------
# Test 4: uniform caption sampling in train mode
# ---------------------------------------------------------------------------

def test_caption_sampling_uniform():
    """Training-mode __getitem__ should sample the five captions with
    approximately uniform frequency over many draws.

    Validates: Requirement 2.3
    """
    captions = ["c0", "c1", "c2", "c3", "c4"]
    image = Image.new("RGB", (256, 256), color="red")
    rows = [{"image": image, "captions": captions}]
    mock_ds = MockHFDataset(rows)

    dataset = Flickr30KDataset(
        hf_dataset=mock_ds,
        indices=[0],
        transform=transforms.ToTensor(),
        is_train=True,
    )

    random.seed(42)
    n_samples = 1000
    counts = Counter()
    for _ in range(n_samples):
        _, caption = dataset[0]
        counts[caption] += 1

    # Each caption should land within ±80 of the expected 200 draws.
    expected = n_samples // len(captions)
    tolerance = 80
    for c in captions:
        assert expected - tolerance <= counts[c] <= expected + tolerance, (
            f"Caption {c!r} occurred {counts[c]} times, expected ~{expected}"
        )
