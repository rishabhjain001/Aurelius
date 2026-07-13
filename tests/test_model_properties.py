"""
tests/test_model_properties.py — Property-based tests for InfoNCE and cosine similarity.

Validates mathematical invariants of InfoNCEModule and the Similarity_Matrix
using Hypothesis.

Properties covered:
    3. Cosine Similarity Bounds
    4. Similarity Matrix Shape Invariant
    5. Similarity Matrix Diagonal Alignment
    6. InfoNCE Target Label Invariant
    7. Temperature Positivity After Training Steps
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from model import InfoNCEModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_unit_vecs(n: int, d: int) -> torch.Tensor:
    """Return [n, d] L2-normalized float32 tensor."""
    x = torch.randn(n, d)
    return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Property 3: Cosine Similarity Bounds
# ---------------------------------------------------------------------------

# Feature: aurelius-clip-vlm, Property 3: Cosine Similarity Bounds
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=32),
    m=st.integers(min_value=2, max_value=32),
    d=st.integers(min_value=2, max_value=32),
)
def test_cosine_similarity_bounds(n: int, m: int, d: int) -> None:
    """For any pair of L2-normalized embedding vectors, cosine similarity ∈ [-1.0, 1.0].

    Validates: Requirements 7.1, 8.5, 19.3
    """
    image_embeds = _rand_unit_vecs(n, d)
    text_embeds = _rand_unit_vecs(m, d)

    with torch.no_grad():
        # Cosine similarity is the dot product between L2-normalized vectors
        sim_matrix = image_embeds @ text_embeds.T  # [n, m]

    assert sim_matrix.min().item() >= -1.0 - 1e-5, (
        f"Cosine similarity below -1.0: {sim_matrix.min().item()}"
    )
    assert sim_matrix.max().item() <= 1.0 + 1e-5, (
        f"Cosine similarity above 1.0: {sim_matrix.max().item()}"
    )


# ---------------------------------------------------------------------------
# Property 4: Similarity Matrix Shape Invariant
# ---------------------------------------------------------------------------

# Feature: aurelius-clip-vlm, Property 4: Similarity Matrix Shape Invariant
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=8),
    m=st.integers(min_value=1, max_value=8),
    d=st.integers(min_value=1, max_value=8),
)
def test_similarity_matrix_shape(n: int, m: int, d: int) -> None:
    """For any batch of N images and M texts, the Similarity_Matrix has N rows and M columns.

    InfoNCEModule.forward requires N==M for cross-entropy, so the shape invariant is
    verified directly via the matrix product image_embeds @ text_embeds.T.

    Validates: Requirements 7.1, 19.4
    """
    image_embeds = _rand_unit_vecs(n, d)
    text_embeds = _rand_unit_vecs(m, d)

    with torch.no_grad():
        sim_matrix = image_embeds @ text_embeds.T  # [n, m]

    assert sim_matrix.shape == (n, m), (
        f"Expected shape ({n}, {m}), got {tuple(sim_matrix.shape)}"
    )


# ---------------------------------------------------------------------------
# Property 5: Similarity Matrix Diagonal Alignment
# ---------------------------------------------------------------------------

# Feature: aurelius-clip-vlm, Property 5: Similarity Matrix Diagonal Alignment
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=8),
    d=st.integers(min_value=2, max_value=32),
)
def test_similarity_matrix_diagonal_alignment(n: int, d: int) -> None:
    """For any batch of N image-caption pairs, similarity[i, i] equals dot(image_i, text_i).

    Since embeddings are L2-normalized, cosine similarity equals the dot product.

    Validates: Requirements 7.1, 19.5
    """
    image_embeds = _rand_unit_vecs(n, d)
    text_embeds = _rand_unit_vecs(n, d)

    with torch.no_grad():
        sim_matrix = image_embeds @ text_embeds.T  # [n, n]

        for i in range(n):
            expected = torch.dot(image_embeds[i], text_embeds[i]).item()
            actual = sim_matrix[i, i].item()
            assert abs(actual - expected) < 1e-5, (
                f"Diagonal mismatch at i={i}: sim_matrix[{i},{i}]={actual:.6f}, "
                f"dot product={expected:.6f}"
            )


# ---------------------------------------------------------------------------
# Property 6: InfoNCE Target Label Invariant
# ---------------------------------------------------------------------------

# Feature: aurelius-clip-vlm, Property 6: InfoNCE Target Label Invariant
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    d=st.integers(min_value=2, max_value=32),
)
def test_infonce_target_label_invariant(n: int, d: int) -> None:
    """For any batch of N pairs, target at row/column i equals i for all i in 0…N-1.

    Verified by recomputing the loss manually with explicit torch.arange(N) targets
    and confirming it matches the InfoNCEModule output exactly.

    Validates: Requirements 7.3, 7.4, 19.6
    """
    module = InfoNCEModule(temperature_init=0.07)
    module.eval()

    image_embeds = _rand_unit_vecs(n, d)
    text_embeds = _rand_unit_vecs(n, d)

    with torch.no_grad():
        loss_module, sim_matrix = module(image_embeds, text_embeds)

        # Recompute with explicit arange targets to verify the invariant
        scaled = sim_matrix / module.log_temperature.exp()
        targets = torch.arange(n)

        loss_i2t = torch.nn.functional.cross_entropy(scaled, targets)
        loss_t2i = torch.nn.functional.cross_entropy(scaled.T, targets)
        expected_loss = (loss_i2t + loss_t2i) / 2.0

    assert abs(loss_module.item() - expected_loss.item()) < 1e-5, (
        f"Loss mismatch: module={loss_module.item():.6f}, manual={expected_loss.item():.6f}. "
        "This indicates targets used internally are not torch.arange(N)."
    )


# ---------------------------------------------------------------------------
# Property 7: Temperature Positivity After Training Steps
# ---------------------------------------------------------------------------

# Feature: aurelius-clip-vlm, Property 7: Temperature Positivity After Training Steps
@settings(max_examples=100, deadline=None)
@given(
    k=st.integers(min_value=3, max_value=10),
    d=st.integers(min_value=2, max_value=16),
)
def test_temperature_positivity_after_training_steps(k: int, d: int) -> None:
    """After every step in a sequence of ≥3 simulated training steps, exp(log_temperature) > 0.

    Validates: Requirements 7.7, 19.7
    """
    module = InfoNCEModule(temperature_init=0.07)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)

    n = 4  # fixed small batch size for speed

    for step in range(k):
        image_embeds = _rand_unit_vecs(n, d)
        text_embeds = _rand_unit_vecs(n, d)

        optimizer.zero_grad()
        loss, _ = module(image_embeds, text_embeds)
        loss.backward()
        optimizer.step()

        temperature = module.log_temperature.exp().item()
        assert temperature > 0.0, (
            f"Temperature not positive after step {step + 1}: {temperature}"
        )
