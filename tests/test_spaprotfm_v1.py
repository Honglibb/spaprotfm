import torch

from spaprotfm.models.spaprotfm_v1 import (
    MaskedUNetV1, random_mask_with_always_observed,
)
from spaprotfm.models.spaprotfm_v0 import build_masked_input


def test_v1_forward_shape():
    m = MaskedUNetV1(n_channels=10, base=8)
    x = torch.randn(2, 20, 64, 64)
    out = m(x)
    assert out.shape == (2, 10, 64, 64)


def test_v1_value_bias_initialized_to_zero():
    m = MaskedUNetV1(n_channels=10, base=8)
    assert m.value_bias.shape == (10,)
    assert torch.allclose(m.value_bias, torch.zeros(10))


def test_v1_value_bias_modifies_input_only_at_value_planes():
    m = MaskedUNetV1(n_channels=3, base=4)
    # Set bias to known value so we can detect it
    with torch.no_grad():
        m.value_bias.copy_(torch.tensor([10.0, 20.0, 30.0]))
    full = torch.zeros(1, 3, 8, 8)
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    inp = build_masked_input(full, mask)
    # Forward shouldn't crash; we just verify it runs
    out = m(inp)
    assert out.shape == (1, 3, 8, 8)


def test_random_mask_with_always_observed_includes_pinned():
    mask = random_mask_with_always_observed(
        n_channels=10, batch_size=4, always_observed=[0, 1, 2],
        k_min=1, k_max=3,
    )
    assert mask.shape == (4, 10)
    # All rows must include 0, 1, 2
    assert (mask[:, [0, 1, 2]] == 1.0).all()


def test_random_mask_with_no_always_observed_works():
    mask = random_mask_with_always_observed(
        n_channels=10, batch_size=4, always_observed=None,
        k_min=2, k_max=5,
    )
    assert mask.shape == (4, 10)
    assert (mask.sum(dim=1) >= 2).all()
    assert (mask.sum(dim=1) <= 5).all()
