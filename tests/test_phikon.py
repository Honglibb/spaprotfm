"""Tests for the Phikon-v2 encoder wrapper.

These tests load the real model the first time they run (downloads ~350 MB
into the project HF cache). Subsequent runs use the cached weights.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from spaprotfm.condition.phikon import (  # noqa: E402
    PHIKON_HIDDEN_DIM,
    PHIKON_NUM_TOKENS,
    PhikonEncoder,
)


@pytest.fixture(scope="module")
def encoder() -> PhikonEncoder:
    return PhikonEncoder()


def test_output_shape(encoder: PhikonEncoder):
    x = torch.rand(2, 3, 224, 224)
    feats = encoder.encode(x)
    assert feats.shape == (2, PHIKON_HIDDEN_DIM, PHIKON_NUM_TOKENS, PHIKON_NUM_TOKENS)


def test_return_cls(encoder: PhikonEncoder):
    x = torch.rand(1, 3, 224, 224)
    feats, cls = encoder.encode(x, return_cls=True)
    assert feats.shape == (1, PHIKON_HIDDEN_DIM, PHIKON_NUM_TOKENS, PHIKON_NUM_TOKENS)
    assert cls.shape == (1, PHIKON_HIDDEN_DIM)


def test_weights_frozen(encoder: PhikonEncoder):
    frozen = [p.requires_grad for p in encoder.model.parameters()]
    assert not any(frozen), "Phikon weights must have requires_grad=False"


def test_eval_mode_after_train_call(encoder: PhikonEncoder):
    encoder.train()
    assert not encoder.model.training, "backbone must stay in eval() after .train()"


def test_rejects_wrong_shape(encoder: PhikonEncoder):
    with pytest.raises(ValueError):
        encoder.encode(torch.rand(1, 3, 128, 128))
    with pytest.raises(ValueError):
        encoder.encode(torch.rand(3, 224, 224))
