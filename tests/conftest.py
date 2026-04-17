import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def tiny_image(rng):
    """Random 64x64x4 image to act as a tiny multichannel patch."""
    return rng.uniform(0, 100, size=(64, 64, 4)).astype(np.float32)
