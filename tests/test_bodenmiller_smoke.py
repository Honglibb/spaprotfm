"""Integration smoke test — only runs if real Bodenmiller data is present.

Run with::

    uv run pytest tests/test_bodenmiller_smoke.py -v -s

The test skips automatically if the data directory or RDS files are absent.
"""

from pathlib import Path

import numpy as np
import pytest

from spaprotfm.data.bodenmiller import load_imc_rds
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image

DATA_DIR = Path("/code/zkgy/hongliyin_computer/data/raw/imc")


@pytest.mark.skipif(not DATA_DIR.exists(), reason="No real IMC data downloaded")
def test_bodenmiller_load_normalize_tile():
    """Load the first available RDS file, normalize, tile, and check shapes."""
    rds_files = sorted(DATA_DIR.glob("*.rds"))
    if not rds_files:
        pytest.skip("No .rds files in IMC data dir")

    rds_path = rds_files[0]
    print(f"\nUsing: {rds_path.name}")

    # ---- load ----
    images = load_imc_rds(rds_path, max_images=3)
    assert len(images) > 0, "Loader returned empty list"

    img = images[0]
    print(f"Loaded '{img.name}': shape={img.image.shape}, n_channels={len(img.channel_names)}")
    print(f"First 5 channels: {img.channel_names[:5]}")

    assert img.image.ndim == 3, f"Expected (H,W,C), got {img.image.ndim}D"
    assert img.image.dtype == np.float32, f"Expected float32, got {img.image.dtype}"
    assert len(img.channel_names) == img.image.shape[2], (
        f"Channel name count {len(img.channel_names)} != image depth {img.image.shape[2]}"
    )

    # ---- normalize ----
    normed = normalize_image(img.image)
    assert normed.shape == img.image.shape, "Normalization changed shape"
    assert normed.min() >= 0.0 - 1e-6, f"Normed min {normed.min()} < 0"
    assert normed.max() <= 1.0 + 1e-6, f"Normed max {normed.max()} > 1"
    assert normed.dtype == np.float32

    # ---- tile ----
    tiles, coords = tile_image(normed, patch_size=128, stride=128)
    print(f"Tiled into {tiles.shape[0]} patches of shape {tiles.shape[1:]}")
    print(f"First 3 tile coords: {coords[:3]}")

    assert tiles.shape[0] > 0, "No tiles produced"
    assert tiles.shape[1] == 128, f"Tile H != 128: {tiles.shape[1]}"
    assert tiles.shape[2] == 128, f"Tile W != 128: {tiles.shape[2]}"
    assert tiles.shape[3] == img.image.shape[2], "Tile channel dim mismatch"
    assert len(coords) == tiles.shape[0], "coords / tiles length mismatch"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="No real IMC data downloaded")
def test_bodenmiller_all_three_datasets():
    """Check that all 3 expected datasets are present and can be loaded."""
    expected_stems = {
        "Damond_2019_Pancreas_images",
        "HochSchulz_2022_Melanoma_images_protein",
        "JacksonFischer_2020_BreastCancer_images_basel",
    }
    present = {f.stem for f in DATA_DIR.glob("*.rds")}
    missing = expected_stems - present
    if missing:
        pytest.skip(f"Missing datasets: {missing}")

    for stem in sorted(expected_stems):
        rds_path = DATA_DIR / f"{stem}.rds"
        print(f"\nLoading {rds_path.name} ...")
        images = load_imc_rds(rds_path, max_images=1)
        assert len(images) >= 1
        img = images[0]
        print(
            f"  '{img.name}': shape={img.image.shape}, "
            f"channels={img.channel_names[:3]}..."
        )
        assert img.image.ndim == 3
        assert img.image.dtype == np.float32
        assert len(img.channel_names) > 0
