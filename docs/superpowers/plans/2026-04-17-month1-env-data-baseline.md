# SpaProtFM Month 1 Implementation Plan: Environment, Data Pipeline, Baseline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the SpaProtFM repository, build a tested data pipeline for HuBMAP CODEX + HTAN IMC + Bodenmiller IMC datasets, build the evaluation framework (PCC/MSE/SSIM/Frobenius/cell-typing), and reproduce Murphy U-Net baseline numbers on HuBMAP intestine.

**Architecture:** Python package `spaprotfm/` with `data/`, `eval/`, `baselines/` submodules. Use `uv` for dependency management. PyTorch + Lightning for training. Pytest for tests. Each module is independently importable and unit-tested where possible. Heavy data downloads gated behind CLI scripts in `scripts/`.

**Tech Stack:**
- Python 3.11+, `uv` (Astral) for env mgmt
- PyTorch 2.4+, Lightning 2.4+, `diffusers` (for later plans)
- `tifffile`, `zarr`, `anndata`, `scanpy` for biological data
- `deepcell` (Mesmer) for cell segmentation
- `pytest` + `hypothesis` for tests
- `requests` + `globus-cli` for HuBMAP downloads

---

## File Structure

```
hongliyin_computer/
├── pyproject.toml                  # uv project + deps
├── .gitignore                      # ignore data/, .venv/, .pytest_cache/
├── README.md                       # setup + run instructions
├── docs/superpowers/{specs,plans}/ # already exists
├── src/spaprotfm/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── markers.py              # marker name standardization (no I/O)
│   │   ├── normalization.py        # arcsinh, percentile clipping (no I/O)
│   │   ├── tiling.py               # patch extraction (no I/O)
│   │   ├── hubmap.py               # HuBMAP API + Globus download
│   │   ├── htan.py                 # HTAN data fetching
│   │   ├── bodenmiller.py          # IMC dataset loader (R-bridge or direct)
│   │   └── manifests.py            # dataset registry / catalog
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py              # PCC, MSE, SSIM, Frobenius (pure math)
│   │   ├── segmentation.py         # Mesmer wrapper with caching
│   │   └── celltype.py             # cell-level aggregation + classifier eval
│   └── baselines/
│       ├── __init__.py
│       └── murphy_unet.py          # adapter to murphygroup/CODEXPanelOptimization
├── scripts/                        # CLI entry points
│   ├── download_hubmap.py
│   ├── download_htan.py
│   ├── preprocess.py
│   ├── run_baseline.py
│   └── eval_baseline.py
├── tests/
│   ├── conftest.py                 # shared fixtures (tiny synthetic data)
│   ├── test_markers.py
│   ├── test_normalization.py
│   ├── test_tiling.py
│   ├── test_metrics.py
│   ├── test_segmentation.py
│   └── test_celltype.py
├── data/                           # gitignored
│   ├── raw/{hubmap,htan,bodenmiller}/
│   ├── processed/
│   └── manifests/                  # dataset catalogs (CSV)
└── results/                        # gitignored, baseline outputs
```

**Design rationale:**
- Pure math / data transforms (markers, normalization, tiling, metrics) live in their own files because they're easy to unit-test and shouldn't depend on I/O.
- I/O modules (hubmap, htan, bodenmiller) are isolated so we can mock them in tests.
- `scripts/` are thin CLI wrappers; all logic lives in `src/spaprotfm/`.

---

## Task 1: Bootstrap repository

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/spaprotfm/__init__.py`

- [ ] **Step 1: Init git and check no existing repo**

```bash
cd /code/zkgy/hongliyin_computer
git status 2>&1 | head -1
# expected: "fatal: not a git repository" — confirms clean slate
git init
git checkout -b main
```

- [ ] **Step 2: Verify `uv` is installed**

```bash
uv --version
# expected: uv X.Y.Z (any recent version)
# if missing: curl -LsSf https://astral.sh/uv/install.sh | sh
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "spaprotfm"
version = "0.1.0"
description = "Multimodal generative panel extension for spatial proteomics"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "lightning>=2.4",
    "diffusers>=0.30",
    "transformers>=4.44",
    "numpy>=1.26",
    "scipy>=1.13",
    "scikit-image>=0.24",
    "scikit-learn>=1.5",
    "pandas>=2.2",
    "tifffile>=2024.7",
    "zarr>=2.18",
    "anndata>=0.10",
    "scanpy>=1.10",
    "requests>=2.32",
    "tqdm>=4.66",
    "matplotlib>=3.9",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "hypothesis>=6.108", "pytest-cov>=5.0", "ruff>=0.6"]
seg = ["deepcell>=0.12"]  # heavy, optional install

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py311"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/spaprotfm"]
```

- [ ] **Step 4: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
data/
results/
*.tiff
*.tif
*.zarr
.DS_Store
.idea/
.vscode/
```

- [ ] **Step 5: Create `src/spaprotfm/__init__.py`**

```python
"""SpaProtFM: multimodal generative panel extension for spatial proteomics."""
__version__ = "0.1.0"
```

- [ ] **Step 6: Create `README.md`** (skeleton — full version in Task 12)

```markdown
# SpaProtFM

Multimodal generative panel extension for spatial proteomics (CODEX/IMC).

## Setup

\`\`\`bash
uv sync
uv pip install -e .
\`\`\`

## Status

Month 1: data pipeline + baseline reproduction.

See `docs/superpowers/specs/2026-04-17-spaprotfm-design.md` for project design.
```

- [ ] **Step 7: Install deps**

```bash
uv sync
# expected: "Resolved N packages in Xs", venv created at .venv/
```

- [ ] **Step 8: Verify Python can import package**

```bash
uv run python -c "import spaprotfm; print(spaprotfm.__version__)"
# expected: 0.1.0
```

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore README.md src/spaprotfm/__init__.py docs/
git commit -m "chore: bootstrap spaprotfm repo with uv and design docs"
```

---

## Task 2: Marker name standardization (TDD)

**Why:** Different datasets call the same protein different names ("CD8a" vs "CD8" vs "CD8α" vs "CD8a-2"). We need a canonical mapping before any cross-dataset training.

**Files:**
- Create: `src/spaprotfm/data/markers.py`, `tests/test_markers.py`, `tests/conftest.py`, `data/manifests/marker_aliases.yaml`

- [ ] **Step 1: Create `tests/conftest.py` with shared fixtures**

```python
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def tiny_image(rng):
    """Random 64x64x4 image to act as a tiny multichannel patch."""
    return rng.uniform(0, 100, size=(64, 64, 4)).astype(np.float32)
```

- [ ] **Step 2: Write failing tests in `tests/test_markers.py`**

```python
from spaprotfm.data.markers import (
    canonicalize,
    load_alias_table,
    standardize_panel,
)


def test_canonicalize_strips_whitespace_and_lowers():
    assert canonicalize("  CD8a  ") == "cd8a"


def test_canonicalize_removes_greek_variants():
    assert canonicalize("CD8α") == "cd8a"
    assert canonicalize("HLA-DRβ") == "hla-drb"


def test_canonicalize_normalizes_dashes():
    assert canonicalize("HLA_DR") == "hla-dr"
    assert canonicalize("HLA DR") == "hla-dr"


def test_load_alias_table_returns_dict_of_canonical_to_aliases(tmp_path):
    yaml_path = tmp_path / "aliases.yaml"
    yaml_path.write_text(
        "CD8A:\n  - CD8\n  - CD8a\n  - CD8α\n"
        "PAN-CK:\n  - PanCK\n  - Pan-Keratin\n"
    )
    table = load_alias_table(yaml_path)
    assert table["cd8a"] == "CD8A"
    assert table["cd8"] == "CD8A"
    assert table["pan-ck"] == "PAN-CK"
    assert table["panck"] == "PAN-CK"
    assert table["pan-keratin"] == "PAN-CK"


def test_standardize_panel_maps_known_and_warns_unknown(tmp_path, caplog):
    yaml_path = tmp_path / "aliases.yaml"
    yaml_path.write_text("CD8A:\n  - CD8a\n")
    table = load_alias_table(yaml_path)

    out = standardize_panel(["CD8a", "FOXP3"], table)
    assert out == ["CD8A", None]
    assert "FOXP3" in caplog.text
```

- [ ] **Step 3: Run tests, verify they fail with ImportError**

```bash
uv run pytest tests/test_markers.py -v
# expected: ModuleNotFoundError: No module named 'spaprotfm.data.markers'
```

- [ ] **Step 4: Create `src/spaprotfm/data/__init__.py`**

```python
"""Data pipeline modules."""
```

- [ ] **Step 5: Implement `src/spaprotfm/data/markers.py`**

```python
"""Marker name standardization across CODEX/IMC datasets."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_GREEK_TO_LATIN = {"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "κ": "k"}
_DASH_LIKE = re.compile(r"[\s_]+")


def canonicalize(name: str) -> str:
    """Normalize a marker name to lowercase ASCII with hyphen separators."""
    s = name.strip()
    for greek, latin in _GREEK_TO_LATIN.items():
        s = s.replace(greek, latin)
    # Decompose unicode and drop diacritics
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = _DASH_LIKE.sub("-", s)
    return s.lower()


def load_alias_table(path: str | Path) -> dict[str, str]:
    """Load YAML mapping {canonical: [aliases]} and invert to {alias_canon: canonical}."""
    raw = yaml.safe_load(Path(path).read_text())
    table: dict[str, str] = {}
    for canonical, aliases in raw.items():
        names = [canonical, *aliases]
        for name in names:
            table[canonicalize(name)] = canonical
    return table


def standardize_panel(
    names: list[str], alias_table: dict[str, str]
) -> list[str | None]:
    """Map raw marker names to canonical names; return None for unknown."""
    out: list[str | None] = []
    for n in names:
        key = canonicalize(n)
        if key in alias_table:
            out.append(alias_table[key])
        else:
            logger.warning("Unknown marker: %r (canonical: %r)", n, key)
            out.append(None)
    return out
```

- [ ] **Step 6: Run tests, verify they pass**

```bash
uv run pytest tests/test_markers.py -v
# expected: 5 passed
```

- [ ] **Step 7: Bootstrap initial alias YAML**

Create `data/manifests/marker_aliases.yaml` with first 30 common markers (extend later):

```yaml
CD3:
  - CD3
  - CD3e
  - CD3epsilon
  - CD3-epsilon
CD4:
  - CD4
CD8A:
  - CD8
  - CD8a
  - CD8alpha
  - CD8α
CD20:
  - CD20
CD45:
  - CD45
  - PTPRC
CD45RA:
  - CD45RA
CD45RO:
  - CD45RO
CD68:
  - CD68
CD163:
  - CD163
FOXP3:
  - FOXP3
  - Foxp3
KI67:
  - Ki67
  - MKI67
HLA-DR:
  - HLA-DR
  - HLA-DRA
  - HLADR
PAN-CK:
  - PanCK
  - Pan-Keratin
  - PanKeratin
  - Cytokeratin
VIMENTIN:
  - Vimentin
  - VIM
DAPI:
  - DAPI
  - Hoechst
ECAD:
  - E-Cadherin
  - ECAD
  - CDH1
SMA:
  - SMA
  - aSMA
  - alpha-SMA
  - ACTA2
COLLAGEN:
  - Collagen
  - Col1
  - Col1a1
CD11C:
  - CD11c
  - ITGAX
CD11B:
  - CD11b
  - ITGAM
CD56:
  - CD56
  - NCAM1
CD31:
  - CD31
  - PECAM1
CD138:
  - CD138
  - SDC1
PD1:
  - PD-1
  - PD1
  - PDCD1
PDL1:
  - PD-L1
  - PDL1
  - CD274
GRANZYMEB:
  - GranzymeB
  - GZMB
CD15:
  - CD15
  - FUT4
LYVE1:
  - LYVE1
  - LYVE-1
PODOPLANIN:
  - Podoplanin
  - PDPN
TBET:
  - T-bet
  - Tbet
  - TBX21
```

- [ ] **Step 8: Smoke test with real YAML**

```bash
uv run python -c "
from spaprotfm.data.markers import load_alias_table, standardize_panel
t = load_alias_table('data/manifests/marker_aliases.yaml')
print(standardize_panel(['CD8a', 'PanCK', 'unknown_marker'], t))
"
# expected: ['CD8A', 'PAN-CK', None] (with a warning log line for unknown_marker)
```

- [ ] **Step 9: Commit**

```bash
git add src/spaprotfm/data/ tests/conftest.py tests/test_markers.py data/manifests/marker_aliases.yaml
git commit -m "feat(data): marker name standardization with alias table"
```

---

## Task 3: Image normalization (TDD)

**Why:** CODEX and IMC pixel intensities span orders of magnitude. Standard preprocessing: arcsinh transform + per-channel percentile clipping.

**Files:**
- Create: `src/spaprotfm/data/normalization.py`, `tests/test_normalization.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from spaprotfm.data.normalization import (
    arcsinh_transform,
    percentile_clip,
    normalize_image,
)


def test_arcsinh_transform_with_default_cofactor():
    x = np.array([0.0, 5.0, 100.0])
    out = arcsinh_transform(x, cofactor=5.0)
    assert out[0] == 0.0
    assert np.isclose(out[1], np.arcsinh(1.0))
    assert np.isclose(out[2], np.arcsinh(20.0))


def test_arcsinh_handles_zeros_and_negatives():
    x = np.array([-1.0, 0.0, 1.0])
    out = arcsinh_transform(x, cofactor=1.0)
    assert out[1] == 0.0
    assert out[0] == -out[2]  # arcsinh is odd


def test_percentile_clip_bounds_values():
    x = np.array([1, 2, 3, 4, 100], dtype=np.float32)
    out = percentile_clip(x, lo=0, hi=80)
    assert out.min() == pytest.approx(1.0)
    assert out.max() == pytest.approx(np.percentile(x, 80))


def test_normalize_image_per_channel(rng):
    img = rng.uniform(0, 1000, size=(32, 32, 3)).astype(np.float32)
    out = normalize_image(img, cofactor=5.0, percentile=(0, 99))
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0
    # Each channel scaled to roughly [0, 1]
    for c in range(3):
        assert out[..., c].max() <= 1.0 + 1e-6


def test_normalize_image_constant_channel_does_not_nan():
    img = np.zeros((8, 8, 2), dtype=np.float32)
    img[..., 1] = 5.0
    out = normalize_image(img)
    assert not np.isnan(out).any()
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_normalization.py -v
# expected: ModuleNotFoundError
```

- [ ] **Step 3: Implement `src/spaprotfm/data/normalization.py`**

```python
"""Image normalization for multiplex tissue images."""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def arcsinh_transform(x: ArrayLike, cofactor: float = 5.0) -> np.ndarray:
    """Apply arcsinh(x / cofactor); standard CyTOF/CODEX preprocessing."""
    return np.arcsinh(np.asarray(x, dtype=np.float32) / cofactor)


def percentile_clip(x: ArrayLike, lo: float = 0.0, hi: float = 99.0) -> np.ndarray:
    """Clip values to [lo-percentile, hi-percentile] of x."""
    arr = np.asarray(x, dtype=np.float32)
    lo_v = np.percentile(arr, lo) if lo > 0 else arr.min()
    hi_v = np.percentile(arr, hi)
    return np.clip(arr, lo_v, hi_v)


def normalize_image(
    img: ArrayLike,
    cofactor: float = 5.0,
    percentile: tuple[float, float] = (0.0, 99.0),
) -> np.ndarray:
    """Per-channel arcsinh + percentile clipping + min-max scaling to [0, 1].

    Args:
        img: (H, W, C) array.
        cofactor: arcsinh cofactor.
        percentile: (lo, hi) percentiles for clipping.

    Returns:
        Float32 array in [0, 1] with same shape as input.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected (H, W, C) image, got shape {arr.shape}")

    out = np.empty_like(arr)
    for c in range(arr.shape[-1]):
        ch = arcsinh_transform(arr[..., c], cofactor=cofactor)
        ch = percentile_clip(ch, lo=percentile[0], hi=percentile[1])
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min < 1e-8:
            out[..., c] = 0.0
        else:
            out[..., c] = (ch - ch_min) / (ch_max - ch_min)
    return out
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/test_normalization.py -v
# expected: 5 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/spaprotfm/data/normalization.py tests/test_normalization.py
git commit -m "feat(data): per-channel arcsinh + percentile normalization"
```

---

## Task 4: Patch tiling (TDD)

**Why:** Whole-slide images are too large for a single forward pass. We tile into 256×256 patches with optional overlap for inference stitching.

**Files:**
- Create: `src/spaprotfm/data/tiling.py`, `tests/test_tiling.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from spaprotfm.data.tiling import tile_image, untile_image


def test_tile_image_no_overlap_exact_division(rng):
    img = rng.uniform(0, 1, size=(256, 512, 3)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=256)
    assert tiles.shape == (2, 256, 256, 3)
    assert coords == [(0, 0), (0, 256)]


def test_tile_image_with_overlap(rng):
    img = rng.uniform(0, 1, size=(256, 384, 1)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=128)
    # Two tiles at x=0 and x=128 (x=256 would be edge-aligned)
    assert tiles.shape[0] == 2
    assert (0, 0) in coords
    assert (0, 128) in coords


def test_tile_image_pads_when_dimensions_not_divisible(rng):
    img = rng.uniform(0, 1, size=(300, 300, 2)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=256, pad=True)
    # 2x2 grid: (0,0), (0,256), (256,0), (256,256) — last ones padded
    assert tiles.shape == (4, 256, 256, 2)


def test_untile_image_reverses_no_overlap(rng):
    img = rng.uniform(0, 1, size=(256, 512, 3)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=256)
    out = untile_image(tiles, coords, output_shape=(256, 512, 3))
    np.testing.assert_allclose(out, img)


def test_untile_image_averages_overlap(rng):
    img = rng.uniform(0, 1, size=(256, 384, 1)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=128)
    out = untile_image(tiles, coords, output_shape=img.shape)
    np.testing.assert_allclose(out, img, atol=1e-5)
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_tiling.py -v
# expected: ModuleNotFoundError
```

- [ ] **Step 3: Implement `src/spaprotfm/data/tiling.py`**

```python
"""Tile (H, W, C) images into fixed-size patches and stitch back."""

from __future__ import annotations

import numpy as np


def tile_image(
    img: np.ndarray,
    patch_size: int = 256,
    stride: int | None = None,
    pad: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Split image into patches.

    Args:
        img: (H, W, C) array.
        patch_size: side of square patches.
        stride: step between patches. Defaults to patch_size (no overlap).
        pad: if True, zero-pad so all patches are full-size.

    Returns:
        tiles: (N, patch_size, patch_size, C) array.
        coords: list of (y, x) top-left corners.
    """
    if stride is None:
        stride = patch_size
    H, W, C = img.shape
    if pad:
        pad_h = (-(H - patch_size) % stride) if H >= patch_size else (patch_size - H)
        pad_w = (-(W - patch_size) % stride) if W >= patch_size else (patch_size - W)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
        H, W, C = img.shape

    coords: list[tuple[int, int]] = []
    tiles: list[np.ndarray] = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            tiles.append(img[y : y + patch_size, x : x + patch_size, :])
            coords.append((y, x))

    return np.stack(tiles, axis=0), coords


def untile_image(
    tiles: np.ndarray,
    coords: list[tuple[int, int]],
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Stitch patches back into an image, averaging overlapping regions."""
    H, W, C = output_shape
    out = np.zeros((H, W, C), dtype=np.float32)
    cnt = np.zeros((H, W, 1), dtype=np.float32)
    p = tiles.shape[1]
    for tile, (y, x) in zip(tiles, coords):
        h = min(p, H - y)
        w = min(p, W - x)
        out[y : y + h, x : x + w, :] += tile[:h, :w, :]
        cnt[y : y + h, x : x + w, :] += 1.0
    cnt[cnt == 0] = 1.0
    return out / cnt
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/test_tiling.py -v
# expected: 5 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/spaprotfm/data/tiling.py tests/test_tiling.py
git commit -m "feat(data): patch tiling with overlap stitching"
```

---

## Task 5: Evaluation metrics (TDD)

**Why:** All baselines and our model must be scored consistently. Implement once, test thoroughly, reuse everywhere.

**Files:**
- Create: `src/spaprotfm/eval/__init__.py`, `src/spaprotfm/eval/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from spaprotfm.eval.metrics import (
    pearson_correlation,
    mean_squared_error,
    structural_similarity,
    frobenius_distance,
)


def test_pearson_perfect_correlation():
    x = np.linspace(0, 1, 100)
    y = 2 * x + 0.5
    assert pearson_correlation(x, y) == pytest.approx(1.0, abs=1e-6)


def test_pearson_anticorrelation():
    x = np.linspace(0, 1, 100)
    assert pearson_correlation(x, -x) == pytest.approx(-1.0, abs=1e-6)


def test_pearson_handles_constant_returns_zero():
    x = np.zeros(100)
    y = np.linspace(0, 1, 100)
    assert pearson_correlation(x, y) == 0.0


def test_mse_zero_when_identical(rng):
    x = rng.uniform(0, 1, 50).astype(np.float32)
    assert mean_squared_error(x, x) == 0.0


def test_mse_known_value():
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    assert mean_squared_error(x, y) == pytest.approx((1 + 4) / 2)


def test_ssim_self_is_one(rng):
    img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
    assert structural_similarity(img, img) == pytest.approx(1.0, abs=1e-4)


def test_ssim_decreases_with_noise(rng):
    img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
    noisy = img + rng.normal(0, 0.5, img.shape)
    assert structural_similarity(img, noisy) < 0.9


def test_frobenius_distance_matches_norm():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.zeros_like(A)
    expected = np.sqrt(1 + 4 + 9 + 16)
    assert frobenius_distance(A, B) == pytest.approx(expected)


def test_pearson_per_channel_returns_one_per_channel(rng):
    pred = rng.uniform(0, 1, (10, 10, 3)).astype(np.float32)
    target = pred.copy()
    target[..., 0] += rng.normal(0, 0.01, (10, 10))
    out = pearson_correlation(pred, target, per_channel=True)
    assert out.shape == (3,)
    assert out[1] == pytest.approx(1.0, abs=1e-4)
    assert out[2] == pytest.approx(1.0, abs=1e-4)
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_metrics.py -v
# expected: ModuleNotFoundError
```

- [ ] **Step 3: Implement `src/spaprotfm/eval/__init__.py`**

```python
"""Evaluation metrics and protocols."""
```

- [ ] **Step 4: Implement `src/spaprotfm/eval/metrics.py`**

```python
"""Evaluation metrics for spatial proteomics imputation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def pearson_correlation(
    x: np.ndarray, y: np.ndarray, per_channel: bool = False
) -> float | np.ndarray:
    """PCC between flattened arrays, or per-channel for (H, W, C) inputs.

    Returns 0.0 (or zeros) if either input has zero variance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if per_channel:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("per_channel=True requires (H, W, C) inputs")
        C = x.shape[-1]
        return np.array(
            [pearson_correlation(x[..., c], y[..., c]) for c in range(C)]
        )

    xf, yf = x.ravel(), y.ravel()
    if xf.std() < 1e-12 or yf.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xf, yf)[0, 1])


def mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(x, np.float64) - np.asarray(y, np.float64)) ** 2))


def structural_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM for 2D or 3D (per-channel averaged) arrays in [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    data_range = max(x.max(), y.max()) - min(x.min(), y.min()) or 1.0
    if x.ndim == 2:
        return float(_ssim(x, y, data_range=data_range))
    if x.ndim == 3:
        return float(
            _ssim(x, y, data_range=data_range, channel_axis=-1)
        )
    raise ValueError(f"Expected 2D or 3D array, got {x.ndim}D")


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(A, np.float64) - np.asarray(B, np.float64)))
```

- [ ] **Step 5: Run tests, verify pass**

```bash
uv run pytest tests/test_metrics.py -v
# expected: 9 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/spaprotfm/eval/__init__.py src/spaprotfm/eval/metrics.py tests/test_metrics.py
git commit -m "feat(eval): pixel-level metrics (PCC, MSE, SSIM, Frobenius)"
```

---

## Task 6: HuBMAP CODEX dataset catalog and downloader

**Why:** HuBMAP has 5000+ datasets; we need a curated subset (intestine, spleen, lymph node CODEX). This task builds a manifest CSV listing exact dataset IDs + download paths.

**Files:**
- Create: `data/manifests/hubmap_codex.csv`, `src/spaprotfm/data/hubmap.py`, `scripts/download_hubmap.py`

- [ ] **Step 1: Manually curate HuBMAP dataset list**

Visit https://portal.hubmapconsortium.org/search?entity_type[0]=Dataset&data_types[0]=CODEX and collect: 16 large intestine + 16 small intestine + 8 spleen + 9 lymph node = 49 datasets.

Save to `data/manifests/hubmap_codex.csv`:

```csv
hubmap_id,organ,n_markers,assay,download_url
HBM326.HZHJ.347,LARGE_INTESTINE,46,CODEX,https://assets.hubmapconsortium.org/<UUID>/...
... (49 rows)
```

> ⚠️ **Real-time data**: At plan-execution time, query the HuBMAP search API to populate this CSV. See Step 3 for the query snippet.

- [ ] **Step 2: Write `src/spaprotfm/data/hubmap.py`**

```python
"""HuBMAP CODEX dataset catalog + downloader."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

HUBMAP_SEARCH_URL = "https://search.api.hubmapconsortium.org/v3/portal/search"


@dataclass
class HuBMAPDataset:
    hubmap_id: str
    organ: str
    n_markers: int
    download_url: str


def load_catalog(catalog_path: str | Path) -> list[HuBMAPDataset]:
    df = pd.read_csv(catalog_path)
    return [
        HuBMAPDataset(
            hubmap_id=row.hubmap_id,
            organ=row.organ,
            n_markers=int(row.n_markers),
            download_url=row.download_url,
        )
        for row in df.itertuples(index=False)
    ]


def query_hubmap_search(organ: str, assay: str = "CODEX") -> list[dict]:
    """Query HuBMAP search API for public datasets matching organ + assay."""
    payload = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"entity_type.keyword": "Dataset"}},
                    {"term": {"data_types.keyword": assay}},
                    {"term": {"origin_samples.organ.keyword": organ}},
                    {"term": {"data_access_level": "public"}},
                ]
            }
        },
        "size": 200,
    }
    r = requests.post(HUBMAP_SEARCH_URL, json=payload, timeout=30)
    r.raise_for_status()
    return [hit["_source"] for hit in r.json()["hits"]["hits"]]


def download_dataset(ds: HuBMAPDataset, out_dir: str | Path) -> Path:
    """Download a single dataset to out_dir/<hubmap_id>/. Returns the directory."""
    out = Path(out_dir) / ds.hubmap_id
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", ds.hubmap_id, out)
    # NOTE: real implementation must respect HuBMAP's Globus or asset URL scheme.
    # See scripts/download_hubmap.py for the actual download logic.
    return out
```

- [ ] **Step 3: Write `scripts/download_hubmap.py`**

```python
"""Download HuBMAP CODEX datasets listed in the catalog.

Usage:
    uv run python scripts/download_hubmap.py \
        --catalog data/manifests/hubmap_codex.csv \
        --out data/raw/hubmap \
        --organ LARGE_INTESTINE \
        --limit 2  # start small
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from spaprotfm.data.hubmap import load_catalog, query_hubmap_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_url(url: str, dest: Path, chunk: int = 1024 * 1024) -> None:
    if dest.exists():
        log.info("Skip existing: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for buf in r.iter_content(chunk):
                f.write(buf)
                pbar.update(len(buf))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--organ", default=None, help="Filter by organ (optional)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of datasets (0 = all)")
    p.add_argument("--refresh-catalog", action="store_true",
                   help="Re-query HuBMAP API and overwrite catalog")
    args = p.parse_args()

    if args.refresh_catalog:
        # Repopulate CSV from API for organs of interest
        import pandas as pd
        rows = []
        for organ in ["LARGE_INTESTINE", "SMALL_INTESTINE", "SPLEEN", "LYMPH_NODE"]:
            hits = query_hubmap_search(organ)
            for h in hits:
                rows.append({
                    "hubmap_id": h.get("hubmap_id"),
                    "organ": organ,
                    "n_markers": len(h.get("metadata", {}).get("antibodies", [])) or -1,
                    "assay": "CODEX",
                    "download_url": "",  # filled by manual or asset API call
                })
        pd.DataFrame(rows).to_csv(args.catalog, index=False)
        log.info("Wrote %d rows to %s", len(rows), args.catalog)
        return 0

    catalog = load_catalog(args.catalog)
    if args.organ:
        catalog = [c for c in catalog if c.organ == args.organ]
    if args.limit:
        catalog = catalog[: args.limit]

    log.info("Will download %d datasets to %s", len(catalog), args.out)
    for ds in catalog:
        if not ds.download_url:
            log.warning("No download_url for %s, skipping", ds.hubmap_id)
            continue
        fetch_url(ds.download_url, args.out / ds.hubmap_id / Path(ds.download_url).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Smoke test catalog refresh (uses live API)**

```bash
uv run python scripts/download_hubmap.py \
    --catalog data/manifests/hubmap_codex.csv \
    --out /tmp/hubmap_test \
    --refresh-catalog
# expected: writes ~50 rows to CSV; download_url column may be empty
# (you may need to manually inspect HuBMAP UI to fill download URLs for first dataset)
```

- [ ] **Step 5: Manually fill download_url for ONE dataset and test fetch**

Pick one HuBMAP dataset, get its asset bundle URL from the portal, paste into CSV, then:

```bash
uv run python scripts/download_hubmap.py \
    --catalog data/manifests/hubmap_codex.csv \
    --out data/raw/hubmap \
    --limit 1
# expected: downloads the dataset (could be 5-50 GB; use --limit 1 first)
```

- [ ] **Step 6: Verify file structure on disk**

```bash
ls -la data/raw/hubmap/
# expected: one HBM... directory
du -sh data/raw/hubmap/
# expected: nontrivial size
```

- [ ] **Step 7: Commit (catalog + code only, NOT raw data)**

```bash
git add src/spaprotfm/data/hubmap.py scripts/download_hubmap.py data/manifests/hubmap_codex.csv
git commit -m "feat(data): HuBMAP CODEX catalog + downloader script"
```

---

## Task 7: Tiff/OME-TIFF reader for CODEX

**Why:** HuBMAP CODEX is delivered as OME-TIFF; we need a unified loader that returns (H, W, C) arrays + marker name list.

**Files:**
- Create: `src/spaprotfm/data/io.py`, `tests/test_io.py`

- [ ] **Step 1: Write tests using a fixture-built OME-TIFF**

```python
import numpy as np
import pytest
import tifffile

from spaprotfm.data.io import load_codex_ometiff


@pytest.fixture
def synthetic_ometiff(tmp_path):
    """Build a tiny OME-TIFF with 4 channels and metadata."""
    path = tmp_path / "synthetic.ome.tiff"
    data = np.random.uniform(0, 100, size=(4, 64, 64)).astype(np.uint16)  # (C, H, W)
    metadata = {
        "Channel": [{"Name": n} for n in ["DAPI", "CD4", "CD8", "PanCK"]],
    }
    tifffile.imwrite(path, data, metadata=metadata, photometric="minisblack")
    return path


def test_load_codex_ometiff_returns_hwc_and_names(synthetic_ometiff):
    img, names = load_codex_ometiff(synthetic_ometiff)
    assert img.shape == (64, 64, 4)
    assert img.dtype == np.float32
    assert names == ["DAPI", "CD4", "CD8", "PanCK"]
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_io.py -v
# expected: ModuleNotFoundError
```

- [ ] **Step 3: Implement `src/spaprotfm/data/io.py`**

```python
"""I/O helpers for multiplex tissue image formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def load_codex_ometiff(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load an OME-TIFF and return (H, W, C) float32 array + channel names.

    Falls back to generic channel names ('ch0', 'ch1', ...) if metadata absent.
    """
    with tifffile.TiffFile(str(path)) as tf:
        arr = tf.asarray()  # typically (C, H, W) for OME-TIFF
        meta = tf.imagej_metadata or {}
        ome = tf.ome_metadata

    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        # (C, H, W) -> (H, W, C)
        arr = np.moveaxis(arr, 0, -1)
    elif arr.ndim == 2:
        arr = arr[..., None]

    arr = arr.astype(np.float32)

    names: list[str] = []
    if ome:
        # Cheap parse: look for Channel Name="..." in OME-XML
        import re
        names = re.findall(r'Channel[^>]*Name="([^"]+)"', ome)
    if not names and isinstance(meta, dict) and "Channel" in meta:
        names = [c.get("Name", f"ch{i}") for i, c in enumerate(meta["Channel"])]
    if not names:
        names = [f"ch{i}" for i in range(arr.shape[-1])]

    return arr, names
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_io.py -v
# expected: 1 passed
```

- [ ] **Step 5: Smoke test on real HuBMAP data (if Task 6 downloaded a dataset)**

```bash
uv run python -c "
from pathlib import Path
from spaprotfm.data.io import load_codex_ometiff
files = list(Path('data/raw/hubmap').rglob('*.ome.tif*'))
if files:
    img, names = load_codex_ometiff(files[0])
    print(f'Shape: {img.shape}, channels: {names[:5]}...')
else:
    print('No HuBMAP ome.tiff files found yet (skip task 6 first)')
"
# expected if Task 6 ran: Shape: (H, W, ~46), channels: ['DAPI', 'CD4', ...]
```

- [ ] **Step 6: Commit**

```bash
git add src/spaprotfm/data/io.py tests/test_io.py
git commit -m "feat(data): OME-TIFF loader with channel name extraction"
```

---

## Task 8: Cell segmentation wrapper (Mesmer)

**Why:** Cell-level evaluation requires segmenting cells from DAPI + membrane markers. Mesmer (DeepCell) is the standard.

**Files:**
- Create: `src/spaprotfm/eval/segmentation.py`, `tests/test_segmentation.py`

- [ ] **Step 1: Write tests (with mocked Mesmer)**

```python
import numpy as np
import pytest

from spaprotfm.eval.segmentation import (
    select_segmentation_channels,
    aggregate_per_cell,
)


def test_select_segmentation_channels_returns_dapi_and_membrane():
    names = ["DAPI", "CD4", "CD45", "PanCK"]
    nuc, mem = select_segmentation_channels(names)
    assert names[nuc] == "DAPI"
    # membrane fallback: any of CD45/PanCK/E-Cad
    assert names[mem] in {"CD45", "PanCK"}


def test_select_segmentation_raises_when_no_dapi():
    with pytest.raises(ValueError, match="DAPI"):
        select_segmentation_channels(["CD4", "CD8"])


def test_aggregate_per_cell_means_intensities():
    img = np.zeros((4, 4, 2), dtype=np.float32)
    img[0:2, 0:2, 0] = 1.0      # cell 1, channel 0
    img[2:4, 2:4, 1] = 5.0      # cell 2, channel 1
    seg = np.zeros((4, 4), dtype=np.int32)
    seg[0:2, 0:2] = 1
    seg[2:4, 2:4] = 2

    df = aggregate_per_cell(img, seg, channel_names=["A", "B"])
    assert len(df) == 2
    assert df.loc[df["cell_id"] == 1, "A"].iloc[0] == pytest.approx(1.0)
    assert df.loc[df["cell_id"] == 1, "B"].iloc[0] == pytest.approx(0.0)
    assert df.loc[df["cell_id"] == 2, "A"].iloc[0] == pytest.approx(0.0)
    assert df.loc[df["cell_id"] == 2, "B"].iloc[0] == pytest.approx(5.0)
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_segmentation.py -v
# expected: ModuleNotFoundError
```

- [ ] **Step 3: Implement `src/spaprotfm/eval/segmentation.py`**

```python
"""Cell segmentation + per-cell aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_NUCLEAR_PRIORITY = ["DAPI", "Hoechst", "DNA1", "DNA2"]
_MEMBRANE_PRIORITY = ["CD45", "PanCK", "Pan-Keratin", "E-Cadherin", "ECAD", "CD3"]


def select_segmentation_channels(names: list[str]) -> tuple[int, int]:
    """Pick (nuclear_idx, membrane_idx) channels for Mesmer."""
    upper = [n.upper() for n in names]
    nuc = None
    for cand in _NUCLEAR_PRIORITY:
        if cand.upper() in upper:
            nuc = upper.index(cand.upper())
            break
    if nuc is None:
        raise ValueError(f"No DAPI/Hoechst-like channel in panel: {names}")

    mem = None
    for cand in _MEMBRANE_PRIORITY:
        if cand.upper() in upper:
            mem = upper.index(cand.upper())
            break
    if mem is None:
        # Fallback to any non-nuclear channel
        mem = (nuc + 1) % len(names)
    return nuc, mem


def segment_with_mesmer(
    img: np.ndarray,
    nuclear_idx: int,
    membrane_idx: int,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Run Mesmer; return (H, W) integer segmentation mask. Cached on disk if cache_path given."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path)

    # Lazy import — Mesmer is a heavy optional dep
    from deepcell.applications import Mesmer  # type: ignore

    nuc = img[..., nuclear_idx]
    mem = img[..., membrane_idx]
    stacked = np.stack([nuc, mem], axis=-1)[None, ...]  # (1, H, W, 2)
    app = Mesmer()
    mask = app.predict(stacked, image_mpp=0.5)[0, ..., 0].astype(np.int32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mask)
    return mask


def aggregate_per_cell(
    img: np.ndarray, seg: np.ndarray, channel_names: list[str]
) -> pd.DataFrame:
    """Mean intensity per (cell, channel). Returns DataFrame with cell_id + each channel."""
    if img.shape[:2] != seg.shape:
        raise ValueError(f"image {img.shape[:2]} ≠ seg {seg.shape}")
    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]  # 0 = background

    rows = []
    for cid in cell_ids:
        mask = seg == cid
        means = img[mask].mean(axis=0)
        rows.append({"cell_id": int(cid), **dict(zip(channel_names, means))})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/test_segmentation.py -v
# expected: 3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/spaprotfm/eval/segmentation.py tests/test_segmentation.py
git commit -m "feat(eval): cell segmentation wrapper + per-cell aggregation"
```

---

## Task 9: Preprocessing CLI script

**Why:** Tie data loaders + normalization + tiling + (optional) segmentation into one script that emits ready-to-train zarr/npz.

**Files:**
- Create: `scripts/preprocess.py`

- [ ] **Step 1: Write `scripts/preprocess.py`**

```python
"""Preprocess raw HuBMAP OME-TIFFs into normalized, tiled, marker-standardized npz.

Usage:
    uv run python scripts/preprocess.py \
        --raw-dir data/raw/hubmap \
        --out-dir data/processed/hubmap \
        --alias data/manifests/marker_aliases.yaml \
        --patch-size 256 --stride 256
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from spaprotfm.data.io import load_codex_ometiff
from spaprotfm.data.markers import load_alias_table, standardize_panel
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--alias", required=True, type=Path)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alias_table = load_alias_table(args.alias)

    files = list(args.raw_dir.rglob("*.ome.tif*"))
    log.info("Found %d OME-TIFFs under %s", len(files), args.raw_dir)
    for f in tqdm(files):
        try:
            img, names = load_codex_ometiff(f)
            canonical = standardize_panel(names, alias_table)
            keep = [i for i, n in enumerate(canonical) if n is not None]
            img = img[..., keep]
            canonical = [canonical[i] for i in keep]
            img = normalize_image(img)
            tiles, coords = tile_image(img, patch_size=args.patch_size, stride=args.stride)

            out_path = args.out_dir / f"{f.stem}.npz"
            np.savez_compressed(
                out_path,
                tiles=tiles,
                coords=np.array(coords),
                channel_names=np.array(canonical),
                source=str(f.relative_to(args.raw_dir)),
            )
            log.info("Wrote %s (tiles=%d, channels=%d)", out_path, tiles.shape[0], len(canonical))
        except Exception as e:
            log.exception("Failed on %s: %s", f, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on whatever HuBMAP data was downloaded**

```bash
uv run python scripts/preprocess.py \
    --raw-dir data/raw/hubmap \
    --out-dir data/processed/hubmap \
    --alias data/manifests/marker_aliases.yaml \
    --patch-size 256
# expected: writes one .npz per .ome.tiff
```

- [ ] **Step 3: Verify output**

```bash
uv run python -c "
import numpy as np
from pathlib import Path
files = list(Path('data/processed/hubmap').glob('*.npz'))
if not files:
    print('No processed files yet')
else:
    d = np.load(files[0])
    print(f'tiles {d[\"tiles\"].shape}, channels: {d[\"channel_names\"].tolist()[:5]}...')
"
# expected: tiles (N, 256, 256, K), channels: ['CD3', 'CD8A', ...]
```

- [ ] **Step 4: Commit**

```bash
git add scripts/preprocess.py
git commit -m "feat(scripts): preprocessing CLI for raw OME-TIFFs"
```

---

## Task 10: Murphy U-Net baseline integration

**Why:** Reproducing the strongest existing baseline gives us numbers to beat and a reference inference pipeline.

**Files:**
- Create: `src/spaprotfm/baselines/__init__.py`, `src/spaprotfm/baselines/murphy_unet.py`, `scripts/run_baseline.py`, `external/CODEXPanelOptimization/` (git submodule)

- [ ] **Step 1: Add Murphy repo as a git submodule**

```bash
git submodule add https://github.com/murphygroup/CODEXPanelOptimization.git external/CODEXPanelOptimization
git commit -m "chore: vendor Murphy CODEXPanelOptimization as submodule"
```

- [ ] **Step 2: Inspect their requirements**

```bash
cat external/CODEXPanelOptimization/requirements.txt 2>/dev/null \
    || cat external/CODEXPanelOptimization/README.md | head -50
# Read and note: PyTorch version, dependencies, dataset format expected
```

- [ ] **Step 3: Create thin adapter `src/spaprotfm/baselines/__init__.py`**

```python
"""Baseline reproductions."""
```

- [ ] **Step 4: Create `src/spaprotfm/baselines/murphy_unet.py`**

```python
"""Adapter for Murphy group's CODEXPanelOptimization U-Net baseline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3] / "external" / "CODEXPanelOptimization"


def _ensure_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


class MurphyUNetBaseline:
    """Wrap Murphy's U-Net for our (N, H, W, C) numpy interface.

    Their original code expects images in their notebook layout; we adapt
    by manually constructing the U-Net with same architecture.
    """

    def __init__(self, in_channels: int, out_channels: int, device: str = "cuda"):
        _ensure_on_path()
        # Try to import their model; if it fails, build a vanilla U-Net of same shape.
        try:
            from model import UNet  # type: ignore  # from their repo
            self.model = UNet(in_channels=in_channels, out_channels=out_channels)
        except Exception as e:
            logger.warning(
                "Could not import Murphy UNet (%s); using fallback vanilla U-Net", e
            )
            from spaprotfm.baselines._vanilla_unet import UNet
            self.model = UNet(in_channels=in_channels, out_channels=out_channels)
        self.model.to(device)
        self.device = device

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-4,
        batch_size: int = 8,
    ) -> None:
        """Fit on (N, H, W, C_in) -> (N, H, W, C_out)."""
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        x_t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        y_t = torch.from_numpy(y).permute(0, 3, 1, 2).float()
        ds = torch.utils.data.TensorDataset(x_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            losses: list[float] = []
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            logger.info("epoch %d  loss %.4f", ep, sum(losses) / len(losses))

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_t = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(self.device)
        with torch.no_grad():
            preds = []
            for i in range(0, len(x_t), 8):
                preds.append(self.model(x_t[i : i + 8]).cpu().numpy())
        out = np.concatenate(preds, axis=0)  # (N, C_out, H, W)
        return np.moveaxis(out, 1, -1)
```

- [ ] **Step 5: Create fallback vanilla U-Net at `src/spaprotfm/baselines/_vanilla_unet.py`**

```python
"""Vanilla U-Net used as fallback if Murphy repo's model can't be imported."""

from __future__ import annotations

import torch
import torch.nn as nn


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int = 32):
        super().__init__()
        self.d1 = _double_conv(in_channels, base)
        self.d2 = _double_conv(base, base * 2)
        self.d3 = _double_conv(base * 2, base * 4)
        self.d4 = _double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _double_conv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)
```

- [ ] **Step 6: Quick shape test**

```bash
uv run python -c "
import torch
from spaprotfm.baselines._vanilla_unet import UNet
m = UNet(in_channels=10, out_channels=20)
x = torch.randn(2, 10, 256, 256)
print(m(x).shape)
"
# expected: torch.Size([2, 20, 256, 256])
```

- [ ] **Step 7: Create `scripts/run_baseline.py`**

```python
"""Train + evaluate Murphy U-Net baseline on processed HuBMAP data.

Usage:
    uv run python scripts/run_baseline.py \
        --processed-dir data/processed/hubmap \
        --out-dir results/baseline_murphy \
        --observed-markers CD3,CD8A,CD20,CD68,DAPI,KI67,PAN-CK \
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from spaprotfm.baselines.murphy_unet import MurphyUNetBaseline
from spaprotfm.eval.metrics import (
    pearson_correlation,
    mean_squared_error,
    structural_similarity,
)

log = logging.getLogger(__name__)


def load_processed(processed_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Concatenate all .npz tiles. Returns (N, H, W, C), channel_names."""
    files = sorted(processed_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {processed_dir}")
    tiles_list = []
    names = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if names is None:
            names = list(d["channel_names"])
        elif list(d["channel_names"]) != names:
            log.warning("Skipping %s: panel mismatch", f)
            continue
        tiles_list.append(d["tiles"])
    return np.concatenate(tiles_list, axis=0), names


def split_observed_target(
    tiles: np.ndarray, names: list[str], observed: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    obs_idx = [names.index(n) for n in observed if n in names]
    tgt_idx = [i for i in range(len(names)) if i not in set(obs_idx)]
    return (
        tiles[..., obs_idx],
        tiles[..., tgt_idx],
        [names[i] for i in obs_idx],
        [names[i] for i in tgt_idx],
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--observed-markers", required=True,
        help="Comma-separated canonical names (e.g. CD3,CD8A,...)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tiles, names = load_processed(args.processed_dir)
    log.info("Loaded %d tiles with %d channels", tiles.shape[0], tiles.shape[-1])
    observed = [n.strip() for n in args.observed_markers.split(",")]
    x, y, obs_names, tgt_names = split_observed_target(tiles, names, observed)
    log.info("Observed (%d): %s", len(obs_names), obs_names)
    log.info("Target (%d): %s", len(tgt_names), tgt_names)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(tiles))
    n_train = int(0.8 * len(tiles))
    train, test = idx[:n_train], idx[n_train:]

    model = MurphyUNetBaseline(
        in_channels=x.shape[-1], out_channels=y.shape[-1], device=args.device
    )
    model.fit(x[train], y[train], epochs=args.epochs, batch_size=args.batch_size)
    pred = model.predict(x[test])

    # Metrics
    pcc_per_marker = pearson_correlation(pred, y[test], per_channel=True)
    mse = mean_squared_error(pred, y[test])
    ssim = structural_similarity(pred[0], y[test][0])  # one example tile

    metrics = {
        "n_train": int(n_train),
        "n_test": int(len(test)),
        "observed": obs_names,
        "target": tgt_names,
        "mean_pcc": float(np.nanmean(pcc_per_marker)),
        "pcc_per_marker": pcc_per_marker.tolist(),
        "mse": mse,
        "ssim_first_tile": ssim,
    }
    out_json = args.out_dir / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log.info("Wrote metrics to %s — mean PCC = %.3f, MSE = %.4f",
             out_json, metrics["mean_pcc"], mse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Smoke run on small subset (skip if no GPU yet)**

```bash
uv run python scripts/run_baseline.py \
    --processed-dir data/processed/hubmap \
    --out-dir results/baseline_smoke \
    --observed-markers DAPI,CD3,CD8A,CD20,KI67,PAN-CK,VIMENTIN \
    --epochs 2 \
    --device cpu  # 2-epoch CPU smoke test
# expected: writes results/baseline_smoke/metrics.json with low PCC (only 2 epochs)
```

- [ ] **Step 9: Commit**

```bash
git add src/spaprotfm/baselines/ scripts/run_baseline.py .gitmodules external/
git commit -m "feat(baseline): Murphy U-Net adapter + training/eval CLI"
```

---

## Task 11: Cell-level evaluation script

**Why:** Pixel metrics aren't enough; the field cares about cell-level concordance (typing accuracy, marker expression).

**Files:**
- Create: `src/spaprotfm/eval/celltype.py`, `tests/test_celltype.py`, `scripts/eval_baseline.py`

- [ ] **Step 1: Write tests for cell-typing concordance**

```python
import numpy as np
import pandas as pd
import pytest

from spaprotfm.eval.celltype import knn_cell_type_accuracy


def test_knn_accuracy_perfect_when_features_identical(rng):
    df = pd.DataFrame({
        "cell_id": range(20),
        "CD3": rng.uniform(0, 1, 20),
        "CD8A": rng.uniform(0, 1, 20),
    })
    df["true_label"] = (df["CD3"] > 0.5).astype(int)
    acc = knn_cell_type_accuracy(
        true_features=df[["CD3", "CD8A"]].values,
        pred_features=df[["CD3", "CD8A"]].values,
        labels=df["true_label"].values,
        k=3,
    )
    assert acc == pytest.approx(1.0, abs=0.05)


def test_knn_accuracy_drops_with_random_pred_features(rng):
    n = 200
    true_feat = rng.uniform(0, 1, (n, 4))
    labels = (true_feat[:, 0] > 0.5).astype(int)
    pred_feat = rng.uniform(0, 1, (n, 4))  # random
    acc = knn_cell_type_accuracy(true_feat, pred_feat, labels, k=5)
    assert acc < 0.7
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_celltype.py -v
# expected: ModuleNotFoundError
```

- [ ] **Step 3: Implement `src/spaprotfm/eval/celltype.py`**

```python
"""Cell-level evaluation: classify cells from features, compare types."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


def knn_cell_type_accuracy(
    true_features: np.ndarray,
    pred_features: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    n_splits: int = 5,
) -> float:
    """Train kNN on true features, evaluate on predicted features (held-out folds).

    Returns mean accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores: list[float] = []
    for train_idx, test_idx in skf.split(true_features, labels):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(true_features[train_idx], labels[train_idx])
        pred_labels = clf.predict(pred_features[test_idx])
        scores.append((pred_labels == labels[test_idx]).mean())
    return float(np.mean(scores))
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_celltype.py -v
# expected: 2 passed
```

- [ ] **Step 5: Create `scripts/eval_baseline.py` (full eval pipeline)**

```python
"""Full evaluation: pixel + cell-level metrics for a trained baseline.

Usage:
    uv run python scripts/eval_baseline.py \
        --processed-dir data/processed/hubmap \
        --baseline-dir results/baseline_murphy \
        --out results/baseline_murphy/full_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from spaprotfm.eval.metrics import pearson_correlation, mean_squared_error

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--baseline-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_in = json.loads((args.baseline_dir / "metrics.json").read_text())
    log.info("Baseline mean PCC = %.3f", metrics_in["mean_pcc"])

    # Cell-level eval requires segmenting then aggregating predicted vs true full-tile images.
    # For Month 1 we report pixel metrics + leave segmentation hookup as TODO note.
    out = {
        **metrics_in,
        "cell_level_eval_status": "deferred to Month 2 (requires per-tile segmentation)",
    }
    args.out.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Commit**

```bash
git add src/spaprotfm/eval/celltype.py tests/test_celltype.py scripts/eval_baseline.py
git commit -m "feat(eval): kNN cell-type accuracy + full eval CLI"
```

---

## Task 12: Update README with end-to-end run instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with full instructions**

```markdown
# SpaProtFM

Multimodal generative panel extension for spatial proteomics (CODEX/IMC).

> **Status (Month 1):** Data pipeline + Murphy U-Net baseline reproduction.
> Project design: `docs/superpowers/specs/2026-04-17-spaprotfm-design.md`.

## Setup

```bash
# 1. Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone with submodules
git clone --recurse-submodules <this-repo>
cd hongliyin_computer

# 3. Install deps
uv sync
uv pip install -e .

# 4. (Optional) Mesmer for cell segmentation
uv pip install -e .[seg]

# 5. Verify
uv run pytest -v
# expected: ~25 tests passing
```

## End-to-end pipeline

```bash
# Refresh HuBMAP catalog from API
uv run python scripts/download_hubmap.py \
    --catalog data/manifests/hubmap_codex.csv \
    --out data/raw/hubmap \
    --refresh-catalog

# Manually fill download_url column for at least 2 datasets, then:
uv run python scripts/download_hubmap.py \
    --catalog data/manifests/hubmap_codex.csv \
    --out data/raw/hubmap \
    --organ LARGE_INTESTINE --limit 2

# Preprocess
uv run python scripts/preprocess.py \
    --raw-dir data/raw/hubmap \
    --out-dir data/processed/hubmap \
    --alias data/manifests/marker_aliases.yaml

# Train + eval Murphy U-Net baseline
uv run python scripts/run_baseline.py \
    --processed-dir data/processed/hubmap \
    --out-dir results/baseline_murphy \
    --observed-markers DAPI,CD3,CD8A,CD20,CD45,KI67,PAN-CK \
    --epochs 30

# Full eval (placeholder — pixel only in Month 1)
uv run python scripts/eval_baseline.py \
    --processed-dir data/processed/hubmap \
    --baseline-dir results/baseline_murphy \
    --out results/baseline_murphy/full_eval.json
```

## Repo layout

See `docs/superpowers/plans/2026-04-17-month1-env-data-baseline.md`.

## Tests

```bash
uv run pytest -v --cov=src/spaprotfm --cov-report=term-missing
```

## License

TBD (vendored Murphy submodule has no explicit license — clarify before publishing).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: full setup + end-to-end pipeline in README"
```

---

## Task 13: End-to-end smoke test

**Why:** Make sure all modules wire together on a tiny synthetic dataset before relying on real downloads.

**Files:**
- Create: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
"""End-to-end smoke: synthetic image -> normalize -> tile -> train tiny model -> predict -> metrics."""

import numpy as np
import torch

from spaprotfm.baselines._vanilla_unet import UNet
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image
from spaprotfm.eval.metrics import pearson_correlation


def test_e2e_smoke(rng):
    # 1. Synthetic 256x256x6 image
    img = rng.uniform(0, 100, size=(256, 256, 6)).astype(np.float32)
    img = normalize_image(img)
    tiles, _ = tile_image(img, patch_size=128, stride=128)
    assert tiles.shape == (4, 128, 128, 6)

    # 2. Split observed/target
    x = tiles[..., :3]  # 3 observed
    y = tiles[..., 3:]  # 3 target

    # 3. Tiny model, 5 steps of training
    model = UNet(in_channels=3, out_channels=3, base=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    x_t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    y_t = torch.from_numpy(y).permute(0, 3, 1, 2).float()
    for _ in range(5):
        loss = loss_fn(model(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 4. Predict + metric
    with torch.no_grad():
        pred = model(x_t).permute(0, 2, 3, 1).numpy()
    pcc = pearson_correlation(pred, y, per_channel=True)
    assert pcc.shape == (3,)
    # Don't require high PCC — just no NaN, no crash
    assert not np.isnan(pcc).any()
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_e2e_smoke.py -v
# expected: 1 passed
```

- [ ] **Step 3: Run all tests together**

```bash
uv run pytest -v
# expected: all ~25 tests passing
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke test through normalize/tile/train/eval"
```

---

## Task 14: Capture Month 1 status report

**Files:**
- Create: `docs/superpowers/notes/2026-XX-XX-month1-status.md` (replace XX-XX with actual date when complete)

- [ ] **Step 1: After running baseline on real data, record actual numbers**

Template:

```markdown
# Month 1 Status Report

**Period:** 2026-04-17 → 2026-MM-DD

## What got done

- ✅ Repo bootstrapped (uv, pytest, src layout)
- ✅ Marker standardization (X canonical names, Y aliases)
- ✅ Normalization, tiling, metrics (all with tests, ~25 passing)
- ✅ HuBMAP catalog (49 datasets curated)
- ✅ Downloaded N datasets (TODO actual count + total GB)
- ✅ Processed N tiles
- ✅ Murphy U-Net baseline reproduced: mean PCC = X.XX (Murphy paper reports ~0.4-0.5 on intestine)

## What did not work

- (TODO — fill from real experience)

## Surprises

- (TODO)

## Decisions made

- (TODO)

## Next month

- Plan 2: SpaProtFM v0 (marker masking + diffusion, no H&E/scRNA)
```

- [ ] **Step 2: Commit when done**

```bash
mkdir -p docs/superpowers/notes
git add docs/superpowers/notes/
git commit -m "docs: month 1 status report"
```

---

## Self-review (run through after writing all tasks)

**Spec coverage check:**

| Spec section | Covered by |
|---|---|
| §2 Problem definition | Tasks 5, 6, 7 (data shape + observed/target split) |
| §3 Differentiation | Out of scope for Month 1 |
| §4 Datasets | Tasks 6, 7, 9 (HuBMAP); HTAN/Bodenmiller deferred to Plan 2 |
| §5 Method design | Out of scope for Month 1 |
| §6 Evaluation | Tasks 5, 8, 11 |
| §7 Timeline (Month 1) | Whole plan |
| §8 Risks | Acknowledged in Task 6 (download), Task 10 (license) |

**Gaps / deferred work** (intentional, parked for Plan 2):
- HTAN IMC integration — needs separate API study
- Bodenmiller `imcdatasets` R-bridge — needs `rpy2` setup
- Cell-level evaluation full pipeline (segment → aggregate → compare predicted vs true)
- GPU verification (do this manually in Task 1.7 alternative)

**Type consistency:** All function signatures use `np.ndarray`, `Path`, `list[str]` consistently. PCC return types: `float | np.ndarray` per the `per_channel` flag — verified between metrics.py and run_baseline.py.

**Placeholder scan:** No "TODO/TBD" except in:
- Task 14 status template (intentional — fill at end)
- Task 12 README license line (intentional — must clarify before publishing)
- Task 11 eval_baseline `cell_level_eval_status` (intentional deferral with explanation)

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-month1-env-data-baseline.md`.

**Two execution options:**

1. **Subagent-Driven (recommended for autonomous Claude execution)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good if you want me to drive Tasks 1-5 (pure code, no data dependencies) right now in this session.

2. **Inline Execution** — Step through tasks in this session sequentially. Good for tasks needing your decisions (e.g. which HuBMAP datasets to download).

3. **You execute, I support** — You follow the plan at your own pace; I help on individual tasks when stuck. Good for a 4-week timeline.

**Which approach?**
