"""Bodenmiller IMC dataset loader.

Reads Bodenmiller CytoImageList RDS files via an R subprocess conversion step.
The R package ``cytomapper`` / ``EBImage`` are NOT required; we use stub S4 class
definitions and ``unclass()`` to extract the raw arrays.

Typical usage::

    images = load_imc_rds("/path/to/images.rds")
    # images is a list[IMCImage]
    img = images[0]
    # img.image: (H, W, C) float32 numpy array
    # img.channel_names: list of C channel name strings
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# R script that converts a CytoImageList RDS to per-image binary files.
# Uses unclass() so EBImage / cytomapper are NOT required.
_R_CONVERSION_SCRIPT = r"""
args <- commandArgs(trailingOnly=TRUE)
rds_path <- args[1]
out_dir  <- args[2]
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

suppressMessages({
    library(methods)
    library(S4Vectors)
})

# Minimal stub S4 classes — avoids loading EBImage / cytomapper
setClass('Image',         contains='array')
setClass('CytoImageList', representation(
    listData        = 'list',
    elementType     = 'character',
    elementMetadata = 'ANY',
    metadata        = 'list'
))

rds      <- readRDS(rds_path)
listdata <- methods::slot(rds, 'listData')
n        <- length(listdata)

# Channel names live in dimnames[[3]] of the first Image
first_dn <- attr(listdata[[1]], 'dimnames')
ch_names <- if (!is.null(first_dn) && length(first_dn) >= 3) first_dn[[3]] else character(0)
writeLines(ch_names, file.path(out_dir, 'channel_names.txt'))

img_names <- names(listdata)
if (is.null(img_names)) img_names <- paste0('img_', seq_len(n))
writeLines(img_names, file.path(out_dir, 'image_names.txt'))

for (i in seq_len(n)) {
    arr <- unclass(listdata[[i]])   # drops S4 class; raw double array
    d   <- dim(arr)
    con <- file(file.path(out_dir, sprintf('img_%04d.bin', i)), 'wb')
    writeBin(as.integer(d),              con, size=4, endian='little')
    writeBin(as.double(as.vector(arr)),  con, size=4, endian='little')
    close(con)
}
cat('OK', n, '\n')
"""

_RSCRIPT_SEARCH = [
    # Check common conda environments that have R
    os.path.expanduser("~/anaconda3/envs/ecdna_R/bin/Rscript"),
    os.path.expanduser("~/miniconda3/envs/ecdna_R/bin/Rscript"),
    "/usr/bin/Rscript",
    "/usr/local/bin/Rscript",
]


def _find_rscript() -> str:
    """Return the path to Rscript, raising RuntimeError if not found."""
    # 1. Honour explicit env var
    env_rscript = os.environ.get("RSCRIPT")
    if env_rscript and os.path.isfile(env_rscript):
        return env_rscript

    # 2. Try known paths
    for path in _RSCRIPT_SEARCH:
        if os.path.isfile(path):
            return path

    # 3. Fall back to PATH
    found = shutil.which("Rscript")
    if found:
        return found

    raise RuntimeError(
        "Rscript not found. Install R (e.g. conda install -c conda-forge r-base) "
        "or set the RSCRIPT environment variable to the Rscript binary path."
    )


def _run_r_conversion(rds_path: Path, out_dir: Path) -> None:
    """Call R to convert a CytoImageList RDS into per-image .bin files."""
    rscript = _find_rscript()

    # Write the embedded R script to a temp file
    with tempfile.NamedTemporaryFile(
        suffix=".R", mode="w", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(_R_CONVERSION_SCRIPT)
        r_script_path = fh.name

    try:
        result = subprocess.run(
            [rscript, "--vanilla", r_script_path, str(rds_path), str(out_dir)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            # Filter known benign R startup messages from stderr
            benign = {"Loading required package", "Attaching package", "The following", "masked"}
            error_lines = [
                ln for ln in stderr.splitlines()
                if not any(ln.startswith(b) for b in benign)
            ]
            raise RuntimeError(
                f"R conversion failed (exit {result.returncode}).\n"
                f"stderr:\n{chr(10).join(error_lines)}\nstdout:\n{stdout}"
            )

        if not stdout.startswith("OK"):
            raise RuntimeError(f"Unexpected R script output: {stdout!r}")

        logger.info("R conversion output: %s", stdout)
    finally:
        os.unlink(r_script_path)


def _read_bin_file(path: Path) -> np.ndarray:
    """Read a single per-image .bin file produced by the R script.

    File format: 3 × int32 (H, W, C) little-endian, then H*W*C × float32.

    Returns
    -------
    np.ndarray
        Shape (H, W, C), dtype float32.
    """
    with open(path, "rb") as fh:
        raw = fh.read(12)
        h, w, c = struct.unpack_from("<iii", raw)
        n_floats = h * w * c
        data = np.frombuffer(fh.read(n_floats * 4), dtype="<f4")

    # R stores arrays in column-major (Fortran) order: dim = (H, W, C)
    # np.frombuffer gives a 1-D view; reshape with Fortran ordering, then
    # transpose axes so that the final layout is (H, W, C) in C order.
    arr = data.reshape((h, w, c), order="F")
    return arr.astype(np.float32, copy=False)


@dataclass
class IMCImage:
    """A single multichannel IMC tissue image."""

    name: str
    image: np.ndarray  # (H, W, C) float32
    channel_names: list[str]


def load_imc_rds(
    path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    max_images: int | None = None,
    force_reconvert: bool = False,
) -> list[IMCImage]:
    """Load a Bodenmiller IMC CytoImageList RDS file into Python.

    The first call for a given RDS file runs an R subprocess to convert the
    data into a flat binary cache directory next to the RDS file (or in
    ``cache_dir`` if supplied).  Subsequent calls read from the cache directly.

    Parameters
    ----------
    path:
        Path to the ``.rds`` file.
    cache_dir:
        Directory for the extracted binary cache.  Defaults to
        ``<rds_stem>_extracted/`` next to the RDS file.
    max_images:
        If set, only load the first ``max_images`` images (useful for smoke
        tests).
    force_reconvert:
        If True, re-run the R conversion even if a cache already exists.

    Returns
    -------
    list[IMCImage]
        Each entry is one tissue image with a (H, W, C) float32 array and
        channel names.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"RDS file not found: {path}")

    # Determine cache directory
    if cache_dir is None:
        cache_dir = path.parent / f"{path.stem}_extracted"
    cache_dir = Path(cache_dir)

    done_marker = cache_dir / "image_names.txt"

    if force_reconvert or not done_marker.exists():
        logger.info("Running R conversion for %s → %s", path.name, cache_dir)
        _run_r_conversion(path, cache_dir)
    else:
        logger.info("Using cached extraction in %s", cache_dir)

    # Load channel names
    ch_file = cache_dir / "channel_names.txt"
    channel_names: list[str] = []
    if ch_file.exists():
        channel_names = [ln.strip() for ln in ch_file.read_text().splitlines() if ln.strip()]

    # Load image names
    img_names = [ln.strip() for ln in done_marker.read_text().splitlines() if ln.strip()]

    # Sort bin files by index
    bin_files = sorted(cache_dir.glob("img_*.bin"))
    if not bin_files:
        raise RuntimeError(f"No img_*.bin files found in cache dir {cache_dir}")

    if max_images is not None:
        bin_files = bin_files[:max_images]
        img_names = img_names[:max_images]

    images: list[IMCImage] = []
    for i, (bfile, iname) in enumerate(zip(bin_files, img_names)):
        arr = _read_bin_file(bfile)
        images.append(IMCImage(name=iname, image=arr, channel_names=channel_names))
        if i == 0:
            logger.info(
                "First image '%s': shape=%s, channels=%d",
                iname,
                arr.shape,
                len(channel_names),
            )

    logger.info("Loaded %d images from %s", len(images), path.name)
    return images
