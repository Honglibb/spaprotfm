"""Cross-dataset transfer heatmap for the SpaProtFM paper.

Renders a 4x3 heatmap where:
  - rows 0..2 = Murphy baseline trained on Damond / HochSchulz / Jackson
  - row 3    = SpaProtFM v2 leave-one-out (never saw the target cohort)
  - cols     = test cohort (Damond / HochSchulz / Jackson)

Each cell is the mean PCC averaged over the paired-seed intersection
(seeds {0, 1, 42}). The top block visualises per-panel Murphy's sharp
in-dist / OOD generalisation gap; the bottom row visualises v2's flat
OOD row — i.e. a single checkpoint covers all three cohorts.

Outputs results/figures/paper/figure_transfer_heatmap.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

R = Path("/home/zkgy/hongliyin_computer/results")
OUT = R / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Damond", "HochSchulz", "Jackson"]
DS_SHORT_LABEL = {"Damond": "Damond", "HochSchulz": "HochSchulz", "Jackson": "Jackson"}
DS_ID = {"Damond": "D", "HochSchulz": "H", "Jackson": "J"}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def _paired_seeds() -> list[int]:
    v2 = set()
    for p in R.glob("xds_pooled_s*"):
        if (p / "metrics.json").exists():
            try:
                v2.add(int(p.name.split("_s")[-1]))
            except ValueError:
                pass
    mu = set()
    for p in R.glob("murphy_xds_Damond_s*"):
        if (p / "metrics.json").exists():
            try:
                mu.add(int(p.name.split("_s")[-1]))
            except ValueError:
                pass
    return sorted(v2 & mu)


def _murphy_matrix(seed: int) -> np.ndarray:
    """3x3: rows=Murphy source, cols=test dataset. NaN if missing."""
    m = np.full((3, 3), np.nan)
    for i, src in enumerate(DATASETS):
        p = R / f"murphy_xds_{src}_s{seed}" / "metrics.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for ds, res in d["eval"]["in_dist"].items():
            m[i, DATASETS.index(ds)] = res["mean_pcc"]
        for ds, res in d["eval"]["ood"].items():
            m[i, DATASETS.index(ds)] = res["mean_pcc"]
    return m


def _v2_loo_row(seed: int) -> np.ndarray:
    """1x3 row: v2 leave-one-out checkpoint OOD mean PCC per target cohort."""
    row = np.full(3, np.nan)
    for j, ds in enumerate(DATASETS):
        p = R / f"xds_heldout{DS_ID[ds]}_s{seed}" / "metrics.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if ds in d["eval"].get("ood", {}):
            row[j] = d["eval"]["ood"][ds]["mean_pcc"]
    return row


def build_matrix() -> tuple[np.ndarray, int]:
    seeds = _paired_seeds()
    if not seeds:
        raise SystemExit("No paired seeds for transfer heatmap")
    murphy_stack = np.stack([_murphy_matrix(s) for s in seeds], axis=0)
    v2_stack = np.stack([_v2_loo_row(s) for s in seeds], axis=0)
    murphy_mean = np.nanmean(murphy_stack, axis=0)
    v2_mean = np.nanmean(v2_stack, axis=0)
    mat = np.vstack([murphy_mean, v2_mean])  # (4, 3)
    return mat, len(seeds)


def main() -> None:
    setup_style()
    mat, n_seeds = build_matrix()

    row_labels = [
        "Murphy (trained on Damond)",
        "Murphy (trained on HochSchulz)",
        "Murphy (trained on Jackson)",
        "SpaProtFM v2 (leave-one-out)",
    ]
    col_labels = [DS_SHORT_LABEL[d] for d in DATASETS]

    fig, ax = plt.subplots(1, 1, figsize=(4.72, 2.8))
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(mat, cmap="viridis", norm=norm, aspect="auto")

    # Annotate each cell with its PCC; use black/white based on lightness.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            # viridis light end ~ 0.65 normalized -> use white below, black above
            t_color = "white" if norm(v) < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=t_color, fontweight="bold")

    # Separator between Murphy block and v2 row
    ax.axhline(2.5, color="white", lw=2.2)
    # Also mark the Murphy diagonal (in-dist cells) with a thin white frame
    for i in range(3):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor="white", lw=0.9,
                               linestyle=":"))

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(4))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("test cohort", labelpad=4)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for s in ax.spines.values():
        s.set_visible(False)

    # Mark the column winner (highest PCC per column) with a bold outline.
    for j in range(3):
        i_best = int(np.nanargmax(mat[:, j]))
        ax.add_patch(Rectangle((j - 0.5, i_best - 0.5), 1, 1,
                               fill=False, edgecolor="#e74c3c", lw=1.6))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.9)
    cbar.set_label("mean PCC (bio targets)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_title(
        f"Cross-dataset panel transfer — mean PCC (n={n_seeds} seeds)",
        fontsize=9, pad=8,
    )

    fig.tight_layout()
    out_pdf = OUT / "figure_transfer_heatmap.pdf"
    out_png = OUT / "figure_transfer_heatmap.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
