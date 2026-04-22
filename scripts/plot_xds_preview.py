"""Preview figure for cross-dataset transfer (seed=42 only).

Shows the OOD transfer comparison: Murphy vs v2 on each held-out dataset.
Paired bar chart + 3x3 matrix heatmap for the transfer table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = Path("/home/zkgy/hongliyin_computer/results")
FIGDIR = R / "figures"

DATASETS = ["Damond", "HochSchulz", "Jackson"]
DS_SHORT = {"Damond": "Damond\n(pancreas)", "HochSchulz": "HochSchulz\n(melanoma)", "Jackson": "Jackson\n(breast)"}


def load_murphy_matrix(seed: int) -> np.ndarray:
    """3x3 matrix: row=train source, col=test dataset."""
    mat = np.full((3, 3), np.nan)
    for i, src in enumerate(DATASETS):
        p = R / f"murphy_xds_{src}_s{seed}/metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        for ds, res in m["eval"]["in_dist"].items():
            mat[i, DATASETS.index(ds)] = res["mean_pcc"]
        for ds, res in m["eval"]["ood"].items():
            mat[i, DATASETS.index(ds)] = res["mean_pcc"]
    return mat


def load_v2_ood(seed: int) -> dict:
    """Return {test_dataset: ood_pcc} from leave-one-out runs."""
    out = {}
    name_map = {"Damond": "D", "HochSchulz": "H", "Jackson": "J"}
    for ds in DATASETS:
        p = R / f"xds_heldout{name_map[ds]}_s{seed}/metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        out[ds] = m["eval"]["ood"][ds]["mean_pcc"]
    return out


def load_v2_pooled(seed: int) -> dict:
    """Pooled v2 in-dist per dataset."""
    p = R / f"xds_pooled_s{seed}/metrics.json"
    if not p.exists():
        return {}
    m = json.load(open(p))
    return {ds: m["eval"]["in_dist"][ds]["mean_pcc"] for ds in DATASETS}


def fig_transfer_comparison(seed: int = 42):
    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])

    # --- Panel 1: Murphy transfer matrix ---
    ax1 = fig.add_subplot(gs[0, 0])
    mat = load_murphy_matrix(seed)
    im = ax1.imshow(mat, cmap="YlOrRd", vmin=0.10, vmax=0.50, aspect="equal")
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels([DS_SHORT[d] for d in DATASETS], fontsize=8)
    ax1.set_yticklabels([DS_SHORT[d] for d in DATASETS], fontsize=8)
    ax1.set_xlabel("Test dataset")
    ax1.set_ylabel("Trained on")
    ax1.set_title(f"Murphy per-source (seed={seed})\nDiag=in-dist, off=OOD transfer", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="white" if mat[i,j] > 0.3 else "black",
                     weight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="mean PCC")

    # --- Panel 2: v2 transfer matrix (using pooled for diag, leave-one-out for off) ---
    ax2 = fig.add_subplot(gs[0, 1])
    mat_v2 = np.full((3, 3), np.nan)
    v2_pool = load_v2_pooled(seed)
    v2_ood = load_v2_ood(seed)
    for i, train in enumerate(DATASETS):
        # Use pooled in-dist for the diagonal
        mat_v2[i, i] = v2_pool.get(train, np.nan)
    # For off-diagonal we use the leave-one-out OOD: train=others, test=test_ds
    # Place in row=any-train-source (we'll duplicate in all non-self rows for the OOD column)
    for j, test in enumerate(DATASETS):
        ood = v2_ood.get(test, np.nan)
        for i in range(3):
            if i != j:
                mat_v2[i, j] = ood
    im2 = ax2.imshow(mat_v2, cmap="YlOrRd", vmin=0.10, vmax=0.50, aspect="equal")
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels([DS_SHORT[d] for d in DATASETS], fontsize=8)
    ax2.set_yticklabels([DS_SHORT[d] for d in DATASETS], fontsize=8)
    ax2.set_xlabel("Test dataset")
    ax2.set_ylabel("Trained on")
    ax2.set_title(f"v2 pooled/LOO (seed={seed})\nDiag=pooled train, off=held-out OOD", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{mat_v2[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="white" if mat_v2[i,j] > 0.3 else "black",
                     weight="bold" if i == j else "normal")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="mean PCC")

    # --- Panel 3: OOD head-to-head ---
    ax3 = fig.add_subplot(gs[0, 2])
    # For each test dataset, Murphy's BEST OOD source vs v2 OOD
    murphy_ood_best = []
    v2_ood_vals = []
    for j, test in enumerate(DATASETS):
        # Murphy OOD = max across off-diagonal rows
        vals = [mat[i, j] for i in range(3) if i != j and not np.isnan(mat[i, j])]
        murphy_ood_best.append(max(vals) if vals else np.nan)
        v2_ood_vals.append(v2_ood.get(test, np.nan))
    x = np.arange(3)
    w = 0.35
    b1 = ax3.bar(x - w/2, murphy_ood_best, w, label="Murphy (best source)",
                  color="#c0504d", edgecolor="k", lw=0.5)
    b2 = ax3.bar(x + w/2, v2_ood_vals, w, label="v2 (pooled LOO)",
                  color="#3c78b4", edgecolor="k", lw=0.5)
    for bar, val in zip(b1, murphy_ood_best):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                 ha="center", fontsize=8)
    for bar, val in zip(b2, v2_ood_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                 ha="center", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.split("\n")[0] for d in [DS_SHORT[d] for d in DATASETS]], fontsize=8)
    ax3.set_ylabel("OOD mean PCC")
    ax3.set_title("OOD head-to-head\n(zero-shot transfer)", fontsize=10)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_axisbelow(True)
    ax3.set_ylim(0, 0.50)

    fig.suptitle(
        "Cross-dataset transfer — SpaProtFM v2 vs Murphy (seed=42 preview; multi-seed pending)",
        y=1.02, fontsize=11, weight="bold"
    )
    fig.tight_layout()
    out = FIGDIR / "xds_preview_s42.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_transfer_comparison(seed=42)
