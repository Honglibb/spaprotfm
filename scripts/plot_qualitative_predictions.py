"""Qualitative prediction figure for the SpaProtFM paper.

For each held-out IMC cohort (Damond, HochSchulz, Jackson), load the
corresponding leave-one-out v2 checkpoint (trained without ever seeing that
cohort), run inference on tiles from the held-out cohort, pick a
representative tile, and render:

  Row = dataset
  cols: [morphology composite] [GT m1] [Pred m1] [GT m2] [Pred m2] [scatter]

where m1=CD45 and m2=Ki67 (diverse textures: diffuse immune infiltrate vs
punctate proliferation). The scatter column is a 2D density of GT vs Pred
per-pixel intensity pooled across all 6 target markers for that tile.

The "representative" tile per dataset is picked as the one with median
mean-target-PCC among the top 50% by tile dynamic range, to avoid showing
either cherry-picked wins or empty tiles.

Outputs results/figures/paper/figure_qualitative.{pdf,png}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from spaprotfm.condition.phikon import PhikonEncoder
from spaprotfm.condition.pseudo_he import (
    synthesize_pseudo_he,
    synthesize_pseudo_he_batch,
)
from spaprotfm.data.bodenmiller import (
    DEFAULT_CANONICAL_MARKERS,
    load_imc_rds,
    project_to_canonical_panel,
)
from spaprotfm.data.normalization import normalize_image
from spaprotfm.data.tiling import tile_image
from spaprotfm.eval.metrics import pearson_correlation
from spaprotfm.models.spaprotfm_v0 import build_masked_input, fixed_mask
from spaprotfm.models.spaprotfm_v2 import MaskedUNetV2

R = Path("/home/zkgy/hongliyin_computer/results")
OUT = R / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

DATASET_RDS = {
    "Damond":     "/code/zkgy/hongliyin_computer/data/raw/imc/Damond_2019_Pancreas_images.rds",
    "HochSchulz": "/code/zkgy/hongliyin_computer/data/raw/imc/HochSchulz_2022_Melanoma_images_protein.rds",
    "Jackson":    "/code/zkgy/hongliyin_computer/data/raw/imc/JacksonFischer_2020_BreastCancer_images_basel.rds",
}
HELDOUT_CKPT = {
    "Damond":     R / "xds_heldoutD_s42" / "model.pt",
    "HochSchulz": R / "xds_heldoutH_s42" / "model.pt",
    "Jackson":    R / "xds_heldoutJ_s42" / "model.pt",
}
DS_TITLE = {
    "Damond": "Damond",
    "HochSchulz": "HochSchulz",
    "Jackson": "Jackson",
}

CANON = DEFAULT_CANONICAL_MARKERS
DNA_IDX = [0, 1]
EVAL_OBS = [0, 1, 2, 3]  # DNA1, DNA2, H3, SMA
TARGET_IDX = [i for i in range(len(CANON)) if i not in EVAL_OBS]  # 6 bio targets
BIO_IDX_FOR_HE = [i for i in range(len(CANON)) if i not in set(DNA_IDX)]

DISPLAY_MARKERS = ["CD45", "CD68"]  # two markers shown as image columns

PATCH = 128
BATCH = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_IMGS_PER_DATASET = 8  # limit for speed; we just need a handful of tiles


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def robust_scale(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def load_model(ckpt: Path) -> MaskedUNetV2:
    model = MaskedUNetV2(
        n_channels=len(CANON), base=48, cond_in=1024, cond_dim=192, cond_grid=14,
    ).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_tile(model: MaskedUNetV2, phikon: PhikonEncoder, tile: np.ndarray) -> np.ndarray:
    """tile: (H, W, C) float32. Returns (H, W, C) predicted full stack."""
    x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    phikon_dtype = next(phikon.model.parameters()).dtype
    rgb = synthesize_pseudo_he_batch(x.detach().cpu(), DNA_IDX, BIO_IDX_FOR_HE).to(DEVICE)
    if phikon_dtype is not None:
        rgb = rgb.to(phikon_dtype)
    with torch.no_grad():
        cond = phikon.encode(rgb).float()
        m = fixed_mask(len(CANON), EVAL_OBS, 1, device=DEVICE)
        inp = build_masked_input(x, m)
        pred = model(inp, cond)
    return pred.squeeze(0).permute(1, 2, 0).cpu().numpy()


def pick_tile(imgs, model, phikon, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Tile all imgs, pick the tile with median mean-target PCC among tiles in
    the top 50% by target-channel dynamic range. Returns (gt_tile, pred_tile).
    """
    candidates = []  # list of (gt_tile, target_range)
    for im in imgs:
        normed = normalize_image(im.image).astype(np.float32)
        tiles, _ = tile_image(normed, patch_size=PATCH, stride=PATCH)
        for t in tiles:
            tgt = t[..., TARGET_IDX]
            dyn = float(np.mean([np.ptp(tgt[..., k]) for k in range(tgt.shape[-1])]))
            candidates.append((t, dyn))
    if not candidates:
        raise RuntimeError("No tiles found")
    # top half by dynamic range
    ranked = sorted(candidates, key=lambda x: -x[1])
    top = ranked[: max(1, len(ranked) // 2)]
    # predict for each and pick median by mean PCC
    preds = []
    pccs = []
    for gt, _ in top:
        pred = predict_tile(model, phikon, gt)
        gt_tgt = gt[..., TARGET_IDX]
        pr_tgt = pred[..., TARGET_IDX]
        per = pearson_correlation(
            pr_tgt.reshape(PATCH, PATCH, -1),
            gt_tgt.reshape(PATCH, PATCH, -1),
            per_channel=True,
        )
        preds.append(pred)
        pccs.append(float(np.nanmean(per)))
    order = np.argsort(pccs)
    mid = order[len(order) // 2]
    return top[mid][0], preds[mid]


def pseudo_he_rgb(tile: np.ndarray) -> np.ndarray:
    """Run the same pseudo-H&E synthesis that feeds the Phikon conditioner.

    Returns (H, W, 3) uint-safe float32 in [0, 1].
    """
    # tile is (H, W, C); synthesize_pseudo_he accepts (C, H, W) or (H, W, C).
    rgb = synthesize_pseudo_he(tile, DNA_IDX, BIO_IDX_FOR_HE)
    return rgb.permute(1, 2, 0).cpu().numpy()


def main() -> None:
    setup_style()
    rng = np.random.default_rng(SEED)

    print(f"[qual] loading Phikon-v2 (shared across datasets) on {DEVICE}")
    phikon = PhikonEncoder(model_id="owkin/phikon-v2").to(DEVICE)
    phikon.eval()

    per_ds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ds in ("Damond", "HochSchulz", "Jackson"):
        print(f"[qual] {ds}: loading images and checkpoint ...")
        raw = load_imc_rds(DATASET_RDS[ds], max_images=MAX_IMGS_PER_DATASET)
        imgs = project_to_canonical_panel(raw, ds)
        model = load_model(HELDOUT_CKPT[ds])
        gt_tile, pred_tile = pick_tile(imgs, model, phikon, rng)
        per_ds[ds] = (gt_tile, pred_tile)
        del model
        torch.cuda.empty_cache()

    print("[qual] rendering figure")
    ncols = 1 + 2 * len(DISPLAY_MARKERS) + 1  # morph, (GT, pred)*N, scatter
    nrows = 3
    # Wider + taller; give the scatter column extra room.
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.2, 3.8),
        gridspec_kw={"width_ratios": [1] * (ncols - 1) + [1.15]},
    )

    col_headers = ["pseudo-H&E\n(Phikon input)"]
    for mk in DISPLAY_MARKERS:
        col_headers += [f"{mk}\nground truth", f"{mk}\nv2 prediction"]
    col_headers += ["per-pixel density\n(6 targets pooled)"]

    for row, ds in enumerate(("Damond", "HochSchulz", "Jackson")):
        gt, pred = per_ds[ds]
        col = 0
        # Pseudo-H&E (same synthesis that feeds Phikon-v2 conditioning)
        axes[row, col].imshow(pseudo_he_rgb(gt))
        # Row label as axis ylabel — will be visible once spines are off but
        # axes remain allocated.
        axes[row, col].set_ylabel(
            DS_TITLE[ds], fontsize=8, fontweight="bold",
            labelpad=6, rotation=90,
        )
        col += 1

        for mk in DISPLAY_MARKERS:
            idx = CANON.index(mk)
            g = robust_scale(gt[..., idx])
            p = robust_scale(pred[..., idx])
            axes[row, col].imshow(g, cmap="magma", vmin=0, vmax=1)
            col += 1
            axes[row, col].imshow(p, cmap="magma", vmin=0, vmax=1)
            col += 1

        # Per-pixel density hexbin (all 6 targets pooled)
        gt_all = gt[..., TARGET_IDX].reshape(-1)
        pr_all = pred[..., TARGET_IDX].reshape(-1)
        ax_s = axes[row, col]
        lo = 0.0
        hi = max(float(gt_all.max()), float(pr_all.max()))
        ax_s.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=0.6, zorder=1)
        ax_s.hexbin(gt_all, pr_all, gridsize=35, cmap="viridis", mincnt=1,
                    bins="log", extent=(lo, hi, lo, hi), zorder=2)
        r = float(np.corrcoef(gt_all, pr_all)[0, 1])
        ax_s.text(0.05, 0.95, f"r = {r:.2f}", transform=ax_s.transAxes,
                  fontsize=7, fontweight="bold", va="top", ha="left",
                  bbox=dict(facecolor="white", edgecolor="#cccccc",
                            alpha=0.9, pad=1.5, lw=0.4))
        ax_s.set_xlim(lo, hi)
        ax_s.set_ylim(lo, hi)
        ax_s.set_aspect("equal")
        ax_s.tick_params(labelsize=6, length=2, pad=1.5)
        # Keep ticks minimal
        ax_s.set_xticks([0, round(hi, 1)])
        ax_s.set_yticks([0, round(hi, 1)])
        if row == 2:
            ax_s.set_xlabel("GT", fontsize=7, labelpad=1)
        if row == 1:
            ax_s.set_ylabel("Pred", fontsize=7, labelpad=1)

    # Strip ticks/spines on image panels (keep scatter panels tidy)
    for row in range(nrows):
        for col in range(ncols - 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for s in axes[row, col].spines.values():
                s.set_visible(False)

    # Column headers above top row
    for col, h in enumerate(col_headers):
        axes[0, col].set_title(h, fontsize=7, pad=4)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.87, bottom=0.09,
                        wspace=0.12, hspace=0.08)
    out_pdf = OUT / "figure_qualitative.pdf"
    out_png = OUT / "figure_qualitative.png"
    # Larger pad so rotated row labels aren't clipped by tight bbox.
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
