"""Figure 1 architecture schematic for the SpaProtFM paper.

Two-path layout:

  IMC tile ──┬── mask target channels ─── masked 10-ch input ──────┐
             └── pseudo-H&E synthesis ─── Phikon-v2 (frozen) ──────┤
                                                                   │
                                                  MaskedUNetV2     │
                                                   ▼               │
                                         predicted 10-ch panel  ◄──┘
                                                   │
                                                   ▼
                                    MSE loss on masked positions

Output results/figures/paper/figure_architecture.{pdf,png}.

Placeholder-quality; final version may want Illustrator polish but this
establishes layout, palette, and information content for the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("/home/zkgy/hongliyin_computer/results/figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito adjacent palette (matches main figures)
C_DATA = "#F4F4F4"; C_DATA_EDGE = "#666666"
C_MASK = "#FFE5A5"; C_MASK_EDGE = "#E69F00"
C_HE = "#F0D0E0"; C_HE_EDGE = "#B55D8E"
C_PHIKON = "#D8E3F0"; C_PHIKON_EDGE = "#0072B2"
C_UNET = "#D9EDD9"; C_UNET_EDGE = "#2a7a2a"
C_OUT = "#FFE9D9"; C_OUT_EDGE = "#D55E00"
C_LOSS = "#F9DADA"; C_LOSS_EDGE = "#B94646"
C_ARROW = "#333333"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def rbox(ax, cx, cy, w, h, fc, ec, lw=0.9):
    """Rounded box centred at (cx, cy)."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2,
    )
    ax.add_patch(box)
    return box


def arrow(ax, x0, y0, x1, y1, lw=0.9, color=C_ARROW, ls="-"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, linestyle=ls,
        zorder=3,
    )
    ax.add_patch(a)


def tc(ax, x, y, s, size=8, color="#222", weight="normal",
       ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=4)


def main() -> None:
    setup_style()

    fig, ax = plt.subplots(1, 1, figsize=(7.09, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # x positions for the two parallel paths
    XL = 3.0   # left column (mask path)
    XR = 7.0   # right column (pseudo-H&E + Phikon path)
    XC = 5.0   # centre (shared: input, model, output)
    BOX_W_WIDE = 4.0
    BOX_W = 3.2
    BOX_H = 0.6

    # Row Y positions (top → bottom)
    Y_INPUT = 5.5
    Y_STEP1 = 4.3   # mask / pseudo-H&E
    Y_STEP2 = 3.1   # masked input / Phikon
    Y_MODEL = 1.8
    Y_OUT = 0.55

    # ---------- (1) Input IMC tile ----------
    rbox(ax, XC, Y_INPUT, BOX_W_WIDE, BOX_H, C_DATA, C_DATA_EDGE)
    tc(ax, XC, Y_INPUT + 0.14,
       "IMC tile — 10-marker canonical panel", size=8.5, weight="bold")
    tc(ax, XC, Y_INPUT - 0.14,
       "DNA1  DNA2  H3  SMA   |   CD20  CD3e  CD45  CD68  Ki67  cPARP",
       size=6.5, color="#444")

    # Split arrows to each column
    arrow(ax, XC - 0.6, Y_INPUT - BOX_H / 2, XL, Y_STEP1 + BOX_H / 2)
    arrow(ax, XC + 0.6, Y_INPUT - BOX_H / 2, XR, Y_STEP1 + BOX_H / 2)

    # ---------- (2a) Left: mask target channels ----------
    rbox(ax, XL, Y_STEP1, BOX_W, BOX_H, C_MASK, C_MASK_EDGE)
    tc(ax, XL, Y_STEP1 + 0.14, "mask target channels",
       size=8, weight="bold")
    tc(ax, XL, Y_STEP1 - 0.14,
       "train: random k-of-6   ·   test: all 6",
       size=6.5, color="#444")

    arrow(ax, XL, Y_STEP1 - BOX_H / 2, XL, Y_STEP2 + BOX_H / 2)

    # ---------- (3a) Left: masked 10-ch input ----------
    rbox(ax, XL, Y_STEP2, BOX_W, BOX_H, C_DATA, C_DATA_EDGE)
    tc(ax, XL, Y_STEP2 + 0.14, "masked 10-channel input",
       size=8, weight="bold")
    tc(ax, XL, Y_STEP2 - 0.14,
       "morphology kept + targets zeroed",
       size=6.5, color="#444")

    # ---------- (2b) Right: pseudo-H&E synthesis ----------
    rbox(ax, XR, Y_STEP1, BOX_W, BOX_H, C_HE, C_HE_EDGE)
    tc(ax, XR, Y_STEP1 + 0.14, "pseudo-H&E synthesis",
       size=8, weight="bold")
    tc(ax, XR, Y_STEP1 - 0.14,
       "DNA → hematoxylin   ·   bio → eosin",
       size=6.5, color="#444")

    arrow(ax, XR, Y_STEP1 - BOX_H / 2, XR, Y_STEP2 + BOX_H / 2)

    # ---------- (3b) Right: Phikon-v2 (frozen) ----------
    rbox(ax, XR, Y_STEP2, BOX_W, BOX_H, C_PHIKON, C_PHIKON_EDGE, lw=1.1)
    tc(ax, XR, Y_STEP2 + 0.14, "Phikon-v2  ❄  (frozen)",
       size=8, weight="bold")
    tc(ax, XR, Y_STEP2 - 0.14,
       "ViT-B   ·   1024-d × 14 × 14 features",
       size=6.5, color="#444")

    # ---------- (4) MaskedUNetV2 ----------
    UNET_W = 5.5
    rbox(ax, XC, Y_MODEL, UNET_W, 0.7, C_UNET, C_UNET_EDGE, lw=1.1)
    tc(ax, XC, Y_MODEL + 0.18, "MaskedUNetV2", size=9, weight="bold")
    tc(ax, XC, Y_MODEL - 0.14,
       "encoder   →   bottleneck  ⊕  Phikon cond   →   decoder",
       size=6.8, color="#333")

    # Converging arrows from both paths into UNet
    arrow(ax, XL, Y_STEP2 - BOX_H / 2, XC - 1.6, Y_MODEL + 0.35)
    arrow(ax, XR, Y_STEP2 - BOX_H / 2, XC + 1.6, Y_MODEL + 0.35,
          color=C_PHIKON_EDGE, lw=1.1)

    # ---------- (5) Output ----------
    rbox(ax, XC, Y_OUT, BOX_W_WIDE, BOX_H, C_OUT, C_OUT_EDGE, lw=1.1)
    tc(ax, XC, Y_OUT + 0.14,
       "predicted 10-channel panel", size=8.5, weight="bold")
    tc(ax, XC, Y_OUT - 0.14,
       "training loss:   MSE on masked positions",
       size=6.8, color="#444")

    arrow(ax, XC, Y_MODEL - 0.35, XC, Y_OUT + BOX_H / 2)

    # Color semantics are legible from box labels (Phikon=frozen, UNet=trainable,
    # pseudo-H&E=deterministic synthesis, mask=masking). Legend removed to free
    # the left edge.

    out_pdf = OUT / "figure_architecture.pdf"
    out_png = OUT / "figure_architecture.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
