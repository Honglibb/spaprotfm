"""BIB submission-ready paper figures for the SpaProtFM manuscript.

Produces vector PDFs (primary) + 600-dpi PNG (for Word drafts) at Briefings in
Bioinformatics column widths:
  - single column  : 86 mm  = 3.39"
  - 1.5 column     : 120 mm = 4.72"
  - double column  : 178 mm = 7.01"

Typography: sans-serif, 8 pt base, 7 pt ticks, 9 pt panel labels bold.
Palette   : Okabe-Ito (colorblind-safe). Murphy = vermillion, v2 = blue, v1 = orange.

Three figures (in the order the story is told in the paper body):

  fig_he_ablation        v1 vs v2 panel sweep — shows Phikon pseudo-H&E is the
                         active ingredient across panel sizes.
  fig_panel_flexibility  single v2 checkpoint vs Murphy retrained per panel size.
  fig_xds_transfer       single v2 checkpoint vs per-panel Murphy on zero-shot
                         transfer to held-out cohorts (THE killer figure).

Outputs go to results/figures/paper/*.{pdf,png}. /results/ is gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

R = Path("/home/zkgy/hongliyin_computer/results")
OUT = R / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# --- Column widths (inches) -------------------------------------------------
W_SINGLE = 3.39
W_ONE_HALF = 4.72
W_DOUBLE = 7.09

# --- Okabe-Ito palette ------------------------------------------------------
C_MURPHY = "#D55E00"  # vermillion
C_V2 = "#0072B2"      # blue
C_V1 = "#E69F00"      # orange
C_GREY = "#666666"
C_MUTE = "#BBBBBB"

DATASETS = ["Damond", "HochSchulz", "Jackson"]
DS_TITLE = {
    "Damond":     "Damond pancreas",
    "HochSchulz": "HochSchulz melanoma",
    "Jackson":    "Jackson breast",
}
DS_SHORT = {"Damond": "D", "HochSchulz": "H", "Jackson": "J"}
XDS_MARKERS = ["CD20", "CD3e", "CD45", "CD68", "Ki67", "cPARP"]


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, label: str, dx: float = -0.17, dy: float = 1.06) -> None:
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"Wrote {OUT}/{name}.{{pdf,png}}")


# =============================================================================
# Figure: H&E ablation — v1 vs v2 panel sweep (3 datasets)
# =============================================================================

HE_CONFIG = [
    ("Damond pancreas",     R / "v1_sweep_damond",     R / "v2_sweep_damond",     0.421),
    ("HochSchulz melanoma", R / "v1_sweep_hochschulz", R / "v2_sweep_hochschulz", 0.493),
    ("Jackson breast",      R / "v1_sweep_jackson",    R / "v2_sweep_jackson",    0.376),
]


def _panel_sweep(path: Path):
    data = json.loads((path / "panel_sweep.json").read_text())
    sizes = sorted({r["panel_size"] for r in data})
    means = [float(np.mean([r["mean_pcc_bio"] for r in data if r["panel_size"] == s]))
             for s in sizes]
    stds = [float(np.std([r["mean_pcc_bio"] for r in data if r["panel_size"] == s]))
            for s in sizes]
    return sizes, means, stds


def fig_he_ablation() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.7), sharey=False)
    for ax, (title, v1_dir, v2_dir, base_pcc), label in zip(axes, HE_CONFIG, "abc"):
        s1, m1, e1 = _panel_sweep(v1_dir)
        s2, m2, e2 = _panel_sweep(v2_dir)
        ax.errorbar(s1, m1, yerr=e1, fmt="-o", color=C_V1, capsize=3,
                    lw=1.3, ms=4.5, label="v1 (no H&E)")
        ax.errorbar(s2, m2, yerr=e2, fmt="-s", color=C_V2, capsize=3,
                    lw=1.3, ms=4.5, label="v2 (+ Phikon)")
        ax.axhline(base_pcc, color=C_GREY, ls="--", lw=0.8, alpha=0.7,
                   label="Murphy (sz=10)")
        ax.set_xlabel("Observed panel size")
        if label == "a":
            ax.set_ylabel("Mean PCC (bio targets)")
        ax.set_title(title)
        ax.set_xticks(sorted(set(s1) | set(s2)))
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        panel_label(ax, label)
    # Shared legend above all panels — avoids per-panel crowding.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.8,
               columnspacing=2.0, borderpad=0.3)
    # Per-panel baseline value annotated at the left end (whitespace above
    # the starting data points) so it does not sit on top of the dashed line.
    for ax, (_, _, _, base_pcc) in zip(axes, HE_CONFIG):
        ax.annotate(f"{base_pcc:.2f}", xy=(ax.get_xlim()[0], base_pcc),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=C_GREY, ha="left", va="bottom")
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.93))
    save(fig, "figure_he_ablation")


# =============================================================================
# Figure: Panel flexibility — single v2 checkpoint vs per-size Murphy
# =============================================================================

FLEX_CONFIG = [
    ("Damond pancreas", "damond",
     [("baseline_damond_sz7", 7),
      ("baseline_damond", 10),
      ("baseline_damond_sz15", 15),
      ("baseline_damond_sz20", 20)]),
    ("HochSchulz melanoma", "hochschulz",
     [("baseline_hochschulz_sz7", 7),
      ("baseline_hochschulz", 10),
      ("baseline_hochschulz_sz15", 15),
      ("baseline_hochschulz_sz20", 20)]),
    ("Jackson breast", "jackson",
     [("baseline_jackson", 10),
      ("baseline_jackson_sz15", 15),
      ("baseline_jackson_sz20", 20)]),
]


def _murphy_mean_excl_dna(path: Path) -> float:
    m = json.loads((path / "metrics.json").read_text())
    per = m["test_pcc_per_marker"]
    return float(np.mean([v for k, v in per.items() if k not in ("DNA1", "DNA2")]))


def fig_panel_flexibility() -> None:
    common_ticks = [3, 7, 10, 15, 20]
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.7), sharey=False)
    for ax, (title, key, murphy_rows), label in zip(axes, FLEX_CONFIG, "abc"):
        sizes, means, stds = _panel_sweep(R / f"v2_sweep_{key}")
        ax.plot(sizes, means, "-o", color=C_V2, lw=1.4, ms=4.5,
                label="SpaProtFM v2 (one checkpoint)", zorder=3)
        ax.fill_between(sizes,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=C_V2, alpha=0.18, lw=0)
        mu_vals = [_murphy_mean_excl_dna(R / d) for d, _ in murphy_rows]
        mu_x = [s for _, s in murphy_rows]
        ax.scatter(mu_x, mu_vals, marker="D", s=42, color=C_MURPHY,
                   edgecolor="k", lw=0.5, zorder=4,
                   label="Murphy (retrained per size)")
        ax.set_xlabel("Observed panel size")
        if label == "a":
            ax.set_ylabel("Mean PCC (bio targets, excl. DNA)")
        ax.set_title(title)
        ax.set_xlim(2.0, 21.0)
        ax.set_xticks(common_ticks)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        panel_label(ax, label)
    # Shared legend above all panels, matches HE ablation layout.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.8,
               columnspacing=2.5, borderpad=0.3)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.93))
    save(fig, "figure_panel_flexibility")


# =============================================================================
# Figure: Cross-dataset transfer
# =============================================================================

NAME_MAP = {"Damond": "D", "HochSchulz": "H", "Jackson": "J"}


def _seeds_v2() -> list[int]:
    seeds = set()
    for p in R.glob("xds_pooled_s*"):
        if (p / "metrics.json").exists():
            try:
                seeds.add(int(p.name.split("_s")[-1]))
            except ValueError:
                pass
    return sorted(seeds)


def _seeds_murphy() -> list[int]:
    seeds = set()
    for p in R.glob("murphy_xds_Damond_s*"):
        if (p / "metrics.json").exists():
            try:
                seeds.add(int(p.name.split("_s")[-1]))
            except ValueError:
                pass
    return sorted(seeds)


def _v2_pooled(seed: int) -> dict:
    p = R / f"xds_pooled_s{seed}/metrics.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())
    return {ds: m["eval"]["in_dist"][ds]["mean_pcc"] for ds in DATASETS
            if ds in m["eval"]["in_dist"]}


def _v2_ood(seed: int) -> dict:
    out = {}
    for ds in DATASETS:
        p = R / f"xds_heldout{NAME_MAP[ds]}_s{seed}/metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        if ds in m["eval"]["ood"]:
            out[ds] = m["eval"]["ood"][ds]["mean_pcc"]
    return out


def _v2_ood_per_marker(seed: int) -> dict:
    out = {}
    for ds in DATASETS:
        p = R / f"xds_heldout{NAME_MAP[ds]}_s{seed}/metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        out[ds] = m["eval"]["ood"][ds]["pcc_per_marker"]
    return out


def _murphy_matrix(seed: int) -> np.ndarray:
    mat = np.full((3, 3), np.nan)
    for i, src in enumerate(DATASETS):
        p = R / f"murphy_xds_{src}_s{seed}/metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        for ds, res in m["eval"]["in_dist"].items():
            mat[i, DATASETS.index(ds)] = res["mean_pcc"]
        for ds, res in m["eval"]["ood"].items():
            mat[i, DATASETS.index(ds)] = res["mean_pcc"]
    return mat


def _murphy_ood_best_per_marker(seed: int) -> dict:
    out = {}
    mat = _murphy_matrix(seed)
    for j, test in enumerate(DATASETS):
        best_i, best = -1, -np.inf
        for i in range(3):
            if i == j or np.isnan(mat[i, j]):
                continue
            if mat[i, j] > best:
                best, best_i = mat[i, j], i
        if best_i < 0:
            continue
        src = DATASETS[best_i]
        m = json.loads((R / f"murphy_xds_{src}_s{seed}/metrics.json").read_text())
        out[test] = m["eval"]["ood"][test]["pcc_per_marker"]
    return out


def _agg(xs):
    a = np.array([x for x in xs if not np.isnan(x)])
    return (float(a.mean()), float(a.std(ddof=0))) if a.size else (np.nan, 0.0)


def fig_xds_transfer() -> None:
    seeds = sorted(set(_seeds_v2()) & set(_seeds_murphy()))
    if not seeds:
        raise SystemExit("No paired xds seeds found")

    v2_pool = {ds: [] for ds in DATASETS}
    v2_ood = {ds: [] for ds in DATASETS}
    m_id = {ds: [] for ds in DATASETS}
    m_ood = {ds: [] for ds in DATASETS}
    for s in seeds:
        for ds, v in _v2_pooled(s).items():
            v2_pool[ds].append(v)
        for ds, v in _v2_ood(s).items():
            v2_ood[ds].append(v)
        mat = _murphy_matrix(s)
        for j, ds in enumerate(DATASETS):
            if not np.isnan(mat[j, j]):
                m_id[ds].append(mat[j, j])
            off = [mat[i, j] for i in range(3) if i != j and not np.isnan(mat[i, j])]
            if off:
                m_ood[ds].append(max(off))

    fig = plt.figure(figsize=(W_DOUBLE, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35)

    # --- (a) OOD per-dataset head-to-head ---
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    w = 0.34
    mu = [_agg(m_ood[ds])[0] for ds in DATASETS]
    mus = [_agg(m_ood[ds])[1] for ds in DATASETS]
    v2 = [_agg(v2_ood[ds])[0] for ds in DATASETS]
    v2s = [_agg(v2_ood[ds])[1] for ds in DATASETS]
    ax.bar(x - w/2, mu, w, yerr=mus, color=C_MURPHY, edgecolor="k", lw=0.4,
           capsize=3, error_kw={"lw": 0.8}, label="Murphy (best source)")
    ax.bar(x + w/2, v2, w, yerr=v2s, color=C_V2, edgecolor="k", lw=0.4,
           capsize=3, error_kw={"lw": 0.8}, label="SpaProtFM v2 (pooled LOO)")
    for i in range(3):
        delta = v2[i] - mu[i]
        y = max(v2[i] + v2s[i], mu[i] + mus[i]) + 0.03
        ax.text(i, y, f"Δ = {delta:+.2f}", ha="center", fontsize=7,
                color="black", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_TITLE[d].replace(" ", "\n", 1) for d in DATASETS])
    ax.set_ylabel("OOD mean PCC")
    ax.set_ylim(0, 0.52)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=1.4, borderpad=0.3)
    panel_label(ax, "a")

    # --- (b) In-dist vs OOD, overall average ---
    ax2 = fig.add_subplot(gs[0, 1])
    all_v2_id = [v for ds in DATASETS for v in v2_pool[ds]]
    all_v2_ood = [v for ds in DATASETS for v in v2_ood[ds]]
    all_m_id = [v for ds in DATASETS for v in m_id[ds]]
    all_m_ood = [v for ds in DATASETS for v in m_ood[ds]]
    xb = np.arange(2)
    wb = 0.34
    vals_m = [_agg(all_m_id)[0], _agg(all_m_ood)[0]]
    errs_m = [_agg(all_m_id)[1], _agg(all_m_ood)[1]]
    vals_v = [_agg(all_v2_id)[0], _agg(all_v2_ood)[0]]
    errs_v = [_agg(all_v2_id)[1], _agg(all_v2_ood)[1]]
    ax2.bar(xb - wb/2, vals_m, wb, yerr=errs_m, color=C_MURPHY, edgecolor="k",
            lw=0.4, capsize=3, error_kw={"lw": 0.8}, label="Murphy")
    ax2.bar(xb + wb/2, vals_v, wb, yerr=errs_v, color=C_V2, edgecolor="k",
            lw=0.4, capsize=3, error_kw={"lw": 0.8}, label="SpaProtFM v2")
    for i in range(2):
        ax2.text(xb[i] - wb/2, vals_m[i] + errs_m[i] + 0.01,
                 f"{vals_m[i]:.2f}", ha="center", fontsize=7, color=C_MURPHY)
        ax2.text(xb[i] + wb/2, vals_v[i] + errs_v[i] + 0.01,
                 f"{vals_v[i]:.2f}", ha="center", fontsize=7,
                 color=C_V2, fontweight="bold")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(["in-distribution", "out-of-distribution"])
    ax2.set_ylabel("mean PCC (averaged over 3 datasets)")
    ax2.set_ylim(0, 0.52)
    ax2.grid(axis="y", alpha=0.25, lw=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", handlelength=1.4, borderpad=0.3)
    panel_label(ax2, "b")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14, wspace=0.35)
    save(fig, "figure_xds_transfer")


# =============================================================================
# Merged Figure 2: method validation (HE ablation + panel flexibility)
# =============================================================================


def fig_method_validation() -> None:
    """Two-row figure combining H&E ablation (top) and panel flexibility (bottom).

    Row 1 (a–c): v1 vs v2 panel sweep, 3 datasets. Grey dashed line = per-dataset
                 Murphy baseline at size=10 (numeric value annotated).
    Row 2 (d–f): single v2 checkpoint (line + band) vs per-size-retrained Murphy
                 diamonds; demonstrates panel flexibility.
    """
    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 4.9), sharey=False)

    # ---------- Row 1: H&E ablation ----------
    for ax, (title, v1_dir, v2_dir, base_pcc), label in zip(axes[0], HE_CONFIG, "abc"):
        s1, m1, e1 = _panel_sweep(v1_dir)
        s2, m2, e2 = _panel_sweep(v2_dir)
        ax.errorbar(s1, m1, yerr=e1, fmt="-o", color=C_V1, capsize=3,
                    lw=1.3, ms=4.5, label="v1 (no H&E)")
        ax.errorbar(s2, m2, yerr=e2, fmt="-s", color=C_V2, capsize=3,
                    lw=1.3, ms=4.5, label="v2 (+ Phikon)")
        ax.axhline(base_pcc, color=C_GREY, ls="--", lw=0.8, alpha=0.7,
                   label="Murphy (sz=10)")
        ax.annotate(f"{base_pcc:.2f}", xy=(float(min(s1)), base_pcc),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=C_GREY, ha="left", va="bottom")
        ax.set_xlabel("Observed panel size")
        if label == "a":
            ax.set_ylabel("Mean PCC (bio targets)")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(sorted(set(s1) | set(s2)))
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        panel_label(ax, label, dx=-0.18, dy=1.08)

    # Row-1 shared legend above top row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    row1_legend = fig.legend(handles, labels, loc="upper center", ncol=3,
                             bbox_to_anchor=(0.5, 0.99), handlelength=1.8,
                             columnspacing=2.0, borderpad=0.3, frameon=False)
    fig.add_artist(row1_legend)

    # ---------- Row 2: panel flexibility ----------
    common_ticks = [3, 7, 10, 15, 20]
    for ax, (title, key, murphy_rows), label in zip(axes[1], FLEX_CONFIG, "def"):
        sizes, means, stds = _panel_sweep(R / f"v2_sweep_{key}")
        ax.plot(sizes, means, "-o", color=C_V2, lw=1.4, ms=4.5,
                label="SpaProtFM v2 (one checkpoint)", zorder=3)
        ax.fill_between(sizes,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=C_V2, alpha=0.18, lw=0)
        mu_vals = [_murphy_mean_excl_dna(R / d) for d, _ in murphy_rows]
        mu_x = [s for _, s in murphy_rows]
        ax.scatter(mu_x, mu_vals, marker="D", s=42, color=C_MURPHY,
                   edgecolor="k", lw=0.5, zorder=4,
                   label="Murphy (retrained per size)")
        ax.set_xlabel("Observed panel size")
        if label == "d":
            ax.set_ylabel("Mean PCC (bio, excl. DNA)")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(2.0, 21.0)
        ax.set_xticks(common_ticks)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        panel_label(ax, label, dx=-0.18, dy=1.08)

    # Row-2 shared legend centered between rows
    handles, labels = axes[1, 0].get_legend_handles_labels()
    row2_legend = fig.legend(handles, labels, loc="center", ncol=2,
                             bbox_to_anchor=(0.5, 0.515), handlelength=1.8,
                             columnspacing=2.5, borderpad=0.3, frameon=False)
    fig.add_artist(row2_legend)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.08,
                        wspace=0.28, hspace=0.78)
    save(fig, "figure_method_validation")


# =============================================================================
# Merged Figure 3: cross-dataset transfer (heatmap + head-to-head + summary)
# =============================================================================

from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def fig_cross_dataset_transfer() -> None:
    """Three-panel figure for the killer cross-dataset transfer story.

    (a) 4×3 transfer heatmap: Murphy-on-{D,H,J} + v2 LOO rows; test cohort cols.
    (b) OOD per-dataset head-to-head bars (v2 vs Murphy best-source).
    (c) Overall in-distribution vs out-of-distribution mean PCC.
    """
    seeds = sorted(set(_seeds_v2()) & set(_seeds_murphy()))
    if not seeds:
        raise SystemExit("No paired xds seeds found")

    # Aggregate
    v2_pool = {ds: [] for ds in DATASETS}
    v2_ood = {ds: [] for ds in DATASETS}
    m_id = {ds: [] for ds in DATASETS}
    m_ood = {ds: [] for ds in DATASETS}
    murphy_stack = []
    v2_loo_stack = []
    for s in seeds:
        for ds, v in _v2_pooled(s).items():
            v2_pool[ds].append(v)
        for ds, v in _v2_ood(s).items():
            v2_ood[ds].append(v)
        mat = _murphy_matrix(s)
        murphy_stack.append(mat)
        for j, ds in enumerate(DATASETS):
            if not np.isnan(mat[j, j]):
                m_id[ds].append(mat[j, j])
            off = [mat[i, j] for i in range(3) if i != j and not np.isnan(mat[i, j])]
            if off:
                m_ood[ds].append(max(off))
        loo_row = np.full(3, np.nan)
        for j, ds in enumerate(DATASETS):
            if ds in _v2_ood(s):
                loo_row[j] = _v2_ood(s)[ds]
        v2_loo_stack.append(loo_row)

    murphy_mean = np.nanmean(np.stack(murphy_stack, axis=0), axis=0)
    v2_loo_mean = np.nanmean(np.stack(v2_loo_stack, axis=0), axis=0)
    heat_mat = np.vstack([murphy_mean, v2_loo_mean])  # (4, 3)

    fig = plt.figure(figsize=(W_DOUBLE, 3.3))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.4, 1.1, 0.85], wspace=0.58,
    )

    # --- (a) heatmap ---
    ax = fig.add_subplot(gs[0, 0])
    vmin = float(np.nanmin(heat_mat))
    vmax = float(np.nanmax(heat_mat))
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(heat_mat, cmap="viridis", norm=norm, aspect="auto")
    for i in range(heat_mat.shape[0]):
        for j in range(heat_mat.shape[1]):
            v = heat_mat[i, j]
            if np.isnan(v):
                continue
            tc = "white" if norm(v) < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")
    ax.axhline(2.5, color="white", lw=2.2)
    for i in range(3):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor="white", lw=0.9, linestyle=":"))
    for j in range(3):
        i_best = int(np.nanargmax(heat_mat[:, j]))
        ax.add_patch(Rectangle((j - 0.5, i_best - 0.5), 1, 1,
                               fill=False, edgecolor="#e74c3c", lw=1.5))
    ax.set_xticks(range(3))
    ax.set_xticklabels([d for d in DATASETS], fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(
        ["Murphy (D)", "Murphy (H)", "Murphy (J)", "SpaProtFM v2 (LOO)"],
        fontsize=7,
    )
    ax.set_xlabel("test cohort", fontsize=7.5, labelpad=3)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, shrink=0.7)
    cbar.set_label("mean PCC", fontsize=6.5, labelpad=2)
    cbar.ax.tick_params(labelsize=6)
    panel_label(ax, "a", dx=-0.22, dy=1.08)

    # --- (b) OOD per-dataset head-to-head ---
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    w = 0.34
    mu = [_agg(m_ood[ds])[0] for ds in DATASETS]
    mus = [_agg(m_ood[ds])[1] for ds in DATASETS]
    v2 = [_agg(v2_ood[ds])[0] for ds in DATASETS]
    v2s = [_agg(v2_ood[ds])[1] for ds in DATASETS]
    ax2.bar(x - w/2, mu, w, yerr=mus, color=C_MURPHY, edgecolor="k", lw=0.4,
            capsize=3, error_kw={"lw": 0.8}, label="Murphy (best src)")
    ax2.bar(x + w/2, v2, w, yerr=v2s, color=C_V2, edgecolor="k", lw=0.4,
            capsize=3, error_kw={"lw": 0.8}, label="v2 (pooled LOO)")
    for i in range(3):
        delta = v2[i] - mu[i]
        y = max(v2[i] + v2s[i], mu[i] + mus[i]) + 0.03
        ax2.text(i, y, f"Δ {delta:+.2f}", ha="center", fontsize=6.5,
                 color="black", style="italic")
    ax2.set_xticks(x)
    ax2.set_xticklabels(DATASETS, fontsize=7, rotation=15, ha="right")
    ax2.set_ylabel("OOD mean PCC")
    ax2.set_ylim(0, 0.52)
    ax2.grid(axis="y", alpha=0.25, lw=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", handlelength=1.4, borderpad=0.3)
    ax2.set_title("zero-shot OOD per cohort", fontsize=8.5)
    panel_label(ax2, "b", dx=-0.2, dy=1.08)

    # --- (c) In-dist vs OOD overall ---
    ax3 = fig.add_subplot(gs[0, 2])
    all_v2_id = [v for ds in DATASETS for v in v2_pool[ds]]
    all_v2_ood = [v for ds in DATASETS for v in v2_ood[ds]]
    all_m_id = [v for ds in DATASETS for v in m_id[ds]]
    all_m_ood = [v for ds in DATASETS for v in m_ood[ds]]
    xb = np.arange(2)
    wb = 0.34
    vals_m = [_agg(all_m_id)[0], _agg(all_m_ood)[0]]
    errs_m = [_agg(all_m_id)[1], _agg(all_m_ood)[1]]
    vals_v = [_agg(all_v2_id)[0], _agg(all_v2_ood)[0]]
    errs_v = [_agg(all_v2_id)[1], _agg(all_v2_ood)[1]]
    ax3.bar(xb - wb/2, vals_m, wb, yerr=errs_m, color=C_MURPHY, edgecolor="k",
            lw=0.4, capsize=3, error_kw={"lw": 0.8}, label="Murphy")
    ax3.bar(xb + wb/2, vals_v, wb, yerr=errs_v, color=C_V2, edgecolor="k",
            lw=0.4, capsize=3, error_kw={"lw": 0.8}, label="v2")
    ax3.set_xticks(xb)
    ax3.set_xticklabels(["in-dist", "OOD"], fontsize=7)
    ax3.set_ylabel("mean PCC (3-dataset avg)")
    ax3.set_ylim(0, 0.52)
    ax3.grid(axis="y", alpha=0.25, lw=0.4)
    ax3.set_axisbelow(True)
    ax3.legend(loc="upper right", handlelength=1.4, borderpad=0.3)
    ax3.set_title("overall", fontsize=8.5)
    panel_label(ax3, "c", dx=-0.28, dy=1.08)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)
    save(fig, "figure_cross_dataset_transfer")


# =============================================================================
# Supplement: per-marker OOD
# =============================================================================

def fig_xds_per_marker() -> None:
    seeds = sorted(set(_seeds_v2()) & set(_seeds_murphy()))
    v2_pm = {ds: {m: [] for m in XDS_MARKERS} for ds in DATASETS}
    mu_pm = {ds: {m: [] for m in XDS_MARKERS} for ds in DATASETS}
    for s in seeds:
        for ds, d in _v2_ood_per_marker(s).items():
            for mk, v in d.items():
                if mk in v2_pm[ds]:
                    v2_pm[ds][mk].append(v)
        for ds, d in _murphy_ood_best_per_marker(s).items():
            for mk, v in d.items():
                if mk in mu_pm[ds]:
                    mu_pm[ds][mk].append(v)

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.6), sharey=True)
    x = np.arange(len(XDS_MARKERS))
    w = 0.34
    for ax, ds, label in zip(axes, DATASETS, "abc"):
        mm = [_agg(mu_pm[ds][k])[0] for k in XDS_MARKERS]
        ms = [_agg(mu_pm[ds][k])[1] for k in XDS_MARKERS]
        vm = [_agg(v2_pm[ds][k])[0] for k in XDS_MARKERS]
        vs = [_agg(v2_pm[ds][k])[1] for k in XDS_MARKERS]
        ax.bar(x - w/2, mm, w, yerr=ms, color=C_MURPHY, edgecolor="k",
               lw=0.4, capsize=3, error_kw={"lw": 1.1},
               label="Murphy (best source)" if label == "c" else None)
        ax.bar(x + w/2, vm, w, yerr=vs, color=C_V2, edgecolor="k",
               lw=0.4, capsize=3, error_kw={"lw": 1.1},
               label="SpaProtFM v2" if label == "c" else None)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(XDS_MARKERS, rotation=25, ha="right")
        ax.set_title(f"OOD on {DS_TITLE[ds]}")
        ax.set_ylim(-0.05, 0.65)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        panel_label(ax, label)
    axes[0].set_ylabel("per-marker PCC (OOD)")
    axes[-1].legend(loc="upper right", handlelength=1.4, borderpad=0.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "figure_xds_per_marker")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    setup_style()
    # Merged main figures for the paper
    fig_method_validation()
    fig_cross_dataset_transfer()
    # Supplementary / alternate-layout figures
    fig_he_ablation()
    fig_panel_flexibility()
    fig_xds_transfer()
    fig_xds_per_marker()
