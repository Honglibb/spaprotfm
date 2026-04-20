"""Generate 3 paper-ready figures for the v2 vs Murphy multi-seed story.

1. Headline: v2+Phikon vs Murphy (fair, DNA excluded), paired across seeds.
2. Phikon ablation: v2+Phikon vs v2-no-cond, paired across seeds.
3. Per-marker scatter: Murphy vs v2 PCC per marker, averaged across seeds.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/home/zkgy/hongliyin_computer/results")
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Damond", "HochSchulz", "Jackson"]
DATASET_TITLE = {
    "Damond": "Damond pancreas (38 ch)",
    "HochSchulz": "HochSchulz melanoma (46 ch)",
    "Jackson": "Jackson breast (45 ch)",
}
SEEDS = [42, 0, 1]


def load_v2_phikon(ds: str) -> dict[int, dict]:
    base = {
        "Damond": ("v2_sweep_damond", "v2_phikon_damond_s0", "v2_phikon_damond_s1"),
        "HochSchulz": ("v2_sweep_hochschulz", "v2_phikon_hochschulz_s0", "v2_phikon_hochschulz_s1"),
        "Jackson": ("v2_sweep_jackson", "v2_phikon_jackson_s0", "v2_phikon_jackson_s1"),
    }[ds]
    return {s: json.load(open(RESULTS / d / "metrics.json")) for s, d in zip(SEEDS, base)}


def load_v2_nocond(ds: str) -> dict[int, dict]:
    base = {
        "Damond": ("v2_nocond_damond", "v2_nocond_damond_s0", "v2_nocond_damond_s1"),
        "HochSchulz": ("v2_nocond_hochschulz", "v2_nocond_hochschulz_s0", "v2_nocond_hochschulz_s1"),
        "Jackson": ("v2_nocond_jackson", "v2_nocond_jackson_s0", "v2_nocond_jackson_s1"),
    }[ds]
    return {s: json.load(open(RESULTS / d / "metrics.json")) for s, d in zip(SEEDS, base)}


def load_murphy(ds: str) -> dict[int, dict]:
    base = {
        "Damond": ("baseline_damond", "baseline_damond_s0", "baseline_damond_s1"),
        "HochSchulz": ("baseline_hochschulz", "baseline_hochschulz_s0", "baseline_hochschulz_s1"),
        "Jackson": ("baseline_jackson", "baseline_jackson_s0", "baseline_jackson_s1"),
    }[ds]
    return {s: json.load(open(RESULTS / d / "metrics.json")) for s, d in zip(SEEDS, base)}


def mean_per_marker(metrics: dict, exclude: tuple = ()) -> float:
    per = metrics["test_pcc_per_marker"]
    vals = [v for k, v in per.items() if k not in exclude]
    return statistics.mean(vals)


def murphy_pcc_incl_dna(metrics: dict) -> float:
    return metrics["test_mean_pcc"]


def v2_pcc_incl_dna(metrics: dict) -> float:
    # v2 test_mean_pcc_bio_targets already includes DNA (DNA is in target_idx, not excluded).
    # For Jackson it also correctly excludes Ru (which is in non_bio_channels / extra_always).
    return metrics["test_mean_pcc_bio_targets"]


def v2_pcc_excl_dna(metrics: dict) -> float:
    return mean_per_marker(metrics, exclude=("DNA1", "DNA2"))


def paired_plot(ax, left_vals: list, right_vals: list, left_label: str, right_label: str,
                colors=("#c0504d", "#3c78b4"), title: str = ""):
    """Paired dot-and-line plot with mean±std bars."""
    n = len(left_vals)
    assert n == len(right_vals)
    xl, xr = 0.0, 1.0
    # Bar: mean
    ax.bar([xl, xr], [np.mean(left_vals), np.mean(right_vals)],
           color=[colors[0], colors[1]], alpha=0.3, width=0.55, edgecolor="none")
    # Error bars: std
    ax.errorbar([xl, xr], [np.mean(left_vals), np.mean(right_vals)],
                yerr=[np.std(left_vals, ddof=1), np.std(right_vals, ddof=1)],
                fmt="none", ecolor="k", capsize=6, lw=1.2)
    # Paired lines + dots
    for i in range(n):
        ax.plot([xl, xr], [left_vals[i], right_vals[i]],
                color="gray", alpha=0.45, lw=1)
    ax.scatter([xl] * n, left_vals, color=colors[0], s=45, zorder=3, edgecolor="k", lw=0.6)
    ax.scatter([xr] * n, right_vals, color=colors[1], s=45, zorder=3, edgecolor="k", lw=0.6)
    ax.set_xticks([xl, xr])
    ax.set_xticklabels([left_label, right_label])
    ax.set_xlim(-0.45, 1.45)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    # Annotate n wins
    wins = sum(1 for l, r in zip(left_vals, right_vals) if r > l)
    ax.text(0.5, 0.97, f"{wins}/{n} paired positive",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#2a7a2a" if wins == n else "gray",
            weight="bold" if wins == n else "normal")


def fig_headline():
    """Headline: v2 vs Murphy paired comparison, BOTH including DNA (the published metric)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        murphy = load_murphy(ds)
        v2 = load_v2_phikon(ds)
        murphy_vals = [murphy_pcc_incl_dna(murphy[s]) for s in SEEDS]
        v2_vals = [v2_pcc_incl_dna(v2[s]) for s in SEEDS]
        paired_plot(ax, murphy_vals, v2_vals,
                    "Murphy\n(per-panel trained)", "SpaProtFM v2\n(one-for-all)",
                    title=DATASET_TITLE[ds])
        ax.set_ylabel("Mean PCC (targets, incl. DNA)" if ds == "Damond" else "")
    fig.suptitle("SpaProtFM v2 vs Murphy baseline — paired, n=3 seeds (both methods predict DNA)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "headline_v2_vs_murphy.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def fig_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        nc = load_v2_nocond(ds)
        ph = load_v2_phikon(ds)
        nc_vals = [v2_pcc_incl_dna(nc[s]) for s in SEEDS]
        ph_vals = [v2_pcc_incl_dna(ph[s]) for s in SEEDS]
        paired_plot(ax, nc_vals, ph_vals,
                    "v2, no Phikon\n(cond=None)", "v2 + Phikon\n(pseudo-H&E)",
                    colors=("#888888", "#3c78b4"),
                    title=DATASET_TITLE[ds])
        ax.set_ylabel("Mean PCC (bio targets)" if ds == "Damond" else "")
    fig.suptitle("Phikon ablation — pseudo-H&E + Phikon-v2 bottleneck fusion contribution (n=3)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "ablation_phikon.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def fig_per_marker():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        murphy = load_murphy(ds)
        v2 = load_v2_phikon(ds)
        # Average per-marker PCC across seeds (both methods)
        m_keys = set(murphy[SEEDS[0]]["test_pcc_per_marker"])
        v_keys = set(v2[SEEDS[0]]["test_pcc_per_marker"])
        shared = sorted(m_keys & v_keys)
        m_mean = {k: np.mean([murphy[s]["test_pcc_per_marker"][k] for s in SEEDS]) for k in shared}
        v_mean = {k: np.mean([v2[s]["test_pcc_per_marker"][k] for s in SEEDS]) for k in shared}
        xs = np.array([m_mean[k] for k in shared])
        ys = np.array([v_mean[k] for k in shared])
        # Diagonal
        lo, hi = min(xs.min(), ys.min()) - 0.05, max(xs.max(), ys.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, alpha=0.6)
        # Points
        diffs = ys - xs
        colors = ["#2a7a2a" if d > 0 else "#c0504d" for d in diffs]
        ax.scatter(xs, ys, s=28, c=colors, alpha=0.8, edgecolor="k", lw=0.4)
        # Annotate top 3 gains and top 3 losses
        idx_gain = np.argsort(diffs)[-3:][::-1]
        idx_loss = np.argsort(diffs)[:3]
        for i in list(idx_gain) + list(idx_loss):
            ax.annotate(shared[i], (xs[i], ys[i]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", alpha=0.85)
        n_gain = int((diffs > 0).sum())
        n_total = len(diffs)
        ax.set_title(f"{DATASET_TITLE[ds]}\n{n_gain}/{n_total} markers v2 > Murphy")
        ax.set_xlabel("Murphy per-marker PCC (mean of 3 seeds)")
        if ds == "Damond":
            ax.set_ylabel("SpaProtFM v2 per-marker PCC (mean of 3 seeds)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Per-marker comparison — points above the diagonal are markers where v2 beats Murphy",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "per_marker_v2_vs_murphy.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_headline()
    fig_ablation()
    fig_per_marker()
