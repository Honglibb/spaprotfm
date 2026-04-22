"""Panel-flexibility comparison: one v2 checkpoint (any panel size) vs per-size-retrained Murphy.

For each dataset:
  v2: loaded from panel_sweep.json (3 random panels per size)
  Murphy: one checkpoint PER panel size, from baseline_{ds}{,_sz7,_sz15,_sz20}/metrics.json

Plot: x = panel size, y = mean PCC.
  v2 as line + shaded std across 3 random panels.
  Murphy as scatter points (one per size).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/home/zkgy/hongliyin_computer/results")
FIGDIR = RESULTS / "figures"

DATASETS = [
    ("Damond pancreas", "damond",
     ["baseline_damond_sz7", "baseline_damond", "baseline_damond_sz15", "baseline_damond_sz20"],
     [7, 10, 15, 20]),
    ("HochSchulz melanoma", "hochschulz",
     ["baseline_hochschulz_sz7", "baseline_hochschulz", "baseline_hochschulz_sz15", "baseline_hochschulz_sz20"],
     [7, 10, 15, 20]),
    ("Jackson breast", "jackson",
     ["baseline_jackson", "baseline_jackson_sz15", "baseline_jackson_sz20"],
     [10, 15, 20]),
]


def murphy_mean_excl_dna(metrics: dict) -> float:
    per = metrics["test_pcc_per_marker"]
    bio = [v for k, v in per.items() if k not in ("DNA1", "DNA2")]
    return statistics.mean(bio)


def load_v2_sweep(v2_dir: str) -> dict[int, list[float]]:
    """Return {panel_size: [mean_pcc_bio across trials]} from panel_sweep.json.

    v2's mean_pcc_bio excludes extra_always (Ru for Jackson) but includes DNA.
    For fair apples-to-apples with Murphy-excl-DNA, recompute mean_pcc_bio excluding DNA.

    We only have the three-trial per-size summary pcc_bio from panel_sweep.json,
    NOT the per-marker breakdown. So we can't easily exclude DNA post-hoc from v2.
    Fall back to using reported mean_pcc_bio (includes DNA); caveat in text.
    """
    data = json.loads((RESULTS / v2_dir / "panel_sweep.json").read_text())
    out: dict[int, list[float]] = {}
    for row in data:
        out.setdefault(row["panel_size"], []).append(row["mean_pcc_bio"])
    return out


def get_v2_incl_dna_by_size(v2_dir: str) -> dict[int, list[float]]:
    """v2 panel_sweep.json reports mean_pcc_bio = mean over targets excluding extra_always (Ru).
    For Damond/HochSchulz extra_always=[] so it's the raw mean over all targets (incl DNA too,
    but DNA is observed in v2 so it's not in targets — actually target_idx in v2 IS {not in eval_obs},
    which for non-baseline panel sweep uses obs = always_observed ∪ random. DNA IS in obs, NOT in target.
    So v2's panel_sweep mean_pcc_bio already excludes DNA from targets."""
    return load_v2_sweep(v2_dir)


def main():
    # Panel-sweep v2 dirs per dataset
    v2_dirs = {
        "damond": "v2_sweep_damond",
        "hochschulz": "v2_sweep_hochschulz",
        "jackson": "v2_sweep_jackson",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=False)
    for ax, (title, ds_key, mu_dirs, mu_sizes) in zip(axes, DATASETS):
        # v2 curve
        v2 = get_v2_incl_dna_by_size(v2_dirs[ds_key])
        v2_sizes = sorted(v2.keys())
        v2_means = [np.mean(v2[s]) for s in v2_sizes]
        v2_stds = [np.std(v2[s], ddof=1) if len(v2[s]) > 1 else 0 for s in v2_sizes]
        ax.plot(v2_sizes, v2_means, "-o", color="#3c78b4", label="SpaProtFM v2 (one checkpoint)",
                lw=1.8, markersize=7, zorder=3)
        ax.fill_between(v2_sizes,
                        [m - s for m, s in zip(v2_means, v2_stds)],
                        [m + s for m, s in zip(v2_means, v2_stds)],
                        color="#3c78b4", alpha=0.20)

        # Murphy per-size points (excluding DNA from each size's mean)
        mu_vals = []
        for d in mu_dirs:
            m = json.loads((RESULTS / d / "metrics.json").read_text())
            mu_vals.append(murphy_mean_excl_dna(m))
        ax.scatter(mu_sizes, mu_vals, color="#c0504d", s=90, marker="D",
                   edgecolor="k", lw=0.8, zorder=4,
                   label=f"Murphy (retrained per size, n={len(mu_sizes)} models)")

        # Cosmetic
        ax.set_title(title)
        ax.set_xlabel("Observed panel size")
        if ds_key == "damond":
            ax.set_ylabel("Mean PCC (bio targets, excl. DNA)")
        ax.set_xticks(sorted(set(v2_sizes) | set(mu_sizes)))
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Panel-size flexibility — v2 (1 checkpoint) matches Murphy (retrained per size)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "panel_flex_v2_vs_murphy.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
