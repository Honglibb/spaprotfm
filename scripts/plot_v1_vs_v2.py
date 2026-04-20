"""Plot SpaProtFM v1 vs v2 panel-size sweeps side-by-side."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = {
    "damond_pancreas": {
        "v1": "results/v1_sweep_damond/panel_sweep.json",
        "v2": "results/v2_sweep_damond/panel_sweep.json",
        "baseline_pcc": 0.421,
        "title": "Damond pancreas (38 ch)",
    },
    "hochschulz_melanoma": {
        "v1": "results/v1_sweep_hochschulz/panel_sweep.json",
        "v2": "results/v2_sweep_hochschulz/panel_sweep.json",
        "baseline_pcc": 0.493,
        "title": "HochSchulz melanoma (46 ch)",
    },
    "jackson_breast": {
        "v1": "results/v1_sweep_jackson/panel_sweep.json",
        "v2": "results/v2_sweep_jackson/panel_sweep.json",
        "baseline_pcc": 0.376,
        "title": "Jackson breast (45 ch, bio)",
    },
}

OUT = Path("results/figures/v1_vs_v2_panel_sweep.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def agg(data: list[dict], key: str = "mean_pcc_bio") -> tuple[list[int], list[float], list[float]]:
    sizes = sorted({r["panel_size"] for r in data})
    means, stds = [], []
    for s in sizes:
        vals = [r[key] for r in data if r["panel_size"] == s]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return sizes, means, stds


fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

for ax, (name, cfg) in zip(axes, DATASETS.items()):
    v1_file = Path(cfg["v1"])
    v2_file = Path(cfg["v2"])
    if not v1_file.exists():
        ax.set_title(f"{cfg['title']} (v1 missing)")
        continue
    v1 = json.loads(v1_file.read_text())
    s1, m1, e1 = agg(v1)
    ax.errorbar(s1, m1, yerr=e1, marker="o", capsize=4, label="v1 (no H&E)",
                color="tab:blue")

    if v2_file.exists():
        v2 = json.loads(v2_file.read_text())
        s2, m2, e2 = agg(v2)
        ax.errorbar(s2, m2, yerr=e2, marker="s", capsize=4,
                    label="v2 (+ Phikon pseudo-H&E)", color="tab:red")

    ax.axhline(cfg["baseline_pcc"], color="grey", linestyle="--", alpha=0.6,
               label=f"Murphy baseline @ size=10 ({cfg['baseline_pcc']:.3f})")
    ax.set_xlabel("Number of observed markers")
    ax.set_ylabel("Test mean PCC (biological targets)")
    ax.set_title(cfg["title"])
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("SpaProtFM v1 vs v2: H&E conditioning impact across panel sizes", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(f"Wrote {OUT}")


# Also print a summary table.
print("\n=== Summary (mean_pcc_bio) ===")
print(f"{'dataset':<22} {'size':>5} {'v1':>8} {'v2':>8} {'Δ':>8}")
for name, cfg in DATASETS.items():
    v1 = json.loads(Path(cfg["v1"]).read_text())
    v2_file = Path(cfg["v2"])
    if not v2_file.exists():
        continue
    v2 = json.loads(v2_file.read_text())
    sizes = sorted({r["panel_size"] for r in v1})
    for s in sizes:
        v1_vals = [r["mean_pcc_bio"] for r in v1 if r["panel_size"] == s]
        v2_vals = [r["mean_pcc_bio"] for r in v2 if r["panel_size"] == s]
        if not v1_vals or not v2_vals:
            continue
        m1 = float(np.mean(v1_vals)); m2 = float(np.mean(v2_vals))
        print(f"{name:<22} {s:>5} {m1:>8.3f} {m2:>8.3f} {m2-m1:>+8.3f}")
