"""Plot SpaProtFM v1 panel-size sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = {
    "damond_pancreas": {"file": "results/v1_sweep_damond/panel_sweep.json", "color": "tab:blue", "baseline_pcc": 0.421},
    "hochschulz_melanoma": {"file": "results/v1_sweep_hochschulz/panel_sweep.json", "color": "tab:orange", "baseline_pcc": 0.493},
    "jackson_breast": {"file": "results/v1_sweep_jackson/panel_sweep.json", "color": "tab:green", "baseline_pcc": 0.376},
}

OUT = Path("results/figures/panel_sweep.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))
for name, cfg in DATASETS.items():
    data = json.loads(Path(cfg["file"]).read_text())
    sizes = sorted({r["panel_size"] for r in data})
    means, stds = [], []
    for s in sizes:
        vals = [r["mean_pcc_bio"] for r in data if r["panel_size"] == s]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4, label=name, color=cfg["color"])
    ax.axhline(cfg["baseline_pcc"], color=cfg["color"], linestyle="--", alpha=0.5,
               label=f"{name} baseline @ size 10")

ax.set_xlabel("Number of observed markers")
ax.set_ylabel("Test mean PCC (biological targets)")
ax.set_title("SpaProtFM v1: one model, any panel size\n(dashed = Murphy baseline trained for size=10 only)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(f"Wrote {OUT}")
