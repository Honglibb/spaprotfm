"""Final figure for cross-dataset transfer with multi-seed error bars.

Aggregates over whatever seeds are available on disk. For v2, reads the
pooled_s{seed} + heldout{D,H,J}_s{seed} runs. For Murphy, reads the
murphy_xds_{src}_s{seed} runs (one Murphy model per training source).

Outputs:
  results/figures/xds_final.png (multi-panel summary)
  results/figures/xds_per_marker.png (per-marker OOD bars)
  results/tables/xds_final_summary.csv (numbers behind the figure)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = Path("/home/zkgy/hongliyin_computer/results")
FIGDIR = R / "figures"
TBLDIR = R / "tables"
TBLDIR.mkdir(exist_ok=True, parents=True)

DATASETS = ["Damond", "HochSchulz", "Jackson"]
DS_LABEL = {"Damond": "Damond\n(pancreas)", "HochSchulz": "HochSchulz\n(melanoma)", "Jackson": "Jackson\n(breast)"}
DS_SHORT = {"Damond": "Damond", "HochSchulz": "HochSchulz", "Jackson": "Jackson"}
NAME_MAP = {"Damond": "D", "HochSchulz": "H", "Jackson": "J"}
MARKERS = ["CD20", "CD3e", "CD45", "CD68", "Ki67", "cPARP"]


def find_seeds_v2() -> list[int]:
    seeds = set()
    for p in R.glob("xds_pooled_s*"):
        if (p / "metrics.json").exists():
            try:
                seeds.add(int(p.name.split("_s")[-1]))
            except ValueError:
                continue
    return sorted(seeds)


def find_seeds_murphy() -> list[int]:
    seeds = set()
    for p in R.glob("murphy_xds_Damond_s*"):
        if (p / "metrics.json").exists():
            try:
                seeds.add(int(p.name.split("_s")[-1]))
            except ValueError:
                continue
    return sorted(seeds)


def load_v2_pooled(seed: int) -> dict:
    """Return {dataset: mean_pcc} from pooled (train-all) in-dist eval."""
    p = R / f"xds_pooled_s{seed}/metrics.json"
    if not p.exists():
        return {}
    m = json.load(open(p))
    return {ds: m["eval"]["in_dist"][ds]["mean_pcc"] for ds in DATASETS
            if ds in m["eval"]["in_dist"]}


def load_v2_ood(seed: int) -> dict:
    """Return {dataset: ood_mean_pcc} from leave-one-out runs."""
    out = {}
    for ds in DATASETS:
        p = R / f"xds_heldout{NAME_MAP[ds]}_s{seed}/metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        if ds in m["eval"]["ood"]:
            out[ds] = m["eval"]["ood"][ds]["mean_pcc"]
    return out


def load_v2_ood_per_marker(seed: int) -> dict:
    """Return {dataset: {marker: pcc}} for each held-out run's OOD eval."""
    out = {}
    for ds in DATASETS:
        p = R / f"xds_heldout{NAME_MAP[ds]}_s{seed}/metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        out[ds] = m["eval"]["ood"][ds]["pcc_per_marker"]
    return out


def load_murphy_matrix(seed: int) -> np.ndarray:
    """3×3: row=train source, col=test dataset (in-dist on diag, OOD off-diag)."""
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


def load_murphy_ood_per_marker_best(seed: int) -> dict:
    """For each held-out test dataset, pick the BEST source's per-marker OOD PCCs."""
    out = {}
    # First get the best source per test by mean_pcc
    mat = load_murphy_matrix(seed)
    for j, test in enumerate(DATASETS):
        best_i, best_val = -1, -np.inf
        for i in range(3):
            if i == j:
                continue
            if np.isnan(mat[i, j]):
                continue
            if mat[i, j] > best_val:
                best_val = mat[i, j]
                best_i = i
        if best_i < 0:
            continue
        src = DATASETS[best_i]
        p = R / f"murphy_xds_{src}_s{seed}/metrics.json"
        m = json.load(open(p))
        out[test] = m["eval"]["ood"][test]["pcc_per_marker"]
    return out


def agg(values: list[float]) -> tuple[float, float]:
    """Return (mean, std). NaN-safe; std from population (ddof=0) to not hide small-n."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if arr.size == 0:
        return (np.nan, 0.0)
    return (float(arr.mean()), float(arr.std(ddof=0)))


def build_multiseed_tables(seeds_v2: list[int], seeds_murphy: list[int]):
    """Return per-test-dataset summary: v2 pooled in-dist, v2 OOD, Murphy in-dist, Murphy best OOD."""
    v2_pooled = {ds: [] for ds in DATASETS}
    v2_ood = {ds: [] for ds in DATASETS}
    for s in seeds_v2:
        p = load_v2_pooled(s)
        o = load_v2_ood(s)
        for ds in DATASETS:
            if ds in p:
                v2_pooled[ds].append(p[ds])
            if ds in o:
                v2_ood[ds].append(o[ds])

    murphy_indist = {ds: [] for ds in DATASETS}  # train==test
    murphy_ood_best = {ds: [] for ds in DATASETS}
    for s in seeds_murphy:
        mat = load_murphy_matrix(s)
        for j, test in enumerate(DATASETS):
            if not np.isnan(mat[j, j]):
                murphy_indist[test].append(mat[j, j])
            off = [mat[i, j] for i in range(3) if i != j and not np.isnan(mat[i, j])]
            if off:
                murphy_ood_best[test].append(max(off))

    return v2_pooled, v2_ood, murphy_indist, murphy_ood_best


def fig_main(seeds_v2: list[int], seeds_murphy: list[int]) -> Path:
    v2_pooled, v2_ood, murphy_indist, murphy_ood_best = build_multiseed_tables(seeds_v2, seeds_murphy)

    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # === Panel A: OOD head-to-head (THE killer plot) ===
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    w = 0.35
    murphy_m = [agg(murphy_ood_best[ds])[0] for ds in DATASETS]
    murphy_s = [agg(murphy_ood_best[ds])[1] for ds in DATASETS]
    v2_m = [agg(v2_ood[ds])[0] for ds in DATASETS]
    v2_s = [agg(v2_ood[ds])[1] for ds in DATASETS]

    b1 = ax.bar(x - w/2, murphy_m, w, yerr=murphy_s, label="Murphy (best source)",
                color="#c0504d", edgecolor="k", lw=0.5, capsize=4, error_kw={"lw": 1})
    b2 = ax.bar(x + w/2, v2_m, w, yerr=v2_s, label="SpaProtFM v2 (pooled LOO)",
                color="#3c78b4", edgecolor="k", lw=0.5, capsize=4, error_kw={"lw": 1})
    for bar, val, s in zip(b1, murphy_m, murphy_s):
        ax.text(bar.get_x() + bar.get_width()/2, val + s + 0.012, f"{val:.2f}",
                ha="center", fontsize=9, color="#c0504d")
    for bar, val, s in zip(b2, v2_m, v2_s):
        ax.text(bar.get_x() + bar.get_width()/2, val + s + 0.012, f"{val:.2f}",
                ha="center", fontsize=9, color="#3c78b4", weight="bold")

    # Paired deltas
    for i in range(3):
        delta = v2_m[i] - murphy_m[i]
        ymax = max(v2_m[i] + v2_s[i], murphy_m[i] + murphy_s[i]) + 0.05
        ax.text(i, ymax, f"Δ={delta:+.2f}", ha="center", fontsize=8.5,
                color="black", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], fontsize=9)
    ax.set_ylabel("OOD mean PCC", fontsize=10)
    ax.set_title(f"OOD zero-shot transfer (n={len(seeds_v2)} seeds)\n"
                 "single SpaProtFM v2 checkpoint vs per-panel Murphy",
                 fontsize=10.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.52)

    # === Panel B: in-dist + OOD gap ===
    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["in-dist", "OOD"]
    v2_id_m = [agg([v for ds in DATASETS for v in v2_pooled[ds]])[0]]
    v2_id_s = [agg([v for ds in DATASETS for v in v2_pooled[ds]])[1]]
    v2_ood_all_m = [agg([v for ds in DATASETS for v in v2_ood[ds]])[0]]
    v2_ood_all_s = [agg([v for ds in DATASETS for v in v2_ood[ds]])[1]]
    m_id_m = [agg([v for ds in DATASETS for v in murphy_indist[ds]])[0]]
    m_id_s = [agg([v for ds in DATASETS for v in murphy_indist[ds]])[1]]
    m_ood_all_m = [agg([v for ds in DATASETS for v in murphy_ood_best[ds]])[0]]
    m_ood_all_s = [agg([v for ds in DATASETS for v in murphy_ood_best[ds]])[1]]

    xb = np.arange(2)
    wb = 0.35
    vals_m = [m_id_m[0], m_ood_all_m[0]]
    errs_m = [m_id_s[0], m_ood_all_s[0]]
    vals_v = [v2_id_m[0], v2_ood_all_m[0]]
    errs_v = [v2_id_s[0], v2_ood_all_s[0]]
    ax2.bar(xb - wb/2, vals_m, wb, yerr=errs_m, label="Murphy", color="#c0504d",
            edgecolor="k", lw=0.5, capsize=4)
    ax2.bar(xb + wb/2, vals_v, wb, yerr=errs_v, label="v2", color="#3c78b4",
            edgecolor="k", lw=0.5, capsize=4)
    for i in range(2):
        ax2.text(xb[i] - wb/2, vals_m[i] + errs_m[i] + 0.012, f"{vals_m[i]:.2f}",
                 ha="center", fontsize=9, color="#c0504d")
        ax2.text(xb[i] + wb/2, vals_v[i] + errs_v[i] + 0.012, f"{vals_v[i]:.2f}",
                 ha="center", fontsize=9, color="#3c78b4", weight="bold")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("mean PCC (averaged over 3 datasets)", fontsize=10)
    ax2.set_title("In-dist vs OOD — overall", fontsize=10.5)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 0.52)

    fig.suptitle(
        f"Cross-dataset panel transfer — SpaProtFM v2 vs per-panel Murphy baseline "
        f"(multi-seed n={len(seeds_v2)})",
        y=1.02, fontsize=12, weight="bold"
    )
    fig.tight_layout()
    out = FIGDIR / "xds_final.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_marker(seeds_v2: list[int], seeds_murphy: list[int]) -> Path:
    """Per-marker OOD PCC for each held-out dataset, v2 vs Murphy best-source."""
    v2_pm = {ds: {m: [] for m in MARKERS} for ds in DATASETS}
    m_pm = {ds: {m: [] for m in MARKERS} for ds in DATASETS}
    for s in seeds_v2:
        pm = load_v2_ood_per_marker(s)
        for ds, d in pm.items():
            for mkr, v in d.items():
                if mkr in v2_pm[ds]:
                    v2_pm[ds][mkr].append(v)
    for s in seeds_murphy:
        pm = load_murphy_ood_per_marker_best(s)
        for ds, d in pm.items():
            for mkr, v in d.items():
                if mkr in m_pm[ds]:
                    m_pm[ds][mkr].append(v)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    x = np.arange(len(MARKERS))
    w = 0.35
    for ax, ds in zip(axes, DATASETS):
        m_mean = [agg(m_pm[ds][mk])[0] for mk in MARKERS]
        m_std = [agg(m_pm[ds][mk])[1] for mk in MARKERS]
        v_mean = [agg(v2_pm[ds][mk])[0] for mk in MARKERS]
        v_std = [agg(v2_pm[ds][mk])[1] for mk in MARKERS]
        ax.bar(x - w/2, m_mean, w, yerr=m_std, label="Murphy (best src)",
               color="#c0504d", edgecolor="k", lw=0.5, capsize=3)
        ax.bar(x + w/2, v_mean, w, yerr=v_std, label="v2",
               color="#3c78b4", edgecolor="k", lw=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(MARKERS, fontsize=9, rotation=20)
        ax.set_title(f"OOD on {DS_SHORT[ds]}", fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.05, 0.65)
        ax.axhline(0, color="k", lw=0.4)
    axes[0].set_ylabel("per-marker PCC (OOD)", fontsize=10)
    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle(f"Per-marker OOD transfer (multi-seed n={len(seeds_v2)})",
                 y=1.03, fontsize=12, weight="bold")
    fig.tight_layout()
    out = FIGDIR / "xds_per_marker.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_csv(seeds_v2: list[int], seeds_murphy: list[int]) -> Path:
    v2_pooled, v2_ood, murphy_indist, murphy_ood_best = build_multiseed_tables(seeds_v2, seeds_murphy)
    out = TBLDIR / "xds_final_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_dataset", "metric",
                    "murphy_mean", "murphy_std", "murphy_n",
                    "v2_mean", "v2_std", "v2_n", "delta (v2-murphy)"])
        for ds in DATASETS:
            mm, ms = agg(murphy_indist[ds])
            vm, vs = agg(v2_pooled[ds])
            w.writerow([ds, "in_dist", f"{mm:.4f}", f"{ms:.4f}",
                        len([x for x in murphy_indist[ds] if not np.isnan(x)]),
                        f"{vm:.4f}", f"{vs:.4f}",
                        len([x for x in v2_pooled[ds] if not np.isnan(x)]),
                        f"{vm - mm:+.4f}"])
            mm, ms = agg(murphy_ood_best[ds])
            vm, vs = agg(v2_ood[ds])
            w.writerow([ds, "ood", f"{mm:.4f}", f"{ms:.4f}",
                        len([x for x in murphy_ood_best[ds] if not np.isnan(x)]),
                        f"{vm:.4f}", f"{vs:.4f}",
                        len([x for x in v2_ood[ds] if not np.isnan(x)]),
                        f"{vm - mm:+.4f}"])
    return out


if __name__ == "__main__":
    seeds_v2 = find_seeds_v2()
    seeds_murphy = find_seeds_murphy()
    # Intersect: only use seeds where BOTH exist so we pair fairly
    seeds_paired = sorted(set(seeds_v2) & set(seeds_murphy))
    print(f"v2 seeds available:     {seeds_v2}")
    print(f"Murphy seeds available: {seeds_murphy}")
    print(f"paired seeds used:      {seeds_paired}")
    if not seeds_paired:
        raise SystemExit("No paired seeds available yet")

    f1 = fig_main(seeds_paired, seeds_paired)
    print(f"Wrote {f1}")
    f2 = fig_per_marker(seeds_paired, seeds_paired)
    print(f"Wrote {f2}")
    f3 = write_summary_csv(seeds_paired, seeds_paired)
    print(f"Wrote {f3}")
