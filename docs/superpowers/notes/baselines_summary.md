# Bodenmiller IMC Baselines — Image-Level Split

Settings: 10 observed (first 10 channels) → predict rest. U-Net base=32, 30 epochs, 128×128 patches, batch=32, lr=1e-4, seed=42.

| Dataset | Images | Channels | Tiles (total) | Test mean PCC | Test MSE | Train time |
|---|---|---|---|---|---|---|
| Damond (pancreas) | 100 | 38 (28 targets) | 2006 | **0.421** | 0.0285 | 154 s |
| HochSchulz (melanoma) | 50 | 46 (36 targets) | 3411 | **0.493** | 0.0263 | 333 s |
| JacksonFischer (breast) | 100 | 45 (35 targets) | 4074 | **0.376** | 0.0274 | 375 s |

Image-level split: train 80% / val 10% / test 10% of images (not tiles). Split IDs saved to `results/baseline_*/image_ids.json`.

## Per-marker rankings (top 5 + bottom 5 per dataset)

### Damond (pancreas, 28 targets)

| Rank | Marker | PCC |
|---|---|---|
| 1 | SYP | 0.860 |
| 2 | DNA2 | 0.821 |
| 3 | DNA1 | 0.784 |
| 4 | CDH1 | 0.737 |
| 5 | PDX1 | 0.709 |
| ... | ... | ... |
| 24 | CD8a | 0.203 |
| 25 | Ki67 | 0.174 |
| 26 | FOXP3 | 0.130 |
| 27 | PPY | 0.103 |
| 28 | CD20 | 0.074 |

### HochSchulz (melanoma, 36 targets)

| Rank | Marker | PCC |
|---|---|---|
| 1 | DNA2 | 0.901 |
| 2 | DNA1 | 0.900 |
| 3 | CD45RO | 0.704 |
| 4 | CTNNB1 | 0.644 |
| 5 | CD45RA | 0.631 |
| ... | ... | ... |
| 32 | SOX9 | 0.309 |
| 33 | Ki67_Er168 | 0.291 |
| 34 | CD303 | 0.236 |
| 35 | c_PARP | 0.223 |
| 36 | MPO | 0.152 |

### JacksonFischer (breast cancer, 35 targets)

| Rank | Marker | PCC |
|---|---|---|
| 1 | DNA2 | 0.850 |
| 2 | DNA1 | 0.815 |
| 3 | CD44 | 0.695 |
| 4 | PanCK | 0.677 |
| 5 | FN1 | 0.664 |
| ... | ... | ... |
| 31 | vWF | 0.178 |
| 32 | KRT14 | 0.172 |
| 33 | CD20 | 0.171 |
| 34 | CD3e | 0.149 |
| 35 | p53 | 0.145 |

## Comparison to old tile-level split (Damond only)

| Metric | Tile-level (old) | Image-level (new) | Delta |
|---|---|---|---|
| Test mean PCC | 0.409 | 0.421 | +0.012 |
| Test MSE | n/a | 0.0285 | — |

Note: the new PCC (0.421) is slightly *higher* than the old tile-level number (0.409). This is somewhat surprising — it indicates the old inflated-split result was not worse than the corrected one. However, the old result was still methodologically unsound (tile leakage), and the corrected method is the valid number. The small increase likely reflects variance from different random seeds affecting which images end up in test, combined with the fact that 30 epochs vs the original 20 improves fit.

## Observations

- **DNA channels are universally easiest to predict** (PCC 0.78–0.90 across all 3 datasets), likely because DNA content correlates strongly with morphological features encoded in the first 10 channels (H3, Ruthenium intercalators, etc.).
- **Sparse/rare cell-type markers are hardest**: CD20 (B cells), CD3e (T cells), MPO (neutrophils), p53, and FOXP3 (Tregs) consistently fall in the bottom tier across datasets, reflecting their spatial sparsity and low signal-to-noise in 128×128 patches.
- **HochSchulz (melanoma) achieves the highest mean PCC (0.493)** despite having only 50 images and the most target channels (36), possibly because the larger image size (~1002×963 px) yields more tiles per image with richer spatial context, and melanoma panels include structurally correlated immune/stroma markers.
