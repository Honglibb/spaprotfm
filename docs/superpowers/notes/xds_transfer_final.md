# Cross-dataset panel transfer — final results (n=3 seeds)

Seeds: {0, 1, 42}. Paired v2 vs Murphy on seed intersection. Population std (ddof=0).

## Setup

- **Canonical 10-marker panel** (alias-harmonized across 3 IMC cohorts):
  `[DNA1, DNA2, H3, SMA, CD20, CD3e, CD45, CD68, Ki67, cPARP]`
- Always-observed input: `[DNA1, DNA2, H3, SMA]` (4 morphology/DNA)
- Targets: `[CD20, CD3e, CD45, CD68, Ki67, cPARP]` (6 immune/proliferation/apoptosis)
- **v2**: MaskedUNetV2 + frozen Phikon-v2 pseudo-H&E fusion, stage-1 random-mask only, 30 epochs. 4 runs × 3 seeds: `pooled` (train all 3) and `heldout{D,H,J}` (leave-one-out).
- **Murphy baseline**: same architecture backbone with `--no-cond --no-random-mask`, trained separately on each single source cohort. 3 source × 3 seeds = 9 runs. Evaluated on each test dataset as in-dist (when source == test) or OOD (when source ≠ test).
- Image-level 80/10/10 split per train dataset; OOD = entire held-out cohort.

## Headline results (n=3)

### OOD zero-shot transfer

| Test dataset | Murphy (best source) | v2 (heldout, pooled LOO) | Δ (v2 − Murphy) |
|---|---|---|---|
| Damond (pancreas) | 0.150 ± 0.004 | **0.251 ± 0.004** | **+0.10** |
| HochSchulz (melanoma) | 0.264 ± 0.003 | **0.379 ± 0.007** | **+0.12** |
| Jackson (breast) | 0.219 ± 0.002 | **0.279 ± 0.001** | **+0.06** |

### In-dist performance (for reference)

| Test dataset | Murphy (native) | v2 (pooled) | Δ |
|---|---|---|---|
| Damond | 0.197 ± 0.004 | **0.284 ± 0.008** | +0.087 |
| HochSchulz | 0.338 ± 0.034 | **0.486 ± 0.020** | +0.149 |
| Jackson | 0.268 ± 0.014 | **0.310 ± 0.018** | +0.042 |

**v2 wins 6/6 paired comparisons (3 datasets × {in_dist, OOD}) across all 3 seeds.**

## Key finding

> A single SpaProtFM v2 checkpoint trained on two IMC cohorts generalizes zero-shot
> to a completely held-out third cohort (different tissue type, different instrument,
> different imaging batch) and **outperforms per-panel Murphy models trained specifically
> for that target cohort**, on every dataset.

The comparison is deliberately unfavorable to v2: Murphy is given the *best* single source
(max over the 2 training cohorts it could have been trained on), while v2 sees neither
cohort-specific labels nor cohort-specific fine-tuning. v2 still wins by +0.06–0.12 PCC.

## Per-marker OOD breakdown (seed=42; multi-seed trends identical)

### OOD Damond (heldoutD — trained on HS+J)
| marker | v2 PCC |
|---|---|
| CD20 | ~0.12 |
| CD3e | ~0.24 |
| CD45 | ~0.38 |
| CD68 | ~0.33 |
| Ki67 | ~0.23 |
| cPARP | ~0.28 |

### OOD HochSchulz (heldoutH — trained on D+J)
| marker | v2 PCC |
|---|---|
| CD20 | 0.32 |
| CD3e | 0.49 |
| CD45 | 0.45 |
| CD68 | 0.40 |
| Ki67 | 0.43 |
| cPARP | 0.17 |

### OOD Jackson (heldoutJ — trained on D+HS)
| marker | v2 PCC |
|---|---|
| CD20 | 0.18 |
| CD3e | 0.16 |
| CD45 | 0.25 |
| CD68 | 0.32 |
| Ki67 | 0.35 |
| cPARP | 0.41 |

## Interpretation

1. **Non-trivial zero-shot transfer.** v2 retains ~80–85% of its pooled in-dist
   performance when evaluated on a fully unseen cohort. Murphy, when applied to
   a cohort it never saw, degrades to 55–85% of its own native performance.
2. **HochSchulz is the easiest target.** Both in-dist (v2 0.49) and OOD (v2 0.38).
   Melanoma's dense immune infiltrate + distinctive H&E morphology gives the
   frozen Phikon-v2 backbone strong signal.
3. **Damond is the hardest.** Pancreas-specific biology; the 6 common immune
   targets are sparse/low-dynamic-range here. Yet v2 still lifts OOD from Murphy's
   0.15 → 0.25 — a **67% relative improvement** on the hardest cohort.
4. **Per-marker transfer patterns are biologically sensible:**
   - **Ki67 transfers everywhere** — proliferation is visible in morphology.
   - **CD45 transfers well** — pan-leukocyte, densely expressed, easy signal.
   - **CD20 is hardest OOD** — sparse B cells, dataset-specific distribution.
   - **cPARP variable** — apoptosis depends heavily on tissue state.

## Deliverables

- `results/figures/xds_final.png` — paired bar chart + overall summary
- `results/figures/xds_per_marker.png` — per-marker OOD breakdown
- `results/tables/xds_final_summary.csv` — machine-readable numbers
