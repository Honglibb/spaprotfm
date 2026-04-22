# Plan 4 — Cross-dataset panel transfer (SpaProtFM v2)

**Goal**: demonstrate that a single v2 checkpoint trained on pooled IMC data generalizes to a held-out dataset with the same panel, making the "one checkpoint for any panel" claim undeniable.

## Shared panel (10 markers, canonical order)

After name harmonization across Damond / HochSchulz / Jackson:

| idx | marker | role | notes |
|-----|--------|------|-------|
| 0 | DNA1 | always-observed | DNA intercalator |
| 1 | DNA2 | always-observed | DNA intercalator |
| 2 | H3 | always-observed | histone (morphology) |
| 3 | SMA | always-observed | smooth muscle (structure) |
| 4 | CD20 | target | B cell |
| 5 | CD3e | target | pan-T cell |
| 6 | CD45 | target | pan-leukocyte |
| 7 | CD68 | target | macrophage |
| 8 | Ki67 | target | proliferation |
| 9 | cPARP | target | apoptosis |

Harmonization table:

| canonical | Damond | HochSchulz | Jackson |
|-----------|--------|------------|---------|
| Ki67 | Ki67 | Ki67_Er168 | Ki67 |
| cPARP | cPARP_cCASP3 | c_PARP | cPARP_cCASP3 |
| all others | identical names | — | — |

## Experimental settings

**Setting 1 — pooled training (foundation-model story)**
- Train on train-split images from ALL 3 datasets, 10-marker shared panel
- Test on held-out images from each dataset separately
- Multi-seed n=3
- Claim: one model handles all 3 tissue types

**Setting 2 — leave-one-dataset-out (transfer story)**
- Train on 2 of 3 datasets
- Test on ALL images of the held-out dataset (unseen at train time)
- 3 folds × 3 seeds = 9 runs
- Claim: model transfers to an entirely new cohort

**Baseline**
- Murphy (per-dataset trained) evaluated on the same 10-marker panel and same held-out test images — for comparison, Murphy transferred across datasets should fail catastrophically (expected).

## Implementation plan

1. New data loader `load_imc_multi_rds()` that:
   - Takes list of RDS paths + canonical marker list
   - Loads each, extracts only shared channels, reorders into canonical order
   - Stacks tiles with a dataset-ID tensor for diagnostics
2. Modify `train_spaprotfm_v2.py` to accept `--rds` as comma-separated list, with `--test-rds` for leave-one-out.
3. Add `--canonical-markers` flag to specify shared panel (10 markers by default).
4. Always-observed = idx [0, 1, 2, 3] (DNA×2, H3, SMA) — richer structural prior than DNA-only.

## Success criteria

- **Setting 1**: v2 pooled performance on each test-split within 0.02 PCC of per-dataset v2.
- **Setting 2**: v2 held-out transfer ≥ 0.25 PCC (bio targets). Murphy cross-dataset transfer ≤ 0.10 (effectively random).
- If Setting 2 succeeds, this is the flagship result for the paper.

## Risks

- **Domain shift**: IMC intensity distributions differ across tissues (pancreas vs melanoma vs breast). Per-dataset z-score normalization may be needed.
- **Small common panel**: only 6 targets, so absolute PCC numbers may be dominated by a few markers.
- **Leakage**: must ensure image-level split within each dataset is preserved when pooling.
