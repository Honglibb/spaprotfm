# SpaProtFM v1 panel-size sweep

**Date**: 2026-04-17
**Goal**: demonstrate v1's *flexibility* — one trained model works at any panel size — versus the per-panel-retrained Murphy baseline.

## Setup

- Model: `MaskedUNetV1` (base=48, per-channel value bias), stage-1 random-mask pre-training (30 ep) + stage-2 fixed-mask fine-tuning (5 ep on size-10 panel for baseline-parity headline).
- After training, **the same checkpoint** is evaluated at 5 panel sizes × 3 random panel draws each = 15 eval points per dataset.
- Baseline reference: Murphy U-Net trained for size-10 panel only.
- Test split: held-out images (image-level, not tile-level).
- Metric: mean Pearson correlation over biological target channels (non-Ru).

## Results (mean PCC over 3 random panel draws)

| Dataset | size=3 | size=7 | size=10 | size=15 | size=20 | Baseline @ size=10 |
|---|---|---|---|---|---|---|
| Damond pancreas (38 ch) | 0.353 ± 0.006 | 0.380 ± 0.012 | **0.397 ± 0.008** | 0.415 ± 0.035 | 0.456 ± 0.031 | 0.421 |
| HochSchulz melanoma (46 ch) | 0.403 ± 0.055 | 0.438 ± 0.009 | **0.462 ± 0.008** | 0.477 ± 0.009 | 0.504 ± 0.010 | 0.493 |
| Jackson breast (45 ch, bio) | 0.219 ± 0.040 | 0.267 ± 0.022 | **0.338 ± 0.018** | 0.349 ± 0.044 | 0.370 ± 0.020 | 0.376 |

Figure: `results/figures/panel_sweep.png`.

## Takeaways

1. **Monotonic scaling.** v1 PCC rises smoothly with panel size on every dataset — the model uses the extra observations.
2. **Small gap at size=10.** Under the baseline's own operating point (fixed size-10 panel), v1-with-*random* 10-channel panels averages 0.02-0.04 PCC below the baseline trained *specifically* for that panel. Expected: the baseline is one-panel-only.
3. **v1 surpasses baseline when given more markers.** At size=20, v1 beats the baseline's size-10 score on Damond (0.456 vs 0.421) and HochSchulz (0.504 vs 0.493).
4. **The paper argument.** Matching baseline at size=10 would require training 5 separate baselines (one per panel size) and picking random panels per size — ≈5 × wall-clock. v1 is one model, any panel. That is the value proposition.

## Data for figure

- `results/v1_sweep_damond/panel_sweep.json`
- `results/v1_sweep_hochschulz/panel_sweep.json`
- `results/v1_sweep_jackson/panel_sweep.json`
- `scripts/plot_panel_sweep.py`

## Next

- Plan 3: add H&E foundation-model condition (UNI / Phikon / Virchow2) to close the remaining size-10 gap.
- Plan 4: add scRNA-seq condition + cross-dataset (train on one IMC cohort, evaluate panel extension on another).
