# SpaProtFM v0 vs Murphy-style Baseline (10 fixed observed)

| Dataset | Baseline PCC | SpaProtFM v0 PCC | Δ | Win? |
|---|---|---|---|---|
| Damond pancreas | 0.421 | 0.385 | -0.036 | No |
| HochSchulz melanoma | 0.493 | 0.470 | -0.023 | No |
| JacksonFischer breast | 0.376 | 0.311 | -0.065 | No |

## Training wall-clock

| Dataset | Baseline | v0 | v0 overhead |
|---|---|---|---|
| Damond pancreas | 154s | 141s | -9% (faster — 30 vs 20 epochs) |
| HochSchulz melanoma | 333s | 295s | -11% |
| JacksonFischer breast | 375s | 324s | -14% |

v0 trains at comparable speed to the baseline despite 2× input channels (2C vs C), because base=32 is the same and the tile counts are similar.

## Per-marker breakdown

### Damond pancreas (38 ch, 28 target)

Top 3 improvements over baseline:
- CD20: +0.041
- FOXP3: +0.005
- PPY: +0.002

Top 3 regressions vs baseline:
- AMY2A: -0.081
- CDH1: -0.084
- PDX1: -0.116

### HochSchulz melanoma (46 ch, 36 target)

Top 3 improvements:
- CTNNB1: +0.017
- (only 1 marker improved)

Top 3 regressions:
- CD45RA: -0.043
- CD68: -0.062
- CD20: -0.124

### JacksonFischer breast (45 ch, 35 target)

No markers improved. Top 3 regressions:
- FN1: -0.162
- DNA1: -0.171
- PanCK: -0.192

## Key takeaways

- v0 did NOT beat the baseline on any of the 3 datasets. The gap is -0.036 / -0.023 / -0.065 PCC.
- HochSchulz is closest; only 1 marker regressed significantly (CD20 -0.12).
- JacksonFischer shows the worst regression; notably DNA1/DNA2 are in the OBSERVED set for the baseline but as unobserved targets for v0 in JacksonFischer (first 10 channels include 6 Ruthenium probes — essentially blank channels), so the effective information content given to v0 at eval time is worse than for the baseline.
- The Jackson failure may be a channel-ordering artifact: the first 10 channels in JacksonFischer are `Ru96–Ru104` (calibration channels) and `H3`, `H3K27me3`, `KRT5` — six Ru channels contain near-zero signal. The baseline also uses these as input, so both models have the same observed channels; but v0 sees a much harder training signal because during random masking a uniform channel draw often picks Ru channels.

## Hypotheses why v0 < baseline, and what to try in v1

1. **Epoch count mismatch**: Baseline uses 20 epochs; v0 used 30. Even so, v0 still underperforms. The val_fixed loss plateaus around epoch 20–25 with no further gain; baseline was already well-converged at 20 epochs. Not the primary cause.

2. **Random masking as multi-task hurts focus**: By training on all masking scenarios simultaneously (k in [3, C-1]), the model must learn C conditional distributions rather than the one fixed distribution the baseline targets. This capacity dilution is the most likely cause of the PCC gap at equal model size.
   - **v1 fix**: increase base to 64 (4× params), OR use a dedicated "eval scenario" fine-tuning phase of 5 epochs with the fixed 10-observed mask after the random-mask pre-training phase.

3. **Loss denominator mismatch**: The unobserved-only loss (averaged over ~(C-10)/C of channels per batch) produces gradients of different scale than the baseline's MSE over all target channels. Normalising by total pixels regardless of mask count might stabilise training.

4. **No channel embedding**: The model receives a mask plane per channel but has no explicit channel identity encoding (e.g., learnable channel embedding added to the feature map). Without it, the model cannot distinguish "I am predicting marker 11 vs marker 35" — it relies entirely on relative spatial context.
   - **v1 fix**: add a learned `(C, 1, 1)` channel bias added to the input before the first conv.

5. **Per-batch random mask (all tiles same mask per batch)**: The current implementation samples one mask per sample (per-row in the mask), but all within a batch see different masks. This is correct but means within a single batch, the loss is noisy across very different conditioning scenarios. Curriculum masking (start with k near C-1, anneal down) may help.

## Conclusion

v0 is a clean proof-of-concept demonstrating the masked-input conditioning interface. It does not yet beat baseline. The closest dataset (HochSchulz, Δ=-0.023) suggests that with a slightly larger model or a fine-tuning phase, v0 could achieve parity. v1 should increase model capacity (base=64) and add a short fixed-mask fine-tuning stage.
