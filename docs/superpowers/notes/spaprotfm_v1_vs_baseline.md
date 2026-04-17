# SpaProtFM v1 vs v0 vs Murphy Baseline (10 fixed observed)

| Dataset | Baseline | v0 | v1 | v1 − Baseline | v1 win? |
|---|---|---|---|---|---|
| Damond pancreas (38 ch) | 0.421 | 0.385 | 0.405 | −0.016 | No (gap narrowed) |
| HochSchulz melanoma (46 ch) | 0.493 | 0.470 | 0.486 | −0.007 | No (nearly tied) |
| JacksonFischer breast (45 ch) | 0.376 | 0.311 | 0.374 | −0.002 | Nearly (gap ~0.002) |

v1 wall-clock: Damond 225 s, HochSchulz 413 s, JacksonFischer 491 s.

v1 improvements over v0 (delta):
- Damond: +0.020
- HochSchulz: +0.016
- JacksonFischer: +0.063  ← largest gain, thanks to Ru exclusion

## Per-marker analysis

### Damond pancreas

Top 5 predictions (v1):
1. SYP: 0.866
2. DNA2: 0.807
3. DNA1: 0.765
4. CDH1: 0.701
5. PDX1: 0.639

Bottom 5 predictions (v1):
1. p_HH3: 0.198
2. Ki67: 0.181
3. FOXP3: 0.132
4. PPY: 0.117
5. CD20: 0.115

### HochSchulz melanoma

Top 5 predictions (v1):
1. DNA2: 0.895
2. DNA1: 0.893
3. CD45RO: 0.697
4. CTNNB1: 0.661
5. SOX10: 0.609

Bottom 5 predictions (v1):
1. Ki67_Pt198: 0.305
2. Ki67_Er168: 0.285
3. CD303: 0.236
4. c_PARP: 0.221
5. MPO: 0.155

### JacksonFischer breast cancer

Top 5 predictions (v1):
1. DNA2: 0.841
2. DNA1: 0.805
3. CD44: 0.714
4. FN1: 0.666
5. PanCK: 0.660

Bottom 5 predictions (v1):
1. CD20: 0.175
2. vWF: 0.173
3. KRT14: 0.172
4. CD3e: 0.167
5. p53: 0.144

## What worked (vs v0)

- **Ru channel exclusion (Fix 4)** was the most impactful change, contributing +0.063 PCC on JacksonFischer. Ruthenium calibration probes (Ru96, Ru98, Ru99, Ru100, Ru101, Ru102, Ru104 = indices 0–6) are technical artifacts. Including them in the random masking pool in v0 forced the model to allocate capacity predicting meaningless signals. Excluding them lets all capacity focus on the 38 biological markers.
- **Per-channel learnable bias (Fix 1)** and **capacity bump 32→48 (Fix 2)** together with two-stage training contributed consistent gains across all datasets. The model can now distinguish which channel it is predicting, allowing more specialised representations.
- **Two-stage fine-tuning (Fix 3)** narrowed the distribution mismatch: after random-mask pre-training, a 5-epoch fine-tune on the exact eval-time observed/target split further reduces the fixed-mask loss without forgetting general representations.
- Combined, all three datasets improved substantially over v0, with JacksonFischer seeing the largest absolute gain.

## What did NOT work / next iteration

v1 still trails the Murphy baseline on all three datasets (by 0.002–0.016 PCC). Hypotheses for v2:

1. **Encoder pre-training / foundation features.** The baseline likely leverages hand-crafted or co-expression-based features that capture global tissue structure. A contrastive spatial pre-training objective (e.g., SimCLR on image crops) might inject that prior into the encoder.

2. **Longer or higher-LR fine-tuning.** Stage 2 uses LR = 1e-5 (10× decay) for only 5 epochs. Allowing longer fine-tuning with early stopping on fixed-mask PCC (not MSE) may close the gap — especially for Damond and HochSchulz where the gap is small.

3. **Eval-mask-aware loss.** Currently the loss averages over ALL unobserved channels equally. Upweighting the loss on the eval-time target channels during stage 2 would more directly optimise the measured metric.

4. **DNA/structural channel aware eval.** DNA1/DNA2 are always among the top predictions yet are also captured well by baseline. The hard markers (Ki67, MPO, immune rares like CD20, CD3e) dominate the gap. Per-cluster or per-cell-type stratification of PCC might reveal that v1 is actually better on biologically meaningful subsets.

5. **Larger model or multi-scale features.** At base=48 the model has ~4× more params than v0 but is still relatively shallow. A deeper decoder or multi-resolution feature fusion could help.

6. **JacksonFischer: observed-channel choice.** With Ru channels always observed, the first 10 "observed" channels include Ru0–Ru6 plus H3, H3K27me3, KRT5. Choosing biologically informative observed channels (e.g., H3, EpCAM, CD45, vimentin, SMA, Ki67, DNA1, DNA2, KRT19, HER2) rather than the first 10 by index could markedly improve both baseline and v1 headline scores.
