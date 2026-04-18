# Plan 3 — H&E foundation-model conditioning for SpaProtFM v2

**Date**: 2026-04-18  
**Goal**: close the remaining 0.02-0.04 PCC gap at size=10 vs the per-panel Murphy baseline by conditioning the masked U-Net on pretrained pathology FM features extracted from pseudo-H&E synthesized from IMC DNA channels.

**Scope**: one training recipe change (add condition branch). Stay on the 3 Bodenmiller IMC datasets. No new data download.

## Why pseudo-H&E, not real H&E

None of Damond / HochSchulz / JacksonFischer has a matched H&E image on the same slide (confirmed by checking Mendeley/Zenodo supplements, 2026-04-18). Options considered:

| Option | Risk | Effort | Status |
|---|---|---|---|
| Real H&E on Murphy CODEX spleen | new dataset, registration, different modality from headline story | high | deferred |
| Pseudo-H&E from IMC DNA channels + Phikon-v2 | method generalizes to *any* IMC dataset; ≈deep-imcyto precedent | low | **chosen** |

Paper framing: "We use pretrained pathology FM features as morphological prior, via pseudo-H&E synthesized from IMC DNA channels. This requires no paired H&E and transfers FM-learned tissue structure to IMC panel extension."

## Architecture (MaskedUNetV2)

```
IMC tile (C, 128, 128)             Pseudo-H&E RGB (3, 224, 224)
        │                                      │
        │                                      ▼
        │                               Phikon-v2 (frozen)
        │                               → (B, 196, 768) cls+patch tokens
        │                                      │
        │                                      ▼
        │                               proj + reshape → (B, 768, 14, 14)
        │                                      │
        │                                      ▼
        │                               conv 1×1 → (B, base*4, 14, 14)
        │                                      │
        │                                      ▼
        │                              bilinear upsample to 16×16
        │                                      │
        │                                      ▼
        ▼                                      │
  MaskedUNetV1 encoder … bottleneck (B, base*4, 16, 16)  ← concat
        │
        ▼
  decoder … (B, C, 128, 128)
```

Design notes:
- Phikon-v2 is **frozen** (grad off). Only the 1×1 projection + U-Net train.
- Pseudo-H&E is 224×224 to match Phikon's ViT patch grid (14×14 = 196 patches).
- Condition is injected at the U-Net bottleneck only (not all layers) to keep parameter count manageable.
- Rest of the model (encoder/decoder, per-channel bias, 2C input) is unchanged from v1.

## Pseudo-H&E synthesis

Per deep-imcyto recipe:
1. Identify DNA channel indices per dataset (we already know: DNA1/DNA2 in all three).
2. Hematoxylin-like (purple-blue) channel = `max(DNA1, DNA2)` (or sum, normalized).
3. Eosin-like (pink) channel = `mean(remaining_bio_channels)` as a tissue-density proxy.
4. Compose RGB: `R = 1 − 0.5·H − 0.3·E`, `G = 1 − 0.8·H − 0.2·E`, `B = 1 − 0.7·H` (approximate Beer-Lambert absorbance inversion, values clipped to [0,1]).
5. Resize to 224×224 with bilinear.

This is deterministic, no new weights, same tile coordinates as the IMC patch.

## Implementation phases

1. **P1** — `src/spaprotfm/condition/pseudo_he.py`: `synthesize_pseudo_he(tile, dna_idx, bio_idx) -> (3, 224, 224)` + tests. Guarantees output shape and [0,1] range on synthetic input.
2. **P2** — `src/spaprotfm/condition/phikon.py`: `PhikonEncoder` Lightning-compatible module that wraps `owkin/phikon-v2`, exposes `encode(rgb) -> (B, 768, 14, 14)` patch-feature map, frozen weights, eval mode. Cache on disk if weights not present. Test loads model + runs on random input + checks output shape.
3. **P3** — `src/spaprotfm/models/spaprotfm_v2.py`: `MaskedUNetV2` with bottleneck injection. Unit tests for shape + gradient isolation (Phikon params stay frozen).
4. **P4** — `scripts/train_spaprotfm_v2.py`: copy train_v1 + add condition branch. Default to Phikon-v2 with disk cache. Support `--panel-sweep` identical to v1.
5. **P5** — Run all 3 datasets with panel sweep. Compare v2 vs v1 at each size.

## Success criteria

Primary: v2 beats v1 at size=10 by ≥0.02 PCC on ≥2 of 3 datasets (bio targets).

Secondary: v2 retains monotonic panel-size scaling (no regression at size 3, 20, etc.).

Stretch: v2 at size=10 meets or beats the Murphy baseline (Damond 0.421, HochSchulz 0.493, Jackson bio 0.376).

## Failure-mode plan

- If Phikon-v2 download/auth fails → fall back to `timm` ResNet50 ImageNet as a cheap ablation (won't be headline, but proves the pipeline).
- If v2 = v1 within noise → ablate injection site (bottleneck vs all skip connections) before giving up on H&E conditioning.
- If pseudo-H&E visually looks wrong → switch to simpler grayscale concatenation of DNA channels.

## Out of scope for Plan 3

- Real H&E experiments (Plan 3.5 if needed).
- scRNA-seq conditioning (Plan 4).
- Cross-dataset panel-extension generalization (Plan 4).
- Diffusion-based generative head (original Plan 2.5 / paper second half).
