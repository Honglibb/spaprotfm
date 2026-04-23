# SpaProtFM

**Cross-cohort masked panel extension for imaging mass cytometry (IMC)
with pseudo-H&E foundation-model conditioning.**

SpaProtFM v2 is a masked U-Net that predicts missing protein channels
from observed ones in multiplexed IMC tiles, conditioned on features
from a frozen pathology foundation model (Phikon-v2) that consumes a
synthetic H&E image generated from the IMC DNA and biomarker channels.
A single checkpoint trained on two Bodenmiller IMC cohorts generalises
zero-shot to a held-out third cohort and out-performs per-cohort
baselines — see the associated manuscript (reference below).

The method requires no paired H&E imaging, no fine-tuning of the
foundation model, and no cohort-specific adaptation.

---

## Install

```bash
uv sync
uv pip install -e .
```

Python ≥ 3.11, PyTorch ≥ 2.4. All other dependencies are in
`pyproject.toml`. GPU with ≥ 16 GB VRAM is recommended for training
(RTX 3090 used in the paper).

## Data

The three IMC cohorts used in the paper are publicly distributed via
Bioconductor's [`imcdatasets`](https://bioconductor.org/packages/imcdatasets/):

- **Damond 2019** (pancreas, T1D) — 38 markers
- **Hoch-Schulz 2022** (melanoma) — 46 markers
- **Jackson-Fischer 2020** (breast) — 45 markers

Download the three `.rds` files and place them under `data/raw/imc/`.
On first use, the loader (`spaprotfm.data.bodenmiller.load_imc_rds`)
converts each `.rds` to a flat per-tile `.npy` cache via an R
sub-process; subsequent runs read from the cache directly.

The loader also handles the canonical 10-marker panel projection used
in the paper:

```
[DNA1, DNA2, H3, SMA, CD20, CD3e, CD45, CD68, Ki67, cPARP]
```

with alias harmonisation across cohorts (e.g. `Ki67_Er168 → Ki67`).

## Pretrained checkpoints

Pretrained weights for the four main checkpoints used in the paper
are archived on Zenodo at [DOI TBD — see §Citation].

| file | training set | use |
|------|------------|-----|
| `xds_pooled_s42/model.pt`    | all 3 cohorts (in-dist) | in-distribution evaluation |
| `xds_heldoutD_s42/model.pt` | HochSchulz + Jackson    | zero-shot OOD on Damond |
| `xds_heldoutH_s42/model.pt` | Damond + Jackson        | zero-shot OOD on HochSchulz |
| `xds_heldoutJ_s42/model.pt` | Damond + HochSchulz     | zero-shot OOD on Jackson |

Each checkpoint is the `state_dict()` of a `MaskedUNetV2` with
`base=48`, `cond_dim=192`, `cond_grid=14`, `cond_in=1024` (Phikon-v2
hidden size). To load:

```python
import torch
from spaprotfm.models.spaprotfm_v2 import MaskedUNetV2

model = MaskedUNetV2(n_channels=10, base=48, cond_dim=192, cond_grid=14)
model.load_state_dict(torch.load("xds_pooled_s42/model.pt", map_location="cpu"))
model.eval()
```

## Reproduce paper results

### Train

Cross-dataset pooled + leave-one-out (4 runs, ~4 GPU-hours each on RTX 3090):

```bash
for seed in 0 1 42; do
  uv run python scripts/train_spaprotfm_v2_xds.py \
      --train-datasets Damond,HochSchulz,Jackson \
      --heldout-datasets "" \
      --out-dir results/xds_pooled_s${seed} --run-name pooled_s${seed} \
      --seed ${seed} --save-checkpoint

  for held in D H J; do
    case $held in
      D) train="HochSchulz,Jackson";;
      H) train="Damond,Jackson";;
      J) train="Damond,HochSchulz";;
    esac
    case $held in D) test=Damond;; H) test=HochSchulz;; J) test=Jackson;; esac
    uv run python scripts/train_spaprotfm_v2_xds.py \
        --train-datasets ${train} --heldout-datasets ${test} \
        --out-dir results/xds_heldout${held}_s${seed} \
        --run-name heldout${held}_s${seed} \
        --seed ${seed} --save-checkpoint
  done
done
```

Murphy baselines (9 runs, ~1 GPU-hour each):

```bash
for seed in 0 1 42; do
  for src in Damond HochSchulz Jackson; do
    uv run python scripts/train_spaprotfm_v2_xds.py \
        --train-datasets ${src} --heldout-datasets Damond,HochSchulz,Jackson \
        --out-dir results/murphy_xds_${src}_s${seed} \
        --run-name murphy_${src}_s${seed} \
        --seed ${seed} --no-cond --no-random-mask
  done
done
```

### Figures

```bash
uv run python scripts/plot_paper_figures.py        # Fig 2, Fig 3, Sup S1
uv run python scripts/plot_qualitative_predictions.py  # Fig 4
uv run python scripts/plot_architecture.py         # Fig 1 (placeholder)
```

All outputs land in `results/figures/paper/` as vector PDF + 600-dpi
PNG at Briefings in Bioinformatics column widths.

## Repository layout

```
src/spaprotfm/           package
  ├─ data/                IMC loaders, tiling, normalisation, canonical panel
  ├─ models/              spaprotfm_v0 (baseline), v1 (masked UNet), v2 (+Phikon)
  ├─ baselines/           vanilla U-Net used by Murphy baseline
  ├─ condition/           Phikon-v2 encoder wrapper + pseudo-H&E synthesis
  └─ eval/                Pearson / MSE metrics
scripts/                 training + plotting CLIs
tests/                   unit tests
docs/
  ├─ manuscript/         paper drafts
  └─ superpowers/        design docs and result notes
```

## Citation

If you use SpaProtFM in your work, please cite:

```bibtex
@article{yin2026spaprotfm,
  author  = {Yin, Hongli},
  title   = {SpaProtFM: pseudo-H\&E-conditioned masked panel extension for
             cross-cohort imaging mass cytometry},
  journal = {Briefings in Bioinformatics},
  year    = {2026},
  note    = {in submission}
}
```

as well as the three source datasets (Damond 2019, Hoch-Schulz 2022,
Jackson 2020) and the Phikon-v2 foundation model — full reference list
in the manuscript.

## License

Code: MIT. Pretrained weights: CC-BY 4.0. Raw data belongs to the
original dataset authors — see the `imcdatasets` Bioconductor package
for terms.
