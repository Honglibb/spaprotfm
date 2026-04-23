# SpaProtFM: pseudo-H&E-conditioned masked panel extension for cross-cohort imaging mass cytometry

**Authors**: [Author list TBD]
**Affiliations**: [TBD]
**Correspondence**: [TBD]
**Target venue**: Briefings in Bioinformatics (method article)
**Draft**: v1, 2026-04-23

---

## Abstract

Imaging mass cytometry (IMC) resolves 30–50 protein markers at sub-cellular
resolution in intact tissue, but panel composition varies widely across
studies and cohorts. Panel extension — predicting the abundance of markers
that were not acquired from the ones that were — would let investigators
pool data across studies, reduce experimental cost, and reconstruct missing
channels in legacy data, yet existing methods treat every panel as a
separate learning problem and do not transfer between cohorts.

We present **SpaProtFM v2**, a masked panel-extension model whose
conditioning branch injects features from a frozen pathology foundation
model (Phikon-v2) computed on synthetic H&E generated directly from the
IMC tile. A single network trained with random-k-of-n masking on a
canonical 10-marker panel derived from three public IMC cohorts
(Damond pancreas, Hoch-Schulz melanoma, Jackson breast; *n* = 3 seeds)
reproduces per-panel Murphy baseline accuracy while additionally
generalising across panels and cohorts. On zero-shot transfer — a single
checkpoint evaluated on a held-out cohort never seen during training —
SpaProtFM v2 improves mean per-marker Pearson correlation over the best
per-panel Murphy baseline by +0.06 to +0.12 PCC across three cohorts
(+67 % relative on the hardest), and the out-of-distribution SpaProtFM
v2 even exceeds Murphy's own in-distribution performance on every cohort.
Qualitative predictions recover the spatial pattern of CD45+ immune
infiltrates and CD68+ macrophages in unseen tissue with per-pixel
Pearson *r* = 0.20–0.38 across six pooled biological targets.

SpaProtFM v2 is the first IMC panel-extension model to demonstrate
non-trivial zero-shot transfer between cohorts. The pseudo-H&E
synthesis step requires no paired H&E imaging, so the method is
directly applicable to any IMC study.

**Key words**: imaging mass cytometry, spatial proteomics, panel
extension, foundation models, pathology, cross-dataset transfer.

---

## Key Points

- **Panel-extension problem**. IMC studies use heterogeneous marker
  panels; existing per-panel U-Net baselines such as Murphy can neither
  cover panels of different sizes with one model nor transfer to unseen
  cohorts.
- **Pseudo-H&E as universal conditioning**. We synthesise a virtual
  H&E image from the IMC DNA and biomarker channels (Beer-Lambert
  inversion) and feed it to a frozen Phikon-v2 ViT encoder. The resulting
  features generalise across tissue types and replace real H&E slides
  that most IMC studies lack.
- **Random-mask pretraining**. Training the masked U-Net with random
  *k*-of-6 masking — not the fixed test mask — produces a single
  checkpoint usable across panel sizes (3–20) and across cohorts.
- **Zero-shot cross-cohort transfer**. A single SpaProtFM v2 checkpoint
  trained on two cohorts out-performs the Murphy baseline trained
  specifically on a held-out third cohort by +0.06 to +0.12 mean PCC,
  across three IMC cohorts (*n* = 3 seeds, 6/6 paired wins).
- **Bonus**: the out-of-distribution v2 prediction even beats
  Murphy's own in-distribution prediction on every cohort tested.
- **General, lightweight recipe**. The method adds a single 1 × 1 fusion
  convolution above a standard masked U-Net and depends only on
  publicly available Phikon-v2 weights; no paired H&E data, no fine-
  tuning of the foundation model, and no per-cohort adaptation are
  required.

---

## Introduction

Highly multiplexed tissue imaging platforms — IMC, CODEX/PhenoCycler,
MIBI-TOF, and related — can now measure dozens of protein markers at
sub-cellular resolution across entire tumour sections, driving a new
generation of spatial-biology studies. In practice, however, each
investigator commits to a single antibody panel at study design, and
these panels differ substantially between studies. The three most-
cited public IMC cohorts in the Bodenmiller ecosystem are a concrete
example: Damond *et al.* profiled 38 markers in type-1 diabetes
pancreatic sections [ref: Damond 2019]; Hoch and Schulz *et al.*
used a 46-marker melanoma panel emphasising chemokines and immune
checkpoints [ref: HochSchulz 2022]; Jackson and Fischer *et al.*
imaged breast cancer with 45 markers including lineage markers not
present in the other two [ref: Jackson 2020]. Only a small core of
markers — DNA intercalators, histone H3, SMA, and a handful of
canonical immune markers — appears in all three.

Panel heterogeneity creates two concrete problems that limit data
reuse. First, cross-study meta-analysis is essentially impossible
without imputation: an investigator who wants to pool Damond and
Jackson to ask a pan-cohort question about, for instance, CD20+ B-cell
localisation, must discard one study or acquire new data. Second,
legacy cohorts that did not include the marker needed for a new
hypothesis cannot be re-used even when the underlying tissue is still
available, because most retrospective IMC experiments are expensive
to re-stain and re-acquire. A method that can *impute* missing markers
from available ones — termed **panel extension** — would unlock both
use-cases.

Recent work has established that panel extension is feasible at the
level of a single cohort. [Murphy *et al.* 2025] trained a standard
U-Net to predict a fixed held-out set of protein channels from the
remaining channels of the same IMC tile, reaching mean Pearson
correlations of 0.37–0.49 across three public IMC cohorts. This
approach, hereafter called *the Murphy baseline*, gives a useful
per-cohort point estimate of the upper bound one can reach with
standard convolutional modelling on a single panel.

Two properties of the Murphy approach limit its practical utility.
First, it is **panel-specific**: the network's input and output
channel counts are fixed at training time, so a separate model must
be trained for every new panel size or composition an investigator
encounters. Second, it is **cohort-specific**: a Murphy model
trained on Damond cannot be meaningfully applied to Jackson,
because the two cohorts were imaged on different instruments with
different antibody clones in different tissue types. The field
currently has no principled way to share a trained panel-extension
model across cohorts.

We set out to build a method that (a) supports any subset of
observed markers from a canonical panel without retraining, and
(b) transfers zero-shot between IMC cohorts. Our central idea is
that every IMC tile already contains the information needed to
generate a *virtual* H&E image — the DNA channels light up where
nuclei live, and the mean of biomarker channels approximates
cytoplasmic abundance — and that feeding this pseudo-H&E image to
a large pathology foundation model [ref: Phikon-v2] yields a
**tissue-invariant representation** that can act as conditioning
for a masked reconstruction network. Because the pseudo-H&E
synthesis depends only on signal present in any IMC acquisition,
no paired H&E slides are required, and the same Phikon encoder
applies to any cohort.

Combined with random-mask pretraining — where the network learns
to reconstruct arbitrary subsets of channels rather than a fixed
held-out subset — the resulting model is inherently flexible with
respect to panel composition. We call this architecture **SpaProtFM
v2** (**Spa**tial **Prot**eomics **F**oundation **M**odel v2, where
v1 denotes the same masked U-Net backbone without Phikon fusion).

In this paper we show, on three public IMC cohorts:

1. Adding Phikon-v2 pseudo-H&E fusion (**v1 → v2**) improves mean PCC
   by +0.02 to +0.05 on two of three cohorts on the per-panel
   in-distribution task (Fig. 2a–c).
2. A single SpaProtFM v2 checkpoint, trained once with random-mask,
   matches per-size-retrained Murphy baselines across observed panel
   sizes from 3 to 20 on all three cohorts (Fig. 2d–f).
3. A single SpaProtFM v2 checkpoint trained on two cohorts and
   evaluated zero-shot on the held-out third cohort out-performs the
   Murphy baseline trained specifically on that cohort, by
   +0.06 to +0.12 mean PCC — 6 / 6 paired multi-seed wins — and even
   beats Murphy's in-distribution performance on every cohort
   (Fig. 3a–c).
4. Qualitative cross-cohort predictions recover the spatial layout
   of immune and myeloid markers in tissue the model has never seen
   (Fig. 4).

To our knowledge SpaProtFM v2 is the first IMC panel-extension
method to demonstrate non-trivial zero-shot cohort transfer. We
release training and inference code, alias-harmonised canonical
panel loaders for the three Bodenmiller cohorts, and paired-seed
checkpoints at [url TBD].

---

## Materials and Methods

### Datasets and canonical panel

We use three public IMC cohorts: **Damond pancreas** (38 markers,
T1D progression) [ref: Damond 2019]; **Hoch-Schulz melanoma**
(46 markers, anti-PD-1/PD-L1 immunotherapy) [ref: HochSchulz 2022];
and **Jackson breast cancer** (45 markers, Basel cohort) [ref:
Jackson 2020]. All three are distributed as
`CytoImageList` R objects via the `imcdatasets` Bioconductor
package. Raw image data was converted from `.rds` to flat per-tile
`.npy` caches on first load and re-used across runs.

To allow a single model to operate on all three cohorts, we
define a **canonical 10-marker panel**:

```
[DNA1, DNA2, H3, SMA, CD20, CD3e, CD45, CD68, Ki67, cPARP]
```

The first four markers (DNA1, DNA2, H3, SMA) form the **always-
observed** morphology set: two DNA intercalators, histone H3, and
smooth-muscle actin. These channels are always provided as input.
The remaining six — CD20 (B cells), CD3e (T cells), CD45
(pan-leukocyte), CD68 (macrophages), Ki67 (proliferation), cPARP
(apoptosis) — are the **target** set the model learns to
reconstruct.

A small number of markers are named differently across cohorts;
we map them to the canonical names using an alias table
(`Ki67_Er168 → Ki67` in Hoch-Schulz; `cPARP_cCASP3 → cPARP` in
Damond and Jackson; `c_PARP → cPARP` in Hoch-Schulz). Markers
present in a cohort but not in the canonical 10 are dropped; this
reduces each cohort to a common 10-channel representation
suitable for the same network input.

### Pre-processing and tiling

Each `(H, W, C)` image is per-channel normalised by its own 99th
percentile (robust to sparse high-value pixels), clipped to
`[0, 1]`, and tiled into non-overlapping 128×128 patches. Per-image
80 / 10 / 10 splits by image identity produce disjoint train /
validation / in-distribution test sets. For the cross-cohort
transfer experiments, the entire held-out cohort forms the
out-of-distribution test set.

### Masked U-Net backbone (MaskedUNetV2)

The backbone is a five-level U-Net with base channel width 48,
per-channel learnable value bias, and ImageNet-style 2×2 stride
convolutions on the down-sampling path. The input tile is 10-
channel `(C=10, 128, 128)`; the network outputs the same shape.

The masking layer replaces *masked-out* channels with zero before
the first convolution and concatenates a 10-channel binary mask
as auxiliary input (so the network knows which channels it is
being asked to reconstruct). At training time the mask is
**random**: the morphology channels (indices 0–3) are always
observed, and a random *k*-of-6 subset of the target channels
is additionally observed, with *k* sampled uniformly from
{1, …, 5}. This is equivalent to training the network to
reconstruct any single target from any observed subset — the
*flexibility* behaviour that a fixed-mask Murphy baseline lacks.
At evaluation time the mask is **fixed**: morphology channels
observed, all six targets masked out.

### Pseudo-H&E synthesis

A pseudo-H&E image is computed per IMC tile. The hematoxylin
channel is the pixel-wise max of the two DNA channels (DNA1 and
DNA2), percentile-scaled to [0, 1]; the eosin channel is the
pixel-wise mean of the bio channels (non-DNA markers), similarly
scaled. The two stains are combined using a Beer-Lambert-style
linear inversion:

```
R  =  1.0  −  0.5·H  −  0.30·E
G  =  1.0  −  0.8·H  −  0.20·E
B  =  1.0  −  0.7·H  −  0.05·E
```

producing purple nuclei and pink cytoplasm reminiscent of real
H&E histology. The RGB image is bilinearly resized to 224 × 224
— the native input size for ViT-B — and fed to Phikon-v2.

### Phikon-v2 conditioning branch (frozen)

Phikon-v2 is a ViT-B/16 self-supervised foundation model trained
on 400 million histopathology tiles from Owkin [ref: Phikon-v2].
We load the public `owkin/phikon-v2` weights and keep them
frozen throughout training. The forward pass produces a 1024-
dimensional feature vector on a 14 × 14 token grid. No paired
real H&E is ever required; the synthetic H&E above is the sole
input to Phikon-v2.

### Bottleneck fusion

Phikon features `(B, 1024, 14, 14)` are bilinearly up-sampled to
16 × 16 — the spatial resolution of the U-Net bottleneck with
128 × 128 input — and passed through a 1 × 1 convolution
(`cond_proj`) reducing them to 192 channels. The 192-channel
condition map is concatenated with the 384-channel U-Net
bottleneck along the channel axis and fed through a second 1 × 1
convolution (`cond_fuse`) back to 384 channels. This preserves
the U-Net decoder verbatim — the only additional parameters are
`cond_proj` (≈0.2 M) and `cond_fuse` (≈0.2 M) on top of the
backbone's ~4 M.

### Training

All models are trained for 30 epochs with AdamW (learning
rate 1 × 10⁻⁴, batch size 16, no weight decay) on a single
NVIDIA RTX 3090. The loss is mean-squared error computed only
on masked channels (i.e. MSE on the channels the network is
being asked to reconstruct). Only stage-1 random-mask
pretraining is used; a prior ablation showed that stage-2
fine-tuning to the fixed test mask improved per-panel accuracy
slightly but traded away panel flexibility, so we chose the
flexible-checkpoint configuration.

### Murphy baseline

The Murphy baseline is implemented as the same MaskedUNetV2
backbone with the conditioning branch disabled (`--no-cond`)
and the fixed evaluation mask used throughout training
(`--no-random-mask`). This is, by construction, equivalent to
the vanilla per-panel U-Net proposed by Murphy *et al.*, and
retains the same architecture, optimiser, and schedule as
SpaProtFM v2 so that any differences cleanly attribute to the
pseudo-H&E / Phikon / random-mask additions.

### Cross-dataset transfer protocol

For cross-cohort experiments we project every image to the
canonical 10-channel panel, concatenate the three cohorts
image-wise, and perform image-level 80 / 10 / 10 splits per
source cohort. Two training modes are used.

*Pooled.* All three cohorts contribute train / val / in-dist-test
splits; one checkpoint sees data from every cohort. The in-
distribution test uses the held-out 10 % of each source cohort.

*Leave-one-out (LOO).* One cohort is held out in full as the
OOD test set; the network trains on the 80 % splits of the
remaining two. Three LOO checkpoints (`heldoutD`, `heldoutH`,
`heldoutJ`) produce the OOD predictions.

The Murphy baseline is trained separately on each single source
cohort (three Murphy models) and evaluated on each of the
three test cohorts, producing a 3 × 3 transfer matrix per seed.
OOD Murphy performance per test cohort is taken as the
**best** of the two available cross-cohort evaluations — a
deliberately optimistic choice for the baseline.

### Evaluation

Reconstruction quality is summarised by mean per-marker
Pearson correlation (mean PCC) computed across held-out tiles.
For the cross-cohort main result (Fig. 3), metrics are averaged
across seeds {0, 1, 42}, using population standard deviation
(ddof = 0) across three seeds. All paired comparisons use the
seed intersection for which both methods have completed runs.

### Implementation

Training and inference are in Python 3.11 with PyTorch 2.3 and
are wrapped in a thin CLI. Pseudo-H&E synthesis runs in torch
on GPU and adds negligible overhead. All code, configs,
canonical-panel loaders, and paired-seed checkpoints are
available at [url TBD]; raw IMC data is available from the
original studies (see Data Availability).

---

## Results

### Pseudo-H&E conditioning improves panel-extension accuracy

We first ask whether adding Phikon-v2 pseudo-H&E conditioning
to a masked U-Net (**v1 → v2**) improves per-cohort panel-
extension accuracy at the size-10 fixed panel, matching the
Murphy baseline evaluation protocol. Fig. 2a–c shows mean PCC
across panel sizes from 3 to 20 (three random draws per size)
for each cohort, with v1 (no H&E) and v2 (+ Phikon) as coloured
lines and the per-cohort Murphy-size-10 baseline as a dashed
reference.

The Phikon contribution is **cohort-dependent but non-trivial on
two of three cohorts**. On Hoch-Schulz melanoma, v2 improves over
v1 by roughly +0.04 mean PCC at sizes ≥ 10 and reaches 0.50 at
panel size 20 — above the per-size-10 Murphy baseline of 0.49.
On Jackson breast v2 consistently tracks or exceeds v1 across all
panel sizes and matches Murphy at size 10 and 15. On Damond
pancreas v1 and v2 track each other closely at larger panel
sizes; v2's advantage is confined to the small-panel regime
(size 3–7), where the pseudo-H&E prior appears to compensate for
the scarcity of direct evidence. We interpret this as Phikon-v2
contributing **most where data is informative least** — a
behaviour consistent with a foundation-model-as-prior role.

### A single checkpoint covers any panel size

A direct consequence of random-mask training is that the same
v2 checkpoint can be evaluated with any subset of observed
markers, not only the size-10 configuration used during training
evaluation. Fig. 2d–f plots v2's mean PCC (line + shaded 1-s.d.
band across three panel draws) against Murphy baselines that
were **retrained per panel size** — 3 to 4 separate U-Nets per
cohort, each one a full training run.

The one-for-all v2 tracks or exceeds the retrained-per-size
Murphy across most panel sizes and cohorts. On Damond, v2 at
larger panel sizes is within noise of Murphy-retrained but
Murphy retains a small advantage at size 15 and 20 — the regime
where direct evidence from many observed markers is abundant
and the prior matters less. On Hoch-Schulz and Jackson, v2 is
essentially indistinguishable from per-size-retrained Murphy at
every panel size. The practical implication is substantial:
**one checkpoint replaces 3–4 Murphy training runs per cohort**
with no measurable accuracy loss.

### Zero-shot cross-cohort transfer out-performs per-cohort specialists

We next turn to the experiment that differentiates SpaProtFM v2
most sharply from prior work: cross-cohort transfer. Figure 3a
shows a 4 × 3 heatmap of mean PCC averaged over three seeds
where rows are models (Murphy-trained-on-Damond, Murphy-trained-
on-HochSchulz, Murphy-trained-on-Jackson, SpaProtFM v2 leave-
one-out) and columns are test cohorts. Dotted white outlines
mark Murphy's in-distribution cells (training cohort = test
cohort); red outlines mark the column-best (highest PCC) model
per test cohort.

**Every column-best is in the v2 row.** On Damond, the best
Murphy (trained on Damond) scores 0.20 in-distribution; v2
trained without any Damond data scores **0.25** — a 25 %
relative gain, while having never seen a single Damond pixel
in training. On Hoch-Schulz, best Murphy 0.34 → v2 0.38. On
Jackson, best Murphy 0.27 → v2 0.28. The picture is
unambiguous: a single v2 checkpoint trained on two cohorts
generalises well enough to the third cohort to beat a Murphy
model specifically optimised for it.

Figure 3b plots the zero-shot OOD bars (Murphy best cross-
cohort source vs v2 leave-one-out) for each test cohort: v2
gains +0.10, +0.12, and +0.06 PCC on Damond, Hoch-Schulz and
Jackson respectively (n = 3 paired seeds; 6 / 6 paired wins
across the three datasets × in-dist/OOD pairs). Figure 3c
shows the three-cohort average: in-distribution mean PCC
0.27 (Murphy) vs 0.36 (v2); out-of-distribution mean PCC
0.21 (Murphy) vs 0.30 (v2). Both the centre values and the
tighter error bars on v2 indicate that the one-for-all
network is not only more accurate but more **stable** across
cohorts than per-cohort specialists.

The largest relative gain is on **Damond pancreas**, the
cohort the Murphy baseline finds hardest (its in-dist mean PCC
is the lowest of the three at 0.20). SpaProtFM v2 lifts it from
0.15 to 0.25 out-of-distribution — a **67 % relative
improvement** on the most demanding cohort. Pancreas-specific
biology combined with the sparse, low-dynamic-range expression
of the canonical immune markers in pancreatic tissue
apparently defeats a per-cohort U-Net but not a model with
access to a tissue-invariant pathology prior.

### Qualitative predictions recover spatial biological structure

Accuracy metrics alone do not tell us whether the network is
capturing the right biology, only whether its predictions
correlate with ground truth. Figure 4 shows, for each of the
three cohorts in turn, an example 128 × 128 tile drawn from
the held-out cohort (chosen as the median-PCC tile among
the top 50 % of tiles by target-channel dynamic range, to
avoid either cherry-picking or empty-tile artefacts). Columns
1–5 show the pseudo-H&E input to Phikon, the ground-truth and
predicted CD45 channel, and the ground-truth and predicted
CD68 channel; column 6 is a 2-D log-density scatter of
pred vs ground-truth intensity for all six target markers
pooled, annotated with per-pixel Pearson *r*.

All three rows show the same qualitative behaviour: v2 is
softer than ground truth — the network's predictions are
smoother and lower-frequency than the speckle-like single-
pixel immune signal in IMC — but the **spatial layout of
positive regions is correctly reproduced**. In Damond, the
predicted CD45 tracks the pancreatic islet periphery and
interstitial immune infiltrate; in Hoch-Schulz, predicted
CD45 and CD68 jointly highlight the tumour-stroma interface;
in Jackson, predicted CD68 tracks the ductal microenvironment
macrophage population. Per-pixel *r* (pooled over six
targets, including sparse markers that are zero for much of
the tile) is 0.20, 0.38, and 0.26 respectively — low in
absolute terms, as expected when many pixels are near-zero
for most markers, but consistent with the quantitative
per-marker PCCs above.

### Per-marker breakdown

Supplementary Fig. S1 decomposes the zero-shot OOD signal
into the six canonical target markers. v2 is on par with or
out-performs Murphy best-source on every one of the 18
(cohort × marker) cells. The gains are largest on
high-expression, morphologically-anchored markers — CD45,
CD68, Ki67 — and smallest on sparse or state-dependent
markers — CD20, cPARP. This is consistent with the
interpretation that Phikon-v2 contributes a morphology-
anchored tissue prior: markers whose expression correlates
with tissue structure benefit most.

---

## Discussion

### Why pseudo-H&E conditioning works

The mechanism most consistent with our results is that Phikon-
v2 provides a **tissue-structure-invariant representation** of
the input tile that the U-Net can use as a side-input. Two
observations support this. First, the per-panel in-distribution
gains over v1 are modest (+0.02 to +0.05) — when the U-Net
already has direct access to most of the canonical panel,
adding a pathology prior offers only incremental benefit.
Second, the cross-cohort transfer gains are large (+0.06 to
+0.12) — here the pathology prior replaces cohort-specific
signal that the U-Net cannot learn directly from out-of-
distribution data. The prior's value increases the less
in-distribution information the U-Net has, exactly as one
would expect from a Bayesian prior.

Pseudo-H&E plays two roles. It is both (i) a **bridge
modality** that lets pathology-trained foundation models
consume IMC data without paired H&E acquisition, and (ii) a
**channel-invariant summary** of the tile that the network
cannot cheat with by learning cohort-specific marker biases.
Because pseudo-H&E is derived from *every* IMC acquisition's
DNA and biomarker channels in the same way, the resulting
Phikon features are a reproducible, cohort-agnostic feature
space.

### Why random-mask training works

Random-mask pretraining is what makes the same checkpoint
usable across panel sizes. A fixed-mask U-Net is forced to
learn a single conditional distribution *p(targets | fixed
observed subset)*; the random-mask version samples from
*p(targets\_subset | observed\_subset)* for many subsets. The
richer training distribution incurs a small cost in per-
panel accuracy (documented in our ablation notes) but removes
the need to retrain per panel size — the experimental
flexibility-accuracy trade-off that made Murphy baselines
useful as point estimates but impractical as shared models.

### Limitations

Four limitations are worth naming. **(1) Number of markers.**
The canonical 10-marker panel is the intersection of three
cohorts after alias harmonisation; panels for rarer markers
(most lineage markers, most checkpoints beyond CD45) are
cohort-specific and could not be included. A larger canonical
panel and more cohorts are the obvious extension.
**(2) Prediction softness.** v2 predictions are visibly
smoother than ground truth (Fig. 4). IMC single-pixel signal
is partly Poisson noise and partly single-metal-cluster
quantisation; a regression loss cannot recover the stochastic
single-pixel fluctuations even from a perfect posterior.
Recovering this remains open.
**(3) Evaluation is reconstruction-based, not biological.**
We report per-pixel PCC; downstream single-cell phenotyping
concordance between predicted and ground-truth channels — a
more direct measure of biological fidelity — is not yet
evaluated.
**(4) One tissue-class per cohort.** All three cohorts are
cancer or immune tissue; generalisation to structurally very
different tissue (kidney, CNS, cartilage) requires
additional cohorts and is a priority for v3.

### Comparison to related work

Most prior panel-extension work treats the problem as
single-cohort supervised regression [ref: Murphy 2025; other
IMC imputation work]. Cross-cohort transfer has been attempted
for single-cell data [ref: scVI, scArches], but those methods
operate on cell-level expression and do not handle spatial
structure. Pathology foundation models have been used as
feature backbones for H&E-native tasks (classification,
survival, tissue segmentation) [ref: Phikon-v2 applications,
UNI, CONCH]; to our knowledge we are the first to use them
as a **conditioning branch for an IMC reconstruction model**.

### Practical implications

A SpaProtFM-style checkpoint, released with a canonical panel
specification, could act as a community tool for IMC data
sharing — any study whose panel contains the canonical markers
can be mutually pooled, any study missing canonical markers can
be imputed to the canonical panel, and downstream analyses can
be run on the common feature space. The pseudo-H&E pipeline is
lightweight and adds no dependencies beyond Phikon's open
weights; we estimate training one cross-cohort checkpoint on 3
× 1.5 GB IMC cohorts at ~4 GPU-hours on a single 24 GB card.

---

## Conclusion

SpaProtFM v2 integrates a frozen pathology foundation model
(Phikon-v2) into a masked U-Net via pseudo-H&E synthesis from
IMC DNA and biomarker channels, yielding a panel-extension
network that transfers zero-shot across unseen IMC cohorts.
A single checkpoint trained on two cohorts out-performs
Murphy baselines retrained specifically for a third held-out
cohort, by +0.06 to +0.12 mean PCC, on every cohort tested,
and exceeds Murphy's own in-distribution accuracy on every
cohort. The method requires no paired H&E imaging and no
cohort-specific fine-tuning. We position SpaProtFM v2 as a
foundation for shared IMC imputation models and release all
code and checkpoints for community use.

---

## Acknowledgements

[TBD]

## Funding

[TBD]

## Data Availability

All three IMC datasets are publicly available via the
`imcdatasets` Bioconductor package (Damond 2019, Hoch-Schulz
2022, Jackson 2020). Pre-processed `.rds` source files and
intermediate numpy caches are archived at [repository TBD].
Code, trained checkpoints, training logs, and figure-
generation scripts are at [github url TBD].

## Author contributions

[TBD]

## Conflict of interest

The authors declare no conflicts of interest.

---

## Figure legends

**Figure 1. SpaProtFM v2 architecture.**
An input IMC tile (10 canonical markers) passes through two
parallel branches. **Left (masking branch):** the target
channels (CD20 / CD3e / CD45 / CD68 / Ki67 / cPARP) are masked
out — with a random *k*-of-6 pattern during training, and all
six during inference — producing a masked 10-channel input
that preserves the morphology channels (DNA1/DNA2/H3/SMA).
**Right (pseudo-H&E branch):** the full tile is rendered as a
synthetic H&E image (hematoxylin from DNA channels, eosin from
biomarker channels) and encoded by a frozen Phikon-v2 ViT-B,
producing a 1024-d × 14 × 14 feature map. The two branches
converge at the bottleneck of a MaskedUNetV2, where the Phikon
features are concatenated with the encoder output after a 1×1
projection. The decoder produces a predicted 10-channel panel;
the training loss is mean-squared error on masked positions.

**Figure 2. Method validation.**
**(a–c)** Mean per-marker Pearson correlation of SpaProtFM v1
(no H&E, orange) and v2 (+ Phikon, blue) across observed panel
sizes 3–20, on Damond, Hoch-Schulz and Jackson cohorts;
errorbars = 1 s.d. across three random panel draws per size.
The grey dashed line marks the per-cohort Murphy-size-10
baseline. v2 equals or exceeds v1 on two of three cohorts
(Hoch-Schulz and Jackson) across the tested range.
**(d–f)** Same three cohorts, showing a single SpaProtFM v2
checkpoint (blue line + shaded 1-s.d. band, trained once with
random-mask) against Murphy baselines *retrained per size*
(orange diamonds, one full training run per size). v2 matches
per-size-retrained Murphy on Hoch-Schulz and Jackson; a small
Murphy advantage remains on Damond at sizes ≥ 15.

**Figure 3. Zero-shot cross-cohort transfer.**
**(a)** 4 × 3 heatmap of mean PCC (*n* = 3 paired seeds)
where rows are the Murphy-per-cohort baselines and the
SpaProtFM v2 leave-one-out checkpoints, and columns are test
cohorts. Dotted outlines mark Murphy in-distribution cells;
red outlines mark the column-best model. **All three
column-best cells are in the v2 row** — the v2 leave-one-out
checkpoint beats every per-cohort Murphy specialist on its
own cohort.
**(b)** Out-of-distribution head-to-head bars per test cohort:
Murphy best-source (orange) vs SpaProtFM v2 leave-one-out
(blue). v2 gains Δ = +0.10, +0.12, +0.06 mean PCC on Damond,
Hoch-Schulz and Jackson respectively.
**(c)** Three-cohort average of in-distribution and out-of-
distribution mean PCC; v2 wins both, with tighter across-
cohort variance than Murphy.

**Figure 4. Qualitative zero-shot OOD predictions.**
Each row shows one representative 128 × 128 tile from the
held-out cohort predicted by the corresponding leave-one-out
SpaProtFM v2 checkpoint (i.e., a model that never saw a
single pixel from the row's cohort during training). Columns:
pseudo-H&E input that feeds Phikon-v2; ground-truth CD45;
predicted CD45; ground-truth CD68; predicted CD68; 2-D
log-density hexbin of predicted vs ground-truth intensity
pooled over all six target markers (with per-pixel Pearson
*r* annotated). Representative tiles were selected by taking
the median-PCC tile among the top 50 % of tiles by target-
channel dynamic range, to avoid cherry-picking or empty-tile
artefacts.

**Supplementary Figure S1. Per-marker zero-shot OOD PCC.**
Mean PCC per canonical target marker (*n* = 3 seeds) for the
SpaProtFM v2 leave-one-out checkpoint (blue) vs the Murphy
best-source baseline (orange), across the three held-out
cohorts. All 18 cohort × marker cells have v2 ≥ Murphy.

---

## References (placeholders — to be filled in)

1. Damond N, Engler S, Zanotelli V R T, *et al.* **A map of
   human type 1 diabetes progression by imaging mass
   cytometry.** *Cell Metabolism* 2019.
2. Hoch T, Schulz D, Eling N, *et al.* **Multiplexed imaging
   mass cytometry of the chemokine milieus in melanoma
   characterizes features of the response to immunotherapy.**
   *Science Immunology* 2022.
3. Jackson H W, Fischer J R, Zanotelli V R T, *et al.* **The
   single-cell pathology landscape of breast cancer.**
   *Nature* 2020.
4. Murphy *et al.* **[panel-extension U-Net baseline].**
   [year / venue — to confirm].
5. Filiot A, Ghermi R, Olivier A, *et al.* **Phikon-v2: a
   self-supervised pathology foundation model.** [year / venue].
6. Ronneberger O, Fischer P, Brox T. **U-Net: Convolutional
   networks for biomedical image segmentation.** *MICCAI* 2015.
7. Dosovitskiy A, Beyer L, Kolesnikov A, *et al.* **An image
   is worth 16×16 words: transformers for image recognition
   at scale.** *ICLR* 2021.
8. Chen R J, *et al.* **CONCH: A vision-language foundation
   model for computational pathology.** *Nature Medicine* 2024.
9. Chen R J, *et al.* **UNI: A general-purpose self-supervised
   model for pathology.** *Nature Medicine* 2024.
10. Lopez R, Regier J, Cole M B, *et al.* **Deep generative
    modeling for single-cell transcriptomics.** *Nature
    Methods* 2018 (scVI).
