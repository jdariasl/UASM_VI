# UASM-VI — Uncertainty-Aware Saliency Maps via Variational Inference

Implementation of **Bayesian LRP-ε** and **MC Bayesian Integrated Gradients**
for VI-BNNs, generating saliency maps with per-pixel uncertainty quantification.

**Reference:** Arias-Londoño, J.D., *Uncertainty-Aware Saliency Maps via Variational Inference* (2025).

---

## Repository layout

```
UASM_VI/
├── models/
│   └── models.py          # BEfficientNet-B6, BVGG-11/16/19, BResNet-18/34/50/101, select_model()
├── explainability/
│   ├── moment_forward.py  # Forward moment propagation (μ, v, e) through BNN
│   ├── bayesian_lrp.py    # Bayesian LRP-ε  → E[R], Var[R], SNR
│   └── bayesian_ig.py     # MC Bayesian Integrated Gradients → E[IG], Var[IG], SNR
├── data/
│   ├── covidx.py          # COVIDx + COVID-QU-Ex loader (3-class chest X-ray)
│   └── cub200.py          # CUB-200-2011 loader + pointing-game evaluator
├── utils/
│   ├── bnn_utils.py       # Unified API for bayesian_torch layer attributes
│   └── visualization.py   # Saliency / SNR / mask figure generation
├── configs/
│   ├── chestxray_efficientnet.yaml
│   ├── cub200_vgg11.yaml
│   ├── cub200_vgg16.yaml
│   └── cub200_resnet50.yaml
├── tune_eps_std.py        # sweep lrp_eps_std on a checkpoint (conservation check)
├── train.py               # ELBO training loop
├── make_report.py         # figures + metrics for the paper (all checkpoints, 6 examples)
├── explain.py             # Single-image explanation entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. Train a Bayesian model

```bash
python train.py --config configs/chestxray_efficientnet.yaml
```

### 2. Generate uncertainty-aware saliency maps

```bash
# Bayesian LRP-ε
python explain.py \
    --config  configs/chestxray_efficientnet.yaml \
    --image   path/to/xray.png \
    --method  lrp \
    --target  2          # 0=normal, 1=pneumonia, 2=COVID-19

# MC Bayesian Integrated Gradients
python explain.py \
    --config  configs/cub200_vgg11.yaml \
    --image   path/to/bird.jpg \
    --method  ig \
    --target  42
```

### 3. Python API

```python
import torch
from models.models import select_model
from explainability.bayesian_lrp import BayesianLRP
from explainability.bayesian_ig  import BayesianIG

model = select_model("vgg11", bayesian=True, n_classes=200)
model.load_state_dict(torch.load("checkpoints/best_model.pt")["state_dict"])
model.eval()

# -- Bayesian LRP --
blrp = BayesianLRP(model, eps=1e-6)
mu_R, v_R, snr = blrp.explain(x, target_class=42)
# mu_R, v_R, snr : (1, C, H, W)

# -- Uncertainty mask (pixels with low variance) --
mask = BayesianLRP.uncertainty_mask(v_R, percentile=75)

# -- MC Bayesian IG --
big = BayesianIG(model, n_samples=20, m_steps=50)
mu_ig, v_ig, snr_ig = big.explain(x, target_class=42)
```

---

## Report figures and metrics

```bash
python make_report.py --config configs/cub200_vgg11.yaml --ckpt_root checkpoints --outdir results/report
# optional: aggregate metrics over 100 more test images (no figures)
python make_report.py --config configs/cub200_vgg11.yaml --n_eval 100
```

Compares every `checkpoints/cub200_vgg11*` run (deterministic, Reparameterization, Flipout) on six test
images (4 correct + 2 misclassified by the reference model). Writes one grid figure per image, the raw
maps (`.npz`), `metrics_examples.csv`, `summary_*.csv` and `summary.md`. Metrics: predictive entropy and
mutual information, pointing game (with chance rate), LRP conservation, LRP/IG agreement, and a
Monte-Carlo validation of the closed-form LRP mean and variance.

`lrp_eps_std` (config) scales an uncertainty-dependent LRP stabiliser. Classical LRP-eps (`0`) can diverge
on BNNs; tune this value on a trained model by watching the conservation column (`lrp_sum_E`).

## Outputs

Each call to `explain.py` writes three images to `--outdir`:

| File | Content |
|------|---------|
| `*_saliency.png` | Input image · E[R] heatmap · Var[R] heatmap |
| `*_snr.png`      | Input image · signed SNR = E[R] / √Var[R] |
| `*_mask.png`     | Input image · reliable-attribution overlay |

---

## Backbone choices

| Model | Architecture | Notes |
|-------|-------------|-------|
| `vgg11/16/19` | VGG-BN | No skip connections: cleanest case for Bayesian LRP |
| `resnet18/34/50/101` | ResNet | Residual blocks handled by a block-aware walker (see below) |
| `efficientnet_b6` | EfficientNet-B6 | MBConv/SE blocks **not yet supported** by the moment walker (future work) |

All support `estimator: Reparameterization` (default) and `estimator: Flipout`.

### Residual connections

At each `y = ReLU(F(x) + S(x))` the two branches are treated as **independent**: forward
variances add, and relevance is split with the delta-method moments of the add node, then
merged (means and variances added) at the block input. BatchNorm is folded into the preceding
conv in the LRP backward pass (`lrp_fuse_bn: true`; set `false` to reproduce the unfolded behaviour).

**Caveat.** Dropping Cov(F,S) *underestimates* the variance in residual nets. On random-weight
tests with small posterior noise the closed-form logit variance was ~1.0x the first-order
weight-space reference for VGG-11 but ~0.3x for ResNet-18/50. `make_report.py` reports
`logit_var_ratio_fo` (first-order reference; only trustworthy for small noise) and
`mc_*` columns (Monte-Carlo, use these at large noise, e.g. MOPED delta 0.5) so you can
quantify the gap on your trained models. Relevance seed is sign(logit); SNR is scale-invariant.

### Variance diagnostic

```bash
python diag_var.py --config configs/cub200_resnet50.yaml \
    --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt --n_images 3 --n_mc 60 --scales 1 0.3 0.1
```
Compares the analytic activation variance with Monte-Carlo (explicit weight draws) layer by layer, for the
posterior std scaled by each factor. Shows where the closed form loses variance (`first layer with ratio < 0.5`)
and whether the cause is large-noise non-linearity (ratio -> 1 as the scale shrinks) or ignored covariances
(ratio stays low; `adjCorr` > 0 means spatially correlated activations through shared conv weights).

`--calibrate_pool_corr` finds, per image, the correlation `rho` inside the global-average-pool window that makes the
analytic logit variance match Monte-Carlo. A value can be set in the config as `lrp_pool_corr` (default `0.0` = independent
positions = previous behaviour). It only corrects the pool; if the tool prints `inf`, even full correlation is not enough
(other covariances matter). Changing it alters the variances that feed the LRP stabiliser, so re-run `tune_eps_std.py` afterwards.

### Variance-filter evaluation

```bash
python filter_eval.py --config configs/cub200_resnet50.yaml --n_images 50 --n_mc 30 --outdir results/filter_eval
```
Tests whether keeping only low-variance pixels helps. For each Bayesian checkpoint it filters the closed-form
`E[R]` map by `variance`, `snr`, `var_adj` (variance given magnitude), `magnitude` and `random` at several coverages and
measures (1) stability over Monte-Carlo weight draws (`jaccard`, `p_pos`, confound-controlled `auc_stable`),
(2) localisation on the CUB parts (`hit`, `lift`), (3) the risk-coverage curves (`filter_curves.png`, with paired
differences to the random and magnitude filters in `filter_eval_summary.md`), and (4) faithfulness by deletion/insertion
(`del_drop`, `ins_gain`). The variance filter is only useful if it beats `random` **and** `magnitude` at equal coverage.

### Tuning `lrp_eps_std`

```bash
python tune_eps_std.py --config configs/cub200_resnet50.yaml --checkpoint checkpoints/cub200_resnet50/best_model.pt
```
Pick the smallest value for which the conservation sum (`lrp_sum_E`) stays near 1 without exploding.

---

## Datasets

| Task | Dataset | Annotations |
|------|---------|-------------|
| Chest X-ray classification | COVIDx | 3 classes |
| Saliency localisation (chest) | COVID-QU-Ex | Pixel-level infection masks |
| Fine-grained classification | CUB-200-2011 | 200 bird species |
| Saliency localisation (birds) | CUB-200-2011 part annotations | Part coordinates → pointing game |

---

## Citation

```bibtex
@article{arias2025uasm,
  title   = {Uncertainty-Aware Saliency Maps via Variational Inference},
  author  = {Arias-Londo{\~n}o, Juli{\'a}n D.},
  year    = {2025},
  note    = {Universidad de Antioquia, Intelligent Information Systems Lab}
}
```

## Where the closed form loses variance: covariance oracle (`diag_cov.py`)

Feeds every selected conv / linear / average-pool layer with exact input statistics from Monte-Carlo
weight draws and measures how much of its output variance each input-covariance structure explains
(`full` = exact check, `diag` = what `moment_forward.py` assumes, `chan` = keep spatial correlation inside a
channel, `space` = keep cross-channel correlation at a position).

```bash
python diag_cov.py --config configs/cub200_resnet50.yaml \
    --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \
    --n_images 2 --n_mc 60 --every 3 --scales 1.0
```

Output: `results/diag_cov/cov_oracle_scale1.csv` and a table on screen. `full` must be ~1; `diag << 1` at a layer
means the diagonal assumption fails at that layer; `gap_frac_chan` / `gap_frac_space` say which correlation
closes the gap. Use `--every 1` (all layers) or `--layers REGEX` to zoom in.

`--ranks 1 5 20 50` (default) adds a low-rank oracle: `lr<k>` = diagonal + the top-k principal components of the
input covariance (columns `ev<k>` give the share of input variance in those components). If a small k brings
`lr<k>` close to 1 the missing covariance is low-rank. The components are fitted on the same draws, so use
`--n_mc` >= 4k (k is capped at n_mc-2):

```bash
python diag_cov.py --config configs/cub200_resnet50.yaml \
    --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \
    --layers "avgpool|fc|layer4" --every 1 --n_images 2 --n_mc 200 --ranks 1 5 20 50
```

For conv and pool layers it also reports `sh`: synthetic draws with the exact marginals, no correlation between
positions and ONE channel-correlation matrix shared by all positions (`sh_lr<k>`: that matrix reduced to k
directions plus a diagonal). `sh` close to `space` means the cross-channel covariance is stationary over
positions (one C x C matrix per layer is enough); `sh_lr<k>` close to `sh` means that matrix is low-rank.

## Calibration of the sign confidence (`filter_eval.py`)

`filter_eval.py` now also reports, for p = Phi(|E[R]|/sqrt(Var[R])) against the fraction of weight draws that
keep the sign of E[R]: reliability curves, ECE, Brier score, the best single scale factor `c` for sigma and a
Gaussian MC oracle (`calibration.csv`, `calibration.png`, section in `filter_eval_summary.md`). To run only this
part (no filter variants, no faithfulness), which is much faster:

```bash
python filter_eval.py --config configs/cub200_resnet50.yaml --n_images 50 --n_mc 30 \
    --calibration_only --outdir results/calibration
```
