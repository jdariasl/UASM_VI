# UASM-VI — Uncertainty-Aware Saliency Maps via Variational Inference

Implementation of **Bayesian LRP-ε** and **MC Bayesian Integrated Gradients**
for VI-BNNs, generating saliency maps with per-pixel uncertainty quantification.

**Reference:** Arias-Londoño, J.D., *Uncertainty-Aware Saliency Maps via Variational Inference* (2025).

---

## Repository layout

```
UASM_VI/
├── models/
│   └── models.py          # BEfficientNet-B6, BVGG-11-BN, factory select_model()
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
│   └── cub200_vgg11.yaml
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
| `vgg11` | VGG-11-BN | **Preferred for Bayesian LRP**: no skip connections |
| `efficientnet_b6` | EfficientNet-B6 | Higher accuracy; MBConv skip connections handled via identity pass |

Both support `estimator: Reparameterization` (default) and `estimator: Flipout`.

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
