"""
explain.py
Entry point for generating uncertainty-aware saliency maps from a trained
VI-BNN using Bayesian LRP-ε and/or MC Bayesian Integrated Gradients.

Usage
-----
    # Bayesian LRP on a single image
    python explain.py \\
        --config  configs/chestxray_efficientnet.yaml \\
        --image   path/to/image.png \\
        --method  lrp \\
        --target  2           # COVID-19 class

    # MC Bayesian IG, save outputs
    python explain.py \\
        --config  configs/cub200_vgg11.yaml \\
        --image   path/to/bird.jpg \\
        --method  ig \\
        --target  42 \\
        --outdir  results/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.models import select_model
from explainability.bayesian_lrp import BayesianLRP
from explainability.bayesian_ig   import BayesianIG
from utils.visualization import (
    save_saliency_figure,
    save_snr_figure,
    overlay_uncertainty_mask,
)


# --------------------------------------------------------------------------- #
#  Image loading
# --------------------------------------------------------------------------- #

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> torch.Tensor:
    """Returns (1, 3, 224, 224) float tensor."""
    img = Image.open(path).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0)


# --------------------------------------------------------------------------- #
#  Explain a single image
# --------------------------------------------------------------------------- #

def explain_image(
    cfg:    dict,
    image:  str,
    method: str,
    target: int,
    outdir: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  method: {method}  |  target class: {target}")

    # ---- Load model ----
    model = select_model(
        backbone    = cfg["backbone"],
        bayesian    = True,
        n_classes   = cfg["n_classes"],
        estimator   = cfg.get("estimator", "Reparameterization"),
        moped_delta = cfg.get("moped_delta", 0.5),
    )
    ckpt_path = cfg.get("checkpoint", "checkpoints/best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model = model.to(device).eval()

    # ---- Load image ----
    x = load_image(image).to(device)
    print(f"Image: {image}  shape: {tuple(x.shape)}")

    # ---- Compute saliency ----
    if method == "lrp":
        explainer = BayesianLRP(model, eps=cfg.get("lrp_eps", 1e-6))
        mu_R, v_R, snr = explainer.explain(x, target)
        mask = BayesianLRP.uncertainty_mask(v_R, percentile=75.0)
    elif method == "ig":
        explainer = BayesianIG(
            model,
            n_samples = cfg.get("n_samples", 20),
            m_steps   = cfg.get("m_steps",   50),
            baseline  = cfg.get("baseline",  "black"),
        )
        mu_R, v_R, snr = explainer.explain(x, target)
        mask = BayesianIG.uncertainty_mask(v_R, percentile=75.0)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'lrp' or 'ig'.")

    # ---- Save outputs ----
    Path(outdir).mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem + f"_{method}_cls{target}"

    save_saliency_figure(
        x=x, mu_R=mu_R, v_R=v_R,
        path=os.path.join(outdir, stem + "_saliency.png"),
        title=f"{method.upper()} – class {target}",
    )
    save_snr_figure(
        x=x, snr=snr,
        path=os.path.join(outdir, stem + "_snr.png"),
        title=f"SNR map – {method.upper()} – class {target}",
    )
    overlay_uncertainty_mask(
        x=x, mask=mask,
        path=os.path.join(outdir, stem + "_mask.png"),
        title=f"Reliable attribution mask (top-25% var. excluded)",
    )

    print(f"Saved outputs to {outdir}/")

    # Quick stats
    print(f"  E[R]  min={mu_R.min():.4f}  max={mu_R.max():.4f}")
    print(f"  Var[R]max={v_R.max():.6f}")
    print(f"  SNR   min={snr.min():.2f}   max={snr.max():.2f}")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty-Aware Saliency Maps")
    parser.add_argument("--config",  required=True,          help="YAML config")
    parser.add_argument("--image",   required=True,          help="Input image path")
    parser.add_argument("--method",  default="lrp",
                        choices=["lrp", "ig"],               help="Explainability method")
    parser.add_argument("--target",  type=int, required=True, help="Target class index")
    parser.add_argument("--outdir",  default="results",       help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    explain_image(cfg, args.image, args.method, args.target, args.outdir)
