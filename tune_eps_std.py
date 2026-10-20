"""
tune_eps_std.py
Pick the LRP stabiliser scale `lrp_eps_std` for a trained Bayesian checkpoint.

Closed-form Bayesian LRP divides by pre-activations that can be close to zero.
`eps_std` damps them in proportion to their own standard deviation:
  * too small  -> relevance explodes (huge |sum R|, NaN), especially in deep nets
  * too large  -> relevance leaks away (sum R -> 0)
The conservation  sum_i E[R_i] * sign(logit)  should be close to 1.  This script
prints it for a grid of values on a few test images; choose the smallest value
whose conservation is stable (close to 1, no NaN) on all of them.

    python tune_eps_std.py --config configs/cub200_resnet50.yaml \
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt
"""
import argparse

import numpy as np
import torch
import yaml

from models.models import select_model
from data.cub200 import CUB200Dataset
from explainability.bayesian_lrp import BayesianLRP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument("--values", type=float, nargs="+",
                    default=[0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    est = (ck.get("cfg", {}) or {}).get("estimator") or cfg.get("estimator", "Reparameterization")
    model = select_model(cfg["backbone"], True, cfg["n_classes"], estimator=est)
    model.load_state_dict(ck.get("state_dict", ck))
    model = model.to(a.device).eval()

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = np.random.RandomState(a.seed).choice(len(ds), a.n_images, replace=False)
    xs = [ds[int(i)][0].unsqueeze(0).to(a.device) for i in idx]
    with torch.no_grad():
        targets = [int(model(x).argmax(1)) for x in xs]

    print(f"{'eps_std':>8} | conservation per image (sum E[R] * sign(logit)) | median  | mean Var[R]")
    for c in a.values:
        lrp = BayesianLRP(model, eps=float(cfg.get("lrp_eps", 1e-6)), eps_std=c,
                          fuse_bn=bool(cfg.get("lrp_fuse_bn", True)),
                          pool_corr=float(cfg.get("lrp_pool_corr", 0.0)))
        cons, var = [], []
        for x, t in zip(xs, targets):
            mu, v, _ = lrp.explain(x, t)
            cons.append(float(mu.sum()) * lrp.last_sign)
            var.append(float(v.mean()))
        cons = np.array(cons)
        txt = " ".join(f"{q:8.2f}" if np.isfinite(q) and abs(q) < 1e4 else "   BLOWUP" for q in cons)
        print(f"{c:8.3f} | {txt} | {np.nanmedian(cons):7.3f} | {np.nanmean(var):.2e}")


if __name__ == "__main__":
    main()
