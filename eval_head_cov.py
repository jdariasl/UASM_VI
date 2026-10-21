"""
eval_head_cov.py
Does the covariance-aware head (BayesianLRPCov) close the variance gap of BayesianLRP?

For a few test images, compares against Monte-Carlo on explicit weight draws w ~ q
(deterministic LRP-eps on each draw, as in make_report.py):

    logit var     analytic / MC   variance of the explained logit
    Var[R] sum    analytic / MC   total relevance variance
    Var[R] med    median over pixels of analytic / MC
    Spearman      rank correlation of the pixel-wise variance maps
    Pearson E[R]  agreement of the mean maps

The head correlations are fitted on --n_fit draws that are NOT used for the
reference (--n_ref draws), so the numbers are not in-sample.

    python eval_head_cov.py --config configs/cub200_resnet50.yaml \\
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt --n_images 3
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data.cub200 import CUB200Dataset
from diag_var import load_model
from explainability.bayesian_lrp import BayesianLRP
from explainability.bayesian_lrp_cov import BayesianLRPCov
from utils.bnn_utils import SampledTwin


def _rank(a: np.ndarray) -> np.ndarray:
    r = np.empty(a.size)
    r[np.argsort(a, kind="stable")] = np.arange(a.size)
    return r


def spearman(a, b) -> float:
    return float(np.corrcoef(_rank(a.ravel()), _rank(b.ravel()))[0, 1])


def pearson(a, b) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@torch.no_grad()
def mc_reference(twin, runner, x, t, n_ref):
    maps, logits = [], []
    for _ in range(n_ref):
        twin.resample()
        logits.append(float(twin.net(x)[0, t]))
        mu, _, _ = runner.explain(x, t)
        maps.append(mu.cpu())
    st = torch.stack(maps)
    return st.mean(0).numpy(), st.var(0, unbiased=True).numpy(), float(np.var(logits, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=3)
    ap.add_argument("--n_fit", type=int, default=60, help="draws used to fit the head correlations")
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--n_ref", type=int, default=100, help="independent draws of the MC reference")
    ap.add_argument("--eps_std", type=float, default=None,
                    help="stabiliser scale of the analytic runs (default: lrp_eps_std of the config); "
                         "the MC reference never uses one, so 0 compares like with like")
    ap.add_argument("--outdir", default="results/eval_head_cov")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    model, est = load_model(cfg, a.checkpoint, dev)
    print(f"Loaded {a.checkpoint}  ({est}, backbone {cfg['backbone']})")
    eps = float(cfg.get("lrp_eps", 1e-6))
    eps_std = a.eps_std if a.eps_std is not None else float(cfg.get("lrp_eps_std", 0.25))
    fuse = bool(cfg.get("lrp_fuse_bn", True))

    twin = SampledTwin(model)
    ref_runner = BayesianLRP(twin.net, eps=eps, eps_std=0.0, fuse_bn=fuse)
    base = BayesianLRP(model, eps=eps, eps_std=eps_std, fuse_bn=fuse,
                       pool_corr=float(cfg.get("lrp_pool_corr", 0.0)))
    cov = BayesianLRPCov(model, eps=eps, eps_std=eps_std, fuse_bn=fuse,
                         n_mc=a.n_fit, rank=a.rank)

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n_images)

    rows = []
    for n, i in enumerate(idx):
        x = ds[i][0].unsqueeze(0).to(dev)
        with torch.no_grad():
            t = int(twin.use_mean()(x).argmax())
        mu_mc, v_mc, lv_mc = mc_reference(twin, ref_runner, x, t, a.n_ref)
        for name, runner in (("diag", base), ("head_cov", cov)):
            with torch.no_grad():
                mu, v, _ = runner.explain(x, t)
            lv = float(runner.last_stats[-1]["v"][0, t])
            mu, v = mu.cpu().numpy(), v.cpu().numpy()
            m = v_mc > 1e-12
            rows.append(dict(
                image=i, target=t, method=name,
                logit_var_ratio=lv / max(lv_mc, 1e-30),
                var_sum_ratio=float(v.sum() / max(v_mc.sum(), 1e-30)),
                var_median_ratio=float(np.median(v[m] / v_mc[m])),
                spearman_var=spearman(v, v_mc),
                pearson_E=pearson(mu, mu_mc),
            ))
            r = rows[-1]
            print(f"[{n + 1}/{a.n_images}] img {i:5d} cls {t:3d} {name:9s} "
                  f"logitVar {r['logit_var_ratio']:.3f}  VarSum {r['var_sum_ratio']:.3f}  "
                  f"VarMed {r['var_median_ratio']:.3f}  rho_S {r['spearman_var']:.3f}  "
                  f"r_E {r['pearson_E']:.3f}", flush=True)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "head_cov_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== mean over images (ratios to MC, 1 = exact) ===")
    keys = ["logit_var_ratio", "var_sum_ratio", "var_median_ratio", "spearman_var", "pearson_E"]
    print(f"{'method':10s}" + "".join(f"{k:>18s}" for k in keys))
    for name in ("diag", "head_cov"):
        sel = [r for r in rows if r["method"] == name]
        print(f"{name:10s}" + "".join(f"{np.mean([r[k] for r in sel]):18.3f}" for k in keys))


if __name__ == "__main__":
    main()
