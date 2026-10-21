"""
eval_fixed_den.py
Does fixing the output denominator (bayesian_lrp_fixed.py) make the closed-form relevance
variance comparable to Monte-Carlo?

Two definitions of the relevance are compared, each against its own Monte-Carlo reference
computed on the SAME explicit weight draws w ~ q (deterministic LRP-eps on each draw):

  sign   initial relevance sign(f_t)       (BayesianLRP; R_j = A_j / z at the last layer)
  fixed  initial relevance f_t / |E[f_t]|  (BayesianLRPFixedDen; R_j = A_j / d, d constant)

Reported per image and stabiliser scale (ratios to Monte-Carlo, 1 = exact):
    var_sum      total relevance variance
    var_med      median over pixels of analytic / MC
    spearman     rank correlation of the pixel-wise variance maps
    pearson_E    correlation of the mean maps;   E_sum: ratio of the summed mean maps
and, per image, the spread of the explained logit over the draws: logit_cv = std/|mean|
and the fraction of draws in which the logit changes sign.  Because the stabiliser of the
analytic runs has no counterpart in the reference (eps_std = 0 there), eps_std is scanned
with --eps_std (several values allowed); the reference is computed once.

    python eval_fixed_den.py --config configs/cub200_resnet50.yaml \\
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \\
        --n_images 5 --n_ref 100 --eps_std 0.05 0.1 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data.cub200 import CUB200Dataset
from diag_var import load_model
from explainability.bayesian_lrp import BayesianLRP
from explainability.bayesian_lrp_fixed import BayesianLRPFixedDen, LRPFixedDenRef
from utils.bnn_utils import SampledTwin


def _rank(a):
    r = np.empty(a.size)
    r[np.argsort(a, kind="stable")] = np.arange(a.size)
    return r


def spearman(a, b):
    return float(np.corrcoef(_rank(a.ravel()), _rank(b.ravel()))[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@torch.no_grad()
def references(twin, sign_run, fixed_run, x, t, d, n_ref):
    ms, mf, logits = [], [], []
    for _ in range(n_ref):
        twin.resample()
        logits.append(float(twin.net(x)[0, t]))
        a, _, _ = sign_run.explain(x, t)
        b, _, _ = fixed_run.explain(x, t, d)
        ms.append(a.cpu())
        mf.append(b.cpu())
    out = []
    for lst in (ms, mf):
        st = torch.stack(lst)
        out.append((st.mean(0).numpy(), st.var(0, unbiased=True).numpy()))
    lg = np.array(logits)
    return out[0], out[1], float(lg.std(ddof=1) / max(abs(lg.mean()), 1e-12)), float((np.sign(lg) != np.sign(lg.mean())).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--n_ref", type=int, default=100, help="weight draws of the Monte-Carlo references")
    ap.add_argument("--eps_std", type=float, nargs="+", default=None,
                    help="stabiliser scales of the analytic runs (default: lrp_eps_std of the config)")
    ap.add_argument("--outdir", default="results/eval_fixed_den")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model, est = load_model(cfg, a.checkpoint, dev)
    print(f"Loaded {a.checkpoint}  ({est}, backbone {cfg['backbone']})")
    eps = float(cfg.get("lrp_eps", 1e-6))
    fuse = bool(cfg.get("lrp_fuse_bn", True))
    scales = a.eps_std if a.eps_std else [float(cfg.get("lrp_eps_std", 0.25))]
    pool_corr = float(cfg.get("lrp_pool_corr", 0.0))

    twin = SampledTwin(model)
    sign_run = BayesianLRP(twin.net, eps=eps, eps_std=0.0, fuse_bn=fuse)
    fixed_run = LRPFixedDenRef(twin.net, eps=eps, eps_std=0.0, fuse_bn=fuse)
    analytic = {s: (BayesianLRP(model, eps=eps, eps_std=s, fuse_bn=fuse, pool_corr=pool_corr),
                    BayesianLRPFixedDen(model, eps=eps, eps_std=s, fuse_bn=fuse, pool_corr=pool_corr))
                for s in scales}

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n_images)
    rows = []
    for n, i in enumerate(idx):
        x = ds[i][0].unsqueeze(0).to(dev)
        with torch.no_grad():
            lg = twin.use_mean()(x)
            t = int(lg.argmax())
            d = abs(float(lg[0, t]))
        (ms, vs), (mf, vf), cv, flip = references(twin, sign_run, fixed_run, x, t, d, a.n_ref)
        print(f"[{n + 1}/{a.n_images}] img {i} cls {t}  mean logit {float(lg[0, t]):.2f}  "
              f"logit CV {cv:.2f}  sign flips {flip:.2%}", flush=True)
        for s in scales:
            for name, runner, (mu_mc, v_mc) in (("sign", analytic[s][0], (ms, vs)),
                                                ("fixed", analytic[s][1], (mf, vf))):
                with torch.no_grad():
                    mu, v, _ = runner.explain(x, t)
                mu, v = mu.cpu().numpy(), v.cpu().numpy()
                m = v_mc > 1e-30
                r = dict(image=i, target=t, eps_std=s, definition=name, logit_cv=cv, sign_flip=flip,
                         var_sum=float(v.sum() / max(v_mc.sum(), 1e-30)),
                         var_med=float(np.median(v[m] / v_mc[m])),
                         spearman=spearman(v, v_mc), pearson_E=pearson(mu, mu_mc),
                         E_sum=float(mu.sum() / mu_mc.sum()) if abs(mu_mc.sum()) > 1e-30 else float("nan"))
                rows.append(r)
                print(f"    eps_std {s:<5g} {name:5s} var_sum {r['var_sum']:9.3g}  var_med {r['var_med']:9.3g}  "
                      f"rho_S {r['spearman']:.3f}  r_E {r['pearson_E']:.3f}  E_sum {r['E_sum']:.3g}", flush=True)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fixed_den_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)

    print("\n=== median over images (mean in brackets); ratios to Monte-Carlo, 1 = exact ===")
    keys = ["var_sum", "var_med", "spearman", "pearson_E", "E_sum"]
    print(f"{'eps_std':>8s} {'def':>6s}" + "".join(f"{k:>22s}" for k in keys))
    for s in scales:
        for name in ("sign", "fixed"):
            sel = [r for r in rows if r["eps_std"] == s and r["definition"] == name]
            print(f"{s:8g} {name:>6s}" + "".join(
                f"{np.nanmedian([r[k] for r in sel]):12.3g} ({np.nanmean([r[k] for r in sel]):7.3g})" for k in keys))
    print(f"\nCSV and run_args.json in {out}/")


if __name__ == "__main__":
    main()
