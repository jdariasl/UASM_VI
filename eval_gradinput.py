"""
eval_gradinput.py
Closed-form Bayesian gradient x input (explainability/bayesian_gradinput.py) against its own
Monte-Carlo reference:  R = x * dF_t/dx / d  on explicit weight draws w ~ q (SampledTwin),
d = |mean-weight logit| for both.  Nothing to tune: the closed form has no stabiliser.

Reported per image (ratios analytic / Monte-Carlo, 1 = exact), then median (mean) over images:
    var_sum      total relevance variance
    var_med      median over pixels of the variance ratio
    spearman     rank correlation of the pixel-wise variance maps
    pearson_E    correlation of the mean maps;    E_sum: ratio of the summed mean maps
    logit_cv     std/|mean| of the explained logit over the draws
and the calibration of the sign confidence  p = Phi(|E[R]|/sqrt(Var[R]))  against the fraction
of draws that keep the sign of the closed-form E[R] (all pixels and the 20% most relevant):
    ECE, Brier of the closed form, of the closed form with the best single rescale c of sigma,
    of the Gaussian oracle fitted on the same draws (in-sample), and of the constant prediction.

    python eval_gradinput.py --config configs/cub200_resnet50.yaml \\
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \\
        --n_images 10 --n_ref 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data.cub200 import CUB200Dataset
from diag_var import load_model
from explainability.bayesian_gradinput import BayesianGradInput, GradInputRef
from utils.bnn_utils import SampledTwin


def _rank(a):
    r = np.empty(a.size)
    r[np.argsort(a, kind="stable")] = np.arange(a.size)
    return r


def spearman(a, b):
    return float(np.corrcoef(_rank(a.ravel()), _rank(b.ravel()))[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _Phi(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def ece_brier(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    n = len(p)
    ece = 0.0
    for k in range(nb):
        s = b == k
        if s.any():
            ece += s.sum() / n * abs(p[s].mean() - y[s].mean())
    return float(ece), float(np.mean((p - y) ** 2))


def calibration(E, V, Y, mmc, smc):
    """E, V: closed form;  Y: observed agreement;  mmc, smc: MC mean / std (sign-aligned)."""
    top = np.argsort(np.argsort(-np.abs(E))) < int(0.2 * E.size)
    out = {}
    for name, sel in (("all", np.ones_like(top)), ("top20", top)):
        absE = torch.from_numpy(np.abs(E[sel]).astype(np.float64))
        sig = torch.from_numpy(np.sqrt(np.maximum(V[sel], 1e-300)).astype(np.float64))
        y = Y[sel].astype(np.float64)
        grid = np.logspace(-1.5, 3.5, 61)
        br = [float(((_Phi(absE / (c * sig)).numpy() - y) ** 2).mean()) for c in grid]
        c = float(grid[int(np.argmin(br))])
        gauss = _Phi(torch.from_numpy(mmc[sel].astype(np.float64)) /
                     torch.from_numpy(np.maximum(smc[sel], 1e-300).astype(np.float64))).numpy()
        res = {}
        for nm, p in (("closed", _Phi(absE / sig).numpy()), ("closed*c", _Phi(absE / (c * sig)).numpy()),
                      ("mc_gauss", gauss)):
            res[nm] = ece_brier(p, y)
        res["const"] = (0.0, float(np.mean((y - y.mean()) ** 2)))
        res["c"], res["mean_agree"], res["mean_p"] = c, float(y.mean()), float(_Phi(absE / sig).numpy().mean())
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=10)
    ap.add_argument("--n_ref", type=int, default=100, help="weight draws of the Monte-Carlo reference")
    ap.add_argument("--outdir", default="results/eval_gradinput")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model, est = load_model(cfg, a.checkpoint, dev)
    print(f"Loaded {a.checkpoint}  ({est}, backbone {cfg['backbone']})")

    twin = SampledTwin(model)
    ref = GradInputRef(twin.net)
    bgi = BayesianGradInput(model, pool_corr=float(cfg.get("lrp_pool_corr", 0.0)))

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n_images)
    rows, cal_rows = [], []
    for n, i in enumerate(idx):
        x = ds[i][0].unsqueeze(0).to(dev)
        with torch.no_grad():
            lg = twin.use_mean()(x)
            t = int(lg.argmax())
            d = abs(float(lg[0, t]))
        draws, logits = [], []
        for _ in range(a.n_ref):
            twin.resample()
            with torch.no_grad():
                logits.append(float(twin.net(x)[0, t]))
            draws.append(ref.explain(x, t, d).cpu())
        MC = torch.stack(draws).numpy()
        mmc, vmc = MC.mean(0), MC.var(0, ddof=1)
        lgt = np.array(logits)
        cv = float(lgt.std(ddof=1) / max(abs(lgt.mean()), 1e-12))

        with torch.no_grad():
            mu, v, _ = bgi.explain(x, t, denom=d)
        mu, v = mu.cpu().numpy(), v.cpu().numpy()
        m = vmc > 1e-30
        r = dict(image=i, target=t, logit_cv=cv,
                 var_sum=float(v.sum() / max(vmc.sum(), 1e-30)),
                 var_med=float(np.median(v[m] / vmc[m])),
                 spearman=spearman(v, vmc), pearson_E=pearson(mu, mmc[0]),
                 E_sum=float(mu.sum() / mmc.sum()) if abs(mmc.sum()) > 1e-30 else float("nan"))
        rows.append(r)
        print(f"[{n + 1}/{a.n_images}] img {i} cls {t}  logit {float(lg[0, t]):.2f} CV {cv:.2f}  "
              f"var_sum {r['var_sum']:.3g}  var_med {r['var_med']:.3g}  rho_S {r['spearman']:.3f}  "
              f"r_E {r['pearson_E']:.3f}  E_sum {r['E_sum']:.3g}", flush=True)

        sE = np.sign(mu)
        agree = ((MC * sE[None]) > 0).mean(0).ravel().astype(np.float32)
        cal = calibration(mu.ravel(), v.ravel(), agree, (sE * mmc).ravel(),
                          np.sqrt(vmc.ravel()))
        for sname, res in cal.items():
            for pred in ("closed", "closed*c", "mc_gauss", "const"):
                cal_rows.append(dict(image=i, pixels=sname, predictor=pred, ece=res[pred][0], brier=res[pred][1],
                                     c=res["c"], mean_agree=res["mean_agree"], mean_p=res["mean_p"]))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("gradinput_eval.csv", rows), ("gradinput_calibration.csv", cal_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)

    print("\n=== median over images (mean in brackets); analytic / Monte-Carlo, 1 = exact ===")
    keys = ["var_sum", "var_med", "spearman", "pearson_E", "E_sum", "logit_cv"]
    print("".join(f"{k:>22s}" for k in keys))
    print("".join(f"{np.nanmedian([r[k] for r in rows]):12.3g} ({np.nanmean([r[k] for r in rows]):7.3g})"
                  for k in keys))

    print("\n=== sign-confidence calibration, pooled over images (Brier; lower is better) ===")
    print(f"{'pixels':>8s} {'predictor':>10s} {'ECE':>8s} {'Brier':>9s} {'c(median)':>10s} {'mean p':>8s} {'mean agree':>11s}")
    for sname in ("all", "top20"):
        for pred in ("closed", "closed*c", "mc_gauss", "const"):
            sel = [r for r in cal_rows if r["pixels"] == sname and r["predictor"] == pred]
            print(f"{sname:>8s} {pred:>10s} {np.mean([r['ece'] for r in sel]):8.3f} "
                  f"{np.mean([r['brier'] for r in sel]):9.4f} {np.median([r['c'] for r in sel]):10.3g} "
                  f"{np.mean([r['mean_p'] for r in sel]):8.3f} {np.mean([r['mean_agree'] for r in sel]):11.3f}")
    print(f"\nCSV and run_args.json in {out}/")


if __name__ == "__main__":
    main()
