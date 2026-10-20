"""
diag_var.py
Where does the closed-form moment propagation lose variance?

For one trained Bayesian checkpoint, compares layer by layer
    analytic  Var[a]  (MomentForwardPass)          vs
    Monte-Carlo Var[a] over explicit weight draws w ~ q   (SampledTwin)
on a few test images, and repeats it with the posterior std rescaled by
several factors.  Two questions are answered:

 1. WHERE (depth): the first layer / block at which the ratio analytic/MC
    drops (ratio = 1 is exact).  A drop at residual-block outputs points to the
    independent-branch assumption; a steady decay along conv layers points to
    the diagonal-covariance assumption.
 2. WHY (noise level): with sigma scaled by 1, 0.3, 0.1 ...
      * ratio -> 1 as the noise shrinks   => large-noise non-linearity
        (Gaussian / delta-method error);
      * ratio stays << 1 at small noise   => ignored covariances
        (correlated activations from shared conv weights, residual branches).
    The first-order weight-space reference  sum_w (df/dw)^2 sigma_w^2  (exact at
    small noise, no independence assumption) is reported for the target logit as
    a third opinion.

Extra column `adj_corr`: mean correlation (over MC draws) between horizontally
adjacent positions of the same channel.  Values clearly > 0 mean activations
are correlated through shared weights, which the closed form assumes away.

Usage
-----
    python diag_var.py --config configs/cub200_resnet50.yaml \\
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \\
        --n_images 3 --n_mc 60 --scales 1 0.3 0.1

Outputs (in --outdir, default results/diag_var):
    layerwise_scale<s>.csv   one row per layer, averaged over images
    summary.csv              target-logit ratios per scale
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.models import select_model
from data.cub200 import CUB200Dataset
from explainability.moment_forward import MomentForwardPass
from utils.bnn_utils import SampledTwin, get_weight_moments, is_bayesian


# --------------------------------------------------------------------------- #
#  Posterior rescaling
# --------------------------------------------------------------------------- #

def _rho_params(model):
    return [(n, p) for n, p in model.named_parameters() if "rho" in n]


@torch.no_grad()
def scale_sigma(model, s: float, saved: dict):
    """sigma <- s * sigma  (rho stored through the inverse softplus)."""
    for n, p in _rho_params(model):
        if n not in saved:
            saved[n] = p.detach().clone()
        sigma = torch.nn.functional.softplus(saved[n]) * s
        p.copy_(torch.log(torch.expm1(sigma.clamp(min=1e-12))))


@torch.no_grad()
def restore_sigma(model, saved: dict):
    for n, p in _rho_params(model):
        if n in saved:
            p.copy_(saved[n])


# --------------------------------------------------------------------------- #
#  Analytic records -> flat list of comparable points
# --------------------------------------------------------------------------- #

def analytic_points(records, names, skip_relu=True):
    """
    Returns list of (name, mu, v) in data-flow order.  Residual records are
    expanded: body leaves, shortcut leaves, then the block output.
    ReLU records are skipped (torchvision re-uses one ReLU module several times
    per block, so a module hook cannot tell the calls apart).
    """
    out = []

    def walk(recs):
        for r in recs:
            layer = r["layer"]
            if layer is None:
                continue
            if r.get("kind") == "residual":
                walk(r["body"])
                walk(r["short"])
                out.append((names.get(id(layer), "block") + ".out", r["mu"], r["v"]))
            else:
                if skip_relu and isinstance(layer, nn.ReLU):
                    continue
                nm = names.get(id(layer))
                if nm is None:                  # implicit Flatten etc.
                    continue
                out.append((nm, r["mu"], r["v"]))

    walk(records)
    return out


# --------------------------------------------------------------------------- #
#  Monte-Carlo with online statistics
# --------------------------------------------------------------------------- #

class OnlineStats:
    """Welford mean/var of a tensor plus adjacent-position covariance."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None
        self.sxy = None        # sum of x * x_shift
        self.sx = None
        self.sy = None

    def update(self, a: torch.Tensor):
        a = a.detach().float()
        self.n += 1
        if self.mean is None:
            self.mean = torch.zeros_like(a)
            self.m2 = torch.zeros_like(a)
        d = a - self.mean
        self.mean += d / self.n
        self.m2 += d * (a - self.mean)
        if a.dim() == 4 and a.shape[-1] > 1:
            x, y = a[..., :-1], a[..., 1:]
            if self.sxy is None:
                self.sxy = torch.zeros_like(x)
                self.sx = torch.zeros_like(x)
                self.sy = torch.zeros_like(x)
            self.sxy += x * y
            self.sx += x
            self.sy += y

    def var(self):
        return self.m2 / max(self.n - 1, 1)

    def adj_corr(self):
        if self.sxy is None:
            return float("nan")
        n = self.n
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        v = self.var()
        vx, vy = v[..., :-1], v[..., 1:]
        ok = (vx > 1e-12) & (vy > 1e-12)
        if ok.sum() == 0:
            return float("nan")
        corr = cov[ok] / torch.sqrt(vx[ok] * vy[ok])
        return float(corr.clamp(-1, 1).mean())


def mc_stats(twin: SampledTwin, bnn, x, names_bnn, point_names, n_mc):
    """Run n_mc weight draws; hooks record the output of every named module."""
    mods = dict(twin.net.named_modules())
    stats = {nm: OnlineStats() for nm in point_names}
    handles = []
    seen_this_draw = set()

    def make_hook(nm):
        def hook(_m, _i, o):
            if nm in seen_this_draw:        # module called more than once: keep first
                return
            seen_this_draw.add(nm)
            stats[nm].update(o)
        return hook

    for nm in point_names:
        base = nm[:-4] if nm.endswith(".out") else nm
        handles.append(mods[base].register_forward_hook(make_hook(nm)))
    try:
        for _ in range(n_mc):
            twin.resample()
            seen_this_draw.clear()
            with torch.no_grad():
                twin.net(x)
    finally:
        for h in handles:
            h.remove()
    return stats


@torch.no_grad()
def _unused():  # pragma: no cover
    pass


def first_order_var(twin: SampledTwin, x, target):
    twin.use_mean()
    for p in twin.net.parameters():
        p.requires_grad_(True)
    twin.net.zero_grad(set_to_none=True)
    twin.net(x)[0, target].backward()
    var = 0.0
    for src, dst in twin.pairs:
        _, sw, _, sb = get_weight_moments(src)
        var += float((dst.weight.grad ** 2 * sw.detach() ** 2).sum())
        if dst.bias is not None and dst.bias.grad is not None and sb is not None:
            var += float((dst.bias.grad ** 2 * sb.detach() ** 2).sum())
    for p in twin.net.parameters():
        p.requires_grad_(False)
        p.grad = None
    return var


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def load_model(cfg, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    ccfg = ck.get("cfg", {}) if isinstance(ck, dict) else {}
    est = ccfg.get("estimator", cfg.get("estimator", "Reparameterization"))
    model = select_model(cfg["backbone"], bayesian=True, n_classes=cfg["n_classes"],
                         estimator=est)
    model.load_state_dict(state)
    return model.to(device).eval(), est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=3)
    ap.add_argument("--n_mc", type=int, default=60, help="weight draws per image and scale")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.3, 0.1])
    ap.add_argument("--pool_corr", type=float, default=None,
                    help="assumed correlation inside average-pool windows for the analytic pass "
                         "(default: lrp_pool_corr from the config, else 0)")
    ap.add_argument("--calibrate_pool_corr", action="store_true",
                    help="per image and scale, find the pool correlation rho in [0,1] that makes the "
                         "analytic target-logit variance equal the Monte-Carlo one; prints the median")
    ap.add_argument("--outdir", default="results/diag_var")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.pool_corr is None:
        a.pool_corr = float(cfg.get("lrp_pool_corr", 0.0))
    if cfg.get("dataset", "cub200") != "cub200":
        raise SystemExit("diag_var.py currently supports the cub200 dataset only")
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model, est = load_model(cfg, a.checkpoint, dev)
    print(f"Loaded {a.checkpoint}  ({est}, backbone {cfg['backbone']})")
    twin = SampledTwin(model)
    names_bnn = {id(m): n for n, m in model.named_modules()}

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n_images)
    xs = [ds[i][0].unsqueeze(0).to(dev) for i in idx]

    saved = {}
    summary = []
    for s in a.scales:
        scale_sigma(model, s, saved)
        per_layer = {}                    # name -> list of dict per image
        order = []
        logit_rows = []
        rhos = []
        for ii, x in enumerate(xs):
            mp = MomentForwardPass(model, pool_corr=a.pool_corr)
            with torch.no_grad():
                recs = mp.run(x)
            pts = analytic_points(recs, names_bnn)
            pnames = [p[0] for p in pts]
            stats = mc_stats(twin, model, x, names_bnn, pnames, a.n_mc)

            for nm, mu_a, v_a in pts:
                st = stats[nm]
                v_mc = st.var()
                mu_mc = st.mean
                tot_mc = float(v_mc.sum())
                ratio_sum = float(v_a.sum()) / max(tot_mc, 1e-30)
                m = v_mc > 1e-12
                ratio_med = float((v_a[m] / v_mc[m]).median()) if m.any() else float("nan")
                rel_mean = float((mu_a - mu_mc).norm() / mu_mc.norm().clamp(min=1e-12))
                noise = math.sqrt(tot_mc) / max(float(mu_mc.norm()), 1e-12)
                per_layer.setdefault(nm, []).append(
                    dict(ratio_sum=ratio_sum, ratio_med=ratio_med, rel_mean_err=rel_mean,
                         mc_noise=noise, adj_corr=st.adj_corr()))
                if nm not in order:
                    order.append(nm)

            # target logit
            last = pts[-1]
            target = int(last[1].argmax())
            lv_a = float(last[2][0, target])
            lv_mc = float(stats[last[0]].var()[0, target])
            fo = first_order_var(twin, x, target)
            logit_rows.append((lv_a, lv_mc, fo))
            if a.calibrate_pool_corr:
                def lvar(rho):
                    with torch.no_grad():
                        r = MomentForwardPass(model, pool_corr=rho).run(x)
                    return float(r[-1]["v"][0, target])
                lo, hi = 0.0, 1.0
                if lvar(hi) < lv_mc:
                    rho_star = float("inf")            # even full correlation is not enough
                elif lvar(lo) >= lv_mc:
                    rho_star = 0.0
                else:
                    for _ in range(14):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if lvar(mid) < lv_mc else (lo, mid)
                    rho_star = 0.5 * (lo + hi)
                rhos.append(rho_star)
                print(f"      calibrated pool_corr for this image: {rho_star:.3f}", flush=True)
            print(f"  scale {s:g}  image {ii + 1}/{len(xs)}: logit var analytic/MC = "
                  f"{lv_a / max(lv_mc, 1e-30):.3g}   first-order/MC = {fo / max(lv_mc, 1e-30):.3g}",
                  flush=True)

        # ---- aggregate over images ----
        rows = []
        for nm in order:
            L = per_layer[nm]
            rows.append(dict(layer=nm, **{k: float(np.nanmean([d[k] for d in L]))
                                          for k in L[0]}))
        with open(out / f"layerwise_scale{s:g}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"\n=== sigma scale {s:g}: analytic/MC variance ratio by layer "
              f"(mean of {len(xs)} images) ===")
        print(f"{'layer':34s} {'ratio':>8s} {'median':>8s} {'meanErr':>8s} {'noise':>7s} {'adjCorr':>8s}")
        first_drop = None
        step = max(1, len(rows) // 40)
        for k, r in enumerate(rows):
            if first_drop is None and r["ratio_sum"] < 0.5:
                first_drop = r["layer"]
            if k % step == 0 or k == len(rows) - 1:
                print(f"{r['layer'][:34]:34s} {r['ratio_sum']:8.3f} {r['ratio_med']:8.3f} "
                      f"{r['rel_mean_err']:8.3f} {r['mc_noise']:7.3f} {r['adj_corr']:8.3f}")
        print(f"first layer with ratio < 0.5: {first_drop}\n")

        la = np.mean([r[0] for r in logit_rows])
        lm = np.mean([r[1] for r in logit_rows])
        lf = np.mean([r[2] for r in logit_rows])
        summary.append(dict(scale=s, analytic_over_mc=la / max(lm, 1e-30),
                            first_order_over_mc=lf / max(lm, 1e-30),
                            first_drop=first_drop or "-",
                            pool_corr_calibrated=(float(np.median(rhos)) if rhos else float("nan"))))
    restore_sigma(model, saved)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print("=== target-logit variance, ratio to Monte-Carlo (1 = exact) ===")
    print(f"{'sigma scale':>12s} {'analytic':>10s} {'first-order':>12s} {'rho*':>7s}   first drop below 0.5")
    for r in summary:
        print(f"{r['scale']:12g} {r['analytic_over_mc']:10.4g} {r['first_order_over_mc']:12.4g} "
              f"{r['pool_corr_calibrated']:7.3f}   {r['first_drop']}")
    if a.calibrate_pool_corr:
        print("rho* = median calibrated pool correlation (inf = even rho=1 underestimates). Put a value in the "
              "config as `lrp_pool_corr` and re-tune lrp_eps_std.")
    print("\nReading: analytic -> 1 as scale shrinks => large-noise nonlinearity; stays << 1 => ignored "
          "covariances.  adjCorr clearly > 0 => correlated activations (shared conv weights).")
    print(f"CSV files in {out}/")


if __name__ == "__main__":
    main()
