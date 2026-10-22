"""
compare_methods.py
Side-by-side evaluation of the two closed-form uncertainty-aware attributions on the SAME images
and the SAME weight draws:

  LRP   Bayesian LRP-eps         (explainability/bayesian_lrp.py,  stabiliser eps_std)
        reference: deterministic LRP-eps (eps_std = 0) on each weight draw
  GxI   Bayesian gradient x input (explainability/bayesian_gradinput.py, no stabiliser)
        reference: autograd x * dF_t/dx / d on each weight draw   (d = |mean-weight logit|)

For every image and method:  var_sum, var_med, spearman, pearson_E (analytic / Monte-Carlo),
the calibration of the sign confidence (ECE, Brier: closed form, closed form with the best single
rescale c of sigma, Gaussian oracle fitted on the draws, constant) for all pixels and the top 20%,
and the wall-clock cost of one closed-form explanation against one Monte-Carlo draw.  Across
methods: rank correlation of the two mean maps and of the two variance maps.  The LRP closed form
is also scanned over --eps_scan (total-variance ratio and variance Spearman) to show its
dependence on the stabiliser; GxI has no such parameter.

Outputs in --outdir
    compare_metrics.csv          per image and method
    compare_calibration.csv      per image, method, pixel set, predictor
    compare_reliability.csv      pooled reliability curves
    compare_eps_scan.csv         LRP closed form vs eps_std
    compare_summary.md           tables for the paper
    cmp_examples.png             mean / closed-form std / Monte-Carlo std maps, both methods
    cmp_variance_fidelity.png    pixel variance, closed form vs Monte-Carlo (log-log)
    cmp_calibration.png          reliability diagrams
    cmp_eps_sensitivity.png      total-variance ratio vs eps_std

    python compare_methods.py --config configs/cub200_resnet50.yaml \\
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \\
        --n_images 10 --n_ref 100 --n_fig 3 --outdir results/compare_resnet50_reparam
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from explainability.bayesian_lrp import BayesianLRP
from explainability.bayesian_gradinput import BayesianGradInput, GradInputRef
from utils.bnn_utils import SampledTwin

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#777777"
METHODS = ("LRP", "GxI")


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #

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
    ece = 0.0
    for k in range(nb):
        s = b == k
        if s.any():
            ece += s.sum() / len(p) * abs(p[s].mean() - y[s].mean())
    return float(ece), float(np.mean((p - y) ** 2))


def calib_one(E, V, Y, mmc, smc):
    """Calibration of p = Phi(|E|/sigma) for all pixels and the 20% most relevant."""
    top = np.argsort(np.argsort(-np.abs(E))) < int(0.2 * E.size)
    res = {}
    for name, sel in (("all", np.ones_like(top)), ("top20", top)):
        absE = torch.from_numpy(np.abs(E[sel]).astype(np.float64))
        sig = torch.from_numpy(np.sqrt(np.maximum(V[sel], 1e-300)).astype(np.float64))
        y = Y[sel].astype(np.float64)
        grid = np.logspace(-1.5, 3.5, 61)
        br = [float(((_Phi(absE / (c * sig)).numpy() - y) ** 2).mean()) for c in grid]
        c = float(grid[int(np.argmin(br))])
        gauss = _Phi(torch.from_numpy(mmc[sel].astype(np.float64)) /
                     torch.from_numpy(np.maximum(smc[sel], 1e-300).astype(np.float64))).numpy()
        preds = {"closed": _Phi(absE / sig).numpy(), "closed*c": _Phi(absE / (c * sig)).numpy(),
                 "mc_gauss": gauss}
        out = {k: ece_brier(p, y) for k, p in preds.items()}
        out["const"] = (0.0, float(np.mean((y - y.mean()) ** 2)))
        out["c"], out["mean_agree"], out["mean_p"] = c, float(y.mean()), float(preds["closed"].mean())
        res[name] = (out, preds, y)
    return res


def reliability(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return [(float(p[b == k].mean()), float(y[b == k].mean()), int((b == k).sum()))
            for k in range(nb) if (b == k).any()]


# --------------------------------------------------------------------------- #
#  core
# --------------------------------------------------------------------------- #

def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def evaluate(model, xs, n_ref, eps, eps_std, fuse, pool_corr, eps_scan, n_fig, dev, log=print):
    """xs: list of (image_id, tensor (1,C,H,W)).  Returns metric rows, pooled data and figure data."""
    twin = SampledTwin(model)
    lrp_ref = BayesianLRP(twin.net, eps=eps, eps_std=0.0, fuse_bn=fuse)
    gxi_ref = GradInputRef(twin.net)
    lrp = BayesianLRP(model, eps=eps, eps_std=eps_std, fuse_bn=fuse, pool_corr=pool_corr)
    gxi = BayesianGradInput(model, pool_corr=pool_corr)
    scan = {s: BayesianLRP(model, eps=eps, eps_std=s, fuse_bn=fuse, pool_corr=pool_corr) for s in eps_scan}

    rows, cal_rows, scan_rows, figs = [], [], [], []
    pooled = {m: {"E": [], "V": [], "Y": []} for m in METHODS}
    var_pts = {m: ([], []) for m in METHODS}
    rel = {m: {} for m in METHODS}

    for n, (i, x) in enumerate(xs):
        with torch.no_grad():
            lg = twin.use_mean()(x)
            t = int(lg.argmax())
            d = abs(float(lg[0, t]))
        dl, dg, logits = [], [], []
        t_mc = 0.0
        for _ in range(n_ref):
            twin.resample()
            _sync(dev); t0 = time.perf_counter()
            with torch.no_grad():
                logits.append(float(twin.net(x)[0, t]))
                dl.append(lrp_ref.explain(x, t)[0].cpu())
            dg.append(gxi_ref.explain(x, t, d).cpu())
            _sync(dev); t_mc += time.perf_counter() - t0
        t_mc /= n_ref * 2.0                                  # seconds per draw, both methods averaged
        lg_arr = np.array(logits)
        cv = float(lg_arr.std(ddof=1) / max(abs(lg_arr.mean()), 1e-12))

        res = {}
        for name, runner, draws in (("LRP", lrp, dl), ("GxI", gxi, dg)):
            _sync(dev); t0 = time.perf_counter()
            with torch.no_grad():
                mu, v, _ = (runner.explain(x, t) if name == "LRP" else runner.explain(x, t, denom=d))
            _sync(dev); t_cf = time.perf_counter() - t0
            MC = torch.stack(draws).numpy()
            mmc, vmc = MC.mean(0), MC.var(0, ddof=1)
            mu, v = mu.cpu().numpy(), v.cpu().numpy()
            m = vmc > 1e-30
            r = dict(image=i, target=t, method=name, logit=float(lg[0, t]), logit_cv=cv,
                     var_sum=float(v.sum() / max(vmc.sum(), 1e-30)),
                     var_med=float(np.median(v[m] / vmc[m])),
                     spearman=spearman(v, vmc), pearson_E=pearson(mu, mmc),
                     t_closed=t_cf, t_mc_draw=t_mc)
            rows.append(r)
            sE = np.sign(mu)
            agree = ((MC * sE[None]) > 0).mean(0).ravel().astype(np.float32)
            cal = calib_one(mu.ravel(), v.ravel(), agree, (sE * mmc).ravel(), np.sqrt(vmc.ravel()))
            for sname, (out, preds, y) in cal.items():
                for pred in ("closed", "closed*c", "mc_gauss", "const"):
                    cal_rows.append(dict(image=i, method=name, pixels=sname, predictor=pred,
                                         ece=out[pred][0], brier=out[pred][1], c=out["c"],
                                         mean_agree=out["mean_agree"], mean_p=out["mean_p"]))
                if sname == "all":
                    # pooled reliability uses all pixels of all images (subsampled to bound memory)
                    pass
            pooled[name]["E"].append(mu.ravel()[::7]); pooled[name]["V"].append(v.ravel()[::7])
            pooled[name]["Y"].append(agree[::7])
            sub = slice(None, None, 11)
            var_pts[name][0].append(v.sum(1).ravel()[sub]); var_pts[name][1].append(vmc.sum(1).ravel()[sub])
            res[name] = dict(mu=mu, v=v, mmc=mmc, vmc=vmc)
            log(f"[{n + 1}/{len(xs)}] img {i} {name:3s} logit {float(lg[0, t]):7.2f} CV {cv:.2f}  "
                f"var_sum {r['var_sum']:.3g}  var_med {r['var_med']:.3g}  rho_S {r['spearman']:.3f}  "
                f"r_E {r['pearson_E']:.3f}  t_closed {t_cf * 1e3:.0f} ms (MC draw {t_mc * 1e3:.0f} ms)")

        # cross-method agreement and LRP stabiliser scan (closed form only)
        a, b = res["LRP"], res["GxI"]
        for r in rows[-2:]:
            r["cross_spearman_E"] = spearman(a["mu"], b["mu"])
            r["cross_spearman_V"] = spearman(a["v"], b["v"])
        for s, runner in scan.items():
            with torch.no_grad():
                _, vs, _ = runner.explain(x, t)
            vs = vs.cpu().numpy()
            scan_rows.append(dict(image=i, eps_std=s, var_sum=float(vs.sum() / max(res["LRP"]["vmc"].sum(), 1e-30)),
                                  spearman=spearman(vs, res["LRP"]["vmc"])))
        if n < n_fig:
            figs.append(dict(image=i, x=x[0].cpu().numpy(), target=t, **{k: res[k] for k in METHODS}))
    return rows, cal_rows, scan_rows, pooled, var_pts, figs


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #

def _denorm(x):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.clip(x.transpose(1, 2, 0) * std + mean, 0, 1)


def fig_examples(figs, path):
    n = len(figs)
    cols = ["image", "LRP: mean", "LRP: std (closed)", "LRP: std (MC)", "GxI: mean", "GxI: std (closed)", "GxI: std (MC)"]
    fig, axs = plt.subplots(n, 7, figsize=(14.5, 2.15 * n), squeeze=False)
    for r, f in enumerate(figs):
        axs[r, 0].imshow(_denorm(f["x"]))
        for k, m in enumerate(METHODS):
            E = f[m]["mu"][0].sum(0)
            lim = np.percentile(np.abs(E), 99.5) + 1e-12
            axs[r, 1 + 3 * k].imshow(E, cmap="RdBu_r", vmin=-lim, vmax=lim)
            sc = np.sqrt(f[m]["v"][0].sum(0))
            sm = np.sqrt(f[m]["vmc"][0].sum(0))
            hi = np.percentile(np.concatenate([sc.ravel(), sm.ravel()]), 99.5) + 1e-12
            axs[r, 2 + 3 * k].imshow(sc, cmap="viridis", vmin=0, vmax=hi)
            axs[r, 3 + 3 * k].imshow(sm, cmap="viridis", vmin=0, vmax=hi)
        for c in range(7):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(cols[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_fidelity(var_pts, path):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, m, col in zip(axs, METHODS, (BLUE, ORANGE)):
        v = np.concatenate(var_pts[m][0]); mc = np.concatenate(var_pts[m][1])
        ok = (v > 0) & (mc > 0)
        v, mc = np.log10(v[ok]), np.log10(mc[ok])
        lo, hi = min(v.min(), mc.min()), max(v.max(), mc.max())
        ax.hexbin(mc, v, gridsize=60, bins="log", cmap="Greys", extent=(lo, hi, lo, hi), mincnt=1)
        ax.plot([lo, hi], [lo, hi], "--", color=col, lw=1.2, label="exact")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Monte-Carlo pixel variance (log10)")
        ax.set_ylabel("closed-form pixel variance (log10)")
        ax.set_title({"LRP": "Bayesian LRP-$\\varepsilon$", "GxI": "Bayesian gradient$\\times$input"}[m], fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_calibration(pooled, path, nb=10):
    sets = ("all pixels", "top-20% relevant")
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 7.4), squeeze=False)
    rel_rows = []
    for j, m in enumerate(METHODS):
        E = np.concatenate(pooled[m]["E"]); V = np.concatenate(pooled[m]["V"]); Y = np.concatenate(pooled[m]["Y"])
        top = np.argsort(np.argsort(-np.abs(E))) < int(0.2 * E.size)
        for i, sel in enumerate((np.ones_like(top), top)):
            absE = torch.from_numpy(np.abs(E[sel]).astype(np.float64))
            sig = torch.from_numpy(np.sqrt(np.maximum(V[sel], 1e-300)).astype(np.float64))
            y = Y[sel].astype(np.float64)
            grid = np.logspace(-1.5, 3.5, 61)
            c = float(grid[int(np.argmin([float(((_Phi(absE / (cc * sig)).numpy() - y) ** 2).mean()) for cc in grid]))])
            ax = axs[i, j]
            ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
            for nm, p, col in (("closed", _Phi(absE / sig).numpy(), BLUE),
                               (f"closed, c={c:.2g}", _Phi(absE / (c * sig)).numpy(), ORANGE)):
                rr = reliability(p, y, nb)
                ece, br = ece_brier(p, y, nb)
                ax.plot([r[0] for r in rr], [r[1] for r in rr], "-o", ms=3, color=col, label=f"{nm} (ECE {ece:.3f})")
                for r in rr:
                    rel_rows.append(dict(method=m, pixels=sets[i], predictor=nm, mean_pred=r[0], mean_agree=r[1], n=r[2]))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{m} - {sets[i]}", fontsize=9)
            ax.set_xlabel("predicted P(sign kept)"); ax.set_ylabel("observed fraction of draws")
            ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return rel_rows


def fig_eps(scan_rows, gxi_ratio, path):
    es = sorted({r["eps_std"] for r in scan_rows})
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
    med = [np.median([r["var_sum"] for r in scan_rows if r["eps_std"] == e]) for e in es]
    lo = [np.min([r["var_sum"] for r in scan_rows if r["eps_std"] == e]) for e in es]
    hi = [np.max([r["var_sum"] for r in scan_rows if r["eps_std"] == e]) for e in es]
    axs[0].fill_between(es, lo, hi, color=BLUE, alpha=0.2)
    axs[0].plot(es, med, "-o", color=BLUE, ms=4, label="LRP (min-max over images)")
    axs[0].axhline(gxi_ratio, color=ORANGE, lw=1.6, label="GxI (no stabiliser)")
    axs[0].axhline(1.0, color=GREY, ls="--", lw=1)
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel(r"stabiliser scale $\varepsilon_{\rm std}$"); axs[0].set_ylabel("total variance, closed form / MC")
    axs[0].legend(fontsize=8)
    sp = [np.median([r["spearman"] for r in scan_rows if r["eps_std"] == e]) for e in es]
    axs[1].plot(es, sp, "-o", color=BLUE, ms=4, label="LRP")
    axs[1].set_xscale("log"); axs[1].set_ylim(0, 1)
    axs[1].set_xlabel(r"stabiliser scale $\varepsilon_{\rm std}$"); axs[1].set_ylabel("Spearman of variance maps (median)")
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(rows, cal_rows, eps_std):
    md = [f"# Method comparison (LRP eps_std = {eps_std:g})\n",
          "Median over images (mean in brackets); ratios analytic/Monte-Carlo, 1 = exact.\n",
          "| method | total var | median var | Spearman (var) | Pearson (mean) | t closed [ms] | t per MC draw [ms] |",
          "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        s = [r for r in rows if r["method"] == m]
        f = lambda k: f"{np.median([r[k] for r in s]):.3g} ({np.mean([r[k] for r in s]):.3g})"
        md.append(f"| {m} | {f('var_sum')} | {f('var_med')} | {f('spearman')} | {f('pearson_E')} | "
                  f"{np.median([r['t_closed'] for r in s]) * 1e3:.0f} | {np.median([r['t_mc_draw'] for r in s]) * 1e3:.0f} |")
    s = [r for r in rows if r["method"] == "LRP"]
    md.append(f"\nCross-method (same images): Spearman of mean maps {np.median([r['cross_spearman_E'] for r in s]):.3f}, "
              f"of variance maps {np.median([r['cross_spearman_V'] for r in s]):.3f} (medians).\n")
    md += ["\n## Sign-confidence calibration (Brier, mean over images)\n",
           "| method | pixels | closed | closed*c | MC-Gaussian (in-sample) | constant | c (median) | mean p | mean agree |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for px in ("all", "top20"):
            g = lambda pr, k="brier": np.mean([r[k] for r in cal_rows if r["method"] == m and r["pixels"] == px and r["predictor"] == pr])
            c = np.median([r["c"] for r in cal_rows if r["method"] == m and r["pixels"] == px and r["predictor"] == "closed"])
            md.append(f"| {m} | {px} | {g('closed'):.4f} | {g('closed*c'):.4f} | {g('mc_gauss'):.4f} | {g('const'):.4f} | "
                      f"{c:.3g} | {g('closed', 'mean_p'):.3f} | {g('closed', 'mean_agree'):.3f} |")
    return "\n".join(md) + "\n"


def main():
    import yaml
    from data.cub200 import CUB200Dataset
    from diag_var import load_model

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=10)
    ap.add_argument("--n_ref", type=int, default=100)
    ap.add_argument("--n_fig", type=int, default=3, help="images shown in cmp_examples.png")
    ap.add_argument("--eps_std", type=float, default=None, help="LRP stabiliser (default: config)")
    ap.add_argument("--eps_scan", type=float, nargs="*", default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = torch.device(a.device)
    torch.manual_seed(a.seed); random.seed(a.seed)
    model, est = load_model(cfg, a.checkpoint, dev)
    print(f"Loaded {a.checkpoint}  ({est}, backbone {cfg['backbone']})")
    eps_std = a.eps_std if a.eps_std is not None else float(cfg.get("lrp_eps_std", 0.25))

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n_images)
    xs = [(i, ds[i][0].unsqueeze(0).to(dev)) for i in idx]

    rows, cal_rows, scan_rows, pooled, var_pts, figs = evaluate(
        model, xs, a.n_ref, float(cfg.get("lrp_eps", 1e-6)), eps_std, bool(cfg.get("lrp_fuse_bn", True)),
        float(cfg.get("lrp_pool_corr", 0.0)), a.eps_scan, a.n_fig, dev)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "compare_metrics.csv", rows)
    write_csv(out / "compare_calibration.csv", cal_rows)
    write_csv(out / "compare_eps_scan.csv", scan_rows)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)
    fig_examples(figs, out / "cmp_examples.png")
    fig_fidelity(var_pts, out / "cmp_variance_fidelity.png")
    write_csv(out / "compare_reliability.csv", fig_calibration(pooled, out / "cmp_calibration.png"))
    gx = np.median([r["var_sum"] for r in rows if r["method"] == "GxI"])
    fig_eps(scan_rows, gx, out / "cmp_eps_sensitivity.png")
    md = summarise(rows, cal_rows, eps_std)
    (out / "compare_summary.md").write_text(md)
    print("\n" + md)
    print(f"Tables, CSVs and figures in {out}/")


if __name__ == "__main__":
    main()
