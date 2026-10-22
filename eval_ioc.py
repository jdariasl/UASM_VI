"""
eval_ioc.py
Importance-of-context index with percentile confidence (IoC_alpha, Arias-Londono & Godino-Llorente,
BSPC 2024) under the VI approach.  Region of interest = the bird bounding box of CUB-200-2011
(the lungs in the original paper); IoC_alpha > 1 means that background pixels that are positive at
the alpha percentile are more relevant than bird pixels (a Clever-Hans signature).

For each test image and each method (LRP, GxI) the index is computed
  (a) from the Monte-Carlo draws of the weights (the paper's estimator, the reference), and
  (b) in closed form from (mean, variance) maps, for each sigma scale c in --scales,
and the agreement over images is reported (mean +- std of each, mean absolute error, correlation).

    python eval_ioc.py --config configs/cub200_resnet50.yaml \\
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \\
        --n_images 30 --n_ref 100 --outdir results/ioc_resnet50_reparam
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
from PIL import Image

from explainability.bayesian_gradinput import BayesianGradInput, GradInputRef
from explainability.bayesian_lrp import BayesianLRP
from explainability.ioc import bbox_mask, ioc_alpha_closed, ioc_alpha_mc
from utils.bnn_utils import SampledTwin


def load_bboxes(root: Path):
    paths, boxes = {}, {}
    for line in open(root / "images.txt"):
        i, p = line.split()
        paths[int(i)] = p
    for line in open(root / "bounding_boxes.txt"):
        i, x, y, w, h = line.split()
        boxes[int(i)] = (float(x), float(y), float(w), float(h))
    return paths, boxes


def main():
    from data.cub200 import CUB200Dataset
    from diag_var import load_model

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=30)
    ap.add_argument("--n_ref", type=int, default=100)
    ap.add_argument("--alphas", type=float, nargs="+", default=[25, 50, 75, 95])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 3.0, 10.0], help="sigma scale c of the closed form")
    ap.add_argument("--eps_std", type=float, default=None)
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
    root = Path(cfg["data_root"])
    paths, boxes = load_bboxes(root)

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n_images)
    twin = SampledTwin(model)
    lrp_ref = BayesianLRP(twin.net, eps=float(cfg.get("lrp_eps", 1e-6)), eps_std=0.0, fuse_bn=bool(cfg.get("lrp_fuse_bn", True)))
    gxi_ref = GradInputRef(twin.net)
    lrp = BayesianLRP(model, eps=float(cfg.get("lrp_eps", 1e-6)), eps_std=eps_std,
                      fuse_bn=bool(cfg.get("lrp_fuse_bn", True)), pool_corr=float(cfg.get("lrp_pool_corr", 0.0)))
    gxi = BayesianGradInput(model, pool_corr=float(cfg.get("lrp_pool_corr", 0.0)))

    rows = []
    for n, i in enumerate(idx):
        x = ds[i][0].unsqueeze(0).to(dev)
        iid = ds.samples[i][2]
        with Image.open(root / "images" / paths[iid]) as im:
            W, H = im.size
        inside = bbox_mask(boxes[iid], (H, W))
        with torch.no_grad():
            lg = twin.use_mean()(x); t = int(lg.argmax()); d = abs(float(lg[0, t]))
        dl, dg = [], []
        for _ in range(a.n_ref):
            twin.resample()
            with torch.no_grad():
                dl.append(lrp_ref.explain(x, t)[0].cpu().numpy())
            dg.append(gxi_ref.explain(x, t, d).cpu().numpy())
        with torch.no_grad():
            cl = lrp.explain(x, t)[:2]
            cg = gxi.explain(x, t, denom=d)[:2]
        for name, draws, (mu, v) in (("LRP", dl, cl), ("GxI", dg, cg)):
            mu, v = mu.cpu().numpy(), v.cpu().numpy()
            for al in a.alphas:
                r = dict(image=i, method=name, alpha=al, bbox_frac=float(inside.mean()),
                         ioc_mc=ioc_alpha_mc(draws, inside, al))
                for c in a.scales:
                    r[f"ioc_closed_c{c:g}"] = ioc_alpha_closed(mu, v, inside, al, c)
                rows.append(r)
        print(f"[{n + 1}/{a.n_images}] img {i} cls {t} bbox {inside.mean():.2f}  " +
              "  ".join(f"{r['method']}@{r['alpha']:g}: MC {r['ioc_mc']:.2f} cl {r['ioc_closed_c1']:.2f}"
                        for r in rows[-2 * len(a.alphas):] if r['alpha'] in (25, 95)), flush=True)

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "ioc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)

    md = ["| method | alpha | IoC MC | " + " | ".join(f"closed c={c:g}" for c in a.scales) + " | MAE c=" +
          " / ".join(f"{c:g}" for c in a.scales) + " | corr c=1 |", "|---|---|---|" + "---|" * (len(a.scales) + 2)]
    for m in ("LRP", "GxI"):
        for al in a.alphas:
            s = [r for r in rows if r["method"] == m and r["alpha"] == al]
            mc = np.array([r["ioc_mc"] for r in s])
            cols, maes = [], []
            for c in a.scales:
                cl_ = np.array([r[f"ioc_closed_c{c:g}"] for r in s])
                ok = np.isfinite(mc) & np.isfinite(cl_)
                cols.append(f"{np.nanmean(cl_):.2f} ± {np.nanstd(cl_):.2f}")
                maes.append(f"{np.mean(np.abs(mc[ok] - cl_[ok])):.2f}" if ok.any() else "nan")
            c1 = np.array([r[f"ioc_closed_c{a.scales[0]:g}"] for r in s]); ok = np.isfinite(mc) & np.isfinite(c1)
            corr = np.corrcoef(mc[ok], c1[ok])[0, 1] if ok.sum() > 2 else float("nan")
            md.append(f"| {m} | {al:g} | {np.nanmean(mc):.2f} ± {np.nanstd(mc):.2f} | " + " | ".join(cols) +
                      f" | {' / '.join(maes)} | {corr:.2f} |")
    (out / "ioc_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md) + f"\n\nCSV and summary in {out}/")


if __name__ == "__main__":
    main()
