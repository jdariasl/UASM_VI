"""
make_report.py
Generate report-ready figures and metrics for the CUB-200-2011 / VGG-11
experiments, comparing every checkpoint found under checkpoints/:

    checkpoints/cub200_vgg11            deterministic baseline
    checkpoints/cub200_vgg11_reparam    VI-BNN, Reparameterization
    checkpoints/cub200_vgg11_flipout    VI-BNN, Flipout

For six test images (4 correctly + 2 incorrectly classified by the reference
model) it produces, per image, one grid figure

    rows    = models
    columns = input | LRP E[R] | LRP Var[R] | LRP SNR | IG E | IG Var | IG SNR

plus metrics for every (image, model) pair, written to CSV and Markdown.

Metrics
-------
Prediction   : predicted class, confidence, predictive entropy, mutual
               information (epistemic part of the entropy; BNNs only).
Localisation : pointing game on the CUB part annotations (peak of the map falls
               within --tol px of a visible part), for the E[.] map and for the
               SNR map, and the chance rate of that test for the same image.
Faithfulness : LRP conservation  sum_i E[R_i]  (equals the unit relevance placed
               on the target logit, up to bias/BN/stabiliser leakage).
Agreement    : Spearman correlation between the LRP and IG maps.
Validation   : (BNNs) closed-form moments versus references that do not share its
               approximations.  logit_var_ratio_fo = analytic Var[logit] / first-order
               weight-space delta method (sampling-free, exact for small noise);
               mc_logit_var_ratio = analytic / Monte-Carlo over weight draws;
               mc_pearson_E / mc_spearman_Var / mc_var_ratio compare the LRP maps
               with Monte-Carlo LRP on explicit w ~ q.  A ratio near 1 is good; for
               residual networks expect < 1 (branches are treated as independent).

Usage
-----
    python make_report.py --config configs/cub200_vgg11.yaml \
        --ckpt_root checkpoints --outdir results/report

    # also compute aggregate metrics (no figures) over 100 more test images
    python make_report.py --config configs/cub200_vgg11.yaml --n_eval 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.models import select_model
from data.cub200 import CUB200Dataset
from explainability.bayesian_lrp import BayesianLRP
from explainability.bayesian_ig import BayesianIG
from utils.bnn_utils import has_bayesian_layers, SampledTwin
from utils.visualization import _denorm_image, _to_np


# =========================================================================== #
#  Loading
# =========================================================================== #

class Entry:
    """A loaded checkpoint plus the explainers built on top of it."""

    def __init__(self, label, model, bayesian, estimator, ckpt_dir):
        self.label, self.model, self.bayesian = label, model, bayesian
        self.estimator, self.ckpt_dir = estimator, ckpt_dir
        self.twin: Optional[SampledTwin] = None
        self.lrp: Optional[BayesianLRP] = None
        self.ig: Optional[BayesianIG] = None


def load_entries(cfg: dict, ckpt_root: Path, device, args) -> List[Entry]:
    prefix = Path(cfg.get("checkpoint_dir", "checkpoints/cub200_vgg11")).name
    for suf in ("_reparam", "_flipout"):          # configs point at one estimator's dir
        if prefix.endswith(suf):
            prefix = prefix[: -len(suf)]
    dirs = sorted(d for d in ckpt_root.iterdir()
                  if d.is_dir() and d.name.startswith(prefix)
                  and (d / "best_model.pt").exists())
    if not dirs:
        raise FileNotFoundError(f"No '{prefix}*/best_model.pt' under {ckpt_root}")

    entries = []
    for d in dirs:
        ck = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
        state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        ccfg = ck.get("cfg", {}) if isinstance(ck, dict) else {}
        bayesian = any(("mu_kernel" in k) or ("mu_weight" in k) for k in state)
        est = ccfg.get("estimator") or ("Flipout" if "flipout" in d.name.lower()
                                        else "Reparameterization")
        label = (f"{est}" if bayesian else "Deterministic")
        model = select_model(cfg["backbone"], bayesian=bayesian,
                             n_classes=cfg["n_classes"], estimator=est)
        model.load_state_dict(state)
        model = model.to(device).eval()
        e = Entry(label, model, bayesian, est, d)
        print(f"  loaded {d.name:28s} -> {label:18s} "
              f"(epoch {ck.get('epoch', '?')}, val_acc {ck.get('val_acc', float('nan')):.3f})")
        entries.append(e)

    # unique labels
    seen = {}
    for e in entries:
        seen[e.label] = seen.get(e.label, 0) + 1
        if seen[e.label] > 1:
            e.label += f" ({e.ckpt_dir.name})"

    for e in entries:
        if e.bayesian:
            e.twin = SampledTwin(e.model)
        e.lrp = BayesianLRP(e.model, eps=float(cfg.get("lrp_eps", 1e-6)),
                            eps_std=args.eps_std, fuse_bn=bool(cfg.get("lrp_fuse_bn", True)))
        e.ig = BayesianIG(e.model, n_samples=int(cfg.get("n_samples", 20)),
                          m_steps=int(cfg.get("m_steps", 50)),
                          baseline=cfg.get("baseline", "black"),
                          use_twin=e.bayesian, twin=e.twin)
    return entries


# =========================================================================== #
#  Prediction
# =========================================================================== #

@torch.no_grad()
def predict(e: Entry, x: torch.Tensor, n_mc: int):
    """Returns dict(pred, conf, probs, entropy, mi)."""
    k = n_mc if e.bayesian else 1
    ps = torch.stack([torch.softmax(e.model(x), dim=1)[0] for _ in range(k)])  # (k, C)
    p = ps.mean(0)
    ent = -(p * torch.log(p + 1e-12)).sum()
    exp_ent = -(ps * torch.log(ps + 1e-12)).sum(1).mean()
    return dict(pred=int(p.argmax()), conf=float(p.max()), probs=p,
                entropy=float(ent), mi=float(ent - exp_ent) if e.bayesian else float("nan"))


# =========================================================================== #
#  Part annotations in the cropped 224x224 frame
# =========================================================================== #

def parts_in_crop(parts: torch.Tensor, orig_w: int, orig_h: int,
                  resize: int = 256, crop: int = 224) -> np.ndarray:
    """Map CUB part coordinates through Resize(256) + CenterCrop(224).
    Returns (n, 2) array of (x, y) for visible parts that stay inside the crop."""
    if parts.shape[0] == 0:
        return np.zeros((0, 2))
    if orig_w <= orig_h:
        new_w, new_h = resize, int(resize * orig_h / orig_w)
    else:
        new_w, new_h = int(resize * orig_w / orig_h), resize
    top, left = int(round((new_h - crop) / 2.0)), int(round((new_w - crop) / 2.0))
    xy = parts[:, 1:3].numpy().astype(np.float64)
    out = np.stack([xy[:, 0] * new_w / orig_w - left,
                    xy[:, 1] * new_h / orig_h - top], axis=1)
    keep = (out[:, 0] >= 0) & (out[:, 0] < crop) & (out[:, 1] >= 0) & (out[:, 1] < crop)
    return out[keep]


def tolerance_mask(pts: np.ndarray, tol: float, size: int = 224) -> np.ndarray:
    """Boolean (size,size) mask of pixels within `tol` px of any part."""
    if len(pts) == 0:
        return np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[0:size, 0:size]
    m = np.zeros((size, size), dtype=bool)
    for x, y in pts:
        m |= (xx - x) ** 2 + (yy - y) ** 2 <= tol ** 2
    return m


# =========================================================================== #
#  Map utilities / statistics
# =========================================================================== #

def chan_sum(t: torch.Tensor) -> np.ndarray:          # (1,C,H,W) -> (H,W)
    return _to_np(t[0]).sum(axis=0)


def chan_mean(t: torch.Tensor) -> np.ndarray:
    return _to_np(t[0]).mean(axis=0)


def _ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a.ravel())).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def peak_hit(m: np.ndarray, mask: np.ndarray) -> float:
    """1 if the arg-max of m lies inside the tolerance mask."""
    if not np.isfinite(m).all():
        return float("nan")
    iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(mask[iy, ix])


# =========================================================================== #
#  Explain one image with one model
# =========================================================================== #

def explain_one(e: Entry, x: torch.Tensor, target: int, seed: int):
    torch.manual_seed(seed)
    out = {}
    mu, v, snr = e.lrp.explain(x, target)
    out["lrp"] = (mu.detach(), v.detach(), snr.detach() if e.bayesian else None)
    last = e.lrp.last_stats[-1]
    out["logit"] = (float(last["mu"][0, target]), float(last["v"][0, target]))
    out["sign"] = e.lrp.last_sign
    mu, v, snr = e.ig.explain(x, target)
    out["ig"] = (mu.detach(), v.detach(), snr.detach() if e.bayesian else None)
    return out


def first_order_logit_var(e: Entry, x: torch.Tensor, target: int) -> float:
    """
    Sampling-free reference for Var[logit]: first-order weight-space delta method
        Var[f] ~= sum_w (df/dw)^2 sigma_w^2,   gradient taken at the mean weights.
    Exact in the small-noise limit and free of any independence assumption, so
    it isolates the error of the closed-form moment propagation.
    """
    from utils.bnn_utils import get_weight_moments
    e.twin.use_mean()
    for p in e.twin.net.parameters():
        p.requires_grad_(True)
    e.twin.net.zero_grad(set_to_none=True)
    e.twin.net(x)[0, target].backward()
    var = 0.0
    for src, dst in e.twin.pairs:
        _, sw, _, sb = get_weight_moments(src)
        var += float((dst.weight.grad ** 2 * sw ** 2).sum())
        if dst.bias is not None and dst.bias.grad is not None:
            var += float((dst.bias.grad ** 2 * sb ** 2).sum())
    for p in e.twin.net.parameters():
        p.requires_grad_(False)
        p.grad = None
    return var


def mc_lrp_validation(e: Entry, xs: List[torch.Tensor], targets: List[int],
                      analytic: List[Tuple[torch.Tensor, torch.Tensor, float, float]],
                      n_mc: int, eps: float, seed: int) -> List[dict]:
    """Closed-form (E[R], Var[R], logit moments) versus Monte-Carlo on explicit w ~ q."""
    torch.manual_seed(seed)
    runner = BayesianLRP(e.twin.net, eps=eps, eps_std=0.0)   # deterministic net
    acc = [[] for _ in xs]
    lg = [[] for _ in xs]
    for _ in range(n_mc):
        e.twin.resample()
        for i, (x, t) in enumerate(zip(xs, targets)):
            with torch.no_grad():
                lg[i].append(float(e.twin.net(x)[0, t]))
            mu, _, _ = runner.explain(x, t)
            acc[i].append(mu.detach().cpu())
    res = []
    for i, (mu_a, v_a, lmu, lvar) in enumerate(analytic):
        st = torch.stack(acc[i])                              # (K,1,C,H,W)
        mu_mc, v_mc = st.mean(0), st.var(0, unbiased=True)
        mu_a, v_a = mu_a.cpu(), v_a.cpu()
        lv_mc = float(np.var(lg[i], ddof=1))
        res.append(dict(
            mc_pearson_E=pearson(mu_a.numpy(), mu_mc.numpy()),
            mc_spearman_Var=spearman(v_a.numpy(), v_mc.numpy()),
            mc_var_ratio=float(v_a.sum() / v_mc.sum().clamp(min=1e-30)),
            mc_logit_var_ratio=lvar / max(lv_mc, 1e-30),
        ))
    return res


# =========================================================================== #
#  Metrics for one (image, model)
# =========================================================================== #

def image_metrics(pred: dict, maps: dict, mask: np.ndarray, hit_base: float,
                  bayesian: bool, true_label: int) -> dict:
    row = dict(pred=pred["pred"], correct=int(pred["pred"] == true_label),
               conf=pred["conf"], entropy=pred["entropy"], mutual_info=pred["mi"],
               chance_hit=hit_base)
    for name in ("lrp", "ig"):
        mu, v, snr = maps[name]
        e_map = chan_sum(mu)
        row[f"{name}_hit_E"] = peak_hit(e_map, mask)
        row[f"{name}_hit_SNR"] = peak_hit(chan_mean(snr), mask) if snr is not None else float("nan")
        row[f"{name}_mean_var"] = float(v.mean()) if bayesian else float("nan")
        # LRP starts at sign(logit): multiply back so conservation reads ~1
        row[f"{name}_sum_E"] = float(mu.sum()) * (maps["sign"] if name == "lrp" else 1.0)
    row["lrp_ig_spearman"] = spearman(chan_sum(maps["lrp"][0]), chan_sum(maps["ig"][0]))
    return row


# =========================================================================== #
#  Figures
# =========================================================================== #

def _sym_lim(a: np.ndarray, q: float = 99.5) -> float:
    return max(float(np.percentile(np.abs(a[np.isfinite(a)]), q)), 1e-12)


def example_figure(path: Path, img: np.ndarray, pts: np.ndarray, title: str,
                   rows: List[Tuple[str, dict, bool, str]]):
    """rows: (label, maps, bayesian, subtitle)."""
    cols = ["Input", "LRP  E[R]", "LRP  Var[R]", "LRP  SNR",
            "IG  E[IG]", "IG  Var[IG]", "IG  SNR"]
    fig, axes = plt.subplots(len(rows), 7, figsize=(2.3 * 7, 2.3 * len(rows) + 0.9),
                             squeeze=False)
    for r, (label, maps, bayes, sub) in enumerate(rows):
        ax = axes[r, 0]
        ax.imshow(img)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#ffd400", edgecolors="k", linewidths=0.4)
        ax.set_ylabel(f"{label}\n{sub}", fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
        for c, name in ((1, "lrp"), (4, "ig")):
            mu, v, snr = maps[name]
            m = chan_sum(mu)
            lim = _sym_lim(m)
            axes[r, c].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
            if bayes:
                axes[r, c + 1].imshow(chan_sum(v), cmap="viridis", vmin=0,
                                      vmax=max(float(np.percentile(chan_sum(v), 99.5)), 1e-30))
                s = chan_mean(snr)
                sl = min(_sym_lim(s), 5.0)
                axes[r, c + 2].imshow(s, cmap="RdBu_r", vmin=-sl, vmax=sl)
            else:
                for k in (c + 1, c + 2):
                    axes[r, k].imshow(np.ones((224, 224, 3)))      # keeps panel geometry aligned
                    axes[r, k].text(0.5, 0.5, "n/a\n(deterministic)", ha="center", va="center",
                                    fontsize=8, color="0.45", transform=axes[r, k].transAxes)
    for r in range(len(rows)):
        for c in range(7):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
#  Output tables
# =========================================================================== #

METRIC_COLS = ["pred", "correct", "conf", "entropy", "mutual_info",
               "lrp_sum_E", "lrp_mean_var", "lrp_hit_E", "lrp_hit_SNR",
               "ig_mean_var", "ig_hit_E", "ig_hit_SNR", "chance_hit",
               "lrp_ig_spearman", "logit_var_ratio_fo", "mc_logit_var_ratio",
               "mc_pearson_E", "mc_spearman_Var", "mc_var_ratio"]


def _fmt(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.3g}" if abs(v) < 1e-2 or abs(v) >= 1e3 else f"{v:.3f}"


def write_csv(path: Path, rows: List[dict], cols: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def md_table(rows: List[dict], cols: List[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def aggregate(rows: List[dict], labels: List[str]) -> List[dict]:
    agg = []
    for lab in labels:
        rs = [r for r in rows if r["model"] == lab]
        a = {"model": lab, "n": len(rs)}
        for c in METRIC_COLS:
            if c in ("pred",):
                continue
            vals = np.array([r.get(c, np.nan) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            a[c] = float(vals.mean()) if len(vals) else float("nan")
        agg.append(a)
    return agg


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cub200_vgg11.yaml")
    ap.add_argument("--ckpt_root", default="checkpoints")
    ap.add_argument("--outdir", default="results/report")
    ap.add_argument("--data_root", default=None, help="override data_root from the config")
    ap.add_argument("--n_examples", type=int, default=6)
    ap.add_argument("--n_candidates", type=int, default=300,
                    help="random test images screened to pick the examples")
    ap.add_argument("--n_eval", type=int, default=0,
                    help="extra test images for aggregate metrics (no figures, no MC validation)")
    ap.add_argument("--n_mc_lrp", type=int, default=30,
                    help="weight draws for the Monte-Carlo validation of closed-form LRP")
    ap.add_argument("--n_mc_pred", type=int, default=10, help="MC passes for predictions")
    ap.add_argument("--eps_std", type=float, default=None,
                    help="uncertainty-scaled LRP stabiliser (default: lrp_eps_std from the config, else 0.25); 0 = classical LRP-eps")
    ap.add_argument("--tol", type=float, default=15.0, help="pointing-game tolerance, px at 224")
    ap.add_argument("--reference", default="Reparameterization",
                    help="model whose predictions select the examples and the target class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.eps_std is None:
        args.eps_std = float(cfg.get("lrp_eps_std", 0.25))
    assert cfg["backbone"] in ("vgg11", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101"), \
        "make_report supports VGG and ResNet (closed-form LRP needs sequential or residual-block networks)"
    device = torch.device(args.device)
    out = Path(args.outdir); (out / "figures").mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()

    print("Loading checkpoints")
    entries = load_entries(cfg, Path(args.ckpt_root), device, args)
    labels = [e.label for e in entries]
    ref = next((e for e in entries if e.label.startswith(args.reference)), entries[0])
    print(f"Reference model for example selection / target class: {ref.label}")

    # ---- data ----
    data_root = args.data_root or cfg["data_root"]
    ds = CUB200Dataset(root=data_root, split="test", return_parts=True)
    cls_names = {}
    cfile = Path(data_root) / "classes.txt"
    if cfile.exists():
        for line in open(cfile):
            i, n = line.strip().split(" ", 1)
            cls_names[int(i) - 1] = n.split(".", 1)[-1].replace("_", " ")
    cname = lambda c: cls_names.get(int(c), str(int(c)))

    def get(idx):
        img, label, parts = ds[idx]
        path = ds.samples[idx][0]
        w, h = Image.open(path).size
        pts = parts_in_crop(parts, w, h)
        return img.unsqueeze(0).to(device), int(label), pts, path

    # ---- screen candidates with the reference model ----
    rng = random.Random(args.seed)
    cand = rng.sample(range(len(ds)), min(args.n_candidates, len(ds)))
    right, wrong = [], []
    print(f"Screening {len(cand)} test images")
    for idx in cand:
        x, y, pts, _ = get(idx)
        if len(pts) < 3:                       # need enough visible parts for the pointing game
            continue
        p = predict(ref, x, args.n_mc_pred)
        (right if p["pred"] == y else wrong).append(idx)
    want_wrong = max(1, args.n_examples // 3)                     # 2 of 6 by default
    n_wrong = min(len(wrong), max(want_wrong, args.n_examples - len(right)))
    n_right = min(len(right), args.n_examples - n_wrong)
    chosen = right[:n_right] + wrong[:n_wrong]
    print(f"  selected {n_right} correct + {n_wrong} misclassified: {chosen}")

    extra = []
    if args.n_eval > 0:
        pool = [i for i in range(len(ds)) if i not in chosen]
        extra = rng.sample(pool, min(args.n_eval, len(pool)))

    # ---- explain ----
    rows, maps_store = [], {}
    for k, idx in enumerate(chosen + extra):
        is_example = k < len(chosen)
        x, y, pts, path = get(idx)
        target = predict(ref, x, args.n_mc_pred)["pred"]
        mask = tolerance_mask(pts, args.tol)
        base = float(mask.mean())
        fig_rows, per_model = [], {}
        for e in entries:
            pr = predict(e, x, args.n_mc_pred)
            mp = explain_one(e, x, target, seed=args.seed + idx)
            m = image_metrics(pr, mp, mask, base, e.bayesian, y)
            m.update(image=idx, tag=("example" if is_example else "eval"),
                     model=e.label, true=y, target=target)
            if e.bayesian:
                m["logit_var"] = mp["logit"][1]
                m["logit_var_ratio_fo"] = mp["logit"][1] / max(first_order_logit_var(e, x, target), 1e-30)
            rows.append(m)
            per_model[e.label] = mp
            sub = f"pred: {cname(pr['pred'])[:18]}\np={pr['conf']:.2f}"
            fig_rows.append((e.label, mp, e.bayesian, sub))
        maps_store[idx] = per_model

        if is_example:
            img = _denorm_image(_to_np(x[0]))
            title = (f"test image {idx}   |   true: {cname(y)}   |   explained class: {cname(target)}"
                     f"{'  (misclassified by reference)' if y != target else ''}")
            example_figure(out / "figures" / f"example_{k+1:02d}_idx{idx}.png", img, pts, title, fig_rows)
            arrs = {}
            for e in entries:
                for n in ("lrp", "ig"):
                    for t, i in (("E", 0), ("Var", 1), ("SNR", 2)):
                        mp_ = per_model[e.label][n][i]
                        if mp_ is not None:
                            key = f"{e.label}__{n}__{t}".replace(" ", "_")
                            arrs[key] = chan_mean(mp_) if t == "SNR" else chan_sum(mp_)
            np.savez_compressed(out / "figures" / f"example_{k+1:02d}_idx{idx}_maps.npz", **arrs)
        print(f"  [{k+1}/{len(chosen)+len(extra)}] image {idx} done  ({time.time()-t0:.0f}s)")

    # ---- Monte-Carlo validation of the closed-form LRP (examples only) ----
    if args.n_mc_lrp > 0:
        xs, ts = [], []
        for idx in chosen:
            x, y, _, _ = get(idx)
            xs.append(x); ts.append(predict(ref, x, args.n_mc_pred)["pred"])
        for e in entries:
            if not e.bayesian:
                continue
            print(f"MC validation of closed-form LRP: {e.label} ({args.n_mc_lrp} draws)")
            an = [(maps_store[idx][e.label]["lrp"][0], maps_store[idx][e.label]["lrp"][1],
                   *maps_store[idx][e.label]["logit"]) for idx in chosen]
            res = mc_lrp_validation(e, xs, ts, an, args.n_mc_lrp,
                                    float(cfg.get("lrp_eps", 1e-6)), args.seed)
            for idx, r in zip(chosen, res):
                for row in rows:
                    if row["image"] == idx and row["model"] == e.label:
                        row.update(r)

    # ---- tables ----
    ex_rows = [r for r in rows if r["tag"] == "example"]
    id_cols = ["image", "model", "true", "target"]
    write_csv(out / "metrics_examples.csv", ex_rows, id_cols + METRIC_COLS)
    summaries = [("Examples", ex_rows)]
    if extra:
        ev_rows = [r for r in rows if r["tag"] == "eval"]
        write_csv(out / "metrics_eval.csv", ev_rows, id_cols + METRIC_COLS)
        summaries.append((f"Aggregate over {len(ev_rows)//len(entries)} extra test images", ev_rows))

    md = ["# Report metrics\n",
          f"Config `{args.config}`, eps_std={args.eps_std}, pointing tolerance {args.tol:g}px, "
          f"IG: K={cfg.get('n_samples', 20)} draws x {cfg.get('m_steps', 50)} steps, "
          f"MC-LRP draws={args.n_mc_lrp}.\n",
          "`*_hit_*`: 1 = peak of the map within tolerance of a visible part. `chance_hit` is the "
          "fraction of pixels within tolerance (expected hit rate of a random peak). "
          "`lrp_sum_E` is the conservation check. `logit_var_ratio_fo` / `mc_logit_var_ratio`: analytic "
          "Var[logit] divided by the first-order / Monte-Carlo reference (1 = exact). `mc_*`: LRP maps vs Monte-Carlo.\n"]
    for title, rs in summaries:
        agg = aggregate(rs, labels)
        write_csv(out / f"summary_{'examples' if title == 'Examples' else 'eval'}.csv", agg,
                  ["model", "n"] + [c for c in METRIC_COLS if c != "pred"])
        md += [f"\n## {title} – mean per model\n",
               md_table(agg, ["model", "n"] + [c for c in METRIC_COLS if c != "pred"])]
    md += ["\n## Per-example metrics\n", md_table(ex_rows, id_cols + [c for c in METRIC_COLS if c != "pred"])]
    (out / "summary.md").write_text("\n".join(md))
    json.dump(vars(args), open(out / "run_args.json", "w"), indent=2)

    print(f"\nDone in {time.time()-t0:.0f}s -> {out}/")
    print((out / "summary.md").read_text().split("## Per-example")[0])


if __name__ == "__main__":
    main()
