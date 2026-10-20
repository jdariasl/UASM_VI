"""
filter_eval.py
Does filtering a saliency map by its estimated variance help?

Idea under test: keep only the pixels whose closed-form Var[R] is low and
highlight those.  Four tests, each at several coverages (fraction of pixels kept):

 1. STABILITY (Monte-Carlo).  Closed-form Var[R] decides the mask; the maps of
    K explicit weight draws w ~ q (deterministic LRP on a SampledTwin) say
    whether the kept highlight really is stable:
        jaccard   mean overlap between the highlight set of the closed-form map
                  and the highlight sets of the individual draws,
        p_pos     fraction of draws in which the highlighted pixels stay positive.
    Also  auc_stable: among the 20 % most relevant pixels, how well does each
    score separate pixels that stay positive in >= 90 % of the draws (AUROC;
    0.5 = no information).  This controls for the confound that variance
    grows with relevance.
 2. LOCALISATION.  Highlight = the n (default 5 % of the image) most relevant
    kept pixels.  hit (peak of the kept map within --tol px of a part), and
    precision / lift (fraction of the highlight on parts, divided by the
    fraction of the image on parts).
 3. RISK-COVERAGE.  The curves of 1, 2 and 4 against coverage, for every
    filter, are written to filter_curves.png / filter_eval_summary.csv, with
    paired differences to the random and to the magnitude filters (mean and 95 %
    CI over images).
 4. FAITHFULNESS.  Delete the highlight (replace by the mean image) and measure
    the relative drop of the target logit (del_drop, higher = more faithful);
    insert it into the mean image and measure the gain (ins_gain).  Evaluated at
    the mean weights.

Filters (each keeps the best `coverage` fraction of pixels)
    variance   lowest closed-form Var[R]                (the idea under test)
    snr        highest |E[R]| / sqrt(Var[R])
    var_adj    lowest Var[R] *given* |E[R]| (residual of log V ~ log|E|),
               i.e. variance with the magnitude confound removed
    magnitude  highest E[R]  (reference: never removes the highlight)
    random     random pixels (3 repeats averaged)  (reference: chance)

Usage
-----
    python filter_eval.py --config configs/cub200_resnet50.yaml \\
        --n_images 50 --n_mc 30 --outdir results/filter_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.cub200 import CUB200Dataset
from explainability.bayesian_lrp import BayesianLRP
from make_report import (load_entries, predict, parts_in_crop, tolerance_mask,
                         chan_sum, spearman)

FILTERS = ["variance", "snr", "var_adj", "magnitude", "random"]
N_RANDOM = 3
METRICS = ["hit", "lift", "p_pos", "jaccard", "del_drop", "ins_gain"]


# --------------------------------------------------------------------------- #
#  Scores and masks
# --------------------------------------------------------------------------- #

def filter_scores(E: np.ndarray, V: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Higher score = keep first.  E, V: (H, W) closed-form mean / variance."""
    absE = np.abs(E)
    Vp = V + 1e-12 * float(V.max()) + 1e-30
    s = {"variance": -V,
         "snr": absE / np.sqrt(Vp),
         "magnitude": E.copy()}
    # variance given magnitude: residual of log V on log|E|
    lv, le = np.log(Vp).ravel(), np.log(absE + 1e-12 * float(absE.max()) + 1e-30).ravel()
    sub = rng.choice(lv.size, size=min(lv.size, 20000), replace=False)
    a, b = np.polyfit(le[sub], lv[sub], 1)
    s["var_adj"] = -(lv - (a * le + b)).reshape(E.shape)
    return s


def keep_mask(score: np.ndarray, q: float) -> np.ndarray:
    n = max(1, int(round(q * score.size)))
    idx = np.argpartition(-score.ravel(), n - 1)[:n]
    m = np.zeros(score.size, dtype=bool)
    m[idx] = True
    return m.reshape(score.shape)


def top_set(M: np.ndarray, K: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask of the n most relevant positive pixels of M inside K."""
    v = np.where(K, M, -np.inf).ravel()
    n = min(n, int(K.sum()))
    idx = np.argpartition(-v, n - 1)[:n]
    idx = idx[v[idx] > 0]
    m = np.zeros(v.size, dtype=bool)
    m[idx] = True
    return m.reshape(M.shape)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(score)).astype(np.float64) + 1.0
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
#  Monte-Carlo maps
# --------------------------------------------------------------------------- #

def mc_maps(e, x, target, n_mc, eps):
    """(K, H, W) pixel relevance maps of deterministic LRP on weight draws."""
    runner = BayesianLRP(e.twin.net, eps=eps, eps_std=0.0)
    maps = []
    for _ in range(n_mc):
        e.twin.resample()
        mu, _, _ = runner.explain(x, target)
        m = chan_sum(mu)
        if np.isfinite(m).all():
            maps.append(m)
    return np.stack(maps) if maps else None


# --------------------------------------------------------------------------- #
#  Per (image, model)
# --------------------------------------------------------------------------- #

def evaluate_pair(e, x, target, parts_mask, args, rng, eps):
    mu, v, _ = e.lrp.explain(x, target)
    E, V = chan_sum(mu), chan_sum(v)
    if not (np.isfinite(E).all() and np.isfinite(V).all()):
        return None
    MC = mc_maps(e, x, target, args.n_mc, eps)
    if MC is None or len(MC) < 5:
        return None
    Hh, Ww = E.shape
    n_top = max(1, int(round(args.top_frac * Hh * Ww)))
    base = float(parts_mask.mean())
    ppos = (MC > 0).mean(0)                              # per-pixel P(relevance > 0)

    out = {"mc_spearman_V": spearman(V, MC.var(0, ddof=1)), "n_mc_ok": len(MC)}

    # ---- confound-controlled AUROC: among the top-20 % relevant pixels -------
    sc0 = filter_scores(E, V, rng)
    top20 = np.argsort(-E.ravel())[: int(0.2 * E.size)]
    stable = ppos.ravel()[top20] >= 0.9
    for name in ("variance", "snr", "var_adj", "magnitude"):
        out[f"auc_stable_{name}"] = auroc(sc0[name].ravel()[top20], stable)
    out["frac_stable_top20"] = float(stable.mean())

    # ---- masks to evaluate ---------------------------------------------------
    variants = []                                        # (filter, coverage, rep, mask)
    for q in args.keep:
        if q >= 1.0:
            variants.append(("all", 1.0, 0, np.ones_like(E, dtype=bool)))
            continue
        for f in FILTERS:
            if f == "random":
                for r in range(N_RANDOM):
                    variants.append((f, q, r, keep_mask(rng.random(E.shape), q)))
            else:
                variants.append((f, q, 0, keep_mask(sc0[f], q)))

    xs_np = x.detach()
    base_img = torch.zeros_like(xs_np)                   # mean image in normalised space
    del_imgs, ins_imgs, recs = [], [], []
    for (f, q, r, K) in variants:
        Hs = top_set(E, K, n_top)
        nH = int(Hs.sum())
        rec = dict(filter=f, coverage=q, rep=r)
        # --- localisation ---
        Ek = np.where(K, E, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(Ek)), E.shape)
        rec["hit"] = float(parts_mask[iy, ix])
        if nH:
            prec = float(parts_mask[Hs].mean())
            rec["lift"] = prec / max(base, 1e-12)
            # --- stability ---
            rec["p_pos"] = float(ppos[Hs].mean())
            js = []
            for k in range(min(len(MC), args.n_jac)):
                Hk = top_set(MC[k], K, n_top)
                inter = float((Hs & Hk).sum())
                js.append(inter / max(float((Hs | Hk).sum()), 1.0))
            rec["jaccard"] = float(np.mean(js))
        else:
            rec.update(lift=float("nan"), p_pos=float("nan"), jaccard=float("nan"))
        m = torch.from_numpy(Hs).to(x.device)[None, None].expand_as(xs_np)
        del_imgs.append(torch.where(m, base_img, xs_np))
        ins_imgs.append(torch.where(m, xs_np, base_img))
        recs.append(rec)

    # ---- faithfulness at the mean weights (batched) ---------------------------
    if e.bayesian:
        e.twin.use_mean()
        net = e.twin.net
    else:
        net = e.model
    with torch.no_grad():
        f0 = float(net(xs_np)[0, target])
        fb = float(net(base_img)[0, target])
        fd = torch.cat([net(torch.cat(del_imgs[i:i + 32]))[:, target]
                        for i in range(0, len(del_imgs), 32)]).cpu().numpy()
        fi = torch.cat([net(torch.cat(ins_imgs[i:i + 32]))[:, target]
                        for i in range(0, len(ins_imgs), 32)]).cpu().numpy()
    scale = max(abs(f0 - fb), 1e-6)
    for rec, d, i in zip(recs, fd, fi):
        rec["del_drop"] = float((f0 - d) / scale)         # 1 = deleting the highlight erases the evidence
        rec["ins_gain"] = float((i - fb) / scale)         # 1 = the highlight alone restores the logit
    out["records"] = recs
    return out


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #

def _avg_reps(recs):
    """Average the random repeats so every (filter, coverage) has one value per image."""
    groups = {}
    for r in recs:
        groups.setdefault((r["filter"], r["coverage"]), []).append(r)
    res = {}
    for k, rs in groups.items():
        res[k] = {m: float(np.nanmean([r[m] for r in rs])) if any(np.isfinite(r[m]) for r in rs)
                  else float("nan") for m in METRICS}
    return res


def mean_sem(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), (float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan")), len(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_root", default="checkpoints")
    ap.add_argument("--outdir", default="results/filter_eval")
    ap.add_argument("--n_images", type=int, default=50)
    ap.add_argument("--n_mc", type=int, default=30, help="weight draws for the stability test")
    ap.add_argument("--n_jac", type=int, default=10, help="draws used for the Jaccard stability")
    ap.add_argument("--keep", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.1],
                    help="coverages (fraction of pixels kept)")
    ap.add_argument("--top_frac", type=float, default=0.05, help="highlight size (fraction of the image)")
    ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--eps_std", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.eps_std is None:
        a.eps_std = float(cfg.get("lrp_eps_std", 0.25))
    dev = torch.device(a.device)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()

    print("Loading checkpoints")
    entries = [e for e in load_entries(cfg, Path(a.ckpt_root), dev,
                                       SimpleNamespace(eps_std=a.eps_std)) if e.bayesian]
    if not entries:
        raise SystemExit("no Bayesian checkpoints found")
    eps = float(cfg.get("lrp_eps", 1e-6))

    ds = CUB200Dataset(root=cfg["data_root"], split="test", return_parts=True)
    order = random.Random(a.seed).sample(range(len(ds)), len(ds))
    per = {e.label: [] for e in entries}                  # per model: list of dict per image
    aux = {e.label: [] for e in entries}
    done = 0
    for idx in order:
        if done >= a.n_images:
            break
        img, label, parts = ds[idx]
        w, h = Image.open(ds.samples[idx][0]).size
        pts = parts_in_crop(parts, w, h)
        if len(pts) < 3:
            continue
        pm = tolerance_mask(pts, a.tol)
        x = img.unsqueeze(0).to(dev)
        ok = True
        res = {}
        for e in entries:
            target = predict(e, x, 5)["pred"]
            r = evaluate_pair(e, x, target, pm, a, rng, eps)
            if r is None:
                ok = False
                break
            res[e.label] = r
        if not ok:
            print(f"  image {idx}: skipped (non-finite maps)")
            continue
        for lab, r in res.items():
            per[lab].append(_avg_reps(r["records"]))
            aux[lab].append({k: v for k, v in r.items() if k != "records"})
        done += 1
        print(f"  [{done}/{a.n_images}] image {idx}  ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ output
    rows, md = [], ["# Variance-filter evaluation\n",
                    f"{done} test images, {a.n_mc} MC draws, highlight = top {a.top_frac:.0%} of the image, "
                    f"tolerance {a.tol:g}px, eps_std {a.eps_std}.\n",
                    "`hit`: peak of the kept map on a part. `lift`: precision of the highlight on parts / chance. "
                    "`p_pos`, `jaccard`: stability of the highlight over weight draws. `del_drop`, `ins_gain`: "
                    "faithfulness (1 = full effect). Differences are paired over images (mean, 95% CI).\n"]
    covs = sorted({c for d in per[entries[0].label] for (_, c) in d}, reverse=True)
    for e in entries:
        lab = e.label
        md.append(f"\n## {lab}\n")
        ax = aux[lab]
        md.append("Variance ranking / confound-controlled stability: "
                  f"mc_spearman_V = {np.nanmean([d['mc_spearman_V'] for d in ax]):.3f}; "
                  f"share of top-20% pixels stable in >=90% of draws = {np.nanmean([d['frac_stable_top20'] for d in ax]):.3f}\n")
        md.append("\n| score | AUROC (stable vs unstable among top-20% relevant pixels) |\n|---|---|")
        for nm in ("variance", "snr", "var_adj", "magnitude"):
            m, s, n = mean_sem([d[f"auc_stable_{nm}"] for d in ax])
            md.append(f"| {nm} | {m:.3f} ± {1.96 * s:.3f} |")
        for q in covs:
            md.append(f"\n**coverage {q:g}**\n")
            md.append("| filter | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS))
            fl = ["all"] if q >= 1.0 else FILTERS
            for f in fl:
                vals = []
                for m_ in METRICS:
                    mm, ss, n = mean_sem([d[(f, q)][m_] for d in per[lab] if (f, q) in d])
                    vals.append(f"{mm:.3f}")
                    rows.append(dict(model=lab, filter=f, coverage=q, metric=m_, mean=mm, sem=ss, n=n))
                md.append(f"| {f} | " + " | ".join(vals) + " |")
            if q < 1.0:
                for ref in ("random", "magnitude"):
                    for f in ("variance", "snr", "var_adj"):
                        cells = []
                        for m_ in METRICS:
                            diff = [d[(f, q)][m_] - d[(ref, q)][m_] for d in per[lab]
                                    if np.isfinite(d[(f, q)][m_]) and np.isfinite(d[(ref, q)][m_])]
                            mm, ss, n = mean_sem(diff)
                            cells.append(f"{mm:+.3f}±{1.96 * ss:.3f}" if n > 1 else "–")
                        md.append(f"| {f} − {ref} | " + " | ".join(cells) + " |")
    (out / "filter_eval_summary.md").write_text("\n".join(md))
    with open(out / "filter_eval_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["model", "filter", "coverage", "metric", "mean", "sem", "n"])
        wr.writeheader()
        wr.writerows(rows)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)

    # ------------------------------------------------------------------ figure
    titles = {"hit": "pointing hit (peak on part)", "lift": "highlight precision lift",
              "p_pos": "P(highlight stays positive)", "jaccard": "Jaccard vs weight draws",
              "del_drop": "deletion: logit drop", "ins_gain": "insertion: logit gain"}
    col = {"variance": "#1f77b4", "snr": "#2ca02c", "var_adj": "#9467bd",
           "magnitude": "#ff7f0e", "random": "#7f7f7f"}
    fig, axs = plt.subplots(len(METRICS), len(entries), figsize=(4.2 * len(entries), 2.4 * len(METRICS)),
                            squeeze=False, sharex=True)
    for j, e in enumerate(entries):
        for i, m_ in enumerate(METRICS):
            ax = axs[i, j]
            for f in FILTERS:
                xs_, ys, es = [], [], []
                for q in covs:
                    key = ("all" if q >= 1.0 else f, q)
                    mm, ss, n = mean_sem([d[key][m_] for d in per[e.label] if key in d])
                    xs_.append(q); ys.append(mm); es.append(0 if not np.isfinite(ss) else ss)
                ys, es = np.array(ys), np.array(es)
                ax.plot(xs_, ys, "-o", ms=3, color=col[f], label=f)
                ax.fill_between(xs_, ys - es, ys + es, color=col[f], alpha=0.15)
            if i == 0:
                ax.set_title(e.label, fontsize=10)
            if j == 0:
                ax.set_ylabel(titles[m_], fontsize=8)
            ax.grid(alpha=0.3)
    axs[-1, 0].set_xlabel("coverage (fraction of pixels kept)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "filter_curves.png", dpi=170)
    plt.close(fig)

    print(f"\nDone in {time.time() - t0:.0f}s -> {out}/")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
