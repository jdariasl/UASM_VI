"""
diag_cov.py
Which covariance does the closed-form moment propagation miss?   (oracle ablation)

For every selected Conv2d / Linear / AvgPool layer L of a trained VI-BNN the script
measures, with *exact* input statistics taken from Monte-Carlo weight draws, how much
of the output variance of L each covariance structure of its input explains.  Because
the weights of L are independent of its input (they are independent of all earlier
weights), the output variance of a linear layer is EXACT given the input samples a_s:

    Var[z] =  Var_s[ M a_s ]                       (spread of the mean-weight output)
            + sum_i sigma_i^2 * E[a_i^2]           (own weight noise; same for all variants)

and the first term is the only one that depends on the input covariance.  Variants
(all use the same MC input samples, only the correlation between inputs is changed by
permuting the draw index of the input, which keeps every marginal exactly):

    full    no change                        -> exact; must reproduce the MC output variance
    diag    analytic  sum_i M_i^2 Var[a_i]   -> what moment_forward.py assumes
            (diag_shuf: the same by shuffling every element independently; sanity check)
    chan    draws permuted independently per CHANNEL
            -> keeps correlation between positions of the same channel (spatial),
               removes correlation between channels
    space   draws permuted independently per POSITION
            -> keeps correlation between channels at the same position,
               removes correlation between positions

Reading the table (columns are ratios to the Monte-Carlo output variance, 1 = exact):
    full ~ 1                    the machinery is right
    diag << 1                   this layer, FED WITH EXACT INPUT MARGINALS, still loses
                                variance: the diagonal assumption is wrong right here
    chan  >> diag               spatial correlation inside a channel explains the gap
    space >> diag               cross-channel correlation explains the gap
    gap_frac_chan / gap_frac_space = (variant - diag) / (full - diag)
                                (they need not add up to 1; both can be large if the
                                gap needs both kinds of correlation)
    A layer with diag ~ 1 only appears to lose variance in diag_var.py because of
    what it inherits from earlier layers.

    sh      shared channel correlation: synthetic Gaussian draws with the exact per-element
            marginals, NO correlation between positions, and ONE channel-correlation matrix
            (C x C, the mean over positions of the per-position correlation matrices) used at
            every position.  If sh ~ space, the cross-channel covariance is stationary over
            positions and can be stored as a single C x C matrix per layer.
    sh_lr<k>  the same with that matrix replaced by its top-k eigen-directions + a diagonal
            (is the shared channel covariance itself low-rank?)
    lr<k>   (--ranks) diagonal + top-k principal components of the full input covariance.

For a Linear layer the 'channels' are the input features and there is no spatial
dimension: 'chan' = diag and 'space' = full.  For the last Linear layer the ratios of the
target logit (largest mean logit) are reported too (columns *_t).

Usage
-----
    python diag_cov.py --config configs/cub200_resnet50.yaml \\
        --checkpoint checkpoints/cub200_resnet50_reparam/best_model.pt \\
        --n_images 2 --n_mc 60 --every 3 --scales 1.0

    # all layers, only the bottleneck convolutions of stage 3
    python diag_cov.py ... --every 1 --layers "layer3\\..*conv"

Outputs (in --outdir, default results/diag_cov):
    cov_oracle_scale<s>.csv     one row per layer, averaged over images
"""

from __future__ import annotations

import argparse
import csv
import re
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data.cub200 import CUB200Dataset
from utils.bnn_utils import SampledTwin, get_weight_moments, conv_kwargs
from diag_var import load_model, scale_sigma, restore_sigma


# --------------------------------------------------------------------------- #
#  Layer list
# --------------------------------------------------------------------------- #

def list_layers(twin: SampledTwin):
    """[(name, kind, twin_module, bayesian_source or None)] in module order."""
    src_of = {id(dst): src for src, dst in twin.pairs}
    out = []
    for name, m in twin.net.named_modules():
        if id(m) in src_of:
            out.append((name, "conv" if isinstance(m, nn.Conv2d) else "linear", m, src_of[id(m)]))
        elif isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            out.append((name, "pool", m, None))
    return out


# --------------------------------------------------------------------------- #
#  MC capture of the input samples and the output variance of ONE layer
# --------------------------------------------------------------------------- #

@torch.no_grad()
def capture(twin: SampledTwin, module, x, n_mc):
    ins, st = [], dict(n=0, mean=None, m2=None)

    def pre(_m, inp):
        ins.append(inp[0].detach().clone())

    def post(_m, _i, o):
        o = o.detach().float()
        st["n"] += 1
        if st["mean"] is None:
            st["mean"] = torch.zeros_like(o)
            st["m2"] = torch.zeros_like(o)
        d = o - st["mean"]
        st["mean"] += d / st["n"]
        st["m2"] += d * (o - st["mean"])

    h1 = module.register_forward_pre_hook(pre)
    h2 = module.register_forward_hook(post)
    try:
        for _ in range(n_mc):
            twin.resample()
            twin.net(x)
    finally:
        h1.remove()
        h2.remove()
    A = torch.cat(ins, 0).float()                         # (S, C, H, W) or (S, D)
    return A, st["mean"], st["m2"] / max(st["n"] - 1, 1)


# --------------------------------------------------------------------------- #
#  Correlation-structure surgery by permuting the draw index
# --------------------------------------------------------------------------- #

def shuffled(A: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
    S, shp = A.shape[0], list(A.shape)
    if mode == "diag":                                     # independent per element
        key = shp
    elif mode == "chan":                                   # independent per channel
        key = [S, shp[1]] + [1] * (A.dim() - 2)
    elif mode == "space":                                  # independent per position
        key = [S, 1] + shp[2:]
    else:
        raise ValueError(mode)
    idx = torch.rand(key, generator=gen, device=A.device).argsort(0).expand(shp)
    return A.gather(0, idx)


@torch.no_grad()
def shared_channel_corr(A: torch.Tensor):
    """Mean over positions of the per-position channel correlation matrix, (C, C), unit diagonal."""
    S, C = A.shape[0], A.shape[1]
    Ac = A - A.mean(0, keepdim=True)
    sd = Ac.std(0, unbiased=True, keepdim=True)
    Z = torch.where(sd > 1e-6, Ac / sd.clamp(min=1e-6), torch.zeros_like(Ac))
    Zf = Z.permute(0, 2, 3, 1).reshape(-1, C)
    R = (Zf.T @ Zf) / max(Zf.shape[0], 1) * (S / max(S - 1, 1))
    d = R.diagonal().clamp(min=1e-8).sqrt()
    return (R / (d[:, None] * d[None, :])).double(), sd


@torch.no_grad()
def shared_variants(A, lin_var, own, ranks, n_rep, gen):
    """Variances under 'one channel-correlation matrix for all positions, no spatial correlation'."""
    S, C = A.shape[0], A.shape[1]
    R, sd = shared_channel_corr(A)
    lam, V = torch.linalg.eigh(R)
    lam, V = lam.flip(0).clamp(min=0), V.flip(1)
    out, ev = {}, {}

    def run(kind_k):
        tot = 0.0
        for _ in range(n_rep):
            if kind_k is None:                                   # full shared matrix
                Lf = (V * lam.sqrt()).float()
                xi = torch.randn(S, C, *A.shape[2:], generator=gen, device=A.device)
                eta = F.conv2d(xi, Lf[:, :, None, None])
            else:                                                # top-k + diagonal remainder
                k = kind_k
                Lk = (V[:, :k] * lam[:k].sqrt()).float()
                dres = (1.0 - (V[:, :k] ** 2 * lam[:k]).sum(1)).clamp(min=0).sqrt().float()
                xi1 = torch.randn(S, k, *A.shape[2:], generator=gen, device=A.device)
                xi2 = torch.randn(S, C, *A.shape[2:], generator=gen, device=A.device)
                eta = F.conv2d(xi1, Lk[:, :, None, None]) + dres.view(1, -1, 1, 1) * xi2
            tot = tot + lin_var(sd * eta)
        return tot / n_rep + own

    out["sh"] = run(None)
    for k in ranks:
        k = int(min(k, C - 1, S - 2))
        out[f"sh_lr{k}"] = run(k)
        ev[f"evsh{k}"] = float(lam[:k].sum() / lam.sum().clamp(min=1e-12))
    return out, ev


@torch.no_grad()
def variants(kind, module, src, A, n_shuffle, gen, ranks=()):
    """Per-element output variances of every variant (dict name -> tensor).

    ranks: for each k, 'lr<k>' = diag + the top-k principal components of the input covariance:
    the draws are split into their rank-k PCA part (kept) and the residual, whose elements
    are then shuffled independently (marginals kept, residual correlations removed).
    Returns (res, ev) where ev['ev<k>'] is the share of the total input variance in the top-k PCs.
    """
    S = A.shape[0]
    ex2 = (A ** 2).mean(0, keepdim=True)
    var_a = A.var(0, unbiased=True, keepdim=True)

    if kind == "pool":
        op = lambda t: module(t)
        own = 0.0
        n_win = A[0].numel() / max(op(A[:1])[0].numel(), 1)
        diag = op(var_a)[0] / n_win                        # sum_i v_i / n^2  for uniform windows
        lin_var = lambda Aq: op(Aq).var(0, unbiased=True)
    else:
        mw, sw, mb, sb = get_weight_moments(src)
        mw, s2 = mw.detach(), sw.detach() ** 2
        if kind == "conv":
            kw = conv_kwargs(module)
            op = lambda t, w: F.conv2d(t, w, None, **kw)
        else:
            op = lambda t, w: F.linear(t, w)
        own = op(ex2, s2)[0]
        if sb is not None:
            own = own + (sb.detach() ** 2).view(-1, *([1] * (own.dim() - 1)))
        diag = op(var_a, mw ** 2)[0] + own
        lin_var = lambda Aq: op(Aq, mw).var(0, unbiased=True)

    def avg(mode):
        return sum(lin_var(shuffled(A, mode, gen)) for _ in range(n_shuffle)) / n_shuffle

    res = {"full": lin_var(A) + own,
           "diag": diag,
           "diag_shuf": avg("diag") + own,
           "chan": avg("chan") + own,
           "space": avg("space") + own}

    ev = {}
    if A.dim() == 4:
        sres, sev = shared_variants(A, lin_var, own, ranks, n_shuffle, gen)
        res.update(sres)
        ev.update(sev)
    if ranks:
        Ac = (A - A.mean(0, keepdim=True)).reshape(S, -1)
        G = (Ac @ Ac.T).double()
        w, U = torch.linalg.eigh(G)                         # ascending
        w, U = w.flip(0).clamp(min=0), U.flip(1).float()
        tot = float(w.sum()) + 1e-300
        for k in ranks:
            k = int(min(k, S - 2))
            ev[f"ev{k}"] = float(w[:k].sum()) / tot
            Uk = U[:, :k]
            L = (Uk @ (Uk.T @ Ac)).reshape(A.shape)
            Rk = (A - A.mean(0, keepdim=True)) - L
            acc_ = 0.0
            for _ in range(n_shuffle):
                Aq = L + shuffled(Rk, "diag", gen)
                acc_ = acc_ + lin_var(Aq)
            res[f"lr{k}"] = acc_ / n_shuffle + own
            del L, Rk, Aq
    return res, ev


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_images", type=int, default=2)
    ap.add_argument("--n_mc", type=int, default=60, help="weight draws per image and layer")
    ap.add_argument("--n_shuffle", type=int, default=3, help="shuffles averaged per structure")
    ap.add_argument("--every", type=int, default=3, help="use every k-th layer (1 = all)")
    ap.add_argument("--layers", default=None, help="regex on the layer name (applied before --every)")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 5, 20, 50],
                    help="ranks k of the low-rank-plus-diagonal oracle (empty = skip); k is capped at n_mc-2, "
                         "so use --n_mc >= 4k for meaningful numbers")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--outdir", default="results/diag_cov")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if cfg.get("dataset", "cub200") != "cub200":
        raise SystemExit("diag_cov.py currently supports the cub200 dataset only")
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model, est = load_model(cfg, a.checkpoint, dev)
    print(f"Loaded {a.checkpoint}  ({est}, backbone {cfg['backbone']})")
    twin = SampledTwin(model)
    layers = list_layers(twin)
    if a.layers:
        layers = [l for l in layers if re.search(a.layers, l[0])]
    layers = layers[::max(1, a.every)]
    last_linear = [l[0] for l in list_layers(twin) if l[1] == "linear"][-1]
    print(f"{len(layers)} layers selected")

    ds = CUB200Dataset(root=cfg["data_root"], split="test")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n_images)
    xs = [ds[i][0].unsqueeze(0).to(dev) for i in idx]

    saved = {}
    for s in a.scales:
        scale_sigma(model, s, saved)
        acc = {}
        order = []
        for ii, x in enumerate(xs):
            for name, kind, mod, src in layers:
                A, _, v_mc = capture(twin, mod, x, a.n_mc)
                if float(A.var(0).max()) < 1e-10:
                    continue                                  # deterministic input (e.g. the stem)
                if kind == "pool":
                    with torch.no_grad():
                        if mod(A[:1]).shape == A[:1].shape:
                            continue                          # identity pooling
                res, ev = variants(kind, mod, src, A, a.n_shuffle, gen, a.ranks)
                row = dict(ev)
                tot = float(v_mc.sum())
                for k, v in res.items():
                    row[k] = float(v.sum()) / max(tot, 1e-30)
                lrs = " ".join(f"lr{k} {row[f'lr{k}']:.3f}" for k in a.ranks if f"lr{k}" in row)
                if name == last_linear:
                    with torch.no_grad():
                        mean_logit = F.linear(A.mean(0, keepdim=True), get_weight_moments(src)[0].detach())
                    t = int(mean_logit[0].argmax())
                    for k, v in res.items():
                        row[k + "_t"] = float(v[t]) / max(float(v_mc[0, t]), 1e-30)
                acc.setdefault(name, []).append(row)
                if name not in order:
                    order.append(name)
                print(f"  scale {s:g} img {ii + 1}/{len(xs)} {name:34s} "
                      f"full {row['full']:.2f} diag {row['diag']:.3f} chan {row['chan']:.3f} "
                      f"space {row['space']:.3f} sh {row.get('sh', float('nan')):.3f} {lrs}", flush=True)
                del A, v_mc, res
                torch.cuda.empty_cache()

        rows = []
        for nm in order:
            L = acc[nm]
            r = dict(layer=nm, kind=next(l[1] for l in layers if l[0] == nm))
            for k in L[0]:
                r[k] = float(np.mean([d[k] for d in L]))
            gap = r["full"] - r["diag"]
            for tag in ("chan", "space"):
                r[f"gap_frac_{tag}"] = (r[tag] - r["diag"]) / gap if gap > 0.05 else float("nan")
            rows.append(r)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("layer", "kind"), k))
        with open(out / f"cov_oracle_scale{s:g}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

        print(f"\n=== sigma scale {s:g}: output variance of one layer fed with exact MC input moments, "
              f"ratio to MC (1 = exact), mean of {len(xs)} images ===")
        print(f"{'layer':32s} {'kind':6s} {'full':>6s} {'diag':>7s} {'chan':>7s} {'space':>7s} "
              f"{'gap%chan':>9s} {'gap%space':>10s}")
        for r in rows:
            print(f"{r['layer'][:32]:32s} {r['kind']:6s} {r['full']:6.2f} {r['diag']:7.3f} {r['chan']:7.3f} "
                  f"{r['space']:7.3f} {r['gap_frac_chan']:9.2f} {r['gap_frac_space']:10.2f}")
        print("\nfull ~ 1 validates the method.  diag << 1 = the diagonal assumption loses variance AT this layer.\n"
              "chan > diag: correlation between positions of one channel explains it; space > diag: correlation\n"
              "between channels at one position explains it.  (Linear layers: chan = diag, space = full.)")
        lr_keys = [f"lr{min(k, a.n_mc - 2)}" for k in a.ranks]
        if lr_keys and lr_keys[0] in rows[0]:
            print("\nLow-rank oracle: diag + top-k principal components of the input covariance "
                  "(ratio to MC; ev = share of input variance in those k PCs)")
            print(f"{'layer':32s} {'diag':>7s} " + " ".join(f"{k:>7s} {'ev' + k[2:]:>6s}" for k in lr_keys)
                  + f" {'full':>6s}")
            for r in rows:
                print(f"{r['layer'][:32]:32s} {r['diag']:7.3f} "
                      + " ".join(f"{r[k]:7.3f} {r['ev' + k[2:]]:6.2f}" for k in lr_keys)
                      + f" {r['full']:6.2f}")
            print("A small k that brings lrK close to 1 means the missing covariance is low-rank "
                  "(carry K tangent vectors).  PCs are fitted on the same draws: use n_mc >> k.")
        sh_keys = [f"sh_lr{min(k, a.n_mc - 2)}" for k in a.ranks]
        sh_rows = [r for r in rows if "sh" in r]
        if sh_rows:
            print("\nShared channel correlation (one C x C matrix for all positions, no spatial correlation); "
                  "sh_lr<k>: that matrix reduced to k directions + diagonal")
            print(f"{'layer':32s} {'diag':>7s} {'space':>7s} {'sh':>7s} "
                  + " ".join(f"{k:>9s}" for k in sh_keys if k in sh_rows[0]))
            for r in sh_rows:
                print(f"{r['layer'][:32]:32s} {r['diag']:7.3f} {r['space']:7.3f} {r['sh']:7.3f} "
                      + " ".join(f"{r[k]:9.3f}" for k in sh_keys if k in r))
            print("sh ~ space: the cross-channel covariance is stationary over positions (one C x C matrix per layer "
                  "suffices); sh_lr<k> ~ sh: that matrix is low-rank.")
        sub = [r for r in rows if "diag_t" in r]
        if sub:
            r = sub[0]
            print(f"target logit at {r['layer']}: full {r['full_t']:.2f}  diag {r['diag_t']:.3f}  "
                  f"chan {r['chan_t']:.3f}  space {r['space_t']:.3f}  "
                  + "  ".join(f"{k} {r[k + '_t']:.3f}" for k in lr_keys if k + "_t" in r))
    restore_sigma(model, saved)
    print(f"CSV files in {out}/")


if __name__ == "__main__":
    main()
