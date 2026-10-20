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
def variants(kind, module, src, A, n_shuffle, gen):
    """Per-element output variances of every variant (dict name -> tensor)."""
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
    return res


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
                res = variants(kind, mod, src, A, a.n_shuffle, gen)
                row = {}
                tot = float(v_mc.sum())
                for k, v in res.items():
                    row[k] = float(v.sum()) / max(tot, 1e-30)
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
                      f"space {row['space']:.3f}", flush=True)
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
        sub = [r for r in rows if "diag_t" in r]
        if sub:
            r = sub[0]
            print(f"target logit at {r['layer']}: full {r['full_t']:.2f}  diag {r['diag_t']:.3f}  "
                  f"chan {r['chan_t']:.3f}  space {r['space_t']:.3f}")
    restore_sigma(model, saved)
    print(f"CSV files in {out}/")


if __name__ == "__main__":
    main()
