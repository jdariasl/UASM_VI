"""
head_cov.py
Covariance statistics of the network head, estimated from a few Monte-Carlo
weight draws, for the covariance-aware moment propagation (moment_forward_cov.py).

The diagonal closure of MomentForwardPass loses most of the variance in the two
last layers (see diag_cov.py):

  * global average pooling  - the activations of one channel are correlated
    across spatial positions (they share the conv weights);
  * last linear layer       - the pooled features share a few common modes of
    variation across weight draws (low-rank cross-feature covariance).

Only the CORRELATION structure is taken from the draws.  The marginal variances
stay those of the analytic propagation, so the correction isolates the effect of
the ignored covariances instead of re-measuring the loss inherited from the body.

    corr_pool[c, p, q]   spatial correlation of channel c between positions p, q
    U, lam               R_f ~= I + U diag(lam) U^T, the correlation matrix of the
                         last-linear input features (lam = eigenvalue - 1 >= 0 of
                         the k leading directions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from utils.bnn_utils import SampledTwin, is_bayesian_linear


@dataclass
class HeadStats:
    pool_name: Optional[str]            # None: no pooling layer needs a correction
    linear_name: str
    corr_pool: Optional[torch.Tensor]   # (C, P, P)
    U: torch.Tensor                     # (D, k)  orthonormal directions
    lam: torch.Tensor                   # (k,)    eigenvalue - 1, >= 0
    n_mc: int


def find_head(model: nn.Module):
    """Names of the global average pool (or None) and of the last Linear layer."""
    pool_name = linear_name = None
    for name, m in model.named_modules():
        if isinstance(m, nn.AdaptiveAvgPool2d) and _is_global(m):
            pool_name = name
        if is_bayesian_linear(m) or isinstance(m, nn.Linear):
            linear_name = name
    if linear_name is None:
        raise ValueError("model has no Linear layer")
    return pool_name, linear_name


def _is_global(pool: nn.AdaptiveAvgPool2d) -> bool:
    o = pool.output_size
    o = (o, o) if isinstance(o, int) else tuple(o)
    return o == (1, 1)


class HeadCovEstimator:
    """
    Estimates HeadStats for an image from `n_mc` weight draws of the posterior.

    rank   : number of leading directions kept for the feature covariance
             (capped at n_mc - 2).
    """

    def __init__(self, model: nn.Module, n_mc: int = 60, rank: int = 20):
        self.model = model
        self.n_mc = int(n_mc)
        self.rank = int(rank)
        self.twin = SampledTwin(model)
        self.pool_name, self.linear_name = find_head(model)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> HeadStats:
        net = self.twin.net
        mods = dict(net.named_modules())
        pool_in, lin_in = [], []
        handles = [mods[self.linear_name].register_forward_pre_hook(
            lambda _m, i: lin_in.append(i[0].detach().flatten(1).float().clone()))]
        if self.pool_name is not None:
            handles.append(mods[self.pool_name].register_forward_pre_hook(
                lambda _m, i: pool_in.append(i[0].detach().float().clone())))
        try:
            for _ in range(self.n_mc):
                self.twin.resample()
                net(x)
        finally:
            for h in handles:
                h.remove()

        F_ = torch.cat(lin_in, 0)                                  # (S, D)
        corr_pool = None
        pool_name = self.pool_name
        if pool_name is not None:
            A = torch.cat(pool_in, 0)                              # (S, C, H, W)
            if A.shape[-2] * A.shape[-1] > 1:
                corr_pool = _spatial_corr(A)
            else:
                pool_name = None
        U, lam = _lowrank_corr(F_, self.rank)
        return HeadStats(pool_name, self.linear_name, corr_pool, U, lam, self.n_mc)


def _standardise(A: torch.Tensor, dim_keep: int = 0):
    """Centre over the draw axis and divide by the sample std (zero where it vanishes)."""
    Ac = A - A.mean(0, keepdim=True)
    sd = Ac.std(0, unbiased=True, keepdim=True)
    Z = torch.where(sd > 1e-6, Ac / sd.clamp(min=1e-6), torch.zeros_like(Ac))
    return Z


def _spatial_corr(A: torch.Tensor) -> torch.Tensor:
    """(S, C, H, W) -> (C, P, P) per-channel correlation between positions; unit diagonal."""
    S, C = A.shape[:2]
    Z = _standardise(A).reshape(S, C, -1)                          # (S, C, P)
    R = torch.einsum("scp,scq->cpq", Z, Z) / max(S - 1, 1)
    P = R.shape[-1]
    eye = torch.eye(P, device=R.device, dtype=torch.bool)
    return torch.where(eye, torch.ones_like(R), R)


def _lowrank_corr(F_: torch.Tensor, rank: int):
    """Leading eigen-directions of the feature correlation matrix, (D, k) and (k,)."""
    S, D = F_.shape
    Z = _standardise(F_)                                           # (S, D)
    k = max(1, min(rank, S - 2, D))
    # eigen-decomposition of R = Z^T Z / (S-1) through the SVD of the thin matrix Z
    _, sv, Vh = torch.linalg.svd(Z, full_matrices=False)
    ev = sv ** 2 / max(S - 1, 1)
    lam = (ev[:k] - 1.0).clamp(min=0.0)
    return Vh[:k].T.contiguous(), lam
