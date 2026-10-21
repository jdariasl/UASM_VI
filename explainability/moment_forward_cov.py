"""
moment_forward_cov.py
Forward moment propagation with a covariance-aware head.

Same as MomentForwardPass (which is left untouched), except at the two layers
where the diagonal closure loses most of the variance:

  * global average pooling: Var[mean_p a_p] = sum_pq rho_pq s_p s_q / P^2, with
    the spatial correlation rho of each channel estimated by HeadCovEstimator
    and the marginal std s of the analytic propagation;
  * last linear layer:  z = M a  with  Cov[a] = S^(1/2) (I + U diag(lam) U^T) S^(1/2),
    S = diag(Var[a]), so that
        Var[z_k] = sum_j (e_j sigma_kj^2 + v_j mu_kj^2)  +  sum_i lam_i (u_i^T (s * mu_k))^2 .

The records of both layers carry the information the backward pass needs
("cov_pool" / "cov_linear"); see bayesian_lrp_cov.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from explainability.moment_forward import MomentForwardPass
from explainability.head_cov import HeadStats
from utils.bnn_utils import is_bayesian_linear, get_weight_moments


def spatial_cov(corr: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """corr (C, P, P), v (C, P)  ->  per-channel covariance (C, P, P) with marginals v."""
    s = v.clamp(min=0.0).sqrt()
    return corr * s[:, :, None] * s[:, None, :]


class CovMomentForwardPass(MomentForwardPass):
    """MomentForwardPass whose global pooling and last linear layer use `head` statistics."""

    def __init__(self, model: nn.Module, head: HeadStats | None = None, pool_corr: float = 0.0):
        super().__init__(model, pool_corr=pool_corr)
        self.head = head

    def set_head(self, head: HeadStats | None):
        self.head = head

    # ------------------------------------------------------------------
    def _head_modules(self):
        if self.head is None:
            return None, None
        mods = dict(self.model.named_modules())
        pool = mods.get(self.head.pool_name) if self.head.pool_name else None
        return pool, mods[self.head.linear_name]

    def _process_leaf(self, layer, mu, v, e):
        pool, linear = self._head_modules()

        if pool is not None and layer is pool and self.head.corr_pool is not None:
            return self._pool_cov(layer, mu, v, e)

        if linear is not None and layer is linear:
            return self._linear_cov(layer, mu, v, e)

        return super()._process_leaf(layer, mu, v, e)

    # ------------------------------------------------------------------
    def _pool_cov(self, layer, mu, v, e):
        """Global average pooling with the estimated per-channel spatial correlation."""
        B, C, H, W = mu.shape
        mu_o = layer(mu)
        cov = spatial_cov(self.head.corr_pool.to(v.device), v[0].reshape(C, H * W))
        v_o = (cov.sum((1, 2)) / (H * W) ** 2).reshape(1, C, 1, 1)
        e_o = mu_o ** 2 + v_o
        self._record(layer, mu_o, v_o, e_o)
        self.layer_stats[-1]["cov_pool"] = self.head.corr_pool
        return mu_o, v_o, e_o

    def _linear_cov(self, layer, mu, v, e):
        """Last linear layer with a diagonal + low-rank input covariance."""
        mu_o, v_o, e_o = super()._process_leaf(layer, mu, v, e)
        rec = self.layer_stats[-1]
        if rec["layer"] is not layer:               # layer could not be processed
            return mu_o, v_o, e_o

        v_in = torch.flatten(v, 1)
        mu_w = get_weight_moments(layer)[0] if is_bayesian_linear(layer) else layer.weight
        U = self.head.U.to(mu_w.device)
        lam = self.head.lam.to(mu_w.device)

        s = v_in.clamp(min=0.0).sqrt()                       # (B, D)
        proj = U.T @ (s[0][:, None] * mu_w.T)                # (k, out)
        v_o = v_o + (lam[:, None] * proj ** 2).sum(0)[None]
        e_o = mu_o ** 2 + v_o

        rec["v"], rec["e"] = v_o.detach(), e_o.detach()
        rec["cov_linear"] = (U, lam)
        return mu_o, v_o, e_o
