"""
bayesian_lrp_cov.py
Bayesian LRP-eps with a covariance-aware head.

BayesianLRP (bayesian_lrp.py) is left untouched.  This subclass

  1. estimates, per explained image, the correlation structure of the head from a
     few Monte-Carlo weight draws (head_cov.HeadCovEstimator),
  2. runs the forward moments with CovMomentForwardPass (corrected variance at the
     global average pool and at the last linear layer), and
  3. propagates the relevance variance backward through those two layers with the
     first-order (delta-method) expression that keeps the covariance terms the
     diagonal version drops.

Last linear layer, for each output k that carries relevance R_k:
    R_j = A_j / z_hat * R_k,   A_j = a_j w_kj,   z = sum_i A_i
    Var[R_j] ~= (A_j/z_hat)^2 Var[R_k]
              + R_k^2/z_hat^2 * ( Var[A_j] - 2 (A_j/z_hat) Cov[A_j, z] + (A_j/z_hat)^2 Var[z] )
with Cov[a] = S^(1/2)(I + U diag(lam) U^T)S^(1/2) (a independent of w):
    Var[A_j]   = e_j sigma_kj^2 + v_j mu_kj^2
    Cov[A_j,z] = e_j sigma_kj^2 + mu_kj (Cov[a] mu_k)_j
    Var[z]     = sum_j e_j sigma_kj^2 + mu_k^T Cov[a] mu_k

Global average pooling, per channel (c_p = a_p / T, T = P z_hat):
    Var[c_p] ~= (Sigma_pp - 2 (a_p/T) (Sigma 1)_p + (a_p/T)^2 1^T Sigma 1) / T^2
    Var[R_p]  = c_p^2 Var[R] + Var[c_p] (Var[R] + E[R]^2)

The z_hat and Var[c_p] terms are first-order corrections that the diagonal
version omits (leading order); they are present here even for a diagonal covariance.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from explainability.bayesian_lrp import (
    BayesianLRP, EPS_DEFAULT, _stabilise, _lrp_conv_step, _lrp_linear_step,
    _lrp_pool_step, _lrp_identity_step,
)
from explainability.head_cov import HeadCovEstimator
from explainability.moment_forward_cov import CovMomentForwardPass, spatial_cov
from utils.bnn_utils import is_bayesian_conv, is_bayesian_linear, get_weight_moments


# ------------------------------------------------------------- backward steps --

def _lrp_linear_step_cov(mu_a, v_a, e_a, mu_R, v_R, layer, eps, eps_std, U, lam
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """LRP-eps backward step of a Linear layer with input covariance S^.5 (I + U lam U^T) S^.5."""
    if is_bayesian_linear(layer):
        mu_w, sigma_w, mu_b, _ = get_weight_moments(layer)
        sigma_w2 = sigma_w ** 2
    else:
        mu_w, mu_b = layer.weight, layer.bias
        sigma_w2 = torch.zeros_like(mu_w)
    U, lam = U.to(mu_w.device), lam.to(mu_w.device)

    s = v_a[0].clamp(min=0.0).sqrt()                        # (in,)
    e = e_a[0]
    mu_z = F.linear(mu_a, mu_w, mu_b)[0]                    # (out,)
    proj = U.T @ (s[:, None] * mu_w.T)                      # (k, out)
    var_z = e @ sigma_w2.T + v_a[0] @ (mu_w ** 2).T + (lam[:, None] * proj ** 2).sum(0)
    hat_z = _stabilise(mu_z, eps, var_z, eps_std)           # (out,)

    mu_R_in = mu_a[0] * (mu_w.T @ (mu_R[0] / hat_z))        # (in,)
    v_R_in = torch.zeros_like(mu_R_in)

    active = ((mu_R[0] != 0) | (v_R[0] != 0)).nonzero().flatten()
    for k in active.tolist():
        w, sg2, zh = mu_w[k], sigma_w2[k], hat_z[k]
        A = mu_a[0] * w
        cov_a_w = v_a[0] * w + s * (U @ (lam * (U.T @ (s * w))))   # Cov[a] mu_k
        var_A = e * sg2 + v_a[0] * w ** 2
        cov_Az = e * sg2 + w * cov_a_w
        q = A / zh
        t_z = (var_A - 2.0 * q * cov_Az + q ** 2 * var_z[k]) * mu_R[0, k] ** 2 / zh ** 2
        t_R = q ** 2 * v_R[0, k]
        v_R_in = v_R_in + (t_R + t_z).clamp(min=0.0)

    return mu_R_in[None], v_R_in[None]


def _lrp_pool_step_cov(mu_a, v_a, mu_R, v_R, corr, eps
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """LRP-eps backward step of a global average pool with per-channel spatial covariance."""
    _, C, H, W = mu_a.shape
    P = H * W
    a = mu_a[0].reshape(C, P)
    T = P * _stabilise(a.mean(1), eps)                       # (C,)
    c = a / T[:, None]

    Sig = spatial_cov(corr.to(a.device), v_a[0].reshape(C, P))
    S1 = Sig.sum(2)                                          # (C, P)
    tot = S1.sum(1)                                          # (C,)
    var_c = (torch.diagonal(Sig, dim1=1, dim2=2) - 2.0 * c * S1 + c ** 2 * tot[:, None]) \
        / T[:, None] ** 2
    var_c = var_c.clamp(min=0.0)

    mR, vR = mu_R.reshape(C, 1), v_R.reshape(C, 1)
    mu_R_in = c * mR
    v_R_in = (c ** 2 * vR + var_c * (vR + mR ** 2)).clamp(min=0.0)
    return mu_R_in.reshape(1, C, H, W), v_R_in.reshape(1, C, H, W)


# ------------------------------------------------------------------ main class --

class BayesianLRPCov(BayesianLRP):
    """
    Bayesian LRP-eps with a covariance-aware head.  Same interface as BayesianLRP:

        blrp = BayesianLRPCov(model, eps=1e-6, eps_std=0.25, n_mc=60, rank=20)
        mu_R, v_R, snr = blrp.explain(x, target_class)

    n_mc : weight draws used per explained image to estimate the head correlations
    rank : leading directions kept for the cross-feature covariance of the last layer
    """

    def __init__(self, model: nn.Module, eps: float = EPS_DEFAULT,
                 eps_std: float = 0.0, fuse_bn: bool = True,
                 n_mc: int = 60, rank: int = 20):
        super().__init__(model, eps=eps, eps_std=eps_std, fuse_bn=fuse_bn)
        self.mp = CovMomentForwardPass(model)
        self.estimator = HeadCovEstimator(model, n_mc=n_mc, rank=rank)
        self.last_head = None

    def explain(self, x: torch.Tensor, target_class: int):
        self.last_head = self.estimator.fit(x)
        self.mp.set_head(self.last_head)
        return super().explain(x, target_class)

    # Same loop as BayesianLRP._backward plus the two covariance-aware layers.
    def _backward(self, records, inp, mu_R, v_R):
        eps, eps_std = self.eps, self.eps_std
        i = len(records) - 1
        while i >= 0:
            rec   = records[i]
            layer = rec["layer"]
            prev  = records[i - 1] if i > 0 else inp

            if rec.get("kind") == "residual":
                mu_R, v_R = self._backward_residual(rec, prev, mu_R, v_R)
                i -= 1
                continue

            if (self.fuse_bn and isinstance(layer, nn.BatchNorm2d) and i > 0
                    and records[i - 1].get("kind") != "residual"
                    and (is_bayesian_conv(records[i - 1]["layer"])
                         or isinstance(records[i - 1]["layer"], nn.Conv2d))):
                conv  = records[i - 1]["layer"]
                prev2 = records[i - 2] if i > 1 else inp
                mu_R, v_R = _lrp_conv_step(prev2["mu"], prev2["v"], prev2["e"],
                                           mu_R, v_R, conv, eps, eps_std, bn=layer)
                i -= 2
                continue

            mu_in, v_in, e_in = prev["mu"], prev["v"], prev["e"]

            if "cov_linear" in rec:
                U, lam = rec["cov_linear"]
                mu_R, v_R = _lrp_linear_step_cov(mu_in, v_in, e_in, mu_R, v_R, layer,
                                                 eps, eps_std, U, lam)
            elif "cov_pool" in rec:
                mu_R, v_R = _lrp_pool_step_cov(mu_in, v_in, mu_R, v_R, rec["cov_pool"], eps)

            elif is_bayesian_linear(layer) or isinstance(layer, nn.Linear):
                mu_R, v_R = _lrp_linear_step(mu_in, v_in, e_in, mu_R, v_R, layer, eps, eps_std)

            elif is_bayesian_conv(layer) or isinstance(layer, nn.Conv2d):
                mu_R, v_R = _lrp_conv_step(mu_in, v_in, e_in, mu_R, v_R, layer, eps, eps_std)

            elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                mu_R, v_R = _lrp_pool_step(mu_in, mu_R, v_R, layer, eps)

            else:
                mu_R, v_R = _lrp_identity_step(mu_R, v_R)
                if isinstance(layer, nn.Flatten) and mu_in.dim() > mu_R.dim():
                    mu_R = mu_R.reshape(mu_in.shape)
                    v_R  = v_R.reshape(mu_in.shape)
            i -= 1
        return mu_R, v_R
