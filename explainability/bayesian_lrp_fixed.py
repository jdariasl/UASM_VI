"""
bayesian_lrp_fixed.py
LRP-eps with a FIXED output denominator ("logit-initialised" relevance).

Why
---
BayesianLRP starts with relevance sign(f) on the explained logit, so that at the
last layer  R_j = A_j / z  with z the (random) logit and A_j = a_j w_j.  When the
spread of the logit over the weights is comparable to its mean (std 1.9 against mean
2.0 in a trained ResNet-50) this ratio is very non-linear: its variance is dominated by
the draws with z near zero, a first-order (delta-method) expansion cannot describe it,
and the estimated Var[R] depends mostly on the stabiliser.

Standard LRP starts with R_t = f_t, which cancels the denominator (R_j = A_j).  Here the
relevance is the same quantity divided by the CONSTANT d = |E[f_t]| (the posterior-mean
logit), so the scale stays comparable between images:

        R_t = f_t / d        =>        R_j = A_j / d        (last layer)

Given the weights the relevance of the last layer is then LINEAR in the activations and
the head weights, and its moments are exact for the marginals:

        E[R_j]   = mu_j mu_wj / d
        Var[R_j] = ( e_j sigma_wj^2 + v_j mu_wj^2 ) / d^2

For a network with global average pooling in front of the last Linear layer the pooled
activation also cancels:  R_{c,p} = a_{c,p} w_c / (P d), with
        E = mu_p mu_wc / (P d),   Var = ( e_p sigma_wc^2 + v_p mu_wc^2 ) / (P d)^2 ,
which needs only the marginal moments of the last feature map; the cross-feature and the
spatial covariances of the head do not enter the marginal Var[R] there.

Everything below the head (the convolutions, residual adds, ...) is unchanged
(BayesianLRP steps: R_k and hat_z_k are still treated as independent in every layer).

Classes
-------
BayesianLRPFixedDen   closed form for a Bayesian model (head step as above).
LRPFixedDenRef        the matching Monte-Carlo reference for a deterministic weight draw:
                      initial relevance f_draw / d  with d passed in (use the mean-weight
                      logit); run it on SampledTwin.net with eps_std = 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from explainability.bayesian_lrp import (
    BayesianLRP, EPS_DEFAULT, _lrp_conv_step, _lrp_linear_step, _lrp_pool_step,
    _lrp_identity_step,
)
from utils.bnn_utils import is_bayesian_conv, is_bayesian_linear, get_weight_moments


def _is_global_pool(layer) -> bool:
    if not isinstance(layer, nn.AdaptiveAvgPool2d):
        return False
    o = layer.output_size
    o = (o, o) if isinstance(o, int) else tuple(o)
    return o == (1, 1)


def _head_row(layer, t):
    """(mean, variance) of the weight row of the explained output."""
    if is_bayesian_linear(layer):
        mu_w, sigma_w, _, _ = get_weight_moments(layer)
        return mu_w[t], sigma_w[t] ** 2
    return layer.weight[t], torch.zeros_like(layer.weight[t])


# --------------------------------------------------------------------------- #

class BayesianLRPFixedDen(BayesianLRP):
    """Same interface as BayesianLRP; fixed output denominator at the last layer."""

    def explain(self, x: torch.Tensor, target_class: int):
        with torch.no_grad():
            stats = self.mp.run(x)
        last = stats[-1]
        mu_z = float(last["mu"][0, target_class])
        self.last_sign = 1.0 if mu_z >= 0 else -1.0
        self.last_denominator = max(abs(mu_z), 1e-12)
        self._target = target_class
        self._ratio = None
        self.last_stats = stats
        mu_R, v_R = self._backward(stats[1:], stats[0], None, None, top=True)
        snr = mu_R / torch.sqrt(v_R + 1e-12)
        return mu_R, v_R, snr

    # Same loop as BayesianLRP._backward, plus the head step (top level only).
    def _backward(self, records, inp, mu_R, v_R, top: bool = False):
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
                self._ratio = None
                i -= 2
                continue

            mu_in, v_in, e_in = prev["mu"], prev["v"], prev["e"]
            is_linear = is_bayesian_linear(layer) or isinstance(layer, nn.Linear)

            if top and i == len(records) - 1 and is_linear:
                # ---- head: R_j = A_j / d, linear in a and w ----
                mu_w, s2 = _head_row(layer, self._target)
                d = self.last_denominator
                mu_r, var_r = mu_w / d, s2 / d ** 2                       # ratio r_j = w_j / d
                mu_R = (mu_in[0] * mu_r)[None]
                v_R = (e_in[0] * var_r + v_in[0] * mu_r ** 2).clamp(min=0.0)[None]
                self._ratio = (mu_r, var_r)

            elif self._ratio is not None and _is_global_pool(layer) and mu_in.dim() == 4 \
                    and mu_in.shape[1] == self._ratio[0].numel():
                # ---- global average pool: the pooled activation cancels, R_cp = a_cp r_c / P ----
                mu_r, var_r = self._ratio
                C, H, W = mu_in.shape[1:]
                P = H * W
                mu_r, var_r = mu_r.view(1, C, 1, 1), var_r.view(1, C, 1, 1)
                mu_R = mu_in * mu_r / P
                v_R = ((e_in * var_r + v_in * mu_r ** 2) / P ** 2).clamp(min=0.0)
                self._ratio = None

            elif is_linear:
                mu_R, v_R = _lrp_linear_step(mu_in, v_in, e_in, mu_R, v_R, layer, eps, eps_std)
                self._ratio = None
            elif is_bayesian_conv(layer) or isinstance(layer, nn.Conv2d):
                mu_R, v_R = _lrp_conv_step(mu_in, v_in, e_in, mu_R, v_R, layer, eps, eps_std)
                self._ratio = None
            elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                mu_R, v_R = _lrp_pool_step(mu_in, mu_R, v_R, layer, eps)
                self._ratio = None
            else:
                mu_R, v_R = _lrp_identity_step(mu_R, v_R)
                if isinstance(layer, nn.Flatten) and mu_in.dim() > mu_R.dim():
                    mu_R = mu_R.reshape(mu_in.shape)
                    v_R  = v_R.reshape(mu_in.shape)
                elif not isinstance(layer, nn.Flatten):
                    self._ratio = None
            i -= 1
        return mu_R, v_R


# --------------------------------------------------------------------------- #

class LRPFixedDenRef(BayesianLRP):
    """
    Deterministic LRP-eps with initial relevance f_t / d (reference for one weight draw).
    Run it on SampledTwin.net with eps_std=0; pass d = |mean-weight logit|.
    """

    def explain(self, x: torch.Tensor, target_class: int, denom: float):
        with torch.no_grad():
            stats = self.mp.run(x)
        last = stats[-1]
        f = float(last["mu"][0, target_class])
        mu_R = torch.zeros_like(last["mu"])
        v_R = torch.zeros_like(last["v"])
        mu_R[0, target_class] = f / max(abs(float(denom)), 1e-12)
        self.last_stats = stats
        self.last_sign = 1.0 if f >= 0 else -1.0
        mu_R, v_R = self._backward(stats[1:], stats[0], mu_R, v_R)
        return mu_R, v_R, mu_R / torch.sqrt(v_R + 1e-12)
