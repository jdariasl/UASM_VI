"""
bayesian_gradinput.py
Closed-form Bayesian gradient x input for VI-BNNs (no denominators, no stabiliser).

Why
---
Bayesian LRP-eps propagates relevance through ratios  c_jk = z_jk / z_k  and treats R_k and
z_k as independent.  In a draw of the weights, however, R_k is proportional to z_k, the
denominator cancels, and for a ReLU network without biases LRP-eps IS gradient x input.  The
closed form therefore describes a different object than the Monte-Carlo reference, and its
1/z^4 terms blow up at near-zero pre-activations (the Var[R] estimate then depends mostly on
the stabiliser).  Here the relevance is defined as

        R_j(w) = x_j * G_j(w) / d ,     G_j = d f_t / d x_j ,     d = |E[f_t]|   (constant)

and the GRADIENT (not a ratio) is propagated backward in closed form.  There is nothing to
stabilise, and the Monte-Carlo reference (GradInputRef: autograd on explicit weight draws)
is exactly the object the closed form approximates.

Backward moment recursion (per element; g = E[G], v = Var[G], e = g^2 + v)
----------------------------------------------------------------------------
 linear / conv  G_in = W^T G_out, weights independent of G_out and of each other:
                    g_in = W_mu^T g,    v_in = W_mu^2^T v + W_sigma^2^T e
                (same expression as the forward pass, with the transposed operator)
 batch-norm     G_in = s G_out                         g s,  v s^2
 ReLU           G_in = 1[z>0] G_out, gate probability p = Phi(mu_z / sigma_z) from the forward
                moments, gate independent of G_out:
                    g_in = p g,    v_in = p v + p (1-p) g^2
 max-pool       routing by the arg-max of the mean activations (0/1 weights)
 avg-pool       G_in = G_out / n                       g/n,  v/n^2
 residual add   y = ReLU(a + b): the same gradient enters both branches; the two gradient
                streams are summed at the block input as independent (means and variances add)
 head           G at the logit = one-hot(t) (deterministic), so at the features
                    g = mu_w[t],  v = sigma_w[t]^2      (exact)

Approximations: (i) the covariance over the summed index k in  G_j = sum_k W_kj G_k is
dropped (v_in uses the sum of variances); (ii) gate and weights/gradients independent;
(iii) the two residual gradient streams independent; (iv) max-pool routing from the mean.
(i) is the gradient-side mirror of the diagonal activation covariance of the forward pass.

Classes
-------
BayesianGradInput   closed form.   mu_R, v_R, snr = BayesianGradInput(model).explain(x, t)
GradInputRef        Monte-Carlo reference for ONE weight draw (deterministic net, autograd).

Input relevance of the explained class is signed so that positive = pushes the logit up
when the mean logit is positive, i.e. R = x * G / |E f_t| (same convention as BayesianLRP).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from explainability.bayesian_lrp import _convT
from explainability.moment_forward import _Phi
from utils.bnn_utils import (
    is_bayesian_conv, is_bayesian_linear, get_weight_moments, conv_kwargs,
)


# --------------------------------------------------------------------------- #
#  per-layer backward steps  (g, v) = mean / variance of dF/d(layer output)
# --------------------------------------------------------------------------- #

def _weights(layer):
    """(mu_w, sigma_w^2) of a Bayesian or deterministic Linear/Conv layer."""
    if is_bayesian_conv(layer) or is_bayesian_linear(layer):
        mu_w, sigma_w, _, _ = get_weight_moments(layer)
        return mu_w, sigma_w ** 2
    return layer.weight, torch.zeros_like(layer.weight)


def _gi_linear(g, v, layer):
    mu_w, s2 = _weights(layer)
    e = g ** 2 + v
    g_in = F.linear(g, mu_w.t())
    v_in = F.linear(v, (mu_w ** 2).t()) + F.linear(e, s2.t())
    return g_in, v_in.clamp(min=0.0)


def _gi_conv(g, v, layer, hw):
    mu_w, s2 = _weights(layer)
    kw = conv_kwargs(layer)
    e = g ** 2 + v
    g_in = _convT(g, mu_w, kw, hw)
    v_in = _convT(v, mu_w ** 2, kw, hw) + _convT(e, s2, kw, hw)
    return g_in, v_in.clamp(min=0.0)


def _gi_bn(g, v, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    shape = (1, -1, 1, 1) if g.dim() == 4 else (1, -1)
    s = s.view(shape)
    return g * s, v * s ** 2


def _gate_prob(pre):
    """P(z > 0) for z ~ N(mu, v) from the forward moments of the pre-activation."""
    sd = torch.sqrt(pre["v"].clamp(min=1e-16))
    return _Phi(pre["mu"] / sd)


def _gi_relu(g, v, pre):
    p = _gate_prob(pre)
    return p * g, (p * v + p * (1.0 - p) * g ** 2).clamp(min=0.0)


def _gi_silu(g, v, pre):
    mu = pre["mu"]
    sig = torch.sigmoid(mu)
    d = sig + mu * sig * (1.0 - sig)
    return d * g, d ** 2 * v


def _gi_pool(mu_a, g, v, layer):
    """Max-pool: 0/1 routing by the mean arg-max.  Avg-pool: 1/n weights (linear)."""
    with torch.enable_grad():
        a = mu_a.detach().clone().requires_grad_(True)
        z = layer(a)
        g_in = torch.autograd.grad(z, a, g, retain_graph=True)[0]
        c_v = torch.autograd.grad(z, a, v)[0]
    if isinstance(layer, nn.MaxPool2d):
        v_in = c_v                                   # (0/1)^2 = 0/1
    else:
        h_in, w_in = mu_a.shape[-2:]
        h_out, w_out = z.shape[-2:]
        n = max(1, (h_in // h_out) * (w_in // w_out))
        v_in = c_v / n                               # grad gives v/n, need v/n^2
    return g_in.detach(), v_in.detach().clamp(min=0.0)


# --------------------------------------------------------------------------- #

class BayesianGradInput:
    """
    Closed-form moments of  R = x * dF_t/dx / |E f_t|  for a VI-BNN.

    Drop-in interface of BayesianLRP: explain(x, t) -> (mu_R, v_R, snr).  The LRP arguments
    eps / eps_std / fuse_bn are accepted and ignored (nothing to stabilise or fuse).
    """

    def __init__(self, model: nn.Module, eps: float = 0.0, eps_std: float = 0.0,
                 fuse_bn: bool = True, pool_corr: float = 0.0):
        from explainability.moment_forward import MomentForwardPass
        self.model = model.eval()
        self.mp = MomentForwardPass(model, pool_corr=pool_corr)
        self.last_stats = None
        self.last_sign = 1.0
        self.last_denominator = 1.0
        self.last_grad = None            # (mean, var) of the input gradient of the last call

    # ------------------------------------------------------------------
    def explain(self, x: torch.Tensor, target_class: int, denom: Optional[float] = None):
        with torch.no_grad():
            stats = self.mp.run(x)
            last = stats[-1]
            mu_f = float(last["mu"][0, target_class])
            self.last_sign = 1.0 if mu_f >= 0 else -1.0
            d = max(abs(mu_f), 1e-12) if denom is None else max(abs(float(denom)), 1e-12)
            self.last_denominator = d
            self.last_stats = stats

            g = torch.zeros_like(last["mu"])
            g[0, target_class] = 1.0
            v = torch.zeros_like(last["v"])
            g, v = self._backward(stats[1:], stats[0], g, v)
            self.last_grad = (g, v)

        mu_R = x * g / d
        v_R = x ** 2 * v / d ** 2
        snr = mu_R / torch.sqrt(v_R + 1e-12)
        return mu_R, v_R, snr

    # ------------------------------------------------------------------
    def _backward(self, records, inp, g, v):
        i = len(records) - 1
        while i >= 0:
            rec = records[i]
            layer = rec["layer"]
            prev = records[i - 1] if i > 0 else inp

            if rec.get("kind") == "residual":
                g, v = self._backward_residual(rec, prev, g, v)

            elif is_bayesian_linear(layer) or isinstance(layer, nn.Linear):
                g, v = _gi_linear(g, v, layer)

            elif is_bayesian_conv(layer) or isinstance(layer, nn.Conv2d):
                g, v = _gi_conv(g, v, layer, prev["mu"].shape[-2:])

            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                g, v = _gi_bn(g, v, layer)

            elif isinstance(layer, nn.ReLU):
                g, v = _gi_relu(g, v, prev)

            elif isinstance(layer, nn.SiLU):
                g, v = _gi_silu(g, v, prev)

            elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                g, v = _gi_pool(prev["mu"], g, v, layer)

            elif isinstance(layer, nn.Flatten):
                if prev["mu"].dim() > g.dim():
                    g, v = g.reshape(prev["mu"].shape), v.reshape(prev["mu"].shape)
            # Dropout and unknown pass-through layers: identity
            i -= 1
        return g, v

    # ------------------------------------------------------------------
    def _backward_residual(self, rec, x_stats, g, v):
        """y = ReLU(a + b): gate at the output, the same gradient into both branches, then sum."""
        pre = {"mu": rec["main"]["mu"] + rec["sc"]["mu"],
               "v": rec["main"]["v"] + rec["sc"]["v"]}
        g_s, v_s = _gi_relu(g, v, pre)
        g_a, v_a = self._backward(rec["body"], x_stats, g_s, v_s)
        if rec["short"]:
            g_b, v_b = self._backward(rec["short"], x_stats, g_s, v_s)
        else:                                          # identity shortcut
            g_b, v_b = g_s, v_s
        return g_a + g_b, v_a + v_b


# --------------------------------------------------------------------------- #

class GradInputRef:
    """
    Monte-Carlo reference for one weight draw: x * dF_t/dx / d with autograd on a deterministic
    network (SampledTwin.net after resample()).  d must be the SAME constant as in the closed
    form (pass the mean-weight or the closed-form |E f_t|).
    """

    def __init__(self, net: nn.Module):
        self.net = net.eval()

    def explain(self, x: torch.Tensor, target_class: int, denom: float) -> torch.Tensor:
        with torch.enable_grad():
            xx = x.detach().clone().requires_grad_(True)
            f = self.net(xx)[0, target_class]
            (gx,) = torch.autograd.grad(f, xx)
        return (x * gx / max(abs(float(denom)), 1e-12)).detach()
