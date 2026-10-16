"""
bnn_utils.py
Utilities for extracting parameters from bayesian_torch Bayesian layers.

bayesian_torch uses different attribute names for Linear vs Conv layers:
  Linear: mu_weight / rho_weight
  Conv:   mu_kernel / rho_kernel
This module wraps that inconsistency behind a unified API.
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Layer type checks
# --------------------------------------------------------------------------- #

def is_bayesian(layer) -> bool:
    name = layer.__class__.__name__
    return "Reparameterization" in name or "Flipout" in name


def is_bayesian_linear(layer) -> bool:
    name = layer.__class__.__name__
    return is_bayesian(layer) and "Linear" in name


def is_bayesian_conv(layer) -> bool:
    name = layer.__class__.__name__
    return is_bayesian(layer) and "Conv" in name


# --------------------------------------------------------------------------- #
# Parameter extraction
# --------------------------------------------------------------------------- #

def get_weight_moments(layer):
    """
    Return (mu_w, sigma_w, mu_b, sigma_b) for a Bayesian layer.

    sigma = softplus(rho) = log(1 + exp(rho))  [bayesian_torch convention]

    For Conv layers  : weight tensor has shape (out, in/groups, kH, kW)
    For Linear layers: weight tensor has shape (out, in)

    Returns None for mu_b / sigma_b when the layer has no bias.
    """
    if is_bayesian_linear(layer):
        mu_w    = layer.mu_weight                          # (out, in)
        sigma_w = F.softplus(layer.rho_weight)             # (out, in)
        if getattr(layer, "mu_bias", None) is not None:
            mu_b    = layer.mu_bias
            sigma_b = F.softplus(layer.rho_bias)
        else:
            mu_b = sigma_b = None

    elif is_bayesian_conv(layer):
        mu_w    = layer.mu_kernel                          # (out, in/g, kH, kW)
        sigma_w = F.softplus(layer.rho_kernel)
        if getattr(layer, "mu_bias", None) is not None:
            mu_b    = layer.mu_bias
            sigma_b = F.softplus(layer.rho_bias)
        else:
            mu_b = sigma_b = None

    else:
        raise TypeError(f"Expected a Bayesian layer, got {layer.__class__.__name__}")

    return mu_w, sigma_w, mu_b, sigma_b


def get_second_moment_weight(sigma_w):
    """sigma_w^2, used in forward moment propagation."""
    return sigma_w ** 2


# --------------------------------------------------------------------------- #
# Conv / Linear metadata (shared between Bayesian and deterministic)
# --------------------------------------------------------------------------- #

def conv_kwargs(layer) -> dict:
    """Return the kwargs needed to call F.conv2d / F.conv_transpose2d."""
    return dict(
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )


# --------------------------------------------------------------------------- #
# Deterministic twin with re-drawable weights
# --------------------------------------------------------------------------- #

def has_bayesian_layers(model) -> bool:
    return any(is_bayesian(m) for m in model.modules())


def has_flipout_layers(model) -> bool:
    return any("Flipout" in m.__class__.__name__ for m in model.modules())


class SampledTwin:
    """
    A deterministic copy of a BNN whose weights can be re-drawn in place from
    the variational posterior  w ~ N(mu, softplus(rho)^2).

    Why it exists
    -------------
    * Flipout draws a different perturbation for every example in a batch, so a
      batched integrated-gradients path would see a different weight sample at
      every alpha.  A twin holds ONE explicit weight realisation for the whole
      path, which is what IG for a fixed w requires.
    * Monte-Carlo estimates of E[R] / Var[R] (to validate the closed-form
      Bayesian LRP) need explicit deterministic networks.

    Both estimators share the same marginal posterior q(w), so the twin is
    estimator-agnostic.

        twin = SampledTwin(bnn)
        twin.resample()           # draw w ~ q
        logits = twin.net(x)
    """

    def __init__(self, bnn):
        import copy
        import torch.nn as nn

        self.pairs = []                       # (bayesian layer, deterministic layer)
        memo = {}                             # id(bayesian layer) -> deterministic twin layer
        for name, old in bnn.named_modules():
            if not is_bayesian(old):
                continue
            if is_bayesian_linear(old):
                new = nn.Linear(old.in_features, old.out_features, bias=getattr(old, "mu_bias", None) is not None)
                dev = old.mu_weight.device
            else:
                new = nn.Conv2d(old.in_channels, old.out_channels, old.kernel_size,
                                stride=old.stride, padding=old.padding,
                                dilation=old.dilation, groups=old.groups,
                                bias=getattr(old, "mu_bias", None) is not None)
                dev = old.mu_kernel.device
            new = new.to(dev)
            memo[id(old)] = new               # deepcopy will reuse `new` instead of
            self.pairs.append((old, new))     # copying the (large) Bayesian layer
        net = copy.deepcopy(bnn, memo)
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def resample(self):
        for src, dst in self.pairs:
            mu_w, sigma_w, mu_b, sigma_b = get_weight_moments(src)
            dst.weight.copy_(mu_w + sigma_w * torch.randn_like(sigma_w))
            if mu_b is not None:
                dst.bias.copy_(mu_b + sigma_b * torch.randn_like(sigma_b))
        return self.net

    @torch.no_grad()
    def use_mean(self):
        for src, dst in self.pairs:
            mu_w, _, mu_b, _ = get_weight_moments(src)
            dst.weight.copy_(mu_w)
            if mu_b is not None:
                dst.bias.copy_(mu_b)
        return self.net
