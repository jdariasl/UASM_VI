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
        if layer.bias:
            mu_b    = layer.mu_bias
            sigma_b = F.softplus(layer.rho_bias)
        else:
            mu_b = sigma_b = None

    elif is_bayesian_conv(layer):
        mu_w    = layer.mu_kernel                          # (out, in/g, kH, kW)
        sigma_w = F.softplus(layer.rho_kernel)
        if layer.bias:
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
