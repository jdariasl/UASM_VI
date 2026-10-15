"""
bayesian_ig.py
MC Bayesian Integrated Gradients for VI-BNNs.

Standard IG integrates the gradient of the output w.r.t. the input along
a straight-line path from a baseline x' to the input x:

    IG_i(x) = (x_i - x'_i) * ∫_0^1 ∂F(x' + α(x-x')) / ∂x_i  dα

For a BNN the model F is stochastic (weights are random variables).  We
approximate the distribution of IG by drawing K weight samples via the
reparameterization trick and computing IG for each sample:

    IG^(k)  ← IG computed with weight sample w^(k) ~ q(w|D)
    E[IG]   ≈ (1/K) Σ_k IG^(k)
    Var[IG] ≈ (1/K-1) Σ_k (IG^(k) - E[IG])^2

The integral along the path is approximated by the Gauss-Legendre
quadrature (default m=50 steps) for accuracy, or by the trapezoidal
rule if preferred.

SNR map  =  E[IG] / sqrt(Var[IG] + ε)

Ref: Arias-Londoño, "Uncertainty-Aware Saliency Maps via Variational Inference"
     Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Path-integral helpers
# --------------------------------------------------------------------------- #

def _interpolate_inputs(
    x:       torch.Tensor,   # (1, C, H, W)
    baseline: torch.Tensor,  # (1, C, H, W)
    m:       int,
) -> torch.Tensor:
    """
    Return m+1 equally-spaced interpolation points along the straight-line
    path from baseline to x.

        x_α = baseline + α * (x - baseline),   α = 0, 1/m, 2/m, …, 1

    Output shape: (m+1, C, H, W).
    """
    alphas = torch.linspace(0.0, 1.0, m + 1,
                            device=x.device, dtype=x.dtype)   # (m+1,)
    # broadcast: (m+1, 1, 1, 1) * (1, C, H, W)
    delta = x - baseline
    path  = baseline + alphas[:, None, None, None] * delta     # (m+1, C, H, W)
    return path


def _integrated_gradients_single(
    model:    nn.Module,
    x:        torch.Tensor,   # (1, C, H, W)
    baseline: torch.Tensor,   # (1, C, H, W)
    target:   int,
    m:        int,
) -> torch.Tensor:
    """
    Compute IG for one deterministic forward pass (weights are fixed, either
    sampled or mean).

    Returns ig : (1, C, H, W)
    """
    path = _interpolate_inputs(x, baseline, m)   # (m+1, C, H, W)
    path = path.requires_grad_(True)

    # Forward on all m+1 points at once
    logits = model(path)                          # (m+1, n_classes)
    target_logits = logits[:, target].sum()       # scalar

    # Gradient of target w.r.t. path points
    grads = torch.autograd.grad(target_logits, path)[0]  # (m+1, C, H, W)

    # Trapezoidal rule along the first axis
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0, keepdim=True) / 2.0  # (1, C, H, W)

    ig = (x - baseline) * avg_grads
    return ig.detach()


# --------------------------------------------------------------------------- #
#  Main class
# --------------------------------------------------------------------------- #

class BayesianIG:
    """
    MC Bayesian Integrated Gradients.

    Parameters
    ----------
    model        : BNN in eval() mode (weights sampled on each forward call
                   when called normally; we use no_grad=False for gradients)
    n_samples    : K — number of MC weight samples
    m_steps      : number of path interpolation steps (trapezoidal rule)
    baseline     : type of baseline ("black" → zeros, "mean" → dataset mean)

    Usage
    -----
        big = BayesianIG(model, n_samples=20, m_steps=50)
        mu_ig, v_ig, snr = big.explain(x, target_class=1)
    """

    def __init__(
        self,
        model:     nn.Module,
        n_samples: int   = 20,
        m_steps:   int   = 50,
        baseline:  str   = "black",
    ):
        self.model     = model.eval()
        self.n_samples = n_samples
        self.m_steps   = m_steps
        self.baseline  = baseline

    # ------------------------------------------------------------------
    def _make_baseline(self, x: torch.Tensor) -> torch.Tensor:
        if self.baseline == "black":
            return torch.zeros_like(x)
        elif self.baseline == "mean":
            # Use the per-channel mean of x as a simple stand-in
            return x.mean(dim=(-1, -2), keepdim=True).expand_as(x)
        elif self.baseline == "white":
            return torch.ones_like(x)
        else:
            raise ValueError(f"Unknown baseline type '{self.baseline}'. "
                             "Choose 'black', 'white', or 'mean'.")

    # ------------------------------------------------------------------
    def explain(
        self,
        x:            torch.Tensor,  # (1, C, H, W)
        target_class: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : (1, C, H, W)  input image
        target_class : int

        Returns
        -------
        mu_ig  : (1, C, H, W)  mean IG attribution
        v_ig   : (1, C, H, W)  variance of IG across weight samples
        snr    : (1, C, H, W)  signed SNR = mu_ig / sqrt(v_ig + 1e-12)
        """
        baseline = self._make_baseline(x)
        samples  = []

        for _ in range(self.n_samples):
            # Each call to the BNN model draws a fresh weight sample
            ig = _integrated_gradients_single(
                self.model, x, baseline, target_class, self.m_steps,
            )
            samples.append(ig)   # (1, C, H, W)

        # Stack along a new sample dimension → (K, 1, C, H, W)
        stacked = torch.stack(samples, dim=0)

        mu_ig = stacked.mean(dim=0)                   # (1, C, H, W)
        v_ig  = stacked.var(dim=0, unbiased=True).clamp(min=0.0)

        snr = mu_ig / torch.sqrt(v_ig + 1e-12)
        return mu_ig, v_ig, snr

    # ------------------------------------------------------------------
    @staticmethod
    def uncertainty_mask(
        v_ig:       torch.Tensor,
        percentile: float = 75.0,
    ) -> torch.Tensor:
        """
        Binary mask: 1 where IG variance is below the given percentile.
        Mirrors BayesianLRP.uncertainty_mask for a unified evaluation API.
        """
        threshold = torch.quantile(v_ig.flatten(), percentile / 100.0)
        return (v_ig <= threshold).float()
