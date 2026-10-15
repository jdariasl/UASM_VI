"""Uncertainty-aware saliency maps via variational inference.

A mean-field Gaussian posterior q(z) = N(mu, sigma^2) is placed over mask
logits z (one per input feature). The mask m = sigmoid(z) blends the input with
a baseline. The posterior is fitted by maximising a Monte-Carlo ELBO using the
reparameterisation trick: expected model score, minus a sparsity penalty,
minus KL(q || N(0, prior_std^2)). The saliency map is E_q[m] and the
uncertainty map is Std_q[m].
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class UASMResult:
    saliency: np.ndarray
    uncertainty: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    elbo_history: np.ndarray


def uncertainty_aware_saliency(
    score_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 300,
    n_samples: int = 8,
    lr: float = 0.05,
    sparsity: float = 0.1,
    prior_std: float = 1.0,
    kl_weight: float = 1e-2,
    n_eval_samples: int = 100,
    seed: Optional[int] = None,
) -> UASMResult:
    """Compute a saliency map with per-feature uncertainty.

    Args:
        score_and_grad: maps an input (same shape as ``x``) to
            ``(score, d score / d input)``.
        x: input array.
        baseline: value used for masked-out features (default zeros).
    """
    x = np.asarray(x, dtype=float)
    if n_steps < 1 or n_samples < 1 or n_eval_samples < 2:
        raise ValueError("n_steps, n_samples must be >= 1 and n_eval_samples >= 2")
    if prior_std <= 0:
        raise ValueError("prior_std must be positive")
    base = np.zeros_like(x) if baseline is None else np.broadcast_to(
        np.asarray(baseline, dtype=float), x.shape)
    rng = np.random.default_rng(seed)

    mu = np.zeros_like(x)
    rho = np.full_like(x, np.log(np.expm1(0.5)))  # sigma = softplus(rho)
    params = [mu, rho]
    m1 = [np.zeros_like(x) for _ in params]
    m2 = [np.zeros_like(x) for _ in params]
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    history = np.empty(n_steps)
    delta = x - base

    for t in range(1, n_steps + 1):
        sigma = np.log1p(np.exp(rho))
        g_mu = np.zeros_like(x)
        g_sigma = np.zeros_like(x)
        elbo = 0.0
        for _ in range(n_samples):
            eps = rng.standard_normal(x.shape)
            m = _sigmoid(mu + sigma * eps)
            score, g_in = score_and_grad(base + m * delta)
            dm = m * (1 - m)
            # ascent direction of score - sparsity * mean(m)
            g_m = (np.asarray(g_in, dtype=float) * delta - sparsity / x.size) * dm
            g_mu += g_m / n_samples
            g_sigma += g_m * eps / n_samples
            elbo += (score - sparsity * m.mean()) / n_samples
        kl = np.log(prior_std / sigma) + (sigma**2 + mu**2) / (2 * prior_std**2) - 0.5
        g_mu -= kl_weight * mu / prior_std**2
        g_sigma -= kl_weight * (sigma / prior_std**2 - 1.0 / sigma)
        elbo -= kl_weight * kl.sum()
        history[t - 1] = elbo

        grads = [-g_mu, -g_sigma * _sigmoid(rho)]  # minimise -ELBO
        for i, (p, g) in enumerate(zip(params, grads)):
            m1[i] = b1 * m1[i] + (1 - b1) * g
            m2[i] = b2 * m2[i] + (1 - b2) * g * g
            p -= lr * (m1[i] / (1 - b1**t)) / (np.sqrt(m2[i] / (1 - b2**t)) + eps_adam)

    sigma = np.log1p(np.exp(rho))
    masks = _sigmoid(mu + sigma * rng.standard_normal((n_eval_samples,) + x.shape))
    return UASMResult(masks.mean(axis=0), masks.std(axis=0), mu, sigma, history)
