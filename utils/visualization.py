"""
visualization.py
Saliency map plotting utilities for uncertainty-aware explanations.

All functions accept tensors of shape (1, C, H, W) or (C, H, W)
and save the resulting figure to disk.

Dependencies: matplotlib, numpy.  No PyTorch import required at plot time.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional


# --------------------------------------------------------------------------- #
#  Tensor → numpy helpers
# --------------------------------------------------------------------------- #

def _to_np(t) -> np.ndarray:
    """Detach / move to CPU and return numpy array."""
    return t.detach().cpu().float().numpy()


def _denorm_image(x_np: np.ndarray) -> np.ndarray:
    """
    Undo ImageNet normalisation and return a (H, W, 3) uint8 array.
    x_np : (C, H, W) in normalised space.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = x_np.transpose(1, 2, 0) * std + mean       # (H, W, C)
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _saliency_to_heatmap(sal: np.ndarray) -> np.ndarray:
    """
    Collapse channel dimension and normalise to [0, 1].
    sal : (C, H, W) or (H, W)
    """
    if sal.ndim == 3:
        sal = np.abs(sal).mean(axis=0)   # (H, W)
    sal = sal - sal.min()
    denom = sal.max()
    if denom > 0:
        sal = sal / denom
    return sal   # (H, W)


# --------------------------------------------------------------------------- #
#  Main figure functions
# --------------------------------------------------------------------------- #

def save_saliency_figure(
    x:     "torch.Tensor",   # (1, C, H, W) input image
    mu_R:  "torch.Tensor",   # (1, C, H, W) mean relevance / attribution
    v_R:   "torch.Tensor",   # (1, C, H, W) variance
    path:  str,
    title: str = "Bayesian Saliency",
    cmap_sal: str = "hot",
    cmap_var: str = "Blues",
    figsize: tuple = (12, 4),
):
    """
    Three-panel figure: original image | mean saliency | variance map.
    """
    x_np    = _to_np(x[0])
    mu_np   = _to_np(mu_R[0])
    v_np    = _to_np(v_R[0])

    img     = _denorm_image(x_np)
    sal_map = _saliency_to_heatmap(mu_np)
    var_map = _saliency_to_heatmap(v_np)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(img)
    axes[0].set_title("Input")

    im1 = axes[1].imshow(sal_map, cmap=cmap_sal, vmin=0, vmax=1)
    axes[1].set_title("E[R]  (mean attribution)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(var_map, cmap=cmap_var, vmin=0, vmax=1)
    axes[2].set_title("Var[R]  (uncertainty)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_snr_figure(
    x:     "torch.Tensor",   # (1, C, H, W)
    snr:   "torch.Tensor",   # (1, C, H, W)
    path:  str,
    title: str = "SNR Map",
    figsize: tuple = (9, 4),
):
    """
    Two-panel figure: original image | signed SNR overlaid on image.
    Positive SNR → red, negative → blue.
    """
    x_np   = _to_np(x[0])
    snr_np = _to_np(snr[0])
    if snr_np.ndim == 3:
        snr_np = snr_np.mean(axis=0)   # (H, W)

    img  = _denorm_image(x_np)
    vmax = max(abs(snr_np.max()), abs(snr_np.min()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(img)
    axes[0].set_title("Input")

    im = axes[1].imshow(snr_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Signed SNR  =  E[R] / √Var[R]")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def overlay_uncertainty_mask(
    x:     "torch.Tensor",   # (1, C, H, W)
    mask:  "torch.Tensor",   # (1, C, H, W) or (1, 1, H, W) binary
    path:  str,
    title: str = "Reliable Attribution Mask",
    figsize: tuple = (9, 4),
):
    """
    Two-panel: original | image with low-uncertainty regions highlighted.
    The mask is 1 where variance is *low* (reliable attribution).
    """
    x_np   = _to_np(x[0])
    m_np   = _to_np(mask[0])
    if m_np.ndim == 3:
        m_np = m_np.mean(axis=0)   # (H, W)

    img    = _denorm_image(x_np)

    # Overlay: dim out uncertain pixels
    overlay = img.copy().astype(np.float32)
    uncertain = m_np < 0.5   # high-variance pixels
    overlay[uncertain] *= 0.3

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(img)
    axes[0].set_title("Input")

    axes[1].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[1].set_title("Reliable pixels (bright) vs uncertain (dim)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_methods(
    x:          "torch.Tensor",   # (1, C, H, W)
    results:    dict,             # {"LRP": (mu, v, snr), "IG": (mu, v, snr), ...}
    path:       str,
    title:      str = "Method Comparison",
    figsize:    Optional[tuple] = None,
):
    """
    Grid figure comparing multiple explainability methods side-by-side.
    Each method contributes three columns: E[R], Var[R], SNR.
    """
    methods = list(results.keys())
    n_cols  = 3 * len(methods) + 1   # input + 3*n_methods
    if figsize is None:
        figsize = (4 * n_cols, 4)

    x_np = _to_np(x[0])
    img  = _denorm_image(x_np)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for col_base, (name, (mu, v, snr)) in enumerate(results.items(), start=0):
        c0 = 1 + col_base * 3
        sal_map = _saliency_to_heatmap(_to_np(mu[0]))
        var_map = _saliency_to_heatmap(_to_np(v[0]))
        snr_np  = _to_np(snr[0])
        if snr_np.ndim == 3:
            snr_np = snr_np.mean(0)
        vmax = max(abs(snr_np.max()), abs(snr_np.min()), 1e-6)

        axes[c0].imshow(sal_map, cmap="hot", vmin=0, vmax=1)
        axes[c0].set_title(f"{name} E[R]")
        axes[c0].axis("off")

        axes[c0+1].imshow(var_map, cmap="Blues", vmin=0, vmax=1)
        axes[c0+1].set_title(f"{name} Var[R]")
        axes[c0+1].axis("off")

        im = axes[c0+2].imshow(snr_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[c0+2].set_title(f"{name} SNR")
        axes[c0+2].axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
