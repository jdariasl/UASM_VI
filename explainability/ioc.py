"""
ioc.py
Importance-of-context index with percentile-based confidence (IoC_alpha) from
    Arias-Londono & Godino-Llorente, Biomed. Signal Process. Control 90 (2024) 105831,
computed (a) from Monte-Carlo attribution maps, as in that paper, and (b) in closed form from the
mean and variance maps of a VI-BNN (BayesianLRP / BayesianGradInput), without sampling.

Definition (paper Eq. 5).  Let R_{m,j} be the relevance of pixel j in the m-th Monte-Carlo map and
Q_alpha the alpha-percentile over m.  With a region of interest `inside` (the lungs there; the
bird bounding box for CUB-200) and its complement `outside`,

    S^alpha_in  = {j in inside  : Q_alpha(R_j) > 0},    S^alpha_out = {j in outside : Q_alpha(R_j) > 0}
    IoC_alpha   = mean_{j in S^alpha_out} Q_alpha(R_j)  /  mean_{j in S^alpha_in} Q_alpha(R_j)

IoC_alpha > 1: the pixels outside the region that are positive at the alpha percentile are, on
average, more relevant than those inside (a Clever-Hans signature).  Small alpha is demanding
(a pixel must be positive in at least 100-alpha % of the draws), large alpha is permissive.

Closed form.  With a Gaussian pixel marginal R_j ~ N(mu_j, c^2 v_j),
    Q_alpha(R_j) = mu_j + z_alpha * c * sqrt(v_j),     z_alpha = Phi^{-1}(alpha/100),
and the sets and means above are evaluated on these values.  alpha = 50 gives z = 0: the index
uses only the mean map.  c >= 1 rescales sigma (the closed-form variance is underestimated; see
the calibration section of the manuscript).  Colour channels are summed: the pixel relevance is
sum_c R_c, with mean sum_c mu_c and (independence over channels assumed) variance sum_c v_c.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Optional

import numpy as np


def _pix(a) -> np.ndarray:
    """(1,C,H,W) / (C,H,W) -> (H,W) by summing the channels; (H,W) unchanged."""
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    return a.sum(0) if a.ndim == 3 else a


def _ioc(q: np.ndarray, inside: np.ndarray) -> float:
    inside = inside.astype(bool)
    pin, pout = q[inside & (q > 0)], q[~inside & (q > 0)]
    if pin.size == 0 or pout.size == 0:
        return float("nan")
    return float(pout.mean() / pin.mean())


def ioc_alpha_mc(draws, inside: np.ndarray, alpha: float) -> float:
    """draws: iterable of M relevance maps (each (1,C,H,W), (C,H,W) or (H,W)); inside: (H,W) bool."""
    R = np.stack([_pix(d) for d in draws])                 # (M,H,W)
    return _ioc(np.percentile(R, alpha, axis=0), inside)


def ioc_alpha_closed(mu, v, inside: np.ndarray, alpha: float, scale: float = 1.0) -> float:
    """Closed-form IoC_alpha from the mean / variance maps (Gaussian pixel marginals)."""
    m = _pix(mu)
    s = np.sqrt(np.maximum(_pix(v), 0.0))
    z = 0.0 if alpha == 50 else NormalDist().inv_cdf(alpha / 100.0)
    return _ioc(m + z * scale * s, inside)


def bbox_mask(bbox, orig_hw, out_size: int = 224, resize: int = 256) -> np.ndarray:
    """
    Bounding box (x, y, w, h) in original pixels -> boolean (out,out) mask after
    Resize(resize) (shorter side) + CenterCrop(out), the CUB-200 evaluation transform.
    """
    H, W = orig_hw
    s = resize / min(H, W)
    nh, nw = round(H * s), round(W * s)
    top, left = (nh - out_size) // 2, (nw - out_size) // 2
    x, y, w, h = bbox
    x0, x1 = int(round(x * s - left)), int(round((x + w) * s - left))
    y0, y1 = int(round(y * s - top)), int(round((y + h) * s - top))
    mk = np.zeros((out_size, out_size), dtype=bool)
    mk[max(y0, 0):max(min(y1, out_size), 0), max(x0, 0):max(min(x1, out_size), 0)] = True
    return mk
