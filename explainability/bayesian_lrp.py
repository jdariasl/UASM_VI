"""
bayesian_lrp.py
Bayesian LRP-ε: closed-form moment propagation of relevance scores.

Given the layer statistics recorded by MomentForwardPass, this module
propagates relevance backward through the network, computing:

    E[R_j]    – expected relevance of neuron j
    Var[R_j]  – variance of that relevance
    SNR_j     – signal-to-noise ratio  E[R_j] / sqrt(Var[R_j])

The SNR map is the primary output: regions with high |E[R_j]| and low
Var[R_j] are both discriminative AND reliably attributed.

Core equations (LRP-ε rule, delta method)
------------------------------------------
Contribution of neuron j to output k:

    c_jk = z_jk / hat_z_k,   z_jk = a_j * w_jk,   hat_z_k = z_k + ε·sign(z_k)

Moments of the contribution (using weight-activation independence under
mean-field VI and the diagonal covariance approximation):

    mu_A     = mu_j * mu_jk                          [mean of z_jk]
    sigma²_A = e_j * sigma²_jk + v_j * mu²_jk       [variance of z_jk]

    E[c_jk] ≈ mu_A / hat_z_k

    Var[c_jk] ≈ [(hat_z_k - mu_A)² * sigma²_A
                  + mu_A² * (sigma²_z_k - sigma²_A)] / hat_z_k^4

Relevance moments (independence of R_k and c_jk across k):

    E[R_j]   = Σ_k  E[c_jk] * E[R_k]
    Var[R_j] = Σ_k  (E[c_jk]² * Var[R_k] + Var[c_jk] * E[R_k]²)

Ref: Arias-Londoño, "Uncertainty-Aware Saliency Maps via Variational Inference"
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bnn_utils import (
    is_bayesian_conv, is_bayesian_linear,
    get_weight_moments, conv_kwargs,
)

EPS_DEFAULT = 1e-6   # LRP stabiliser


# ------------------------------------------------------------------ helpers --

def _softplus(rho: torch.Tensor) -> torch.Tensor:
    return F.softplus(rho)


def _stabilise(z: torch.Tensor, eps: float,
               v_z: Optional[torch.Tensor] = None,
               eps_std: float = 0.0) -> torch.Tensor:
    """
    hat_z = z + sign(z) * (eps + eps_std * sqrt(Var[z])),  sign(0) := +1.

    The classical LRP-eps stabiliser (eps_std = 0) is unreliable for a BNN:
    pre-activations with mean ~0 and large variance receive relevance through
    E[ReLU(z)] > 0 and are then divided by a near-zero denominator.  Scaling
    the stabiliser with the pre-activation's own standard deviation damps
    exactly those unreliable units.
    """
    sign = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    shift = eps
    if v_z is not None and eps_std > 0.0:
        shift = eps + eps_std * torch.sqrt(v_z.clamp(min=0.0))
    return z + sign * shift


# ------------------------------------------------- per-layer LRP step -------

def _lrp_linear_step(
    mu_a:   torch.Tensor,   # (B, in)   mean activations feeding this layer
    v_a:    torch.Tensor,   # (B, in)
    e_a:    torch.Tensor,   # (B, in)
    mu_R:   torch.Tensor,   # (B, out)  mean relevance of output neurons
    v_R:    torch.Tensor,   # (B, out)  variance of output neurons
    layer:  nn.Module,
    eps:    float,
    eps_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bayesian LRP-ε backward step for a Linear layer.

    Returns (mu_R_in, v_R_in) of shape (B, in).
    """
    if is_bayesian_linear(layer):
        mu_w, sigma_w, mu_b, _ = get_weight_moments(layer)
        sigma_w2 = sigma_w ** 2
    else:
        mu_w     = layer.weight          # (out, in)
        sigma_w2 = torch.zeros_like(mu_w)
        mu_b     = layer.bias

    # ---- Forward moments of pre-activations (needed for hat_z_k) ----
    mu_z  = F.linear(mu_a, mu_w, mu_b)   # (B, out)

    # ---- Variance of z_k = Σ_j z_jk ----
    mu_w2    = mu_w ** 2
    sigma_z2 = F.linear(e_a, sigma_w2) + F.linear(v_a, mu_w2)      # (B, out)
    hat_z = _stabilise(mu_z, eps, sigma_z2, eps_std)               # (B, out)

    # ---- Per-connection moments ----
    # mu_A_{jk}     = mu_a_j * mu_w_{kj}   → shape (B, in, out) – too large
    # We avoid the (B, in, out) tensor by computing the sums directly.

    # E[R_j] = Σ_k  (mu_a_j * mu_w_kj / hat_z_k) * E[R_k]
    #        = mu_a_j * Σ_k (mu_w_kj * E[R_k] / hat_z_k)
    #        = mu_a_j * [mu_w^T @ (mu_R / hat_z)]_j

    ratio_mu = mu_R / hat_z                        # (B, out)
    mu_R_in  = mu_a * F.linear(ratio_mu, mu_w.t()) # (B, in)

    # Var[c_jk] = [(hat_z_k - mu_A)^2 * sigma²_A + mu_A^2 * (sigma²_z - sigma²_A)]
    #             / hat_z_k^4
    #
    # Summing Var[c_jk] * E[R_k]^2 over k:
    #   = Σ_k  Var[c_jk] * mu_R_k^2
    #
    # We compute this via broadcasting-efficient expressions.
    # sigma²_{A,jk} = e_j * sigma²_jk + v_j * mu²_jk

    # term1 for Var[R_j]:
    #   Σ_k  E[c_jk]^2 * Var[R_k]
    #   = Σ_k (mu_a_j * mu_w_kj / hat_z_k)^2 * v_R_k
    #   = mu_a_j^2 * Σ_k (mu_w_kj^2 * v_R_k / hat_z_k^2)
    ratio_var = v_R / (hat_z ** 2)                 # (B, out)
    t1 = (mu_a ** 2) * F.linear(ratio_var, mu_w2.t())          # (B, in)

    # term2 for Var[R_j]:
    #   Σ_k  Var[c_jk] * mu_R_k^2
    # ≈ Σ_k  [sigma²_{A,jk} / hat_z_k^2] * mu_R_k^2     (leading-order)
    # = Σ_k  [(e_j * sigma²_jk + v_j * mu²_jk) / hat_z_k^2] * mu_R_k^2
    # = e_j * Σ_k [sigma²_jk * mu_R_k^2 / hat_z_k^2]
    # + v_j * Σ_k [mu²_jk   * mu_R_k^2 / hat_z_k^2]

    ratio_mR2 = (mu_R ** 2) / (hat_z ** 2)        # (B, out)
    t2 = e_a * F.linear(ratio_mR2, sigma_w2.t()) \
       + v_a * F.linear(ratio_mR2, mu_w2.t())          # (B, in)

    v_R_in = (t1 + t2).clamp(min=0.0)

    return mu_R_in, v_R_in


def _convT(x, w, kw, out_hw, groups_override=None):
    """
    Transposed convolution that maps back to a given spatial size `out_hw`.
    For stride > 1 several input sizes produce the same conv output size, so
    conv_transpose2d needs an explicit output_padding.
    """
    stride, pad, dil = kw["stride"], kw["padding"], kw["dilation"]
    stride = stride if isinstance(stride, tuple) else (stride, stride)
    pad    = pad    if isinstance(pad, tuple)    else (pad, pad)
    dil    = dil    if isinstance(dil, tuple)    else (dil, dil)
    op = []
    for d in range(2):
        base = (x.shape[-2 + d] - 1) * stride[d] - 2 * pad[d] + dil[d] * (w.shape[-2 + d] - 1) + 1
        op.append(int(out_hw[d]) - base)
    return F.conv_transpose2d(x, w, None, stride=stride, padding=pad,
                              output_padding=tuple(op), dilation=dil,
                              groups=kw["groups"])


def _lrp_conv_step(
    mu_a:  torch.Tensor,   # (B, C_in, H, W)
    v_a:   torch.Tensor,
    e_a:   torch.Tensor,
    mu_R:  torch.Tensor,   # (B, C_out, H', W')
    v_R:   torch.Tensor,
    layer: nn.Module,
    eps:   float,
    eps_std: float = 0.0,
    bn:    Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bayesian LRP-ε backward step for a Conv2d layer.

    The LRP-ε recurrence for convolutions mirrors the linear case via
    transposed convolution (conv_transpose2d).

    bn : an eval-mode BatchNorm2d that directly follows the convolution.  It is
         folded into the layer (canonisation): BN(z) = s*z + t is linear, so
         the fused layer has weights s*W (std s*sigma_W) and bias s*(b-m)+beta.
         Relevance arriving at the BN output is then redistributed with the
         correct denominator, which identity treatment of BN does not do.

    Returns (mu_R_in, v_R_in) of shape (B, C_in, H, W).
    """
    if is_bayesian_conv(layer):
        mu_w, sigma_w, mu_b, _ = get_weight_moments(layer)
        sigma_w2 = sigma_w ** 2
    else:
        mu_w     = layer.weight
        sigma_w2 = torch.zeros_like(mu_w)
        mu_b     = layer.bias

    kw = conv_kwargs(layer)

    if bn is not None:
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)        # (C_out,)
        b0 = mu_b if mu_b is not None else torch.zeros_like(scale)
        mu_b     = scale * (b0 - bn.running_mean) + bn.bias
        mu_w     = mu_w * scale.view(-1, 1, 1, 1)
        sigma_w2 = sigma_w2 * (scale ** 2).view(-1, 1, 1, 1)

    # Forward with mean weights to get hat_z
    mu_z  = F.conv2d(mu_a, mu_w, mu_b, **kw)      # (B, C_out, H', W')
    sigma_z2 = F.conv2d(e_a, sigma_w2, None, **kw) \
             + F.conv2d(v_a, mu_w ** 2, None, **kw)
    hat_z = _stabilise(mu_z, eps, sigma_z2, eps_std)

    # ---- E[R_in] via transposed convolution ----
    hw = mu_a.shape[-2:]
    ratio_mu = mu_R / hat_z                       # (B, C_out, H', W')

    # Σ_k (mu_w_k * mu_a) / hat_z_k * mu_R_k  =  mu_a * conv_T(ratio_mu, mu_w)
    mu_R_in = mu_a * _convT(ratio_mu, mu_w, kw, hw)

    # ---- Var[R_in] ----
    ratio_var = v_R / (hat_z ** 2)
    ratio_mR2 = (mu_R ** 2) / (hat_z ** 2)

    # term1: mu_a^2 * conv_T(v_R / hat_z^2, mu_w^2)
    mu_w2 = mu_w ** 2
    t1 = (mu_a ** 2) * _convT(ratio_var, mu_w2, kw, hw)

    # term2: e_a * conv_T(mu_R^2/hat_z^2, sigma²_w) + v_a * conv_T(..., mu²_w)
    t2 = e_a * _convT(ratio_mR2, sigma_w2, kw, hw) \
       + v_a * _convT(ratio_mR2, mu_w2, kw, hw)

    v_R_in = (t1 + t2).clamp(min=0.0)

    return mu_R_in, v_R_in


def _lrp_pool_step(mu_a, mu_R, v_R, layer, eps):
    """
    LRP-eps through MaxPool2d / AdaptiveAvgPool2d.

    Relevance of each pooled unit is redistributed to its inputs in proportion
    to their contribution: for max-pool this is winner-take-all (all relevance
    goes to the arg-max input), for average pooling it is proportional to a_j/n.
    The redistribution is obtained with autograd on the (linear-in-routing)
    pooling operator, so it works for any kernel/stride/padding.

    Variance: the variance of a pooled unit's relevance is routed along the
    same paths; for average pooling the routing weights enter squared (1/n).
    """
    with torch.enable_grad():
        a = mu_a.detach().clone().requires_grad_(True)
        z = layer(a)
        hat_z = _stabilise(z.detach(), eps)
        c_mu = torch.autograd.grad(z, a, mu_R / hat_z, retain_graph=True)[0]
        c_v  = torch.autograd.grad(z, a, v_R / hat_z ** 2)[0]
    mu_R_in = mu_a * c_mu
    if isinstance(layer, nn.MaxPool2d):
        v_R_in = (mu_a ** 2) * c_v          # 0/1 routing: weights squared = weights
    else:
        h_in, w_in   = mu_a.shape[-2:]
        h_out, w_out = z.shape[-2:]
        n = max(1, (h_in // h_out) * (w_in // w_out))
        v_R_in = (mu_a ** 2) * c_v / n      # (1/n)^2 summed over n units = (1/n) * sum of (1/n)
    return mu_R_in, v_R_in.clamp(min=0.0)


def _lrp_identity_step(mu_R, v_R):
    """Pass-through for BN, activation, pooling, flatten."""
    return mu_R, v_R


# --------------------------------------------------------- main class --------

class BayesianLRP:
    """
    Compute Bayesian LRP-ε saliency maps for a VI-BNN.

    Parameters
    ----------
    model  : Bayesian nn.Module (BEfficientNet or BVGG), must be in eval()
    eps    : LRP stabiliser (default 1e-6)
    eps_std: scale of the uncertainty-dependent stabiliser, hat_z = z + sign(z)(eps + eps_std*std(z));
             0 recovers classical LRP-eps, ~0.5-1 is recommended for BNNs
    fuse_bn: fold every eval-mode BatchNorm into the convolution before it in the
             backward pass (standard LRP canonisation; strongly recommended for
             ResNets).  False treats BN as identity for relevance.

    Supported networks: sequential CNNs (VGG) and torchvision ResNets
    (BasicBlock / Bottleneck).  At every residual add the relevance is split
    between the two branches with the delta-method moments of the contribution
    ratios, and the two relevance streams are summed at the block input,
    treating the branches as independent (means and variances add).

    Usage
    -----
        blrp = BayesianLRP(model, eps=1e-6)
        mu_R, v_R, snr = blrp.explain(x, target_class)

        # x   : (1, C, H, W) input image
        # mu_R: (1, C, H, W) expected relevance map (same spatial dims as x)
        # v_R : (1, C, H, W) variance of relevance map
        # snr : (1, C, H, W) signal-to-noise ratio  mu_R / sqrt(v_R)
    """

    def __init__(self, model: nn.Module, eps: float = EPS_DEFAULT,
                 eps_std: float = 0.0, fuse_bn: bool = True,
                 pool_corr: float = 0.0):
        from explainability.moment_forward import MomentForwardPass
        self.model    = model.eval()
        self.eps      = eps
        self.eps_std  = eps_std
        self.fuse_bn  = fuse_bn
        self.mp       = MomentForwardPass(model, pool_corr=pool_corr)
        self.last_stats = None          # forward moments of the last explain() call
        self.last_sign  = 1.0           # sign of the explained logit (start relevance)

    # ------------------------------------------------------------------
    def explain(self, x: torch.Tensor, target_class: int):
        """
        Parameters
        ----------
        x            : (1, C, H, W)
        target_class : int, the class whose relevance we explain

        Returns
        -------
        mu_R  : (1, C, H, W)  expected relevance
        v_R   : (1, C, H, W)  relevance variance
        snr   : (1, C, H, W)  signed SNR = mu_R / sqrt(v_R + 1e-12)
        """
        # --- Forward moment propagation ---
        with torch.no_grad():
            stats = self.mp.run(x)

        # --- Initialise relevance at the output layer ---
        # Only the logit of the target class receives relevance = 1.
        last = stats[-1]
        mu_R_out = torch.zeros_like(last["mu"])
        v_R_out  = torch.zeros_like(last["v"])
        # Start with relevance sign(mean logit) on the target logit.  Then
        # R_j = z_jT / |z_T|, i.e. positive relevance always means "pushes this
        # logit up", whatever the sign of the logit (a +1 start would flip every
        # map when the logit is negative, e.g. when explaining a non-predicted
        # class).  Variance terms depend on mu_R^2 and are unaffected.
        self.last_sign = 1.0 if float(last["mu"][0, target_class]) >= 0 else -1.0
        mu_R_out[0, target_class] = self.last_sign

        mu_R = mu_R_out
        v_R  = v_R_out

        # --- Walk backward (recursive: residual blocks contain sub-sequences) ---
        self.last_stats = stats
        mu_R, v_R = self._backward(stats[1:], stats[0], mu_R, v_R)

        snr = mu_R / torch.sqrt(v_R + 1e-12)
        return mu_R, v_R, snr

    # ------------------------------------------------------------------
    def _backward(self, records, inp, mu_R, v_R):
        """
        Propagate relevance backward through a forward-ordered list of records.

        records : list of stat dicts as produced by MomentForwardPass
        inp     : stat dict of the tensor entering records[0]
        """
        eps, eps_std = self.eps, self.eps_std
        i = len(records) - 1
        while i >= 0:
            rec   = records[i]
            layer = rec["layer"]
            prev  = records[i - 1] if i > 0 else inp

            # ---- residual block ----
            if rec.get("kind") == "residual":
                mu_R, v_R = self._backward_residual(rec, prev, mu_R, v_R)
                i -= 1
                continue

            # ---- conv followed by BN: fuse (the BN record is the one we are at) ----
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

            if is_bayesian_linear(layer) or isinstance(layer, nn.Linear):
                mu_R, v_R = _lrp_linear_step(mu_in, v_in, e_in, mu_R, v_R, layer, eps, eps_std)

            elif is_bayesian_conv(layer) or isinstance(layer, nn.Conv2d):
                mu_R, v_R = _lrp_conv_step(mu_in, v_in, e_in, mu_R, v_R, layer, eps, eps_std)

            elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                mu_R, v_R = _lrp_pool_step(mu_in, mu_R, v_R, layer, eps)

            else:
                # BN (when not fused), activation, flatten, dropout → identity
                mu_R, v_R = _lrp_identity_step(mu_R, v_R)
                if isinstance(layer, nn.Flatten) and mu_in.dim() > mu_R.dim():
                    mu_R = mu_R.reshape(mu_in.shape)
                    v_R  = v_R.reshape(mu_in.shape)
            i -= 1
        return mu_R, v_R

    # ------------------------------------------------------------------
    def _backward_residual(self, rec, x_stats, mu_R, v_R):
        """
        Relevance through  y = ReLU(a + b),  a = F(x) (body), b = S(x) (shortcut).

        1. The final ReLU is identity for relevance.
        2. Split at the add with LRP-eps contributions  c_a = a/hat_s,
           c_b = b/hat_s,  hat_s = (a+b) + sign*(eps + eps_std*std).  With a, b
           independent, the delta method gives

               E[c_a] = mu_a / hat_s,        E[c_b] = mu_b / hat_s,
               Var[c_a] = Var[c_b] = (mu_b^2 v_a + mu_a^2 v_b) / hat_s^4

           (c_a + c_b = 1, so both ratios have the same variance), and

               E[R_a]   = E[c_a] E[R],
               Var[R_a] = E[c_a]^2 Var[R] + Var[c_a] (Var[R] + E[R]^2)   (same for b).
        3. Each stream goes backward through its own sub-sequence.
        4. The streams meet again at the block input and are summed, treating
           them as independent:  means and variances add.
        """
        a, va = rec["main"]["mu"], rec["main"]["v"]
        b, vb = rec["sc"]["mu"],   rec["sc"]["v"]

        hat   = _stabilise(a + b, self.eps, va + vb, self.eps_std)
        c_a   = a / hat
        c_b   = b / hat
        var_c = (b ** 2 * va + a ** 2 * vb) / hat ** 4

        # Var[cR] = Var[c](V + M^2) + E[c]^2 V   (independent c and R; paper Eq. eps_R_var)
        mu_Ra, v_Ra = c_a * mu_R, c_a ** 2 * v_R + var_c * (v_R + mu_R ** 2)
        mu_Rb, v_Rb = c_b * mu_R, c_b ** 2 * v_R + var_c * (v_R + mu_R ** 2)

        mu_a_in, v_a_in = self._backward(rec["body"], x_stats, mu_Ra, v_Ra)
        if rec["short"]:
            mu_b_in, v_b_in = self._backward(rec["short"], x_stats, mu_Rb, v_Rb)
        else:                                   # identity shortcut
            mu_b_in, v_b_in = mu_Rb, v_Rb

        return mu_a_in + mu_b_in, v_a_in + v_b_in

    # ------------------------------------------------------------------
    @staticmethod
    def uncertainty_mask(v_R: torch.Tensor,
                         percentile: float = 75.0) -> torch.Tensor:
        """
        Binary mask: 1 where variance is below the given percentile.
        Use this to highlight only the reliably attributed pixels.
        """
        threshold = torch.quantile(v_R.flatten(), percentile / 100.0)
        return (v_R <= threshold).float()
