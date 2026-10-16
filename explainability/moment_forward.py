"""
moment_forward.py
Forward moment propagation through a VI-BNN.

For each neuron j in every layer we track:
    mu_j  = E[a_j]          (mean activation)
    v_j   = Var[a_j]        (variance of activation)
    e_j   = E[a_j^2]        (second moment  = mu_j^2 + v_j)

These quantities flow through the network layer by layer.

Supported layer types
---------------------
- Bayesian Conv2d  (Reparameterization / Flipout)
- Bayesian Linear  (Reparameterization / Flipout)
- Deterministic Conv2d, Linear, BatchNorm2d, BatchNorm1d
- ReLU    – exact Gaussian closed-form (Φ, φ)
- SiLU    – first-order delta method (good when v_j is small)
- MaxPool2d, AdaptiveAvgPool2d, Flatten, Dropout (pass-through / exact)

The forward pass is performed on the *mean* model (mu_w only) to obtain
activation maps, and the variance is propagated analytically alongside.

Usage
-----
    mp = MomentForwardPass(model)
    # x : (1, C, H, W)  single image (batch_size=1 recommended for LRP)
    layer_stats = mp.run(x)
    # layer_stats[i] = {"mu": Tensor, "v": Tensor, "e": Tensor}
"""

from __future__ import annotations

import math
import warnings
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bnn_utils import (
    is_bayesian, is_bayesian_conv, is_bayesian_linear,
    get_weight_moments, conv_kwargs,
)

# --------------------------------------------------- supported containers ----
try:
    from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
    from torchvision.models.vgg import VGG as _TVVGG
    _RESIDUAL_BLOCKS = (BasicBlock, Bottleneck)
    # containers whose children are stored in forward order, so a plain
    # depth-first walk reproduces the data flow
    _SEQUENTIAL_CONTAINERS = (nn.Sequential, ResNet, _TVVGG)
except ImportError:                                  # torchvision not installed
    _RESIDUAL_BLOCKS = ()
    _SEQUENTIAL_CONTAINERS = (nn.Sequential,)

_WARNED = set()


def _warn_unsupported(module: nn.Module):
    """Modules with children that we cannot walk in data-flow order."""
    name = type(module).__name__
    if name not in _WARNED:
        _WARNED.add(name)
        warnings.warn(
            f"MomentForwardPass: '{name}' is not a known sequential container or "
            "residual block. Its children are walked in definition order, which "
            "may not match its forward(); results for this network are unreliable.",
            stacklevel=3)


# ------------------------------------------------------------------ helpers --

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi(alpha: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF evaluated at alpha."""
    return torch.exp(-0.5 * alpha ** 2) / _SQRT_2PI


def _Phi(alpha: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF evaluated at alpha."""
    return 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))


def _relu_moments(mu: torch.Tensor, v: torch.Tensor):
    """
    Exact first and second moment of max(0, z) when z ~ N(mu, v).

    Returns (mu_out, v_out, e_out).
    """
    eps = 1e-8
    std   = torch.sqrt(v.clamp(min=eps))
    alpha = mu / std

    phi_a = _phi(alpha)
    Phi_a = _Phi(alpha)

    mu_out = mu * Phi_a + std * phi_a
    e_out  = (mu ** 2 + v) * Phi_a + mu * std * phi_a
    v_out  = (e_out - mu_out ** 2).clamp(min=0.0)
    return mu_out, v_out, e_out


def _silu_moments(mu: torch.Tensor, v: torch.Tensor):
    """
    First-order delta-method approximation for SiLU(z) = z * sigmoid(z).

    mu_out ≈ SiLU(mu)
    v_out  ≈ SiLU'(mu)^2 * v

    Valid when v is small (typical for well-trained BNNs).
    """
    sig  = torch.sigmoid(mu)
    silu = mu * sig
    # derivative: SiLU'(x) = sigma(x) + x*sigma(x)*(1-sigma(x))
    dsilu = sig + mu * sig * (1.0 - sig)

    mu_out = silu
    v_out  = (dsilu ** 2) * v
    e_out  = mu_out ** 2 + v_out
    return mu_out, v_out, e_out


# ------------------------------------------------------- layer propagators --

def _prop_bayesian_linear(mu_in: torch.Tensor, v_in: torch.Tensor,
                           e_in: torch.Tensor, layer) -> tuple:
    """
    Forward moment propagation through a Bayesian Linear layer.

    mu_in, v_in, e_in : (B, in_features)

    mu_z_k  = sum_j mu_jk * mu_j + b_k
    v_z_k   = sum_j (e_j * sigma_jk^2 + v_j * mu_jk^2) + sigma_b_k^2
    """
    mu_w, sigma_w, mu_b, sigma_b = get_weight_moments(layer)
    sigma_w2 = sigma_w ** 2

    # (B, out) = (B, in) @ (in, out)^T
    mu_z  = F.linear(mu_in, mu_w, mu_b)
    v_z   = F.linear(e_in, sigma_w2) + F.linear(v_in, mu_w ** 2)
    if sigma_b is not None:
        v_z = v_z + (sigma_b ** 2).unsqueeze(0)

    e_z = mu_z ** 2 + v_z
    return mu_z, v_z, e_z


def _prop_bayesian_conv(mu_in: torch.Tensor, v_in: torch.Tensor,
                         e_in: torch.Tensor, layer) -> tuple:
    """
    Forward moment propagation through a Bayesian Conv2d layer.

    mu_in, v_in, e_in : (B, C_in, H, W)

    mu_z   = conv(mu_in, mu_w) + b          [standard conv with mean weights]
    v_z    = conv(e_in, sigma_w^2)           [second-moment term]
           + conv(v_in, mu_w^2)              [variance-of-activation term]
           + sigma_b^2  [broadcast over spatial]

    The two convolutions share the same stride/padding/groups as the layer.
    """
    mu_w, sigma_w, mu_b, sigma_b = get_weight_moments(layer)
    sigma_w2 = sigma_w ** 2
    kw       = conv_kwargs(layer)

    mu_z = F.conv2d(mu_in, mu_w, mu_b, **kw)
    v_z  = F.conv2d(e_in,  sigma_w2, None, **kw) \
         + F.conv2d(v_in,  mu_w ** 2, None, **kw)
    if sigma_b is not None:
        v_z = v_z + (sigma_b ** 2)[None, :, None, None]

    e_z = mu_z ** 2 + v_z
    return mu_z, v_z, e_z


def _prop_det_conv(mu_in, v_in, e_in, layer):
    """Deterministic Conv2d: weight variance is zero."""
    w  = layer.weight
    b  = layer.bias
    kw = conv_kwargs(layer)
    mu_z = F.conv2d(mu_in, w, b, **kw)
    v_z  = F.conv2d(v_in,  w ** 2, None, **kw)
    e_z  = mu_z ** 2 + v_z
    return mu_z, v_z, e_z


def _prop_det_linear(mu_in, v_in, e_in, layer):
    mu_z = F.linear(mu_in, layer.weight, layer.bias)
    v_z  = F.linear(v_in,  layer.weight ** 2)
    e_z  = mu_z ** 2 + v_z
    return mu_z, v_z, e_z


def _prop_bn(mu_in, v_in, e_in, layer):
    """
    BatchNorm (eval mode): deterministic affine transform.

    BN(z) = gamma/s * z + (beta - gamma * mean / s)   where s = sqrt(var + eps)
    => linear in z, so moments transform exactly.
    """
    gamma = layer.weight          # (C,)
    beta  = layer.bias            # (C,)
    rm    = layer.running_mean    # (C,)
    rv    = layer.running_var     # (C,)
    eps   = layer.eps

    s     = torch.sqrt(rv + eps)  # (C,)
    scale = gamma / s             # (C,)

    if mu_in.dim() == 4:          # Conv BN  (B, C, H, W)
        scale = scale[None, :, None, None]
        bias  = (beta - gamma * rm / s)[None, :, None, None]
    else:                         # Linear BN (B, C)
        scale = scale[None, :]
        bias  = (beta - gamma * rm / s)[None, :]

    mu_out = scale * mu_in + bias
    v_out  = (scale ** 2) * v_in
    e_out  = mu_out ** 2 + v_out
    return mu_out, v_out, e_out


def _prop_relu(mu_in, v_in, e_in):
    return _relu_moments(mu_in, v_in)


def _prop_silu(mu_in, v_in, e_in):
    return _silu_moments(mu_in, v_in)


def _prop_pool(mu_in, v_in, e_in, layer):
    """
    Pooling layers.

    MaxPool2d : we approximate by propagating the mean; variance is
                preserved (conservative – pooling reduces variance in
                practice but bounding it from above is safer for LRP).
    AdaptiveAvgPool2d : exact – AvgPool is linear.
    """
    if isinstance(layer, nn.AdaptiveAvgPool2d):
        h_in, w_in = mu_in.shape[-2], mu_in.shape[-1]
        mu_out = layer(mu_in)
        h_out, w_out = mu_out.shape[-2], mu_out.shape[-1]
        n = (h_in // h_out) * (w_in // w_out)
        v_out  = layer(v_in) / n
        e_out  = mu_out ** 2 + v_out
        return mu_out, v_out, e_out
    elif isinstance(layer, nn.MaxPool2d):
        mu_out = layer(mu_in)
        v_out  = layer(v_in)           # conservative upper bound
        e_out  = mu_out ** 2 + v_out
        return mu_out, v_out, e_out
    else:
        raise NotImplementedError(f"Pooling type {type(layer)} not supported.")


# --------------------------------------------------------- main class --------

class MomentForwardPass:
    """
    Run a moment-propagation forward pass through a BNN and record
    (mu, v, e) at every layer.

    Parameters
    ----------
    model : nn.Module
        A Bayesian model returned by models.py (BEfficientNet or BVGG).
        Must be in eval() mode before calling run().

    Records
    -------
    layer_stats : list of dict
        layer_stats[i] = {"layer": nn.Module,
                          "mu": Tensor, "v": Tensor, "e": Tensor}
        in the order the layers were visited.  Index 0 is the input.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_stats: List[Dict] = []

    # ------------------------------------------------------------------
    def run(self, x: torch.Tensor) -> List[Dict]:
        """
        x : (1, C, H, W)  – single image, no grad needed.

        Returns layer_stats list.
        """
        self.layer_stats = []
        self.model.eval()

        # Input statistics (zero variance at the input)
        mu = x.clone()
        v  = torch.zeros_like(x)
        e  = mu ** 2

        self._record(None, mu, v, e)

        # Walk the model
        self._walk(self.model, mu, v, e)

        return self.layer_stats

    # ------------------------------------------------------------------
    def _record(self, layer, mu, v, e):
        self.layer_stats.append({"layer": layer,
                                  "mu": mu.detach(),
                                  "v":  v.detach(),
                                  "e":  e.detach()})

    # ------------------------------------------------------------------
    def _walk(self, module: nn.Module, mu, v, e):
        """
        Recursively walk module children in forward order.
        Each leaf layer is processed; containers just recurse.
        """
        children = list(module.named_children())

        if not children:
            # Leaf – process it
            mu, v, e = self._process_leaf(module, mu, v, e)
            return mu, v, e

        if isinstance(module, _RESIDUAL_BLOCKS):
            return self._process_residual(module, mu, v, e)

        if module is not self.model and not isinstance(module, _SEQUENTIAL_CONTAINERS):
            _warn_unsupported(module)

        for _, child in children:
            mu, v, e = self._walk(child, mu, v, e)

        return mu, v, e

    # ------------------------------------------------------------------
    def _process_residual(self, block, mu, v, e):
        """
        torchvision BasicBlock / Bottleneck:   y = ReLU( F(x) + S(x) )

        F is the conv-BN(-ReLU) body, S the identity or the 1x1 projection
        (downsample).  Both are propagated from the SAME input moments.  The
        two branch outputs are then summed assuming they are independent, so
        means AND variances add:

            mu_y = mu_F + mu_S,        v_y = v_F + v_S .

        (The branches share the input x and its upstream weights, so this
        ignores Cov[F(x), S(x)]; make_report quantifies the resulting error
        against Monte-Carlo.)  The block is stored as ONE record that keeps its
        sub-records, because the backward pass must split relevance at the add.
        """
        if isinstance(block, BasicBlock):
            body_mods = [block.conv1, block.bn1, block.relu, block.conv2, block.bn2]
        else:  # Bottleneck
            body_mods = [block.conv1, block.bn1, block.relu, block.conv2, block.bn2,
                         block.relu, block.conv3, block.bn3]

        outer = self.layer_stats

        self.layer_stats = []
        mb, vb, eb = mu, v, e
        for m in body_mods:
            mb, vb, eb = self._walk(m, mb, vb, eb)
        body = self.layer_stats

        self.layer_stats = []
        ms, vs, es = mu, v, e
        if block.downsample is not None:
            for m in block.downsample:
                ms, vs, es = self._walk(m, ms, vs, es)
        short = self.layer_stats                      # empty list = identity shortcut

        self.layer_stats = outer

        mu_s = mb + ms
        v_s  = vb + vs
        mu_o, v_o, e_o = _relu_moments(mu_s, v_s)     # block output = ReLU(sum)

        d = lambda a, b, c: {"mu": a.detach(), "v": b.detach(), "e": c.detach()}
        outer.append({
            "layer": block, "kind": "residual",
            "mu": mu_o.detach(), "v": v_o.detach(), "e": e_o.detach(),
            "body": body, "short": short,
            "main": d(mb, vb, eb), "sc": d(ms, vs, es),
        })
        return mu_o, v_o, e_o

    # ------------------------------------------------------------------
    def _process_leaf(self, layer, mu, v, e):
        # torchvision models (VGG, EfficientNet) flatten with the *functional*
        # torch.flatten between avgpool and the classifier, so it is invisible
        # to the module walk.  Insert an explicit Flatten record when a Linear
        # layer receives a spatial tensor.
        if isinstance(layer, nn.Linear) or is_bayesian_linear(layer):
            if mu.dim() > 2:
                mu, v, e = (torch.flatten(t, 1) for t in (mu, v, e))
                self._record(nn.Flatten(), mu, v, e)

        # ---- Bayesian layers ----
        if is_bayesian_conv(layer):
            mu, v, e = _prop_bayesian_conv(mu, v, e, layer)
            self._record(layer, mu, v, e)

        elif is_bayesian_linear(layer):
            mu, v, e = _prop_bayesian_linear(mu, v, e, layer)
            self._record(layer, mu, v, e)

        # ---- Deterministic layers ----
        elif isinstance(layer, nn.Conv2d):
            mu, v, e = _prop_det_conv(mu, v, e, layer)
            self._record(layer, mu, v, e)

        elif isinstance(layer, nn.Linear):
            mu, v, e = _prop_det_linear(mu, v, e, layer)
            self._record(layer, mu, v, e)

        elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mu, v, e = _prop_bn(mu, v, e, layer)
            self._record(layer, mu, v, e)

        elif isinstance(layer, nn.ReLU):
            mu, v, e = _prop_relu(mu, v, e)
            self._record(layer, mu, v, e)

        elif isinstance(layer, nn.SiLU):
            mu, v, e = _prop_silu(mu, v, e)
            self._record(layer, mu, v, e)

        elif isinstance(layer, (nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
            mu, v, e = _prop_pool(mu, v, e, layer)
            self._record(layer, mu, v, e)

        elif isinstance(layer, nn.Flatten):
            mu = torch.flatten(mu, 1)
            v  = torch.flatten(v,  1)
            e  = torch.flatten(e,  1)
            self._record(layer, mu, v, e)

        elif isinstance(layer, nn.Dropout):
            # Dropout is disabled in eval mode – identity
            self._record(layer, mu, v, e)

        else:
            # Unknown leaf: pass through silently (e.g. StochasticDepth,
            # SqueezeExcitation – use mean only, zero additional variance)
            with torch.no_grad():
                try:
                    mu = layer(mu)
                    e  = mu ** 2
                    v  = torch.zeros_like(mu)
                    self._record(layer, mu, v, e)
                except Exception:
                    pass  # skip layers that can't handle float tensors

        return mu, v, e
