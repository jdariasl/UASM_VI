"""
models.py
Deterministic and Bayesian versions of EfficientNet-B6 and VGG-11.

Both backbones are converted to BNNs via dnn_to_bnn (MOPED initialisation):
  - pretrained ImageNet weights are loaded first
  - dnn_to_bnn replaces every Conv2d / Linear with its Bayesian equivalent
  - posterior means are initialised from the pretrained weights (moped_enable=True)

Estimator can be switched via the `estimator` argument:
  "Reparameterization"  (default, matches COVID_BayesianNET)
  "Flipout"
"""

import torch
import torch.nn as nn
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn


# --------------------------------------------------------------------------- #
# Prior / posterior hyper-parameters
# --------------------------------------------------------------------------- #

def bnn_prior_parameters(estimator: str = "Reparameterization",
                         moped_delta: float = 0.5) -> dict:
    """
    Return the const_bnn_prior_parameters dict expected by dnn_to_bnn.

    estimator : "Reparameterization" | "Flipout"
    moped_delta : perturbation fraction for MOPED sigma init (0 < delta <= 1)
    """
    assert estimator in ("Reparameterization", "Flipout"), \
        f"Unknown estimator '{estimator}'. Choose 'Reparameterization' or 'Flipout'."
    return {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": estimator,
        "moped_enable": True,   # init posterior means from pretrained weights
        "moped_delta": moped_delta,
    }


# --------------------------------------------------------------------------- #
# EfficientNet-B6
# --------------------------------------------------------------------------- #

def EfficientNet(n_classes: int = 3) -> nn.Module:
    """Deterministic EfficientNet-B6 with a custom classification head."""
    model = torchvision.models.efficientnet_b6(weights="DEFAULT")
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=2304, out_features=n_classes),
    )
    return model


def BEfficientNet(n_classes: int = 3,
                  saved_model: str = "",
                  estimator: str = "Reparameterization",
                  moped_delta: float = 0.5) -> nn.Module:
    """
    Bayesian EfficientNet-B6.

    saved_model : path to a deterministic checkpoint whose weights are used
                  to warm-start MOPED (recommended).  Leave empty to start
                  from raw ImageNet pretrained weights.
    estimator   : "Reparameterization" | "Flipout"
    """
    model = EfficientNet(n_classes=n_classes)
    if saved_model:
        checkpoint = torch.load(saved_model, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)
    dnn_to_bnn(model, bnn_prior_parameters(estimator, moped_delta))
    return model


# --------------------------------------------------------------------------- #
# VGG-11
# --------------------------------------------------------------------------- #

def VGG(n_classes: int = 3) -> nn.Module:
    """
    Deterministic VGG-11 with batch normalisation.
    The classifier head is replaced to match n_classes.
    """
    model = torchvision.models.vgg11_bn(weights="DEFAULT")
    model.classifier[6] = nn.Linear(4096, n_classes)
    return model


def BVGG(n_classes: int = 3,
         saved_model: str = "",
         estimator: str = "Reparameterization",
         moped_delta: float = 0.5) -> nn.Module:
    """
    Bayesian VGG-11-BN.

    VGG-11 has no residual connections, making it the cleanest architecture
    for Bayesian LRP-ε: relevance propagates strictly layer-by-layer without
    any splitting at skip connections.

    saved_model : optional deterministic checkpoint for MOPED warm-start.
    estimator   : "Reparameterization" | "Flipout"
    """
    model = VGG(n_classes=n_classes)
    if saved_model:
        checkpoint = torch.load(saved_model, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)
    dnn_to_bnn(model, bnn_prior_parameters(estimator, moped_delta))
    return model


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

BACKBONE_REGISTRY = {
    "efficientnet_b6":   (EfficientNet, BEfficientNet),
    "vgg11":             (VGG,          BVGG),
}


def select_model(backbone: str,
                 bayesian: bool = True,
                 n_classes: int = 3,
                 saved_model: str = "",
                 estimator: str = "Reparameterization",
                 moped_delta: float = 0.5) -> nn.Module:
    """
    Unified model factory.

    backbone    : "efficientnet_b6" | "vgg11"
    bayesian    : True for BNN, False for deterministic baseline
    n_classes   : number of output classes
    saved_model : path to a pretrained checkpoint (for MOPED warm-start)
    estimator   : "Reparameterization" | "Flipout"  (ignored when bayesian=False)
    moped_delta : MOPED delta (ignored when bayesian=False)
    """
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    det_fn, bnn_fn = BACKBONE_REGISTRY[backbone]
    if bayesian:
        return bnn_fn(n_classes=n_classes,
                      saved_model=saved_model,
                      estimator=estimator,
                      moped_delta=moped_delta)
    else:
        return det_fn(n_classes=n_classes)
