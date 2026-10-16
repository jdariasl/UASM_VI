"""
train.py
ELBO training loop for VI-BNNs on COVIDx or CUB-200-2011.

Loss
----
    L = CE(logits, y)  +  (1/N) * KL(q||p)

The KL term is scaled by 1/N (dataset size) so that the total ELBO
is roughly dataset-scale-independent.  bayesian_torch accumulates the KL
internally during each forward pass; we retrieve it via
    model.kl_loss()   (available after registering the hook, or via
                       get_kl_loss utility from bayesian_torch)

Usage
-----
    python train.py --config configs/chestxray_efficientnet.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# project imports
from models.models import select_model
from data.covidx import build_covidx_loaders
from data.cub200  import CUB200Dataset


# --------------------------------------------------------------------------- #
#  KL retrieval helper
# --------------------------------------------------------------------------- #

def get_kl_loss(model: nn.Module) -> torch.Tensor:
    """
    Sum the kl_loss() from every Bayesian layer in the model.
    bayesian_torch stores KL per-layer after each forward pass.
    """
    kl = torch.zeros(1, device=next(model.parameters()).device)
    for module in model.modules():
        if hasattr(module, "kl_loss"):
            kl = kl + module.kl_loss()
    return kl


# --------------------------------------------------------------------------- #
#  Training / evaluation
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    kl_weight:  float,
    n_train:    int,
) -> dict:
    model.train()
    total_loss = total_ce = total_kl = correct = n = 0

    for batch in loader:
        # Support optional mask return from COVIDxDataset
        if len(batch) == 3:
            imgs, labels, _ = batch
        else:
            imgs, labels = batch

        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        ce  = criterion(logits, labels)
        kl  = get_kl_loss(model) / n_train
        loss = ce + kl_weight * kl

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_ce   += ce.item()  * bs
        total_kl   += kl.item()  * bs
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        n          += bs

    return {
        "loss":  total_loss / n,
        "ce":    total_ce   / n,
        "kl":    total_kl   / n,
        "acc":   correct    / n,
    }


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    n_mc:      int = 5,
) -> dict:
    """
    MC evaluation: average logits over n_mc weight samples per image.
    This mirrors the inference used in COVID_BayesianNET.
    """
    model.eval()
    total_loss = correct = n = 0

    for batch in loader:
        if len(batch) == 3:
            imgs, labels, _ = batch
        else:
            imgs, labels = batch

        imgs   = imgs.to(device)
        labels = labels.to(device)

        # MC average
        logits_sum = torch.zeros(imgs.size(0),
                                 next(iter(model.parameters())).shape[0]
                                 if False else 0,
                                 device=device)
        out0 = model(imgs)
        logits_sum = out0.clone()
        for _ in range(n_mc - 1):
            logits_sum = logits_sum + model(imgs)
        logits_avg = logits_sum / n_mc

        loss = criterion(logits_avg, labels)
        preds = logits_avg.argmax(dim=1)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (preds == labels).sum().item()
        n          += bs

    return {"loss": total_loss / n, "acc": correct / n}


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model ----
    model = select_model(
        backbone    = cfg["backbone"],
        bayesian    = cfg.get("bayesian", True),
        n_classes   = cfg["n_classes"],
        saved_model = cfg.get("saved_model", ""),
        estimator   = cfg.get("estimator", "Reparameterization"),
        moped_delta = cfg.get("moped_delta", 0.5),
    ).to(device)

    # ---- Data ----
    dataset = cfg.get("dataset", "covidx")
    if dataset == "covidx":
        train_loader, val_loader, test_loader = build_covidx_loaders(
            root        = cfg["data_root"],
            batch_size  = cfg.get("batch_size", 32),
            num_workers = cfg.get("num_workers", 4),
            mask_root   = cfg.get("mask_root", None),
        )
        n_train = len(train_loader.dataset)
    elif dataset == "cub200":
        from torch.utils.data import DataLoader
        train_ds = CUB200Dataset(root=cfg["data_root"], split="train")
        val_ds   = CUB200Dataset(root=cfg["data_root"], split="val")
        test_ds  = CUB200Dataset(root=cfg["data_root"], split="test")
        bs = cfg.get("batch_size", 32)
        nw = cfg.get("num_workers", 4)
        train_loader = DataLoader(train_ds, bs, shuffle=True,  num_workers=nw, pin_memory=True)
        val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=nw, pin_memory=True)
        test_loader  = DataLoader(test_ds,  bs, shuffle=False, num_workers=nw, pin_memory=True)
        n_train = len(train_ds)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    # ---- Optimiser ----
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-4)),
                       weight_decay=float(cfg.get("weight_decay", 0.0)))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 30), eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()
    kl_weight = cfg.get("kl_weight", 1.0)

    # ---- Checkpoint dir ----
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, cfg.get("epochs", 30) + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, kl_weight, n_train,
        )
        val_stats = evaluate(model, val_loader, criterion, device,
                             n_mc=cfg.get("n_mc_eval", 5))

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_stats['loss']:.4f} "
            f"(ce={train_stats['ce']:.4f} kl={train_stats['kl']:.4f}) "
            f"acc={train_stats['acc']:.3f} | "
            f"val loss={val_stats['loss']:.4f} acc={val_stats['acc']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({"epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_acc": best_val_acc,
                        "cfg": cfg}, ckpt_path)
            print(f"  → Saved best checkpoint (val_acc={best_val_acc:.3f})")

    # ---- Test evaluation ----
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_stats = evaluate(model, test_loader, criterion, device,
                          n_mc=cfg.get("n_mc_eval", 10))
    print(f"\nTest accuracy: {test_stats['acc']:.4f}")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
