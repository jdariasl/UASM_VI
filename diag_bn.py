"""Diagnose train/eval mismatch of a Bayesian checkpoint: evaluate val accuracy with
BN in eval mode (normal) vs BN using batch statistics.  A big jump for the second
confirms the BatchNorm / weight-noise explanation.
Usage: python diag_bn.py --config configs/cub200_resnet50.yaml --checkpoint checkpoints/<run>/best_model.pt
"""
import argparse, yaml, torch, torch.nn as nn
from models.models import select_model
from data.cub200 import CUB200Dataset
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
ap.add_argument("--n_batches", type=int, default=20)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); dev = "cuda" if torch.cuda.is_available() else "cpu"
m = select_model(cfg["backbone"], cfg.get("bayesian", True), cfg["n_classes"],
                 estimator=cfg.get("estimator", "Reparameterization"),
                 moped_delta=float(cfg.get("moped_delta", 0.5))).to(dev)
ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
m.load_state_dict(ck.get("state_dict", ck))
ds = CUB200Dataset(root=cfg["data_root"], split="val")
dl = DataLoader(ds, 64, shuffle=True, num_workers=4)
for mode in ("eval", "batch-stats"):
    m.eval()
    if mode == "batch-stats":
        for x in m.modules():
            if isinstance(x, nn.modules.batchnorm._BatchNorm): x.train(); x.momentum = 0.0
    c = n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            if i >= a.n_batches: break
            p = m(x.to(dev)).argmax(1).cpu(); c += (p == y).sum().item(); n += len(y)
    print(f"BN {mode:12s} val acc = {c/n:.3f}")
