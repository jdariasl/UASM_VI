"""
covidx.py
Dataset loader for COVIDx and COVID-QU-Ex chest X-ray datasets.

COVIDx (used in COVID_BayesianNET)
-----------------------------------
Three-class classification: 0=Normal, 1=Pneumonia, 2=COVID-19.
Dataset files are plain-text manifests with one line per sample:
    <patient_id>  <filename>  <label>  <source>

COVID-QU-Ex (for saliency validation)
---------------------------------------
Same three-class labeling but adds pixel-level lung/infection segmentation
masks stored alongside the images.  The mask folder mirrors the image folder.

Usage
-----
    # Classification only
    ds = COVIDxDataset(root="data/covidx", split="train")

    # With segmentation masks (for saliency evaluation)
    ds = COVIDxDataset(root="data/covid_qu_ex", split="test",
                       mask_root="data/covid_qu_ex/masks")
    img, label, mask = ds[0]   # mask is None when mask_root is not given
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# --------------------------------------------------------------------------- #
#  Label maps
# --------------------------------------------------------------------------- #

LABEL_MAP = {"normal": 0, "pneumonia": 1, "COVID-19": 2, "covid-19": 2}

# Default ImageNet normalisation (EfficientNet / VGG were pre-trained on it)
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_MASK_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


# --------------------------------------------------------------------------- #
#  COVIDxDataset
# --------------------------------------------------------------------------- #

class COVIDxDataset(Dataset):
    """
    Parameters
    ----------
    root       : directory that contains train/test image folders and
                 manifest files (train_split.txt / test_split.txt)
    split      : "train" | "val" | "test"
    transform  : torchvision transforms applied to the image
                 (defaults to 224×224 + ImageNet normalisation)
    mask_root  : (optional) directory that mirrors root but holds binary PNG
                 masks for infection/lung segmentation (COVID-QU-Ex layout)
    """

    def __init__(
        self,
        root:       str,
        split:      str = "train",
        transform:  Optional[Callable] = None,
        mask_root:  Optional[str] = None,
    ):
        super().__init__()
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or _DEFAULT_TRANSFORM
        self.mask_root = Path(mask_root) if mask_root else None

        self.samples: List[Tuple[Path, int]] = []
        self._load_manifest()

    # ------------------------------------------------------------------
    def _load_manifest(self):
        """
        COVIDx manifest format (space-separated):
            patient_id  filename  label  source
        COVID-QU-Ex may use a CSV or a sub-folder structure; we support both.
        """
        # Try plain-text manifest first
        manifest = self.root / f"{self.split}_split.txt"
        if manifest.exists():
            with open(manifest) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    fname, raw_label = parts[1], parts[2].lower()
                    label = LABEL_MAP.get(raw_label, -1)
                    if label < 0:
                        continue
                    img_path = self.root / self.split / fname
                    if img_path.exists():
                        self.samples.append((img_path, label))
            return

        # Fallback: sub-folder layout  root/split/normal|pneumonia|COVID-19/
        split_dir = self.root / self.split
        if split_dir.is_dir():
            for cls_name, cls_idx in LABEL_MAP.items():
                cls_dir = split_dir / cls_name
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpg")):
                    self.samples.append((img_path, cls_idx))
            return

        raise FileNotFoundError(
            f"Could not find a COVIDx manifest or sub-folder layout under '{self.root}'."
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if self.mask_root is not None:
            # COVID-QU-Ex: mask lives at mask_root / split / filename (PNG, grayscale)
            rel = img_path.relative_to(self.root)
            mask_path = self.mask_root / rel.parent / img_path.name
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                mask = _MASK_TRANSFORM(mask)           # (1, H, W) float in [0, 1]
            else:
                mask = torch.zeros(1, 224, 224)
            return img, label, mask

        return img, label


# --------------------------------------------------------------------------- #
#  Convenience factory
# --------------------------------------------------------------------------- #

def build_covidx_loaders(
    root:        str,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    mask_root:   Optional[str] = None,
):
    """
    Returns (train_loader, val_loader, test_loader).

    If mask_root is provided the dataset returns (img, label, mask) tuples;
    the val/test loaders always include masks when mask_root is set.
    """
    from torch.utils.data import DataLoader

    splits = {}
    for split in ("train", "val", "test"):
        try:
            ds = COVIDxDataset(root=root, split=split, mask_root=mask_root)
            splits[split] = ds
        except FileNotFoundError:
            splits[split] = None

    def _loader(ds, shuffle):
        if ds is None:
            return None
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True, drop_last=(shuffle))

    return (
        _loader(splits["train"], shuffle=True),
        _loader(splits["val"],   shuffle=False),
        _loader(splits["test"],  shuffle=False),
    )
