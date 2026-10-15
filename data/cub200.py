"""
cub200.py
Dataset loader for CUB-200-2011.

CUB-200-2011 is a fine-grained bird classification dataset with 200 classes
and per-image part annotations.  The part annotations are used for the
*pointing game* evaluation of saliency maps:

    Pointing game accuracy = I(argmax(saliency) ∈ annotated_part_region)

Directory layout expected (standard CUB download):
    root/
        images/                     # images/<class_folder>/<filename>.jpg
        images.txt                  # <img_id> <relative_path>
        image_class_labels.txt      # <img_id> <class_id>   (1-indexed)
        train_test_split.txt        # <img_id> <is_train>
        parts/
            part_locs.txt           # <img_id> <part_id> <x> <y> <visible>

Usage
-----
    ds = CUB200Dataset(root="data/CUB_200_2011", split="train")
    img, label = ds[0]

    # With part annotations (for pointing-game evaluation)
    ds = CUB200Dataset(root="data/CUB_200_2011", split="test",
                       return_parts=True)
    img, label, parts = ds[0]
    # parts: (n_parts, 3) tensor  [part_id, x, y]  – only visible parts
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# --------------------------------------------------------------------------- #
#  Default transforms
# --------------------------------------------------------------------------- #

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------- #
#  CUB200Dataset
# --------------------------------------------------------------------------- #

class CUB200Dataset(Dataset):
    """
    Parameters
    ----------
    root         : CUB_200_2011 root directory
    split        : "train" | "val" | "test"
                   "val" uses the last 10 % of the training set as held-out
    transform    : image transforms (defaults to random-crop for train, else
                   center-crop)
    return_parts : if True, __getitem__ returns (img, label, parts_tensor)
    """

    def __init__(
        self,
        root:         str,
        split:        str = "train",
        transform:    Optional[Callable] = None,
        return_parts: bool = False,
    ):
        super().__init__()
        self.root         = Path(root)
        self.split        = split
        self.return_parts = return_parts

        if transform is not None:
            self.transform = transform
        else:
            self.transform = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM

        self.samples:    List[Tuple[Path, int]] = []
        self.part_locs:  Dict[int, torch.Tensor] = {}   # img_id → (n, 3) [pid, x, y]

        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        root = self.root

        # img_id → relative image path
        id2path: Dict[int, str] = {}
        with open(root / "images.txt") as f:
            for line in f:
                iid, path = line.strip().split()
                id2path[int(iid)] = path

        # img_id → class label (0-indexed)
        id2label: Dict[int, int] = {}
        with open(root / "image_class_labels.txt") as f:
            for line in f:
                iid, cls = line.strip().split()
                id2label[int(iid)] = int(cls) - 1  # 1-indexed → 0-indexed

        # img_id → is_train (1) or test (0)
        id2split: Dict[int, int] = {}
        with open(root / "train_test_split.txt") as f:
            for line in f:
                iid, flag = line.strip().split()
                id2split[int(iid)] = int(flag)

        all_ids = sorted(id2path.keys())
        train_ids = [i for i in all_ids if id2split[i] == 1]
        test_ids  = [i for i in all_ids if id2split[i] == 0]

        # 10 % of training → val
        n_val  = max(1, len(train_ids) // 10)
        val_ids = train_ids[-n_val:]
        train_ids = train_ids[:-n_val]

        if self.split == "train":
            selected = train_ids
        elif self.split == "val":
            selected = val_ids
        else:
            selected = test_ids

        for iid in selected:
            img_path = root / "images" / id2path[iid]
            self.samples.append((img_path, id2label[iid], iid))

        # Part locations (visible parts only) – loaded lazily or eagerly
        if self.return_parts:
            self._load_parts(set(s[2] for s in self.samples))

    # ------------------------------------------------------------------
    def _load_parts(self, img_ids):
        part_file = self.root / "parts" / "part_locs.txt"
        raw: Dict[int, List[List[float]]] = {i: [] for i in img_ids}
        with open(part_file) as f:
            for line in f:
                parts = line.strip().split()
                iid, pid, x, y, visible = int(parts[0]), int(parts[1]), \
                                          float(parts[2]), float(parts[3]), int(parts[4])
                if iid in raw and visible:
                    raw[iid].append([pid, x, y])
        for iid, rows in raw.items():
            if rows:
                self.part_locs[iid] = torch.tensor(rows, dtype=torch.float32)  # (n, 3)
            else:
                self.part_locs[iid] = torch.zeros(0, 3)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path, label, iid = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if self.return_parts:
            parts = self.part_locs.get(iid, torch.zeros(0, 3))
            return img, label, parts

        return img, label


# --------------------------------------------------------------------------- #
#  Pointing-game evaluator
# --------------------------------------------------------------------------- #

def pointing_game_hit(
    saliency:  torch.Tensor,   # (C, H, W) or (H, W)
    parts:     torch.Tensor,   # (n_parts, 3) [part_id, x_orig, y_orig]
    img_size:  Tuple[int, int] = (224, 224),
    orig_size: Tuple[int, int] = (None, None),  # original image H, W before resize
    tolerance: int = 15,                         # pixel radius (in saliency space)
) -> bool:
    """
    Returns True if the maximum saliency pixel falls within `tolerance`
    pixels of at least one visible part annotation.

    The part coordinates are originally in the full-resolution image space.
    We rescale them to match `img_size`.
    """
    if parts.shape[0] == 0:
        return False

    sal = saliency.abs()
    if sal.dim() == 3:
        sal = sal.mean(dim=0)   # (H, W)

    # Peak location in saliency map
    peak = sal.argmax()
    ph   = int(peak) // img_size[1]
    pw   = int(peak) % img_size[1]

    # Rescale part coordinates
    sx = img_size[1] / orig_size[1] if orig_size[1] else 1.0
    sy = img_size[0] / orig_size[0] if orig_size[0] else 1.0

    for row in parts:
        _, x_orig, y_orig = row.tolist()
        px = x_orig * sx
        py = y_orig * sy
        dist = ((ph - py) ** 2 + (pw - px) ** 2) ** 0.5
        if dist <= tolerance:
            return True
    return False
