"""Picklable dataset + transforms for task4_kaggle_nb.py.

DataLoader workers on Windows are spawned (not forked), so the Dataset class and any custom transform must live in an
importable module instead of the notebook's __main__.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class AddGaussianNoise:
    """Additive Gaussian noise on a [0, 1] tensor, sigma drawn per image (test has more noisy frames)."""

    def __init__(self, max_sigma: float = 0.06):
        self.max_sigma = max_sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = float(torch.empty(1).uniform_(0, self.max_sigma))
        return (x + torch.randn_like(x) * sigma).clamp_(0, 1)


class ImgDS(Dataset):
    def __init__(self, root, ids, split, labels=None, tf=None):
        self.root, self.ids, self.split, self.labels, self.tf = Path(root), list(ids), split, labels, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.root / "data" / self.split / f"{self.ids[i]}.jpg").convert("RGB")
        return self.tf(img), (-1 if self.labels is None else int(self.labels[i]))
