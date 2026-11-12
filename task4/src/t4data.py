"""Task 4 data access for the robust-model round (2026-11-12).

Reads the TA's fixed csv format (`id`, `file_name`, `label`); rows stay in csv order, so test arrays are in ANS order.
Images are cached once as uint8 arrays (N, 120, 230, 3) in `task4/outputs/img_{split}_u8.npy` (gitignored).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import paths  # noqa: E402

DATA = paths.task_data(4)
OUT = paths.outputs_dir(4)
LABELS = ["7-11k", "fiveN", "freefried", "pube", "roblock"]  # index = submitted integer (datacard)
H, W = 120, 230


def load_csv(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{split}.csv")
    df["stem"] = df["file_name"].str.removesuffix(".jpg")
    return df


def images(split: str) -> np.ndarray:
    """uint8 (N, 120, 230, 3) in csv row order; built on first use."""
    f = OUT / f"img_{split}_u8.npy"
    df = load_csv(split)
    if f.exists():
        a = np.load(f, mmap_mode="r")
        if len(a) == len(df):
            return a
    a = np.zeros((len(df), H, W, 3), dtype=np.uint8)
    for i, name in enumerate(df["file_name"]):
        im = Image.open(paths.task4_images(split) / name).convert("RGB")
        assert im.size == (W, H), (name, im.size)
        a[i] = np.asarray(im)
    np.save(f, a)
    return a


def train_labels() -> np.ndarray:
    return load_csv("train")["label"].to_numpy().astype(int)


def old_test_order_to_new(arr: np.ndarray) -> np.ndarray:
    """Reindex an array saved in the OLD test.csv row order (before 2026-11-11) to the new ANS order."""
    old = pd.read_csv(DATA.parent / "_old_2026-11-08" / "04_image_classification" / "test.csv")["id"].astype(str)
    pos = pd.Series(np.arange(len(old)), index=old.values)
    return arr[pos.reindex(load_csv("test")["stem"]).to_numpy()]


if __name__ == "__main__":
    for s in ("train", "test"):
        a = images(s)
        print(s, a.shape, a.dtype, round(float(np.asarray(a[:2000]).mean()), 2))
