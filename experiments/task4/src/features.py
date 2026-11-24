"""Task 4 CPU fallback: frozen pretrained-backbone embeddings + image hashes, cached to task4/outputs/.

    python task4/src/features.py [--backbone convnext_tiny]

Writes
    outputs/feat_<backbone>_{train,test}.npy  float32 (N, D) global-average-pooled embeddings, csv row order
    outputs/img_meta_{train,test}.csv         id, width, height, mean_r/g/b, ahash (16x8 grayscale average hash)
Pretrained ImageNet weights are allowed for Task 4 only (project rule 3).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as tm
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import paths, utils  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BACKBONES = {
    "convnext_tiny": (tm.convnext_tiny, tm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (tm.efficientnet_b0, tm.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def ahash(img: Image.Image) -> str:
    g = np.asarray(img.convert("L").resize((16, 8), Image.BILINEAR), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return np.packbits(bits).tobytes().hex()


def load_split(split: str, backbone, batch: int = 64):
    df = pd.read_csv(paths.task_data(4) / f"{split}.csv")
    img_dir = paths.task4_images(split)
    feats, meta, buf = [], [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            feats.append(backbone(x).numpy().astype(np.float32))
        buf.clear()

    t0 = time.time()
    for i, img_id in enumerate(df["id"]):
        img = Image.open(img_dir / f"{img_id}.jpg").convert("RGB")
        a = np.asarray(img, dtype=np.float32) / 255.0
        meta.append((img_id, img.width, img.height, *a.reshape(-1, 3).mean(0), ahash(img)))
        buf.append((a - MEAN) / STD)
        if len(buf) == batch:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"  {split} {i + 1}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    if buf:
        flush()
    meta = pd.DataFrame(meta, columns=["id", "width", "height", "mean_r", "mean_g", "mean_b", "ahash"])
    return np.concatenate(feats), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_tiny", choices=list(BACKBONES))
    args = ap.parse_args()
    utils.seed_everything()
    torch.set_num_threads(6)
    ctor, weights = BACKBONES[args.backbone]
    model = ctor(weights=weights)
    # drop the classification layer -> pooled embedding
    if args.backbone.startswith("convnext"):
        model.classifier[2] = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()
    model = model.eval().to(memory_format=torch.channels_last)
    out = paths.outputs_dir(4)
    for split in ("train", "test"):
        f, meta = load_split(split, model)
        np.save(out / f"feat_{args.backbone}_{split}.npy", f)
        meta.to_csv(out / f"img_meta_{split}.csv", index=False)
        print(split, f.shape, flush=True)


if __name__ == "__main__":
    main()
