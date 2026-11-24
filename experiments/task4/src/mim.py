"""In-domain masked image modelling for task 4 (2026-11-13). TRAIN images only; labels are not used here.

Why: the only pretraining that helped on the shifted test set is masked-image pretraining (ConvNeXt-V2 FCMAE: LB task 4
+0.017), while CLIP, a bigger V2 and CAFormer did not. This continues masked pretraining on the 31.5k training frames
(game content + the dataset's distortions) before the usual supervised fine-tune (`ft.py --init outputs/mim_<name>.pt`).

Method (SimMIM-style, dense masking for a ConvNet): split the 224x448 input into 32x32 patches (7 x 14), replace 60 %
of them with zeros (= the mean colour after normalisation), encode with the ConvNeXt, predict every patch's pixels with
a 1x1 conv + pixel shuffle from the stride-32 feature map, and minimise the MSE on the masked patches only, against
per-patch normalised pixels (as in MAE / FCMAE). Light augmentation only (random crop + rare small rotation).
The test images are never loaded.

Usage (GPU venv, from task4/src):
  python mim.py --name v2bssl --backbone convnextv2_base.fcmae --epochs 10
Output: outputs/mim_<name>.pt (encoder weights, no head), outputs/mim_<name>.log
"""
import argparse
import math
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import t4data

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--backbone", required=True)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--mask", type=float, default=0.6)
p.add_argument("--patch", type=int, default=32)
p.add_argument("--h", type=int, default=224)
p.add_argument("--w", type=int, default=448)
p.add_argument("--drop-path", type=float, default=0.1)
p.add_argument("--limit", type=int, default=0, help="smoke test: use this many train rows and 20 steps")
p.add_argument("--seed", type=int, default=42)
A = p.parse_args()

DEV = "cuda"
OUT = t4data.OUT
LOG = OUT / f"mim_{A.name}.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


torch.manual_seed(A.seed)
np.random.seed(A.seed)
torch.backends.cudnn.benchmark = True

X = torch.from_numpy(np.ascontiguousarray(t4data.images("train")))  # uint8 (N, 120, 230, 3); test never loaded
idx_all = np.arange(len(X)) if not A.limit else np.random.default_rng(A.seed).choice(len(X), A.limit, replace=False)
_cfg = timm.get_pretrained_cfg(A.backbone)
MEAN = torch.tensor(_cfg.mean, device=DEV).view(1, 3, 1, 1)
STD = torch.tensor(_cfg.std, device=DEV).view(1, 3, 1, 1)
ASPECT = t4data.W / t4data.H
GH, GW = A.h // A.patch, A.w // A.patch


def augment(xb):
    """Random crop (area 0.6-1) + small rotation 30 % of the time, resampled to (h, w); returns normalised floats."""
    x = xb.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    B = x.shape[0]
    U = lambda lo, hi: torch.empty(B, device=DEV).uniform_(lo, hi)  # noqa: E731
    area = U(0.6, 1.0)
    fw = fh = area.sqrt()
    tx, ty = (1 - fw) * U(-1, 1), (1 - fh) * U(-1, 1)
    rot = torch.where(U(0, 1) < 0.3, U(-8, 8), torch.zeros(B, device=DEV)) * math.pi / 180
    cos, sin = torch.cos(rot), torch.sin(rot)
    theta = torch.stack([torch.stack([fw * cos, -fh * sin / ASPECT, tx], 1),
                         torch.stack([fw * sin * ASPECT, fh * cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 3, A.h, A.w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (x - MEAN) / STD


def patchify(x):  # (B, 3, H, W) -> (B, GH*GW, P*P*3)
    B = x.shape[0]
    x = x.reshape(B, 3, GH, A.patch, GW, A.patch).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, GH * GW, A.patch * A.patch * 3)


class MIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model(A.backbone, pretrained=True, num_classes=0, drop_path_rate=A.drop_path)
        self.enc.set_grad_checkpointing(True)
        c = self.enc.num_features
        self.dec = nn.Conv2d(c, A.patch * A.patch * 3, 1)
        nn.init.zeros_(self.dec.weight)  # start at loss ≈ 1 without large gradients into the pretrained encoder
        nn.init.zeros_(self.dec.bias)

    def forward(self, x, mask):  # mask: (B, GH, GW) bool, True = hidden
        m = mask.repeat_interleave(A.patch, 1).repeat_interleave(A.patch, 2).unsqueeze(1)
        f = self.enc.forward_features(x.masked_fill(m, 0.0))
        assert f.shape[-2:] == (GH, GW), f.shape
        return F.pixel_shuffle(self.dec(f), A.patch)  # (B, 3, H, W)


model = MIM().to(DEV).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=A.wd, betas=(0.9, 0.95))
steps_per_epoch = len(idx_all) // A.batch if not A.limit else 20
total = A.epochs * steps_per_epoch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")
scaler = torch.amp.GradScaler("cuda")
n_mask = int(round(A.mask * GH * GW))
log(f"=== mim {A.name}: {vars(A)} | grid {GH}x{GW}, {n_mask} masked patches, {len(idx_all)} train images")
for ep in range(A.epochs):
    model.train()
    t0, tot = time.time(), 0.0
    perm = np.random.permutation(idx_all)
    for b in range(steps_per_epoch):
        bi = torch.as_tensor(perm[(b * A.batch) % len(perm):(b * A.batch) % len(perm) + A.batch])
        x = augment(X[bi])
        B = x.shape[0]
        mask = torch.zeros(B, GH * GW, dtype=torch.bool, device=DEV)
        mask.scatter_(1, torch.rand(B, GH * GW, device=DEV).argsort(1)[:, :n_mask], True)
        mask = mask.view(B, GH, GW)
        tgt = patchify(x)
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.var(-1, keepdim=True) + 1e-6).sqrt()
        with torch.autocast(DEV):
            pred = model(x.contiguous(memory_format=torch.channels_last), mask)
        err = ((patchify(pred.float()) - tgt) ** 2).mean(-1)  # (B, GH*GW)
        mflat = mask.view(B, -1).float()
        loss = (err * mflat).sum() / mflat.sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item()
    log(f"ep {ep}: masked-patch loss {tot / steps_per_epoch:.4f}  {time.time() - t0:.0f}s  "
        f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
    if not A.limit:
        torch.save({k: v.cpu() for k, v in model.enc.state_dict().items() if not k.startswith("head.")},
                   OUT / f"mim_{A.name}.pt")
log("done")
