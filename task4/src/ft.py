"""Task 4 fine-tuning with scene-held-out validation (2026-11-12). Pretrained backbones are allowed for task 4 only.

Images come from the uint8 cache (t4data.images) and are augmented on the GPU, so there are no DataLoader workers.
Each fold model is trained on 4/5 of the scene groups, scored on the held-out scenes (OOF, last epoch, no best-epoch
picking) and also predicts the test set; the saved test probabilities are the fold average, in ANS order.
Test images are only ever passed through the trained network (no fitting on test).

Usage (GPU venv, from task4/src):
  python ft.py --name cnxt_t --backbone convnext_tiny.fb_in22k_ft_in1k [--folds 0,1] [--epochs 6] [--cv scene|random]
Outputs: outputs/oof_<name>.npy, outputs/test_<name>.npy, outputs/ft_<name>.json (per-fold macro F1), outputs/ft_<name>.log
"""
import argparse
import json
import math
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import scene_cv
import t4data

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--backbone", required=True)
p.add_argument("--cv", default="scene", choices=["scene", "hard", "random", "full"])
p.add_argument("--folds", default="0,1,2,3,4")
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--batch", type=int, default=48)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--h", type=int, default=224)
p.add_argument("--w", type=int, default=448)
p.add_argument("--drop-path", type=float, default=0.1)
p.add_argument("--layer-decay", type=float, default=1.0, help="layer-wise lr decay (timm param groups); 1 = off")
p.add_argument("--mix", type=float, default=0.0, help="probability of mixup/cutmix per batch")
p.add_argument("--hue", type=float, default=0.05, help="max hue rotation as a fraction of the colour circle")
p.add_argument("--gray", type=float, default=0.0, help="probability of random grayscale")
p.add_argument("--invert", type=float, default=0.05)
p.add_argument("--rot-max", type=float, default=8.0, help="max random rotation in degrees (the data itself has up to ~30)")
p.add_argument("--rot-p", type=float, default=0.3, help="probability of a random rotation")
p.add_argument("--chperm", type=float, default=0.0, help="probability of a random RGB channel permutation (the data has them)")
p.add_argument("--scale-min", type=float, default=0.75, help="min area fraction of the random crop")
p.add_argument("--erase", type=float, default=0.0, help="probability of erasing one random box")
p.add_argument("--balance", type=float, default=0.5, help="class weight = (max count / count) ** balance")
p.add_argument("--scale-max", type=float, default=1.0, help="max area fraction of the random crop")
p.add_argument("--tta-grid", action="store_true", help="predict with full frame + 3x2 grid of training-size crops")
p.add_argument("--scene-balance",type=float, default=0.0,
               help="sample train rows with weight (scene size) ** -alpha within each class (0 = off)")
p.add_argument("--scene-k", type=int, default=10, help="CLIP k-means scenes per class used by --scene-balance")
p.add_argument("--ema",type=float, default=0.0, help="EMA decay of weights (0 = off); EMA weights are used for prediction")
p.add_argument("--smooth", type=float, default=0.1)
p.add_argument("--no-test", action="store_true")
p.add_argument("--limit", type=int, default=0, help="smoke test: use this many train rows")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--grad-ckpt", action="store_true")
p.add_argument("--init", default="", help="encoder state dict loaded after creating the model (e.g. outputs/mim_<name>.pt)")
A = p.parse_args()

DEV = "cuda"
OUT = t4data.OUT
LOG = OUT / f"ft_{A.name}.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


torch.manual_seed(A.seed)
np.random.seed(A.seed)
torch.backends.cudnn.benchmark = True

y_all = t4data.train_labels()
Xtr = torch.from_numpy(np.ascontiguousarray(t4data.images("train")))  # uint8 (N, 120, 230, 3), CPU RAM
FOLDS = ([(np.arange(len(y_all)), np.array([], dtype=int))] if A.cv == "full" else
         {"scene": scene_cv.scene_folds, "hard": scene_cv.hard_folds, "random": scene_cv.random_folds}[A.cv]())
_bf = OUT / "bright_train.npy"
NONDARK = np.load(_bf) >= 25 if _bf.exists() else np.ones(len(y_all), bool)  # near-black frames carry no game content


def f1_both(idx, prob):
    nd = NONDARK[idx]
    return (f1_score(y_all[idx], prob.argmax(1), average="macro"),
            f1_score(y_all[idx][nd], prob[nd].argmax(1), average="macro"))
if A.limit:
    keep = set(np.random.default_rng(A.seed).choice(len(y_all), A.limit, replace=False).tolist())
    FOLDS = [(np.array([i for i in tr if i in keep]), np.array([i for i in va if i in keep])) for tr, va in FOLDS]


# ---------------------------------------------------------------- GPU transforms
def make_model():
    kw = dict(pretrained=True, num_classes=5, drop_path_rate=A.drop_path)
    try:
        m = timm.create_model(A.backbone, img_size=(A.h, A.w), **kw)  # ViT-style models need the input size
    except TypeError:
        m = timm.create_model(A.backbone, **kw)
    if A.init:  # in-domain pretrained encoder (mim.py); the classifier head stays freshly initialised
        miss, unexp = m.load_state_dict(torch.load(A.init, map_location="cpu"), strict=False)
        log(f"init from {A.init}: {len(miss)} missing keys (head expected), {len(unexp)} unexpected")
    if A.grad_ckpt and hasattr(m, "set_grad_checkpointing"):
        m.set_grad_checkpointing(True)
    return m.to(DEV).to(memory_format=torch.channels_last)


_cfg = timm.get_pretrained_cfg(A.backbone)
MEAN = torch.tensor(_cfg.mean, device=DEV).view(1, 3, 1, 1)
STD = torch.tensor(_cfg.std, device=DEV).view(1, 3, 1, 1)
ASPECT = t4data.W / t4data.H


def warp(x, fw, fh, tx, ty, rot):
    """Affine resample to (A.h, A.w). fw/fh: fraction of the source covered; tx/ty: centre shift; rot in radians."""
    cos, sin = torch.cos(rot), torch.sin(rot)
    theta = torch.stack([torch.stack([fw * cos, -fh * sin / ASPECT, tx], 1),
                         torch.stack([fw * sin * ASPECT, fh * cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (x.shape[0], 3, A.h, A.w), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def to_float(xb):
    return xb.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)


def eval_tf(xb, zoom=1.0):
    x = to_float(xb)
    B = x.shape[0]
    one, zero = torch.full((B,), 1 / zoom, device=DEV), torch.zeros(B, device=DEV)
    return ((warp(x, one, one, zero, zero, zero) - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def train_tf(xb):
    x = to_float(xb)
    B = x.shape[0]
    U = lambda lo, hi: torch.empty(B, device=DEV).uniform_(lo, hi)  # noqa: E731
    area, ratio = U(A.scale_min, A.scale_max), torch.exp(U(math.log(1.7 / ASPECT), math.log(2.2 / ASPECT)))
    fw, fh = (area * ratio).sqrt().clamp(max=1.0), (area / ratio).sqrt().clamp(max=1.0)
    tx, ty = (1 - fw) * U(-1, 1), (1 - fh) * U(-1, 1)
    rot = torch.where(U(0, 1) < A.rot_p, U(-A.rot_max, A.rot_max), torch.zeros(B, device=DEV)) * math.pi / 180
    x = warp(x, fw, fh, tx, ty, rot)
    # colour jitter (p = 0.8), skewed darker like the data
    on = (U(0, 1) < 0.8).float().view(B, 1, 1, 1)
    f = lambda lo, hi: (1 + on * (U(lo, hi).view(B, 1, 1, 1) - 1))  # noqa: E731
    x = x * f(0.5, 1.2)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * f(0.6, 1.3) + m
    g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = g + (x - g) * f(0.7, 1.3)
    if A.hue > 0:  # rotate chroma in YIQ
        h = (on.view(B) * U(-A.hue, A.hue)) * 2 * math.pi
        c, s = torch.cos(h).view(B, 1, 1), torch.sin(h).view(B, 1, 1)
        yy = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        ii = 0.596 * x[:, 0] - 0.274 * x[:, 1] - 0.322 * x[:, 2]
        qq = 0.211 * x[:, 0] - 0.523 * x[:, 1] + 0.312 * x[:, 2]
        ii, qq = ii * c - qq * s, ii * s + qq * c
        x = torch.stack([yy + 0.956 * ii + 0.621 * qq, yy - 0.272 * ii - 0.647 * qq, yy - 1.106 * ii + 1.703 * qq], 1)
    x = x.clamp(0, 1)
    if A.gray > 0:
        gm = (U(0, 1) < A.gray).view(B, 1, 1, 1)
        x = torch.where(gm, (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x), x)
    x = torch.where((U(0, 1) < A.invert).view(B, 1, 1, 1), 1 - x, x)
    if A.chperm > 0:  # random RGB channel order per image
        perm = torch.rand(B, 3, device=DEV).argsort(1)
        x = torch.where((U(0, 1) < A.chperm).view(B, 1, 1, 1), x.gather(1, perm.view(B, 3, 1, 1).expand_as(x)), x)
    sig = torch.where(U(0, 1) < 0.3, U(0, 0.06), torch.zeros(B, device=DEV)).view(B, 1, 1, 1)
    x = (x + torch.randn_like(x) * sig).clamp(0, 1)
    if A.erase > 0:
        for i in torch.nonzero(U(0, 1) < A.erase).flatten().tolist():
            eh, ew = int(A.h * np.random.uniform(0.15, 0.45)), int(A.w * np.random.uniform(0.15, 0.45))
            t, l = np.random.randint(0, A.h - eh), np.random.randint(0, A.w - ew)
            x[i, :, t:t + eh, l:l + ew] = torch.rand(3, 1, 1, device=DEV)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def mix_batch(x, y1h):
    """mixup or cutmix (50/50) with lambda ~ Beta(1, 1) for cutmix, Beta(0.4, 0.4) for mixup."""
    perm = torch.randperm(x.shape[0], device=DEV)
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(0.4, 0.4))
        return lam * x + (1 - lam) * x[perm], lam * y1h + (1 - lam) * y1h[perm]
    lam = float(np.random.beta(1.0, 1.0))
    ch, cw = int(A.h * math.sqrt(1 - lam)), int(A.w * math.sqrt(1 - lam))
    t, l = np.random.randint(0, A.h - ch + 1), np.random.randint(0, A.w - cw + 1)
    x = x.clone()
    x[:, :, t:t + ch, l:l + cw] = x[perm][:, :, t:t + ch, l:l + cw]
    lam = 1 - ch * cw / (A.h * A.w)
    return x, lam * y1h + (1 - lam) * y1h[perm]


def crop_tf(xb, fw, fh, tx, ty):
    x = to_float(xb)
    B = x.shape[0]
    c = lambda v: torch.full((B,), float(v), device=DEV)  # noqa: E731
    return ((warp(x, c(fw), c(fh), c(tx), c(ty), c(0.0)) - MEAN) / STD).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, X, idx=None, zooms=(1.0, 1.08, 1.15)):
    """Average of softmax over the zoom views; with --tta-grid also over a 3 x 2 grid of crops of the training crop
    size (area = mean of [scale_min, scale_max]), so a model trained on small crops is evaluated on what it saw."""
    model.eval()
    n = len(X) if idx is None else len(idx)
    out = torch.zeros(n, 5, device=DEV)
    views = [("zoom", z) for z in zooms]
    if A.tta_grid and zooms != (1.0,):  # (1.0,) = the quick per-epoch check
        side = math.sqrt((A.scale_min + A.scale_max) / 2)
        views = [("zoom", 1.0)] + [("crop", (tx, ty)) for tx in np.linspace(-(1 - side), 1 - side, 3)
                                   for ty in np.linspace(-(1 - side), 1 - side, 2)]
    for s in range(0, n, A.batch * 2):
        sl = slice(s, s + A.batch * 2)
        xb = X[sl] if idx is None else X[torch.as_tensor(idx[sl])]
        for kind, v in views:
            xin = eval_tf(xb, v) if kind == "zoom" else crop_tf(xb, side, side, v[0], v[1])
            with torch.autocast(DEV):
                out[sl] += model(xin).float().softmax(1) / len(views)
    return out.cpu().numpy()


def train_fold(k, tr, va):
    model = make_model()
    if A.layer_decay < 1:
        from timm.optim import create_optimizer_v2
        opt = create_optimizer_v2(model, opt="adamw", lr=A.lr, weight_decay=A.wd, layer_decay=A.layer_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=A.wd)
    steps_per_epoch = len(tr) // A.batch
    total = A.epochs * steps_per_epoch
    max_lrs = [g["lr"] * g.get("lr_scale", 1.0) for g in opt.param_groups]  # timm stores layer decay as lr_scale
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, total_steps=total, pct_start=0.1, anneal_strategy="cos")
    counts = np.bincount(y_all[tr], minlength=5)
    cw = torch.tensor((counts.max() / counts) ** A.balance, dtype=torch.float32, device=DEV)
    scaler = torch.amp.GradScaler("cuda")
    ema = None
    if A.ema > 0:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=A.ema, use_warmup=True)
    ytr = torch.as_tensor(y_all)
    scene_p = None
    if A.scene_balance > 0:
        import scene_cv
        g = scene_cv.scene_groups(A.scene_k)[tr]
        _, inv, gcnt = np.unique(g, return_inverse=True, return_counts=True)
        w = gcnt[inv].astype(float) ** -A.scene_balance
        ycls = y_all[tr]
        for c in range(5):  # keep each class's total sampling mass = its share of the training rows
            m = ycls == c
            w[m] *= m.sum() / w[m].sum()
        scene_p = w / w.sum()
    step = 0
    for ep in range(A.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        if A.scene_balance > 0:  # rare train scenes drawn more often (with replacement), class shares unchanged
            perm = torch.as_tensor(np.random.choice(tr, len(tr), replace=True, p=scene_p))
        else:
            perm = torch.as_tensor(np.random.permutation(tr))
        for b in range(steps_per_epoch):
            bi = perm[b * A.batch:(b + 1) * A.batch]
            x = train_tf(Xtr[bi])
            y1h = F.one_hot(ytr[bi].to(DEV), 5).float()
            y1h = y1h * (1 - A.smooth) + A.smooth / 5
            if A.mix > 0 and np.random.rand() < A.mix:
                x, y1h = mix_batch(x, y1h)
            with torch.autocast(DEV):
                logp = F.log_softmax(model(x).float(), 1)
            w = (y1h * cw).sum(1)
            loss = (-(y1h * logp).sum(1) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if ema is not None:
                ema.update(model, step=step)
            tot += loss.item()
        msg = f"fold {k} ep {ep}: loss {tot / steps_per_epoch:.4f}"
        if len(va):
            f1, f1n = f1_both(va, predict(model, Xtr, va, zooms=(1.0,)))
            msg += f"  val macro F1 {f1:.4f} (non-dark {f1n:.4f})"
        if ema is not None and len(va):
            pe = predict(ema.module, Xtr, va, zooms=(1.0,))
            msg += f"  (EMA {f1_score(y_all[va], pe.argmax(1), average='macro'):.4f})"
        log(msg + f"  {time.time() - t0:.0f}s  {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
    return ema.module if ema is not None else model


f_oof, f_test, f_done = OUT / f"oof_{A.name}.npy", OUT / f"test_{A.name}.npy", OUT / f"ft_{A.name}.json"
log(f"=== {A.name}: {vars(A)}")
done = json.loads(f_done.read_text()) if f_done.exists() and not A.limit else {}
oof = np.load(f_oof) if f_oof.exists() and done else np.zeros((len(y_all), 5), dtype=np.float32)
Xte = None if A.no_test or A.limit else torch.from_numpy(np.ascontiguousarray(t4data.images("test")))
test_sum = np.load(f_test) * len(done) if f_test.exists() and done else (None if Xte is None else np.zeros((len(Xte), 5), dtype=np.float32))
for k in [int(v) for v in A.folds.split(",")]:
    if str(k) in done:
        log(f"fold {k}: already done ({done[str(k)]:.4f})")
        continue
    tr, va = FOLDS[k]
    model = train_fold(k, tr, va)
    score = 0.0
    if len(va):
        oof[va] = predict(model, Xtr, va)
        score, score_nd = f1_both(va, oof[va])
        rec = [round(float((oof[va][y_all[va] == c].argmax(1) == c).mean()), 3) for c in range(5)]
        log(f"fold {k}: held-out macro F1 with TTA {score:.4f} | non-dark {score_nd:.4f}  recall {rec}")
        score = score_nd
    if A.limit:
        break
    if Xte is not None:
        test_sum += predict(model, Xte)
    done[str(k)] = float(score)
    np.save(f_oof, oof)
    if Xte is not None:
        np.save(f_test, test_sum / len(done))
    f_done.write_text(json.dumps(done))
    del model
    torch.cuda.empty_cache()
if done:
    v = list(done.values())
    log(f"=== {A.name} [{A.cv} folds {sorted(done)}]: macro F1 mean {np.mean(v):.4f} +- {np.std(v):.4f}  {np.round(v, 4).tolist()}")
