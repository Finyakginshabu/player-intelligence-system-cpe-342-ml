# %% [markdown]
# # Task 4: Game Title Detection, GPU fine-tuning round 1 (local RTX / Kaggle / Colab)
#
# Superseded by `task4_final.ipynb` (round R, ConvNeXt-S on all images); kept for the report's history.
#
# - Pretrained **ConvNeXt-Tiny** (timm, ImageNet-22k → 1k), fine-tuned end-to-end. Pretrained weights are
#   allowed for Task 4 only.
# - 5-fold **StratifiedGroupKFold**: images with an identical 16×8 average hash share a group, so
#   near-duplicates never sit on both sides of a split (otherwise CV is inflated).
# - Input 224×448 (upscaled from 120×230, aspect kept). Text-safe augmentation: small scale/crop jitter and rotation
#   (black borders, as in the data), colour jitter skewed **darker** + Gaussian noise + occasional colour inversion
#   (adversarial validation: test is darker and noisier, AUC 0.768), blur. **No horizontal flip** (HUD text).
# - Label smoothing 0.1 + class-weighted CE (inverse sqrt frequency), AdamW + one-cycle cosine, AMP, channels_last.
# - TTA: identity + 2 mild zooms. Outputs `oof_convnext_ft.npy`, `test_convnext_ft.npy`,
#   `final_pred_convnext_ft.csv` (the repo's `final_pred.csv` is left alone, so the repo still matches the LB).
# - Resumable: finished folds are recorded in `outputs/convnext_ft_done.json` and skipped on a rerun.
#
# **Local (Windows, RTX):** once, `.venv-gpu\Scripts\python -m ipykernel install --sys-prefix --name t4gpu`; then `cd task4` and
# `..\.venv-gpu\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1
# --ExecutePreprocessor.kernel_name=t4gpu task4_kaggle.ipynb` (the default `python3` kernel may be the CPU-only system Python).
# Progress: `outputs/convnext_ft.log`. Smoke test first: set `T4_FOLDS=0 T4_EPOCHS=1 T4_LIMIT=3000`.
# **Kaggle/Colab:** upload `data/04_image_classification/` (train.csv, test.csv, data/train, data/test) and set
# `TASK4_DATA` if it isn't found automatically. Source: `task4/src/task4_kaggle_nb.py` (jupytext).
#
# **Labels:** train.csv holds strings. Submitted integers follow the datacard
# `{7-11k:0, fiveN:1, freefried:2, pube:3, roblock:4}` (alphabetical order, see README "Conflicts").

# %%
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import timm  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchvision import transforms as T

for cand in (Path.cwd() / "src", Path.cwd() / "task4" / "src"):
    if (cand / "imgds.py").exists():
        sys.path.insert(0, str(cand))
try:
    from imgds import AddGaussianNoise, ImgDS  # importable module -> works with spawned workers (Windows)
except ImportError:  # Kaggle/Colab without the repo: Linux forks workers, so inline classes are fine
    from torch.utils.data import Dataset

    class AddGaussianNoise:
        def __init__(self, max_sigma=0.06):
            self.max_sigma = max_sigma

        def __call__(self, x):
            return (x + torch.randn_like(x) * float(torch.empty(1).uniform_(0, self.max_sigma))).clamp_(0, 1)

    class ImgDS(Dataset):
        def __init__(self, root, ids, split, labels=None, tf=None):
            self.root, self.ids, self.split, self.labels, self.tf = Path(root), list(ids), split, labels, tf

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            img = Image.open(self.root / "data" / self.split / f"{self.ids[i]}.jpg").convert("RGB")
            return self.tf(img), (-1 if self.labels is None else int(self.labels[i]))

SEED = 42
LABELS = ["7-11k", "fiveN", "freefried", "pube", "roblock"]  # index = submitted integer (datacard)
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
IMG_H, IMG_W = 224, 448
EPOCHS = int(os.environ.get("T4_EPOCHS", 8))
BATCH = int(os.environ.get("T4_BATCH", 48))
LR, WD = 2e-4, 0.05
N_FOLDS = 5
FOLDS_TO_RUN = [int(f) for f in os.environ.get("T4_FOLDS", "0,1,2,3,4").split(",")]
LIMIT = int(os.environ.get("T4_LIMIT", 0))  # >0: smoke test on a random subset of train rows (no outputs saved)
NUM_WORKERS = int(os.environ.get("T4_WORKERS", 6))
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def find_data_dir() -> Path:
    if os.environ.get("TASK4_DATA"):
        return Path(os.environ["TASK4_DATA"])
    for base in [Path("../data/04_image_classification"), Path("data/04_image_classification"),
                 Path("/kaggle/input"), Path("/content/drive/MyDrive")]:
        if base.exists():
            hits = [p.parent for p in base.rglob("train.csv") if (p.parent / "data" / "train").exists()]
            if hits:
                return hits[0]
    raise FileNotFoundError("set TASK4_DATA to the folder holding train.csv, test.csv and data/{train,test}/")


LOG = OUT / ("convnext_ft_smoke.log" if LIMIT else "convnext_ft.log")


def log(msg: str) -> None:
    """print + append to outputs/convnext_ft.log (nbconvert hides live output; tail the log instead)."""
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


DATA_DIR = find_data_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# A CPU run would take days: fail fast (e.g. when Jupyter picked a kernel with CPU-only torch).
assert DEVICE == "cuda" or os.environ.get("T4_ALLOW_CPU"), f"no CUDA in {sys.executable} (torch {torch.__version__})"
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True  # faster; GPU runs are not bit-reproducible anyway
log(f"{DATA_DIR} | {sys.executable} | torch {torch.__version__} | {DEVICE} "
    f"{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''} | epochs {EPOCHS} batch {BATCH} folds {FOLDS_TO_RUN} "
    f"limit {LIMIT or '-'} workers {NUM_WORKERS}")

# %% [markdown]
# ## 1. Data, labels, near-duplicate groups

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
# TA data fix (2026-11-11): csvs are now id (PLY#/ANS#), file_name, label (int). Use the image name as the id and
# rebuild the string class from the integer label, so the rest of the notebook is unchanged.
for _df in (train, test):
    if "file_name" in _df.columns:
        _df["id"] = _df["file_name"].str.removesuffix(".jpg")
if "label" in train.columns:
    train["class"] = [LABELS[int(i)] for i in train["label"]]
assert set(train["class"]) == set(LABELS), set(train["class"])
y = train["class"].map({c: i for i, c in enumerate(LABELS)}).to_numpy()
print(train.shape, test.shape)
print(pd.Series(y).value_counts().sort_index().rename(dict(enumerate(LABELS))))


def ahash(path) -> str:
    g = np.asarray(Image.open(path).convert("L").resize((16, 8), Image.BILINEAR), dtype=np.float32)
    return np.packbits((g > g.mean()).flatten()).tobytes().hex()


meta_csv = OUT / "img_meta_train.csv"  # written by src/features.py (same hash); recompute if absent
if meta_csv.exists() and (pd.read_csv(meta_csv)["id"].astype(str).values == train["id"].astype(str).values).all():
    hashes = pd.read_csv(meta_csv)["ahash"]
else:
    hashes = pd.Series([ahash(DATA_DIR / "data" / "train" / f"{i}.jpg") for i in train["id"]])
groups = hashes.factorize()[0]
dup = hashes.duplicated(keep=False)
print(f"rows in a duplicate-hash group: {dup.mean():.3%}; groups with mixed labels: "
      f"{(pd.Series(y)[dup].groupby(hashes[dup]).nunique() > 1).sum()}")
FOLDS = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train, y, groups))
if LIMIT:
    rng = np.random.default_rng(SEED)
    keep = set(rng.choice(len(train), LIMIT, replace=False))
    FOLDS = [(np.array([i for i in tr if i in keep]), np.array([i for i in va if i in keep])) for tr, va in FOLDS]

# %% [markdown]
# ## 2. Transforms

# %%
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = T.Compose([
    T.RandomResizedCrop((IMG_H, IMG_W), scale=(0.75, 1.0), ratio=(1.7, 2.2)),
    T.RandomApply([T.RandomRotation(8, fill=0)], p=0.3),
    T.RandomApply([T.ColorJitter(brightness=(0.5, 1.2), contrast=(0.6, 1.3), saturation=0.3, hue=0.05)], p=0.8),
    T.RandomInvert(p=0.05),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
    T.ToTensor(),
    T.RandomApply([AddGaussianNoise(0.06)], p=0.3),
    NORM,
])
TTA = [  # identity + mild zooms (resize larger, centre-crop back)
    T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor(), NORM]),
    T.Compose([T.Resize((int(IMG_H * 1.08), int(IMG_W * 1.08))), T.CenterCrop((IMG_H, IMG_W)), T.ToTensor(), NORM]),
    T.Compose([T.Resize((int(IMG_H * 1.15), int(IMG_W * 1.15))), T.CenterCrop((IMG_H, IMG_W)), T.ToTensor(), NORM]),
]


def loader(ids, split, labels, tf, shuffle):
    return DataLoader(ImgDS(DATA_DIR, ids, split, labels, tf), batch_size=BATCH if shuffle else BATCH * 2,
                      shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True, drop_last=shuffle,
                      persistent_workers=NUM_WORKERS > 0)


# %% [markdown]
# ## 3. Train / predict helpers

# %%
counts = np.bincount(y, minlength=len(LABELS))
CLASS_W = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float32)
print("class weights", CLASS_W.numpy().round(3))


@torch.no_grad()
def predict(model, ids, split):
    model.eval()
    probs = np.zeros((len(ids), len(LABELS)), dtype=np.float32)
    for tf in TTA:
        out = []
        for xb, _ in loader(ids, split, None, tf, shuffle=False):
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                xb = xb.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
                out.append(model(xb).float().softmax(1).cpu().numpy())
        probs += np.concatenate(out) / len(TTA)
    return probs


def train_fold(k, tr, va):
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=len(LABELS))
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    dl = loader(train["id"].iloc[tr], "train", y[tr], train_tf, shuffle=True)
    steps = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    crit = nn.CrossEntropyLoss(weight=CLASS_W.to(DEVICE), label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    best, best_state = -1.0, None
    va_eval = loader(train["id"].iloc[va], "train", y[va], TTA[0], shuffle=False)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            yb = yb.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        model.eval(); preds = []
        with torch.no_grad():
            for xb, _ in va_eval:
                with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                    xb = xb.to(DEVICE).contiguous(memory_format=torch.channels_last)
                    preds.append(model(xb).argmax(1).cpu().numpy())
        f1 = f1_score(y[va], np.concatenate(preds), average="macro")
        mem = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0
        log(f"fold {k} ep {ep}: loss {tot / len(dl):.4f}  val macro F1 {f1:.4f}  ({time.time() - t0:.0f}s, "
            f"{len(tr) / (time.time() - t0):.0f} img/s incl. val, peak mem {mem:.1f} GB)")
        # NOTE: picking the best epoch on the validation fold makes the OOF score slightly optimistic
        if f1 > best:
            best, best_state = f1, {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


# %% [markdown]
# ## 4. 5-fold CV (OOF with TTA) + fold-averaged test predictions (resumable)

# %%
f_oof, f_test, f_done = OUT / "oof_convnext_ft.npy", OUT / "test_convnext_ft.npy", OUT / "convnext_ft_done.json"
resume = not LIMIT and f_done.exists()
done = json.loads(f_done.read_text()) if resume else {}
oof = np.load(f_oof) if resume else np.zeros((len(train), len(LABELS)), dtype=np.float32)
test_sum = np.load(f_test) * len(done) if resume and done else np.zeros((len(test), len(LABELS)), dtype=np.float32)
for k in FOLDS_TO_RUN:
    if str(k) in done:
        log(f"fold {k}: already done (F1 {done[str(k)]:.4f}), skipped")
        continue
    tr, va = FOLDS[k]
    model, best = train_fold(k, tr, va)
    oof[va] = predict(model, train["id"].iloc[va], "train")
    score = f1_score(y[va], oof[va].argmax(1), average="macro")
    log(f"fold {k}: best-epoch F1 {best:.4f} | with TTA {score:.4f}")
    if LIMIT:
        log("smoke test: skipping test prediction and saving")
        break
    test_sum += predict(model, test["id"], "test")
    done[str(k)] = float(score)
    np.save(f_oof, oof); np.save(f_test, test_sum / len(done)); f_done.write_text(json.dumps(done))
    del model; torch.cuda.empty_cache()

if done and not LIMIT:
    idx = np.concatenate([FOLDS[int(k)][1] for k in done])
    log(f"CV macro F1 (TTA, {len(done)} folds): {f1_score(y[idx], oof[idx].argmax(1), average='macro'):.4f} "
        f"(fold mean {np.mean(list(done.values())):.4f} +- {np.std(list(done.values())):.4f})")

# %% [markdown]
# ## 5. Write `final_pred_convnext_ft.csv` (id, prediction as integer 0–4)
# Kept separate from `final_pred.csv` so the repo still matches the LB. Promote it after the task-4 format is settled.

# %%
if len(done) == N_FOLDS:
    test_prob = test_sum / len(done)
    final = pd.DataFrame({"id": test["id"], "prediction": test_prob.argmax(1).astype(int)})
    final.to_csv(OUT / "final_pred_convnext_ft.csv", index=False)
    print(final["prediction"].value_counts(normalize=True).sort_index().rename(dict(enumerate(LABELS))).round(3))
else:
    print(f"{len(done)}/{N_FOLDS} folds done -> final_pred_convnext_ft.csv not written yet")
