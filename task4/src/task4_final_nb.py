# %% [markdown]
# # Task 4: Game Title Detection, final model (round R, 2026-11-12)
#
# - **Type:** 5-class image classification, 230×120 JPEG frames. **Metric:** macro F1. Pretrained weights are allowed
#   for task 4 only.
# - **Problem:** the round-1 ConvNeXt-T fine-tune scored CV 0.98 (random folds) but ≈ 0.85 on the LB. The test set holds
#   scene types that are rare in train (adversarial AUC 0.80 on CLIP embeddings), so random folds overstate accuracy.
# - **Validation used here:** scenes = k-means clusters of CLIP ViT-L/14 embeddings inside each class; whole scenes are
#   held out (`src/scene_cv.py`). Near-black frames (no game content, 0.8% of test) are excluded from the comparison.
# - **Final:** ConvNeXt-S (ImageNet-22k) fine-tuned on **all** training images, 4 seeds, probabilities averaged
#   (`src/ft.py --cv full`). Second candidate: the same plus one ConvNeXt-B (CLIP LAION-2B) model, 50/50.
# - Test images only ever pass through the trained networks (no fitting on test, project rule 2).
#
# Training runs on the GPU venv (`.venv-gpu`) via `src/ft.py`; this notebook collects the saved results. Set
# `RETRAIN = True` to rerun every training command (≈ 2.5 h on an RTX 5070).
# Source: `src/task4_final_nb.py` (jupytext percent format).

# %%
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
while not (ROOT / "common").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "task4" / "src"))

import numpy as np
import pandas as pd

import t4data
from common import utils

TASK = 4
OUT = t4data.OUT
LABELS = t4data.LABELS
RETRAIN = False
GPU_PY = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
SEEDS = [42, 7, 123, 2026]
SMALL = "--backbone convnext_small.fb_in22k_ft_in1k --epochs 6 --batch 48 --lr 2e-4 --drop-path 0.2 --grad-ckpt"
BASECLIP = ("--backbone convnext_base.clip_laion2b_augreg_ft_in12k_in1k --epochs 6 --batch 32 --lr 1e-4 "
            "--drop-path 0.2 --grad-ckpt")

# %% [markdown]
# ## 1. Data (TA's fixed csv format: `id`, `file_name`, `label`)

# %%
train, test = t4data.load_csv("train"), t4data.load_csv("test")
y = train["label"].to_numpy()
print(train.shape, test.shape, "| test ids", test["id"].iloc[0], "...", test["id"].iloc[-1])
print(pd.Series(y).value_counts(normalize=True).sort_index().rename(dict(enumerate(LABELS))).round(3))

# %% [markdown]
# ## 2. Model choice on held-out scenes (hard fold 4 = the hardest region of every class held out)
# Macro F1 on non-dark held-out rows, 6 epochs each. Runs: `python src/ft.py --name <name> --cv hard --folds 4 ...`.

# %%
nondark = np.load(OUT / "bright_train.npy") >= 25
bench = {"h_cnxt_t_base": "ConvNeXt-T in22k (round-1 recipe)", "h_t_aug": "ConvNeXt-T + heavy augmentation",
         "h_small": "ConvNeXt-S in22k", "h_base22k": "ConvNeXt-B in22k", "h_baseclip": "ConvNeXt-B CLIP LAION-2B",
         "h_dinov3B": "DINOv3 ViT-B/16", "h_clipB": "CLIP ViT-B/16"}
import scene_cv  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

va = scene_cv.hard_folds()[4][1]
va_nd = va[nondark[va]]
rows = []
for name, desc in bench.items():
    f = OUT / f"oof_{name}.npy"
    if f.exists():
        p = np.load(f)
        rows.append({"model": desc, "all held-out": f1_score(y[va], p[va].argmax(1), average="macro"),
                     "non-dark held-out": f1_score(y[va_nd], p[va_nd].argmax(1), average="macro")})
pd.DataFrame(rows).sort_values("non-dark held-out", ascending=False).round(4)

# %% [markdown]
# ## 3. Final training on all images (skipped unless `RETRAIN`)

# %%
if RETRAIN:
    for s in SEEDS:
        subprocess.run(f'"{GPU_PY}" ft.py --name f_small_s{s} --cv full --seed {s} {SMALL}', shell=True, check=True,
                       cwd=ROOT / "task4" / "src")
    subprocess.run(f'"{GPU_PY}" ft.py --name f_baseclip_s42 --cv full --seed 42 {BASECLIP}', shell=True, check=True,
                   cwd=ROOT / "task4" / "src")
small = [np.load(OUT / f"test_f_small_s{s}.npy") for s in SEEDS]
p_small = np.mean(small, 0)
print("seeds:", len(small), "| pairwise argmax agreement between seeds:",
      np.round([(small[i].argmax(1) == small[j].argmax(1)).mean() for i in range(len(small)) for j in range(i + 1, len(small))], 4))

# %% [markdown]
# ## 4. Write `outputs/final_pred.csv` (id = image name, integers 0–4) + the blend candidate

# %%
pred = p_small.argmax(1)
final = utils.write_final_pred(TASK, test["stem"], pred)
np.save(OUT / "test_final_small4.npy", p_small)
old = t4data.old_test_order_to_new(np.load(OUT / "test_convnext_ft.npy"))  # round-1 model, in ANS order
print("agreement with the round-1 ConvNeXt-T (LB ≈ 0.85):", round(float((pred == old.argmax(1)).mean()), 4))
print("mean max-prob: final", round(float(p_small.max(1).mean()), 3), "| round 1", round(float(old.max(1).mean()), 3))
mix = pd.DataFrame({"train labels": np.bincount(y, minlength=5) / len(y),
                    "round-1 test preds": np.bincount(old.argmax(1), minlength=5) / len(old),
                    "final test preds": np.bincount(pred, minlength=5) / len(pred)}, index=LABELS)
f_clip = OUT / "test_f_baseclip_s42.npy"
if f_clip.exists():
    p_blend = 0.5 * p_small + 0.5 * np.load(f_clip)
    pd.DataFrame({"id": test["stem"], "prediction": p_blend.argmax(1)}).to_csv(OUT / "pred_small4_baseclip.csv", index=False)
    mix["blend test preds"] = np.bincount(p_blend.argmax(1), minlength=5) / len(p_blend)
    print("blend vs final: rows that differ", int((p_blend.argmax(1) != pred).sum()))
mix.round(3)
