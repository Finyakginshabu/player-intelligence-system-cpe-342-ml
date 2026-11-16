# %% [markdown]
# # Task 4: Game Title Detection, final model (round R, 2026-11-12)
#
# - **Type:** 5-class image classification, 230×120 JPEG frames. **Metric:** macro F1. Pretrained weights are allowed
#   for task 4 only.
# - **Problem:** the round-1 ConvNeXt-T fine-tune scored CV 0.98 (random folds) but ≈ 0.85 on the LB. The test set holds
#   scene types that are rare in train (adversarial AUC 0.80 on CLIP embeddings), so random folds overstate accuracy.
# - **Validation used here:** scenes = k-means clusters of CLIP ViT-L/14 embeddings inside each class; whole scenes are
#   held out (`src/scene_cv.py`). Near-black frames (no game content, 0.8% of test) are excluded from the comparison.
# - **Final (v43, LB 0.85865):** eight pretraining families at equal weight, seeds averaged inside each family:
#   ConvNeXt-S in22k × 4, ConvNeXt-V2-B FCMAE × 3, EVA-02-B × 4, EVA-02-L × 4, EVA-02-L M38M × 2, DINOv3-B × 2,
#   DINOv2-L × 2, DINOv3-L × 1, each fine-tuned on **all** training images (`src/ft.py --cv full`); built by
#   `src/blend_v41plus.py` (v41 = the first six with 1 seed for M38M / DINOv3-B, LB 0.85780). Families after the first
#   two were chosen by LB family tests (held-out scenes did not predict the LB gains). Earlier steps kept for the report:
#   ConvNeXt-S alone (v24, LB 0.84731, `outputs/pred_small4.csv`), S + V2-B (v26, LB 0.85066).
# - Test images only ever pass through the trained networks (no fitting on test, project rule 2).
#
# Training runs on the GPU venv (`.venv-gpu`) via `src/ft.py`; this notebook collects the saved results. Set
# `RETRAIN = True` to rerun every training command (≈ 5 h on an RTX 5070: S ≈ 22 min × 4, V2-B ≈ 67 min × 2, B-CLIP ≈ 47 min).
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
V2_SEEDS = [42, 7]
V2BASE = "--backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4 --drop-path 0.2 --grad-ckpt"

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
         "h_small": "ConvNeXt-S in22k", "h_small_s7": "ConvNeXt-S in22k, seed 7", "h_small_e10": "ConvNeXt-S, 10 epochs",
         "h_small384_hr": "ConvNeXt-S 384 weights, 288x576 input", "h_small_in12k": "ConvNeXt-S in12k",
         "h_small_dinov3": "ConvNeXt-S DINOv3", "h_base22k": "ConvNeXt-B in22k", "h_baseclip": "ConvNeXt-B CLIP LAION-2B",
         "h_v2base": "ConvNeXt-V2-B FCMAE in22k", "h_v2base_s7": "ConvNeXt-V2-B, seed 7", "h_v2large": "ConvNeXt-V2-L",
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
o = lambda n: np.load(OUT / f"oof_{n}.npy")  # noqa: E731
blends = {"S (2 seeds)": (o("h_small") + o("h_small_s7")) / 2}
blends["V2-B (2 seeds)"] = (o("h_v2base") + o("h_v2base_s7")) / 2
blends["0.5 S + 0.5 V2-B (final recipe)"] = (blends["S (2 seeds)"] + blends["V2-B (2 seeds)"]) / 2
for name, p in blends.items():
    rows.append({"model": "blend: " + name, "all held-out": f1_score(y[va], p[va].argmax(1), average="macro"),
                 "non-dark held-out": f1_score(y[va_nd], p[va_nd].argmax(1), average="macro")})
pd.DataFrame(rows).sort_values("non-dark held-out", ascending=False).round(4)

# %% [markdown]
# ## 3. Final training on all images (skipped unless `RETRAIN`)

# %%
if RETRAIN:
    for s in SEEDS:
        subprocess.run(f'"{GPU_PY}" ft.py --name f_small_s{s} --cv full --seed {s} {SMALL}', shell=True, check=True,
                       cwd=ROOT / "task4" / "src")
    for s in V2_SEEDS:
        subprocess.run(f'"{GPU_PY}" ft.py --name f_v2base_s{s} --cv full --seed {s} {V2BASE}', shell=True, check=True,
                       cwd=ROOT / "task4" / "src")
    subprocess.run(f'"{GPU_PY}" ft.py --name f_baseclip_s42 --cv full --seed 42 {BASECLIP}', shell=True, check=True,
                   cwd=ROOT / "task4" / "src")
small = [np.load(OUT / f"test_f_small_s{s}.npy") for s in SEEDS]
p_small = np.mean(small, 0)
v2 = [np.load(OUT / f"test_f_v2base_s{s}.npy") for s in V2_SEEDS]
p_v2 = np.mean(v2, 0)
print("S seeds:", len(small), "| pairwise argmax agreement between seeds:",
      np.round([(small[i].argmax(1) == small[j].argmax(1)).mean() for i in range(len(small)) for j in range(i + 1, len(small))], 4))
print("V2-B seeds:", len(v2), "| agreement:", round(float((v2[0].argmax(1) == v2[1].argmax(1)).mean()), 4),
      "| S vs V2-B agreement:", round(float((p_small.argmax(1) == p_v2.argmax(1)).mean()), 4))

# %% [markdown]
# ## 4. Write `outputs/final_pred.csv` (id = image name, integers 0–4) + the alternatives

# %%
p_v26 = 0.5 * p_small + 0.5 * p_v2  # v26 (LB 0.85066), kept as an intermediate result
np.save(OUT / "test_blend_s4_v2b2.npy", p_v26)
# Final = v43 (LB 0.85865): v41's 6 families (+ 2nd seeds) + DINOv2-L + DINOv3-L, 8 families at equal weight
subprocess.run([sys.executable, "blend_v41plus.py", "--final", "v43_t4_seeds_d2l_d3l", "seeds2",
                "f_dinov2l_s42,f_dinov2l_s7", "f_dinov3l_s42"], check=True, cwd=ROOT / "task4" / "src")
pred = pd.read_csv(OUT / "final_pred.csv")["prediction"].to_numpy()
np.save(OUT / "test_final_small4.npy", p_small)
pd.DataFrame({"id": test["stem"], "prediction": p_small.argmax(1)}).to_csv(OUT / "pred_small4.csv", index=False)
print("final vs ConvNeXt-S alone (v24): rows that differ", int((pred != p_small.argmax(1)).sum()))
old = t4data.old_test_order_to_new(np.load(OUT / "test_convnext_ft.npy"))  # round-1 model, in ANS order
print("agreement with the round-1 ConvNeXt-T (LB ≈ 0.85):", round(float((pred == old.argmax(1)).mean()), 4))
print("mean max-prob: final", round(float(p_small.max(1).mean()), 3), "| round 1", round(float(old.max(1).mean()), 3))
mix = pd.DataFrame({"train labels": np.bincount(y, minlength=5) / len(y),
                    "round-1 test preds": np.bincount(old.argmax(1), minlength=5) / len(old),
                    "S only test preds": np.bincount(p_small.argmax(1), minlength=5) / len(pred),
                    "final test preds": np.bincount(pred, minlength=5) / len(pred)}, index=LABELS)
f_clip = OUT / "test_f_baseclip_s42.npy"
if f_clip.exists():  # older alternative (S + B-CLIP = S alone on held-out scenes)
    p_blend = 0.5 * p_small + 0.5 * np.load(f_clip)
    pd.DataFrame({"id": test["stem"], "prediction": p_blend.argmax(1)}).to_csv(OUT / "pred_small4_baseclip.csv", index=False)
mix.round(3)
