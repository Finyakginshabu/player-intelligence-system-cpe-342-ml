"""Rebuild the v41 task-4 blend (best LB file, sub_v41_t4_m38_dino.csv = 0.85780) and write outputs/final_pred.csv.

Six pretraining families, equal weight 1/6 each; within a family the seeds are averaged first. Every model was fine-tuned
on all training images (`ft.py --cv full`); test images only pass through the trained networks (rule 2).
  ConvNeXt-S in22k            f_small_s{42,7,123,2026}
  ConvNeXt-V2-B FCMAE in22k   f_v2base_s{42,7,123}        (s2026 diverged, never used)
  EVA-02-B MIM in22k          f_eva02b_s{42,7,123,2026}
  EVA-02-L MIM in22k          f_eva02l_s{42,7,123,2026}
  EVA-02-L MIM M38M           f_eva02lm38_s42
  DINOv3 ViT-B LVD-1689M      f_dinov3b_lr2e4_s42
Needs outputs/test_<name>.npy (GPU machine, gitignored). Checks the result against outputs/pred_v41_m38_dino.csv.
Usage (from task4/src): python blend_v41.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import t4data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import utils  # noqa: E402

OUT = t4data.OUT
FAMILIES = {
    "ConvNeXt-S": [f"f_small_s{s}" for s in (42, 7, 123, 2026)],
    "ConvNeXt-V2-B": [f"f_v2base_s{s}" for s in (42, 7, 123)],
    "EVA-02-B": [f"f_eva02b_s{s}" for s in (42, 7, 123, 2026)],
    "EVA-02-L": [f"f_eva02l_s{s}" for s in (42, 7, 123, 2026)],
    "EVA-02-L M38M": ["f_eva02lm38_s42"],
    "DINOv3-B": ["f_dinov3b_lr2e4_s42"],
}

fam = {k: np.mean([np.load(OUT / f"test_{n}.npy") for n in names], 0) for k, names in FAMILIES.items()}
p = np.mean(list(fam.values()), 0)
pred = p.argmax(1)
for k, q in fam.items():
    print(f"{k:14s} agreement with the blend {(q.argmax(1) == pred).mean():.4f}")

ref = pd.read_csv(OUT / "pred_v41_m38_dino.csv")["prediction"].to_numpy()
print("rows differing from the uploaded v41:", int((pred != ref).sum()))
np.save(OUT / "test_blend_v41.npy", p)
utils.write_final_pred(4, t4data.load_csv("test")["stem"], pred)
