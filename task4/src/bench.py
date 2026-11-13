"""Score every hard-fold-4 OOF on the benchmark: macro F1 on all / non-dark held-out rows. Usage: python bench.py [names]"""
import sys

import numpy as np
from sklearn.metrics import f1_score

import scene_cv
import t4data

y = t4data.train_labels()
va = scene_cv.hard_folds()[4][1]
va_nd = va[(np.load(t4data.OUT / "bright_train.npy") >= 25)[va]]
names = sys.argv[1:] or sorted(f.stem[4:] for f in t4data.OUT.glob("oof_h_*.npy"))
for n in names:
    f = t4data.OUT / f"oof_{n}.npy"
    if not f.exists():
        continue
    p = np.load(f)
    if not p[va].any():
        continue
    print(f"{n:22s} all {f1_score(y[va], p[va].argmax(1), average='macro'):.4f}  non-dark "
          f"{f1_score(y[va_nd], p[va_nd].argmax(1), average='macro'):.4f}")
