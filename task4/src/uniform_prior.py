"""Uniform-prior adjustment for task 4 (2026-11-13). Uses NO test information.

Macro F1 weights the 5 classes equally, so the natural decision rule assumes equal class priors. ft.py trains with class
weights (max/count)^0.5, so the networks learn an effective prior ∝ count^0.5 (7-11k ≈ 0.29, roblock ≈ 0.15). Dividing the
probabilities by that prior (and renormalising) gives the equal-prior posterior. Checked on held-out train scenes (hard
folds 3 and 4, non-dark), both at the train mix and re-weighted to an equal class mix.
Usage (GPU venv): python task4/src/uniform_prior.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import scene_cv
import t4data

OUT = t4data.OUT
y = t4data.train_labels()
cnt = np.bincount(y, minlength=5).astype(float)
prior = cnt ** 0.5 / (cnt ** 0.5).sum()


def adjust(p, tau=1.0):
    q = p / prior ** tau
    return q / q.sum(1, keepdims=True)


if __name__ == "__main__":
    print("effective train prior", prior.round(3))
    o = lambda n: np.load(OUT / f"oof_{n}.npy")  # noqa: E731
    nd = np.load(OUT / "bright_train.npy") >= 25
    for fold, names in ((4, ["h_small", "h_small_s7", "h_v2base", "h_v2base_s7"]), (3, ["h3_small", "h3_v2base"])):
        va = scene_cv.hard_folds()[fold][1]
        va = va[nd[va]]
        p = np.mean([o(n) for n in names], 0)
        w_eq = (1 / cnt[y[va]])  # sample weights giving an equal class mix
        for tau in (0, 0.5, 1.0):
            a = adjust(p[va], tau).argmax(1)
            print(f"hard fold {fold}  tau {tau:.1f}  train mix {f1_score(y[va], a, average='macro'):.4f}"
                  f"  equal mix {f1_score(y[va], a, average='macro', sample_weight=w_eq):.4f}")
    test = t4data.load_csv("test")
    p = np.load(OUT / "test_blend_s4_v2b3.npy")
    a = adjust(p).argmax(1)
    print("test rows changed vs v28:", int((a != p.argmax(1)).sum()), "| test mix", np.round(np.bincount(a, minlength=5) / len(a), 3))
    pd.DataFrame({"id": test["stem"], "prediction": a}).to_csv(OUT / "pred_s4_v2b3_uniform.csv", index=False)
