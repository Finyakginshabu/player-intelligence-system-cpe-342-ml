"""Class-prior correction for task 4 (2026-11-12), decided on TRAIN data only.

Train is imbalanced (7-11k 39%), macro F1 weights the five games equally, and nothing says the test mix equals the
train mix. Dividing the predicted probabilities by train_prior ** tau moves the decision rule from "train mix" (tau = 0)
toward "all games equally likely" (tau = 1). tau is judged on held-out-scene OOF predictions with the validation rows
re-weighted to an equal class mix; the test set is not used.
"""
import sys

import numpy as np
from sklearn.metrics import f1_score

import scene_cv
import t4data


def adjust(p: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    q = p / prior ** tau
    return q / q.sum(1, keepdims=True)


def scores(y, p, w=None):
    return f1_score(y, p.argmax(1), average="macro", sample_weight=w)


if __name__ == "__main__":
    y = t4data.train_labels()
    prior = np.bincount(y, minlength=5) / len(y)
    for name in sys.argv[1:]:
        oof = np.load(t4data.OUT / f"oof_{name}.npy")
        m = oof.sum(1) > 0  # rows that were actually predicted
        yy, pp = y[m], oof[m]
        w = (1 / prior)[yy]  # equal-class-mix weights
        print(f"{name}: rows {m.sum()}")
        for tau in (0, 0.25, 0.5, 0.75, 1.0):
            q = adjust(pp, prior, tau)
            print(f"  tau {tau:4}: macro F1 train-mix {scores(yy, q):.4f} | equal-mix {scores(yy, q, w):.4f} | pred mix {np.round(np.bincount(q.argmax(1), minlength=5) / len(q), 3).tolist()}")
