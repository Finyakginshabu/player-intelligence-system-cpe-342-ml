"""Label-free estimate of test accuracy for a task-4 blend (2026-11-13).

Held-out (hard fold 4, non-dark) gives accuracy as a function of the blend's confidence (max probability). Applying that
table to the blend's TEST confidences gives the accuracy the test predictions would have if the confidence->accuracy
relation carries over. No test labels, nothing fitted on test: it only reads the confidence of predictions already made.
VALIDATION FAILED (2026-11-13): level is roughly right (v24 0.878 vs LB 0.873, v26 0.887 vs 0.890) but it cannot rank
blends (v27 0.890 vs LB 0.878; a single seed scores above its 4-seed average): overconfident models look better.
Kept as a documented negative result. Do NOT use it to choose files.
Usage: implied(heldout_probs_of_the_same_recipe, test_probs) -> float
"""
import numpy as np

import scene_cv
import t4data

Y = t4data.train_labels()
_va = scene_cv.hard_folds()[4][1]
VA = _va[(np.load(t4data.OUT / "bright_train.npy") >= 25)[_va]]
EDGES = np.array([0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 1.01])


def implied(oof, test, edges=EDGES):
    c, ok = oof[VA].max(1), oof[VA].argmax(1) == Y[VA]
    ct = test.max(1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m, mt = (c >= lo) & (c < hi), (ct >= lo) & (ct < hi)
        acc = ok[m].mean() if m.sum() >= 20 else ok[(c < hi)].mean()
        tot += acc * mt.mean()
    return float(tot)
