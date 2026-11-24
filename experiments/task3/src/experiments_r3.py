"""Task 3 round 3 (autopilot 2026-11-08): two-stage conditional-median model.

MAE-optimal prediction = conditional median of y | x. With p = P(y > 0 | x) and F+ the distribution of y | y > 0, x:
    median = 0                      if p <= 0.5
           = F+^-1((p - 0.5) / p)   otherwise.
Stage 1: LightGBM binary classifier for p. Stage 2: LightGBM quantile regressors on spenders only (log1p target;
quantiles commute with monotone transforms) at levels ALPHAS covering [0, 0.5]; per-row interpolation at q(p).
Same folds as task3_nb.py. Everything is fit on the training fold only.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils

TASK = 3
utils.seed_everything()
DATA = paths.task_data(TASK)
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["spending_30d"].values.astype(float)
FEATURES = [c for c in test.columns if c != "player_id"]
X, X_test = train[FEATURES], test[FEATURES]
bins = np.zeros(len(y), dtype=int)
nz = y > 0
bins[nz] = 1 + pd.qcut(y[nz], 5, labels=False)
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(X, bins))

BASE = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=utils.SEED,
            deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
ALPHAS = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])


def fit_predict(params, Xtr, ytr, Xva, yva, Xte):
    dtr = lgb.Dataset(Xtr, ytr)
    m = lgb.train(params, dtr, 5000, valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xva, num_iteration=m.best_iteration), m.predict(Xte, num_iteration=m.best_iteration), m.best_iteration


def combine(p, Q, gate=0.5):
    """p: P(y>0); Q: (n, len(ALPHAS)) spender quantiles on the raw scale. Median of the zero-inflated mixture."""
    Q = np.sort(Q, axis=1)  # fix quantile crossing
    q = np.clip((p - 0.5) / np.maximum(p, 1e-9), 0, ALPHAS[-1])
    out = np.array([np.interp(qi, ALPHAS, Qi) for qi, Qi in zip(q, Q)])
    return np.where(p <= gate, 0.0, out)


t0 = time.time()
oof_p, test_p = np.zeros(len(X)), np.zeros(len(X_test))
oof_Q, test_Q = np.zeros((len(X), len(ALPHAS))), np.zeros((len(X_test), len(ALPHAS)))
for k, (tr, va) in enumerate(FOLDS):
    a, b, it = fit_predict({**BASE, "objective": "binary"}, X.iloc[tr], nz[tr].astype(int), X.iloc[va],
                           nz[va].astype(int), X_test)
    oof_p[va], test_p = a, test_p + b / len(FOLDS)
    trp, vap = tr[nz[tr]], va[nz[va]]  # spenders only
    its = [it]
    for j, al in enumerate(ALPHAS):
        a, b, it = fit_predict({**BASE, "objective": "quantile", "alpha": al}, X.iloc[trp], np.log1p(y[trp]),
                               X.iloc[vap], np.log1p(y[vap]), pd.concat([X.iloc[va], X_test]))
        oof_Q[va, j] = np.expm1(b[:len(va)])
        test_Q[:, j] += np.expm1(b[len(va):]) / len(FOLDS)
        its.append(it)
    print(f"fold {k}: iters {its}  ({time.time() - t0:.0f}s)", flush=True)

np.save(paths.outputs_dir(TASK) / "oof_twostage_parts.npy", np.c_[oof_p, oof_Q])
np.save(paths.outputs_dir(TASK) / "test_twostage_parts.npy", np.c_[test_p, test_Q])

oof = combine(oof_p, oof_Q)
pred = combine(test_p, test_Q)
utils.save_preds(TASK, "twostage", oof, pred)

base_oof, base_test = utils.load_preds(TASK, "lgbm_l1")
fs = lambda o: [metrics.nmae(y[va], o[va]) for _, va in FOLDS]  # noqa: E731
f_two, f_base = fs(oof), fs(base_oof)
print(f"twostage  NMAE {metrics.nmae(y, oof):.4f}  fold mean {np.mean(f_two):.4f} +- {np.std(f_two):.4f}  {np.round(f_two, 4)}")
print(f"lgbm_l1   NMAE {metrics.nmae(y, base_oof):.4f}  fold mean {np.mean(f_base):.4f} +- {np.std(f_base):.4f}  {np.round(f_base, 4)}")
print("per-fold delta", np.round(np.array(f_two) - np.array(f_base), 4))
print(f"zero share twostage OOF {np.mean(oof == 0):.3f} test {np.mean(pred == 0):.3f} | true {np.mean(y == 0):.3f}")

# Nested blend weight with the L1 model (fit w on 4 folds, score on the 5th)
W = np.round(np.arange(0, 1.01, 0.1), 2)
nested = []
for tr, va in FOLDS:
    w = W[int(np.argmax([metrics.nmae(y[tr], w * oof[tr] + (1 - w) * base_oof[tr]) for w in W]))]
    nested.append((w, metrics.nmae(y[va], w * oof[va] + (1 - w) * base_oof[va])))
print("nested blend (w on twostage, fold score):", [(w, round(s, 4)) for w, s in nested],
      f"mean {np.mean([s for _, s in nested]):.4f} +- {np.std([s for _, s in nested]):.4f}")
print(f"runtime {time.time() - t0:.0f}s")
