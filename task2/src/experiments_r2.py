"""Task 2 round 2 (autopilot 2026-11-08): latent-axis composite features.

EDA: segments are a 2x2 of engagement (Grinder/Whale high) x spending (Social/Whale high), each axis spread over
~10 noisy features with ~8% NaN each. Trees split on one feature at a time, so an averaged composite should
denoise the axis. Composites = nanmean of z-scores (mean/std fit on the training fold only) -> no leakage.
Same folds / LightGBM params as the baseline in task2_nb.py.
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

TASK = 2
utils.seed_everything()
DATA = paths.task_data(TASK)
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["segment"].astype(int).values

CATS = [c for c in test.columns if not pd.api.types.is_numeric_dtype(test[c]) and c != "player_id"]
NOISE = ["random_metric_1", "random_metric_2", "random_metric_3"]
FEATURES = [c for c in test.columns if c not in ["player_id", *NOISE]]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_composites  # noqa: E402


def prep(df):
    X = df[FEATURES].copy()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=sorted(train[c].dropna().unique()))
    return X


LGB_PARAMS = dict(
    objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    seed=utils.SEED, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1,
)


def run(name, composites, params=LGB_PARAMS, save=True):
    X_all, X_test_all = prep(train), prep(test)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(X_all, y))
    oof, test_prob, fold_scores = np.zeros((len(X_all), 4)), np.zeros((len(X_test_all), 4)), []
    t0 = time.time()
    for tr, va in folds:
        Xtr, Xva, Xte = X_all.iloc[tr], X_all.iloc[va], X_test_all
        if composites:
            Xtr, Xva, Xte = add_composites(Xtr, [Xtr, Xva, Xte])
        dtr = lgb.Dataset(Xtr, y[tr])
        m = lgb.train(params, dtr, 5000, valid_sets=[lgb.Dataset(Xva, y[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        test_prob += m.predict(Xte, num_iteration=m.best_iteration) / len(folds)
        fold_scores.append(metrics.macro_f1(y[va], oof[va].argmax(1)))
    if save:
        utils.save_preds(TASK, name, oof, test_prob)
    res = dict(model=name, macro_f1=metrics.macro_f1(y, oof.argmax(1)), fold_mean=np.mean(fold_scores),
               fold_std=np.std(fold_scores), folds=np.round(fold_scores, 4).tolist(), runtime_s=round(time.time() - t0))
    print(res, flush=True)
    return res, oof, test_prob


if __name__ == "__main__":
    which = sys.argv[1:] or ["comp"]
    if "base" in which:
        run("lgbm_check", composites=False, save=False)
    if "comp" in which:
        run("lgbm_comp", composites=True)
