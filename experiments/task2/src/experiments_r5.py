"""Task 2 round 5 (2026-11-10): add player_id as a feature, following the TA's reference code (TA/Task2).

player_id numbers come in contiguous segment blocks (~97% pure; train and test interleave over the same range), so
the ID number is a near-direct segment signal. Two ways to use it, same 5 folds, everything fit in-fold:
  snb_pid   naive Bayes (round 4) + the ID number as one more continuous feature (smoothed per-class density)
  lgbm_pid  LightGBM + composites (round 2) + the ID number
Run: python task2/src/experiments_r5.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils
from features import add_composites
from nb import SmoothNB

TASK = 2
utils.seed_everything()
D = paths.task_data(TASK)
train, test = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
for df in (train, test):
    df["player_num"] = df["player_id"].str.extract(r"(\d+)", expand=False).astype(float)
y = train["segment"].astype(int).to_numpy()
CATS = [c for c in test.columns if not pd.api.types.is_numeric_dtype(test[c]) and c != "player_id"]
NOISE = ["random_metric_1", "random_metric_2", "random_metric_3"]
BASE = [c for c in test.columns if c not in ["player_id", "player_num", *NOISE]]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=utils.SEED).split(train, y))
LGB_P = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=31, min_child_samples=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=utils.SEED,
             deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)


def report(name, oof):
    fs = [metrics.macro_f1(y[va], oof[va].argmax(1)) for _, va in FOLDS]
    print(f"{name:10s} macro F1 {metrics.macro_f1(y, oof.argmax(1)):.4f} | {np.mean(fs):.4f} ± {np.std(fs):.4f} "
          f"| folds {np.round(fs, 4).tolist()}", flush=True)


def run_nb(feats, name, n_bins=128, sigma=4.0):
    oof, tp = np.zeros((len(y), 4)), np.zeros((len(test), 4))
    for tr, va in FOLDS:
        m = SmoothNB(CATS, 4, n_bins=n_bins, sigma=sigma).fit(train.iloc[tr][feats], y[tr])
        oof[va] = m.predict_proba(train.iloc[va][feats])
        tp += m.predict_proba(test[feats]) / len(FOLDS)
    report(name, oof)
    utils.save_preds(TASK, name, oof, tp)
    return oof, tp


def prep(df, feats):
    X = df[feats].copy()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=sorted(train[c].dropna().unique()))
    return X


def run_lgb(feats, name):
    X, Xt = prep(train, feats), prep(test, feats)
    oof, tp = np.zeros((len(y), 4)), np.zeros((len(test), 4))
    for tr, va in FOLDS:
        Xtr, Xva, Xte = add_composites(X.iloc[tr], [X.iloc[tr], X.iloc[va], Xt])
        dtr = lgb.Dataset(Xtr, y[tr])
        m = lgb.train(LGB_P, dtr, 5000, valid_sets=[lgb.Dataset(Xva, y[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        tp += m.predict(Xte, num_iteration=m.best_iteration) / len(FOLDS)
    report(name, oof)
    utils.save_preds(TASK, name, oof, tp)
    return oof, tp


if __name__ == "__main__":
    t0 = time.time()
    print("reference:")
    report("snb", utils.load_preds(TASK, "snb")[0])
    run_nb(["player_num"], "nb_pidonly", sigma=1.0)
    for s in (1.0, 2.0, 4.0):
        run_nb(BASE + ["player_num"], f"snb_pid_s{s:g}") if s != 4.0 else run_nb(BASE + ["player_num"], "snb_pid")
    run_lgb(BASE + ["player_num"], "lgbm_pid")
    print(f"({time.time() - t0:.0f}s)")
