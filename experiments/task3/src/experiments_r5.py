"""Task 3 round 5 (autopilot 2026-11-09): variance reduction for the two-stage conditional median.

Round-4 analysis: the amount residual (log MAD 0.178) and the weak gate (AUC 0.78) look like noise in the data, the
0.5 gate is optimal and the quantile interpolation scheme does not matter (raw / log / normal-score interpolation
all give 0.78777). What is left is model variance: average several seeds of each stage (and add CatBoost for the
gate), then combine exactly as in round 3. Same folds; everything fit on the training fold only.
Run from repo root: python task3/src/experiments_r5.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils

TASK = 3
utils.seed_everything()
DATA = paths.task_data(TASK)
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["spending_30d"].values.astype(float)
FEATURES = [c for c in test.columns if c != "player_id"]
CATS = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(train[c])]
X, X_test = train[FEATURES].copy(), test[FEATURES].copy()
for c in CATS:  # LightGBM needs category dtype (fixed levels from train)
    lv = sorted(train[c].dropna().unique())
    X[c], X_test[c] = pd.Categorical(X[c], categories=lv), pd.Categorical(X_test[c], categories=lv)
nz = y > 0
bins = np.zeros(len(y), dtype=int)
bins[nz] = 1 + pd.qcut(y[nz], 5, labels=False)
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(X, bins))
BASE = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
ALPHAS = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
SEEDS = [42, 43, 44]


def fit_predict(params, Xtr, ytr, Xva, yva, Xte):
    dtr = lgb.Dataset(Xtr, ytr)
    m = lgb.train(params, dtr, 5000, valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xva, num_iteration=m.best_iteration), m.predict(Xte, num_iteration=m.best_iteration)


def combine(p, Q, gate=0.5):
    Q = np.sort(Q, axis=1)
    q = np.clip((p - 0.5) / np.maximum(p, 1e-9), 0, ALPHAS[-1])
    j = np.clip(np.searchsorted(ALPHAS, q, side="right") - 1, 0, len(ALPHAS) - 2)
    t = np.clip((q - ALPHAS[j]) / (ALPHAS[j + 1] - ALPHAS[j]), 0, 1)
    r = np.arange(len(q))
    return np.where(p <= gate, 0.0, Q[r, j] * (1 - t) + Q[r, j + 1] * t)


def fold_scores(o):
    return [metrics.nmae(y[va], o[va]) for _, va in FOLDS]


def stage1_lgb(seed):
    o, t = np.zeros(len(y)), np.zeros(len(X_test))
    for tr, va in FOLDS:
        a, b = fit_predict({**BASE, "objective": "binary", "seed": seed}, X.iloc[tr], nz[tr].astype(int),
                           X.iloc[va], nz[va].astype(int), X_test)
        o[va], t = a, t + b / len(FOLDS)
    return o, t


def stage1_cat(seed):
    from catboost import CatBoostClassifier
    o, t = np.zeros(len(y)), np.zeros(len(X_test))
    Xc, Xtc = train[FEATURES].copy(), test[FEATURES].copy()
    for c in CATS:
        Xc[c], Xtc[c] = Xc[c].fillna("NA").astype(str), Xtc[c].fillna("NA").astype(str)
    for tr, va in FOLDS:
        m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3, eval_metric="Logloss",
                               random_seed=seed, od_type="Iter", od_wait=200, verbose=0, thread_count=-1,
                               cat_features=CATS)
        m.fit(Xc.iloc[tr], nz[tr].astype(int), eval_set=(Xc.iloc[va], nz[va].astype(int)), use_best_model=True)
        o[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        t += m.predict_proba(Xtc)[:, 1] / len(FOLDS)
    return o, t


def stage2_lgb(seed):
    """Spender quantiles (raw scale) for every row: fit on spenders of the training fold (log1p target)."""
    oQ, tQ = np.zeros((len(y), len(ALPHAS))), np.zeros((len(X_test), len(ALPHAS)))
    for tr, va in FOLDS:
        trp, vap = tr[nz[tr]], va[nz[va]]
        for j, al in enumerate(ALPHAS):
            _, b = fit_predict({**BASE, "objective": "quantile", "alpha": al, "seed": seed}, X.iloc[trp],
                               np.log1p(y[trp]), X.iloc[vap], np.log1p(y[vap]), pd.concat([X.iloc[va], X_test]))
            oQ[va, j] = b[:len(va)]          # log scale; averaged across seeds before expm1
            tQ[:, j] += b[len(va):] / len(FOLDS)
    return oQ, tQ


if __name__ == "__main__":
    t0 = time.time()
    ref = utils.load_preds(TASK, "twostage")[0]
    f_ref = fold_scores(ref)
    print(f"reference twostage: {metrics.nmae(y, ref):.5f} | folds {np.round(f_ref, 4).tolist()} ± {np.std(f_ref):.4f}")
    P1, T1 = [], []
    for s in SEEDS:
        o, t = stage1_lgb(s)
        P1.append(o), T1.append(t)
        print(f"  stage1 lgb seed {s}: AUC {roc_auc_score(nz, o):.4f} logloss {log_loss(nz, o):.4f} ({time.time() - t0:.0f}s)", flush=True)
    oc, tc = stage1_cat(42)
    print(f"  stage1 cat: AUC {roc_auc_score(nz, oc):.4f} logloss {log_loss(nz, oc):.4f} ({time.time() - t0:.0f}s)", flush=True)
    p_lgb, t_lgb = np.mean(P1, axis=0), np.mean(T1, axis=0)
    print(f"  stage1 lgb x{len(SEEDS)}: logloss {log_loss(nz, p_lgb):.4f} | lgb+cat: {log_loss(nz, (p_lgb + oc) / 2):.4f}")
    Q1, TQ = [], []
    for s in SEEDS:
        o, t = stage2_lgb(s)
        Q1.append(o), TQ.append(t)
        print(f"  stage2 seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    Q_ms, TQ_ms = np.expm1(np.mean(Q1, axis=0)), np.expm1(np.mean(TQ, axis=0))
    Q_s0 = np.expm1(Q1[0])
    variants = {
        "1 seed (= round 3 recipe)": (P1[0], Q_s0),
        "stage1 x3": (p_lgb, Q_s0),
        "stage2 x3": (P1[0], Q_ms),
        "both x3": (p_lgb, Q_ms),
        "both x3 + cat gate": ((p_lgb + oc) / 2, Q_ms),
    }
    for name, (p, Q) in variants.items():
        o = combine(p, Q)
        f = fold_scores(o)
        d = np.array(f) - np.array(f_ref)
        print(f"{name:28s} NMAE {metrics.nmae(y, o):.5f} | mean {np.mean(f):.5f} ± {np.std(f):.4f} | d per fold {np.round(d, 4).tolist()}")
    np.save(paths.outputs_dir(TASK) / "oof_twostage_ms_parts.npy", np.c_[(p_lgb + oc) / 2, p_lgb, Q_ms])
    np.save(paths.outputs_dir(TASK) / "test_twostage_ms_parts.npy", np.c_[(t_lgb + tc) / 2, t_lgb, TQ_ms])
    best_o, best_t = combine((p_lgb + oc) / 2, Q_ms), combine((t_lgb + tc) / 2, TQ_ms)
    utils.save_preds(TASK, "twostage_ms_cat", best_o, best_t)
    utils.save_preds(TASK, "twostage_ms", combine(p_lgb, Q_ms), combine(t_lgb, TQ_ms))
    print(f"runtime {time.time() - t0:.0f}s")
