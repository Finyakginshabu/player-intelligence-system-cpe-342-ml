"""Task 3 round 8 (2026-11-19): fill missing values from exact identities between columns.

The data obey two exact row identities (relative error < 1e-15 on complete rows):
    historical_spending   = max(total_transactions, 1) * avg_transaction_value   (tt = 0 rows store avg = historical)
    purchases_on_discount = total_transactions * discount_rate_used
Each of these columns is about 5.8% missing at random, so most gaps can be recovered exactly from the same row. The fill
uses only the row's own features (no target, no other rows), so it is applied identically to train and test.
Compared here: the two-stage conditional median (task3_nb.py section 4c, single seed) on raw vs filled features, same folds.

Usage: python task3/src/experiments_r8.py
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import metrics, paths, utils  # noqa: E402

TASK = 3
H, TT, AVG, POD, DR = ("historical_spending", "total_transactions", "avg_transaction_value", "purchases_on_discount",
                       "discount_rate_used")


def fill_identities(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    before = d[[H, TT, AVG, POD, DR]].isna().sum()
    for _ in range(3):  # a value filled by one identity can unlock the other
        # total_transactions: exact from the discount identity, else the nearest integer of historical / avg
        m = d[TT].isna() & d[POD].notna() & d[DR].notna() & (d[DR] > 0)
        d.loc[m, TT] = np.round(d.loc[m, POD] / d.loc[m, DR])
        m = d[TT].isna() & d[H].notna() & d[AVG].notna() & (d[AVG] > 0)
        r = np.round(d.loc[m, H] / d.loc[m, AVG])
        d.loc[m[m].index[r.values > 1], TT] = r[r.values > 1]  # a ratio of 1 means tt = 0 or 1: left missing
        tt1 = d[TT].clip(lower=1)
        m = d[H].isna() & d[TT].notna() & d[AVG].notna()
        d.loc[m, H] = tt1[m] * d.loc[m, AVG]
        m = d[AVG].isna() & d[TT].notna() & d[H].notna()
        d.loc[m, AVG] = d.loc[m, H] / tt1[m]
        m = d[POD].isna() & d[TT].notna() & d[DR].notna()
        d.loc[m, POD] = d.loc[m, TT] * d.loc[m, DR]
        m = d[DR].isna() & d[TT].notna() & d[POD].notna() & (d[TT] > 0)
        d.loc[m, DR] = d.loc[m, POD] / d.loc[m, TT]
    after = d[[H, TT, AVG, POD, DR]].isna().sum()
    return d, {c: (int(before[c]), int(after[c])) for c in before.index}


def check_identities(df: pd.DataFrame) -> float:
    m = df[[H, TT, AVG]].notna().all(axis=1)
    x = df[m]
    rel = (x[H] - x[TT].clip(lower=1) * x[AVG]).abs() / x[H].abs().clip(lower=1e-9)
    m2 = df[[POD, TT, DR]].notna().all(axis=1)
    x2 = df[m2]
    rel2 = (x2[POD] - x2[TT] * x2[DR]).abs() / x2[POD].abs().clip(lower=1e-9)
    return float(max(rel.max(), rel2[x2[POD] > 0].max()))


BASE = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, seed=utils.SEED, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
ALPHAS_Q = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])


def fit_predict(params, Xtr, ytr, Xva, yva, Xte):
    dtr = lgb.Dataset(Xtr, ytr)
    m = lgb.train(params, dtr, 5000, valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xva, num_iteration=m.best_iteration), m.predict(Xte, num_iteration=m.best_iteration)


def mixture_median(p, Q):
    Q = np.sort(Q, axis=1)
    q = np.clip((p - 0.5) / np.maximum(p, 1e-9), 0, ALPHAS_Q[-1])
    med = np.array([np.interp(qi, ALPHAS_Q, Qi) for qi, Qi in zip(q, Q)])
    return np.where(p <= 0.5, 0.0, med)


def twostage(X, y, X_test, folds, seed=utils.SEED):
    params = {**BASE, "seed": seed}
    nz = y > 0
    oof_p, test_p = np.zeros(len(X)), np.zeros(len(X_test))
    oof_Q, test_Q = np.zeros((len(X), len(ALPHAS_Q))), np.zeros((len(X_test), len(ALPHAS_Q)))
    for tr, va in folds:
        a, b = fit_predict({**params, "objective": "binary"}, X.iloc[tr], nz[tr].astype(int), X.iloc[va],
                           nz[va].astype(int), X_test)
        oof_p[va], test_p = a, test_p + b / len(folds)
        trp, vap = tr[nz[tr]], va[nz[va]]
        for j, al in enumerate(ALPHAS_Q):
            _, b = fit_predict({**params, "objective": "quantile", "alpha": al}, X.iloc[trp], np.log1p(y[trp]),
                               X.iloc[vap], np.log1p(y[vap]), pd.concat([X.iloc[va], X_test]))
            oof_Q[va, j] = np.expm1(b[:len(va)])
            test_Q[:, j] += np.expm1(b[len(va):]) / len(folds)
    return mixture_median(oof_p, oof_Q), mixture_median(test_p, test_Q), (oof_p, oof_Q, test_p, test_Q)


if __name__ == "__main__":
    data = paths.task_data(TASK)
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    y = train["spending_30d"].values.astype(float)
    feats = [c for c in test.columns if c != "player_id"]
    for c in ("primary_game", "platform", "segment", "seasonal_spending_pattern", "competitive_rank"):
        if c in feats and train[c].dtype == object:
            cats = pd.Categorical(pd.concat([train[c], test[c]])).categories
            train[c], test[c] = pd.Categorical(train[c], cats), pd.Categorical(test[c], cats)
    bins = np.zeros(len(y), dtype=int)
    bins[y > 0] = 1 + pd.qcut(y[y > 0], 5, labels=False)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(train, bins))
    ftr, ntr = fill_identities(train)
    fte, nte = fill_identities(test)
    print("missing before -> after (train):", ntr)
    print("missing before -> after (test): ", nte)
    print(f"identity check after filling: max relative error {check_identities(ftr):.1e} (train) "
          f"{check_identities(fte):.1e} (test)")
    res = {}
    for name, (a, b) in {"raw": (train, test), "filled": (ftr, fte)}.items():
        t0 = time.time()
        oof, pred, parts = twostage(a[feats], y, b[feats], folds)
        f = [metrics.nmae(y[va], oof[va]) for _, va in folds]
        res[name] = (oof, pred, f)
        print(f"twostage {name:7s} NMAE {metrics.nmae(y, oof):.5f} +- {np.std(f):.4f}  folds {np.round(f, 4).tolist()}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        utils.save_preds(TASK, f"twostage_r8_{name}", oof, pred)
    d = np.array(res["filled"][2]) - np.array(res["raw"][2])
    print(f"filled - raw: {metrics.nmae(y, res['filled'][0]) - metrics.nmae(y, res['raw'][0]):+.5f} | by fold "
          f"{np.round(d, 5).tolist()} | folds up {int((d > 0).sum())}/5")
