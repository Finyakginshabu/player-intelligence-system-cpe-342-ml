"""Task 3 round 8b (2026-11-19): stage 2 of the two-stage median on the spending RATIO to the player's own history.

Among spenders, log(y / prev_month_spending) is close to a per-segment offset plus noise (corr(log y, log prev) 0.98), so
quantile models of t = log1p(y) - log1p(ref) have an easier target than log1p(y). ref is a function of the row's features
only: prev_month_spending; if missing, historical_spending x the training median of prev / historical (0.30-0.40 by
construction); if both are missing, 0 (then t = log1p(y), as before). Quantiles of y follow as expm1(q_t + log1p(ref)),
since ref is fixed given x. Features: identity-filled (experiments_r8.fill_identities). Same folds and settings as r8.

Usage: python task3/src/experiments_r8b.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_r8 as r8  # noqa: E402
from experiments_r8 import ALPHAS_Q, BASE, fit_predict, mixture_median, metrics, paths, utils  # noqa: E402


def reference(df: pd.DataFrame, ratio: float) -> np.ndarray:
    ref = df["prev_month_spending"].to_numpy(float)
    alt = df["historical_spending"].to_numpy(float) * ratio
    ref = np.where(np.isnan(ref), alt, ref)
    return np.log1p(np.nan_to_num(ref, nan=0.0))


def twostage_ratio(X, y, X_test, lr, lr_test, folds, seed=utils.SEED):
    params = {**BASE, "seed": seed}
    nz = y > 0
    oof_p, test_p = np.zeros(len(X)), np.zeros(len(X_test))
    oof_Q, test_Q = np.zeros((len(X), len(ALPHAS_Q))), np.zeros((len(X_test), len(ALPHAS_Q)))
    t = np.log1p(y) - lr
    for tr, va in folds:
        a, b = fit_predict({**params, "objective": "binary"}, X.iloc[tr], nz[tr].astype(int), X.iloc[va],
                           nz[va].astype(int), X_test)
        oof_p[va], test_p = a, test_p + b / len(folds)
        trp, vap = tr[nz[tr]], va[nz[va]]
        for j, al in enumerate(ALPHAS_Q):
            _, b = fit_predict({**params, "objective": "quantile", "alpha": al}, X.iloc[trp], t[trp],
                               X.iloc[vap], t[vap], pd.concat([X.iloc[va], X_test]))
            oof_Q[va, j] = np.expm1(b[:len(va)] + lr[va])
            test_Q[:, j] += np.expm1(b[len(va):] + lr_test) / len(folds)
    oof_Q, test_Q = np.clip(oof_Q, 0, None), np.clip(test_Q, 0, None)
    return mixture_median(oof_p, oof_Q), mixture_median(test_p, test_Q), (oof_p, oof_Q, test_p, test_Q)


if __name__ == "__main__":
    data = paths.task_data(3)
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    y = train["spending_30d"].values.astype(float)
    feats = [c for c in test.columns if c != "player_id"]
    bins = np.zeros(len(y), dtype=int)
    bins[y > 0] = 1 + pd.qcut(y[y > 0], 5, labels=False)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(train, bins))
    ftr, _ = r8.fill_identities(train)
    fte, _ = r8.fill_identities(test)
    ratio = float((ftr["prev_month_spending"] / ftr["historical_spending"]).median())
    lr, lr_test = reference(ftr, ratio), reference(fte, ratio)
    print(f"prev / historical median {ratio:.4f} | rows with no reference: train {(lr == 0).sum()}, test {(lr_test == 0).sum()}")
    t0 = time.time()
    oof, pred, _ = twostage_ratio(ftr[feats], y, fte[feats], lr, lr_test, folds)
    f = np.array([metrics.nmae(y[va], oof[va]) for _, va in folds])
    print(f"twostage filled+ratio NMAE {metrics.nmae(y, oof):.5f} +- {f.std():.4f}  folds {np.round(f, 4).tolist()}"
          f"  ({time.time() - t0:.0f}s)")
    utils.save_preds(3, "twostage_r8_ratio", oof, pred)
    ref_oof = utils.load_preds(3, "twostage_r8_filled")[0]
    g = np.array([metrics.nmae(y[va], ref_oof[va]) for _, va in folds])
    d = f - g
    print(f"ratio - filled: {metrics.nmae(y, oof) - metrics.nmae(y, ref_oof):+.5f} | by fold {np.round(d, 5).tolist()} "
          f"| folds up {int((d > 0).sum())}/5")
