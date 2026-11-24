"""Task 1 round 5 (autopilot 2026-11-09): binary-F2 threshold under TEST-like missingness.

Test has different NaN rates than train (headshot 2.7% -> 14.1%, win_rate 5.0% -> 13.7%, ...), and rows with a
missing strong feature get less confident probabilities. The binary-F2 threshold (0.159) was tuned on validation
rows with train-like missingness. Here every validation fold is ALSO predicted after re-masking it to the test NaN
rates (only per-column NaN rates of test are used, no test rows or labels), and two rules are compared on the
re-masked folds, nested (threshold from the other 4 folds):
  (a) threshold tuned on the clean OOF  (current practice)
  (b) threshold tuned on the re-masked OOF
Model = the round-3 final (LightGBM, 15 signal features, 5 seeds). Run: python task1/src/experiments_r5.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np

from common import metrics, utils

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments_r3 import FOLDS, LGB_P, SEEDS, SIGNAL, test, train, y  # noqa: E402

rate_tr, rate_te = train[SIGNAL].isna().mean(), test[SIGNAL].isna().mean()


def remask(df, rng):
    """Add NaNs so each column's NaN rate reaches the test rate (columns already above it are left alone)."""
    out = df.copy()
    for c in SIGNAL:
        cur = out[c].isna().mean()
        if rate_te[c] > cur:
            p = (rate_te[c] - cur) / (1 - cur)
            out.loc[out[c].notna() & (rng.random(len(out)) < p), c] = np.nan
    return out


def nested(p_eval, p_tune):
    fs, ts = [], []
    for k, (_, va) in enumerate(FOLDS):
        inner = np.concatenate([FOLDS[j][1] for j in range(5) if j != k])
        t, _ = metrics.best_fbeta_threshold(y[inner], p_tune[inner], 2, average="binary")
        ts.append(round(t, 4))
        fs.append(metrics.f2(y[va], (p_eval[va] >= t).astype(int), average="binary"))
    return np.mean(fs), np.std(fs), np.round(fs, 4).tolist(), ts


if __name__ == "__main__":
    print("NaN rate train -> test (signal features, |diff| > 1pp):")
    d = (rate_te - rate_tr)
    print((100 * rate_tr[d.abs() > 0.01]).round(1).astype(str).str.cat((100 * rate_te[d.abs() > 0.01]).round(1).astype(str), sep=" -> ").to_string())
    t0 = time.time()
    clean, masked = np.zeros(len(y)), np.zeros(len(y))
    for k, (trn, va) in enumerate(FOLDS):
        Xva_m = remask(train.iloc[va][SIGNAL], np.random.default_rng(42 + k))
        for s in SEEDS:
            m = lgb.train({**LGB_P, "seed": s}, lgb.Dataset(train.iloc[trn][SIGNAL], y[trn]), 5000,
                          valid_sets=[lgb.Dataset(train.iloc[va][SIGNAL], y[va])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            clean[va] += m.predict(train.iloc[va][SIGNAL], num_iteration=m.best_iteration) / len(SEEDS)
            masked[va] += m.predict(Xva_m, num_iteration=m.best_iteration) / len(SEEDS)
    print(f"({time.time() - t0:.0f}s)  clean OOF reproduces round 3: "
          f"{np.allclose(clean, utils.load_preds(1, 'lgbm_sig_5seed')[0])}")
    for name, pe, pt in (("clean folds, clean thr (round 3)", clean, clean),
                         ("re-masked folds, clean thr (a)", masked, clean),
                         ("re-masked folds, re-masked thr (b)", masked, masked)):
        mu, sd, f, t = nested(pe, pt)
        print(f"{name:36s} {mu:.4f} ± {sd:.4f} folds {f} thr {t}")
    for name, p in (("clean", clean), ("re-masked", masked)):
        t, f = metrics.best_fbeta_threshold(y, p, 2, average="binary")
        print(f"full-OOF optimum on {name:9s}: thr {t:.4f}, F2 {f:.4f}, positive rate {(p >= t).mean():.4f}")
    p_test = utils.load_preds(1, "lgbm_sig_5seed")[1]
    for t in (0.1589, metrics.best_fbeta_threshold(y, masked, 2, average="binary")[0]):
        print(f"test positive rate at thr {t:.4f}: {(p_test >= t).mean():.4f}")
    np.save(ROOT / "task1" / "outputs" / "oof_lgbm_sig_5seed_remasked.npy", masked)
