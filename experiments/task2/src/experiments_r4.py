"""Task 2 round 4 (autopilot 2026-11-09): additive models for conditionally independent features.

Finding: within each class the 41 features are uncorrelated (max |Spearman| 0.019-0.050, mean 0.004-0.008; pooled
up to 0.16) and missingness is independent of the class. So p(x | c) = prod_f p(x_f | c): naive Bayes is the
correctly specified model, the Bayes-optimal log-odds is additive, and a missing feature should simply be skipped
(exact marginalisation), not imputed.

Models (same 5 folds as every task-2 model, everything fit on the training fold only):
  stumps  LightGBM with num_leaves=2 (boosted additive model / GAM), native NaN
  nb      binned naive Bayes: per-class frequencies over in-fold quantile bins (B bins, Laplace alpha),
          categoricals = per-class level frequencies, NaN skipped
Blends with lgbm_comp are scored with nested CV (weights chosen on the other 4 folds' OOF).
Run: python task2/src/experiments_r4.py [stumps] [nb] [blend]
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
K = 4
CATS = [c for c in test.columns if not pd.api.types.is_numeric_dtype(test[c]) and c != "player_id"]
NOISE = ["random_metric_1", "random_metric_2", "random_metric_3"]
FEATURES = [c for c in test.columns if c not in ["player_id", *NOISE]]
NUMS = [c for c in FEATURES if c not in CATS]
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(train, y))


def fold_report(name, oof, t0=None):
    fs = [metrics.macro_f1(y[va], oof[va].argmax(1)) for _, va in FOLDS]
    res = dict(model=name, macro_f1=round(metrics.macro_f1(y, oof.argmax(1)), 4), fold_mean=round(np.mean(fs), 4),
               fold_std=round(np.std(fs), 4), folds=np.round(fs, 4).tolist())
    if t0 is not None:
        res["runtime_s"] = round(time.time() - t0)
    print(res, flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------
# LightGBM stumps
# ----------------------------------------------------------------------------------------------------------
def prep(df):
    X = df[FEATURES].copy()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=sorted(train[c].dropna().unique()))
    return X


STUMP_PARAMS = dict(objective="multiclass", num_class=K, learning_rate=0.05, num_leaves=2, max_depth=1,
                    min_child_samples=100, feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1,
                    lambda_l2=1.0, max_bin=255, seed=utils.SEED, deterministic=True, force_row_wise=True,
                    n_jobs=-1, verbose=-1)


def run_stumps(name="lgbm_stumps", params=STUMP_PARAMS):
    X, Xt = prep(train), prep(test)
    oof, tp, iters = np.zeros((len(X), K)), np.zeros((len(Xt), K)), []
    t0 = time.time()
    for tr, va in FOLDS:
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        m = lgb.train(params, dtr, 20000, valid_sets=[lgb.Dataset(X.iloc[va], y[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        tp += m.predict(Xt, num_iteration=m.best_iteration) / len(FOLDS)
        iters.append(m.best_iteration)
    print("  best_iter per fold:", iters)
    utils.save_preds(TASK, name, oof, tp)
    return fold_report(name, oof, t0), oof, tp


# ----------------------------------------------------------------------------------------------------------
# Binned naive Bayes with exact NaN marginalisation
# ----------------------------------------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb as nbmod  # noqa: E402  (BinnedNB / SmoothNB live in task2/src/nb.py, shared with the notebook)


def BinnedNB(n_bins=32, alpha=1.0):
    return nbmod.BinnedNB(CATS, K, n_bins, alpha)


def SmoothNB(n_bins=128, sigma=4.0, alpha=1.0):
    return nbmod.SmoothNB(CATS, K, n_bins, sigma, alpha)


def mi_select(Xtr, ytr, min_mi):
    """Features whose binned mutual information with the class (training fold only) is >= min_mi nats."""
    keep = []
    for c in FEATURES:
        v = Xtr[c]
        ok = v.notna().to_numpy()
        if c in CATS:
            codes = pd.Categorical(v[ok]).codes
        else:
            x = v[ok].to_numpy(float)
            e = np.unique(np.quantile(x, np.linspace(0, 1, 33)[1:-1]))
            codes = np.searchsorted(e, x, side="right")
        T = pd.crosstab(codes, ytr[ok]).to_numpy(float)
        P = T / T.sum()
        pr, pc = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        mi = (P[nz] * np.log(P[nz] / (pr @ pc)[nz])).sum()
        if mi >= min_mi:
            keep.append(c)
    return keep


def run_nb(name, n_bins=32, alpha=1.0, save=True, sigma=None, min_mi=None):
    oof, tp = np.zeros((len(train), K)), np.zeros((len(test), K))
    t0 = time.time()
    n_kept = []
    for tr, va in FOLDS:
        feats = FEATURES if min_mi is None else mi_select(train.iloc[tr], y[tr], min_mi)
        n_kept.append(len(feats))
        if sigma is None:
            m = BinnedNB(n_bins, alpha).fit(train.iloc[tr][feats], y[tr])
        else:
            m = SmoothNB(n_bins, sigma, alpha).fit(train.iloc[tr][feats], y[tr])
        oof[va] = m.predict_proba(train.iloc[va][feats])
        tp += m.predict_proba(test[feats]) / len(FOLDS)
    if min_mi is not None:
        print(f"  features kept per fold (min MI {min_mi}): {n_kept}")
    if save:
        utils.save_preds(TASK, name, oof, tp)
    return fold_report(name, oof, t0), oof, tp


def nested_offsets(oof, n_rounds=3, grid=np.linspace(-0.6, 0.6, 25)):
    """Per-class additive log-offsets (coordinate ascent on macro F1), chosen on 4 folds, scored on the 5th."""
    L = np.log(np.clip(oof, 1e-12, 1))

    def fit(idx):
        b = np.zeros(K)
        for _ in range(n_rounds):
            for k in range(1, K):
                best = max(grid, key=lambda g: metrics.macro_f1(y[idx], (L[idx] + np.r_[b[:k], g, b[k + 1:]]).argmax(1)))
                b[k] = best
        return b

    fs, bs = [], []
    for k, (_, va) in enumerate(FOLDS):
        inner = np.concatenate([FOLDS[j][1] for j in range(len(FOLDS)) if j != k])
        b = fit(inner)
        bs.append(np.round(b, 2).tolist())
        fs.append(metrics.macro_f1(y[va], (L[va] + b).argmax(1)))
    print(f"  nested per-class offsets: {np.mean(fs):.4f} ± {np.std(fs):.4f} folds {np.round(fs, 4).tolist()} | offsets {bs}")
    return np.mean(fs), fit(np.arange(len(y)))


# ----------------------------------------------------------------------------------------------------------
# Nested blend: weights picked on 4 folds' OOF, scored on the 5th
# ----------------------------------------------------------------------------------------------------------
def nested_blend(parts, grid=np.linspace(0, 1, 21), log_space=True):
    """parts: dict name -> oof probs. Two-model or three-model weight grid; log_space = blend log-probs."""
    names = list(parts)
    P = [np.log(np.clip(parts[n], 1e-12, 1)) if log_space else parts[n] for n in names]
    if len(names) == 2:
        W = [(w, 1 - w) for w in grid]
    else:
        W = [(a, b, 1 - a - b) for a in grid for b in grid if a + b <= 1 + 1e-9]
    fold_scores, chosen = [], []
    for k, (_, va) in enumerate(FOLDS):
        inner = np.concatenate([FOLDS[j][1] for j in range(len(FOLDS)) if j != k])
        best = max(W, key=lambda w: metrics.macro_f1(y[inner], sum(wi * p[inner] for wi, p in zip(w, P)).argmax(1)))
        chosen.append(best)
        fold_scores.append(metrics.macro_f1(y[va], sum(wi * p[va] for wi, p in zip(best, P)).argmax(1)))
    full = max(W, key=lambda w: metrics.macro_f1(y, sum(wi * p for wi, p in zip(w, P)).argmax(1)))
    print(f"  nested blend {names} ({'log' if log_space else 'prob'}): {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}"
          f" folds {np.round(fold_scores, 4).tolist()} | weights per fold {[tuple(np.round(w, 2)) for w in chosen]}"
          f" | full-OOF weights {tuple(np.round(full, 2))}", flush=True)
    return np.mean(fold_scores), np.std(fold_scores), full


if __name__ == "__main__":
    which = sys.argv[1:] or ["stumps", "nb", "blend"]
    ref, _ = utils.load_preds(TASK, "lgbm_comp")
    print("reference lgbm_comp:")
    fold_report("lgbm_comp", ref)
    if "nbgrid" in which:
        for nbins in (8, 16, 32, 64):
            for alpha in (1.0, 10.0):
                run_nb(f"nb_b{nbins}_a{alpha:g}", nbins, alpha, save=False)
    if "snbgrid" in which:
        for nbins, sigma in ((128, 2), (128, 4), (128, 8), (256, 8), (256, 16)):
            run_nb(f"snb_b{nbins}_s{sigma}", nbins, 1.0, save=False, sigma=sigma)
    if "nb" in which:
        run_nb("nb", 32, 1.0)
        run_nb("snb", 128, 1.0, sigma=4.0)
    if "mi" in which:
        for mm in (0.0005, 0.001, 0.002):
            run_nb(f"snb_mi{mm:g}", 128, 1.0, save=False, sigma=4.0, min_mi=mm)
    if "offsets" in which:
        nested_offsets(utils.load_preds(TASK, "snb")[0])
    if "stumps" in which:
        run_stumps()
    if "blend" in which:
        parts = {"lgbm_comp": ref}
        for n in ("snb", "lgbm_stumps"):
            try:
                parts[n] = utils.load_preds(TASK, n)[0]
            except FileNotFoundError:
                pass
        for n in [p for p in parts if p != "lgbm_comp"]:
            nested_blend({"lgbm_comp": ref, n: parts[n]})
        if len(parts) == 3:
            nested_blend(parts)
