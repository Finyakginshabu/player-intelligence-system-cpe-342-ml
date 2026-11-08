# %% [markdown]
# # Task 3: Monthly Spending Prediction
#
# - **Type:** Regression, target `spending_30d` (THB), zero-inflated and heavy-tailed
# - **Metric:** NMAE = 1 / (1 + Σ|y−ŷ| / Σy). Minimising MAE → the optimal prediction is the
#   **conditional median**, so we use L1 objectives (never MSE).
# - Post-processing tuned on OOF: global multiplier α and zero-gate τ (predict 0 when ŷ < τ).
#
# Final notebook: must run top-to-bottom. Source: `src/task3_nb.py` (jupytext percent format).

# %%
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
while not (ROOT / "common").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils

TASK = 3
utils.seed_everything()
DATA = paths.task_data(TASK)
OUT = paths.outputs_dir(TASK)

# %% [markdown]
# ## 1. Data card & load

# %%
print(paths.datacard(TASK)[:500])
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["spending_30d"].values.astype(float)
print("train", train.shape, "test", test.shape, "| zero frac", (y == 0).mean().round(4),
      "| median", np.median(y), "| mean", y.mean().round(1))

# %% [markdown]
# ## 2. EDA (details in `experiments.md`)
# 48% zeros, median 79 THB vs mean 10,497 THB. `segment` (Task 2 label, given as a feature) is the strongest
# signal: P(spend > 0) is 0.28 / 0.52 / 0.69 / 0.93 for Casual / Grinder / Social / Whale.

# %%
FEATURES = [c for c in test.columns if c != "player_id"]
train.groupby("segment")["spending_30d"].agg(["size", lambda s: (s > 0).mean(), "median", "mean"]).round(2)

# %% [markdown]
# ## 3. Features & folds
# Raw features, NaN handled natively. Folds are stratified on spend bins (zero + quintiles of non-zero spend)
# so every fold sees the same share of zeros and big spenders.

# %%
X, X_test = train[FEATURES], test[FEATURES]
bins = np.zeros(len(y), dtype=int)
nz = y > 0
bins[nz] = 1 + pd.qcut(y[nz], 5, labels=False)
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(X, bins))

BASE = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=utils.SEED,
            deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)


def run_cv(name, params, target_fn=lambda v: v, inverse_fn=lambda v: v):
    t0 = time.time()
    oof, pred = np.zeros(len(X)), np.zeros(len(X_test))
    for tr, va in FOLDS:
        dtr = lgb.Dataset(X.iloc[tr], target_fn(y[tr]))
        dva = lgb.Dataset(X.iloc[va], target_fn(y[va]), reference=dtr)
        m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = np.clip(inverse_fn(m.predict(X.iloc[va], num_iteration=m.best_iteration)), 0, None)
        pred += np.clip(inverse_fn(m.predict(X_test, num_iteration=m.best_iteration)), 0, None) / len(FOLDS)
    folds = [metrics.nmae(y[va], oof[va]) for _, va in FOLDS]
    print(f"{name:10s} NMAE {metrics.nmae(y, oof):.4f} +- {np.std(folds):.4f}  ({time.time() - t0:.0f}s)")
    utils.save_preds(TASK, name, oof, pred)
    return oof, pred


# %% [markdown]
# ## 4. Cross-validation
# (a) L1 on the raw target; (b) L1 on log1p(target), inverted with expm1. The median commutes with any monotone
# transform, so (b) still estimates the conditional median, with a much better-behaved target.

# %%
print("constant-median baseline NMAE", round(metrics.nmae(y, np.full_like(y, np.median(y))), 4))
res = {}
res["lgbm_l1"] = run_cv("lgbm_l1", {**BASE, "objective": "l1"})
res["lgbm_logl1"] = run_cv("lgbm_logl1", {**BASE, "objective": "l1"}, np.log1p, np.expm1)

# %% [markdown]
# ### (c) Two-stage conditional median (round 3, 2026-11-08: +0.0038 NMAE, every fold up)
# With p = P(y > 0 | x) and F+ = distribution of y among spenders, the median of the zero-inflated mixture is
# 0 if p ≤ 0.5, else F+⁻¹((p − 0.5)/p). Stage 1: LightGBM binary classifier for p. Stage 2: LightGBM quantile
# regressors on spenders only (log1p target; quantiles commute with monotone transforms) at levels covering
# [0, 0.5], interpolated per row at q(p). This uses the zero/non-zero structure explicitly instead of asking
# one L1 model to learn it.

# %%
ALPHAS_Q = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])


def fit_predict(params, Xtr, ytr, Xva, yva, Xte):
    dtr = lgb.Dataset(Xtr, ytr)
    m = lgb.train(params, dtr, 5000, valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xva, num_iteration=m.best_iteration), m.predict(Xte, num_iteration=m.best_iteration)


def mixture_median(p, Q):
    Q = np.sort(Q, axis=1)  # fix quantile crossing
    q = np.clip((p - 0.5) / np.maximum(p, 1e-9), 0, ALPHAS_Q[-1])
    med = np.array([np.interp(qi, ALPHAS_Q, Qi) for qi, Qi in zip(q, Q)])
    return np.where(p <= 0.5, 0.0, med)


t0 = time.time()
nz = y > 0
oof_p, test_p = np.zeros(len(X)), np.zeros(len(X_test))
oof_Q, test_Q = np.zeros((len(X), len(ALPHAS_Q))), np.zeros((len(X_test), len(ALPHAS_Q)))
for tr, va in FOLDS:
    a, b = fit_predict({**BASE, "objective": "binary"}, X.iloc[tr], nz[tr].astype(int), X.iloc[va],
                       nz[va].astype(int), X_test)
    oof_p[va], test_p = a, test_p + b / len(FOLDS)
    trp, vap = tr[nz[tr]], va[nz[va]]  # spenders only
    for j, al in enumerate(ALPHAS_Q):
        _, b = fit_predict({**BASE, "objective": "quantile", "alpha": al}, X.iloc[trp], np.log1p(y[trp]),
                           X.iloc[vap], np.log1p(y[vap]), pd.concat([X.iloc[va], X_test]))
        oof_Q[va, j] = np.expm1(b[:len(va)])
        test_Q[:, j] += np.expm1(b[len(va):]) / len(FOLDS)
oof_two, test_two = mixture_median(oof_p, oof_Q), mixture_median(test_p, test_Q)
folds_two = [metrics.nmae(y[va], oof_two[va]) for _, va in FOLDS]
print(f"twostage   NMAE {metrics.nmae(y, oof_two):.4f} +- {np.std(folds_two):.4f}  ({time.time() - t0:.0f}s)")
utils.save_preds(TASK, "twostage", oof_two, test_two)
res["twostage"] = (oof_two, test_two)

# %% [markdown]
# ### Post-processing: multiplier α and zero-gate τ (nested: fit on 4 folds' OOF, score on the 5th)

# %%
ALPHAS = np.round(np.arange(0.70, 1.31, 0.02), 2)
TAUS = np.r_[0, np.geomspace(1, 2000, 40)]


def apply_pp(p, a, t):
    return np.where(p < t, 0.0, p * a)


def fit_pp(p, yt):
    best = (-1, 1.0, 0.0)
    for a in ALPHAS:
        for t in TAUS:
            s = metrics.nmae(yt, apply_pp(p, a, t))
            if s > best[0]:
                best = (s, a, t)
    return best


summary = []
for name, (oof, pred) in res.items():
    nested = []
    for tr, va in FOLDS:
        _, a, t = fit_pp(oof[tr], y[tr])
        nested.append(metrics.nmae(y[va], apply_pp(oof[va], a, t)))
    s_full, a, t = fit_pp(oof, y)
    summary.append(dict(model=name, nmae_raw=metrics.nmae(y, oof), nmae_pp_nested=np.mean(nested),
                        fold_std=np.std(nested), nmae_pp_full=s_full, alpha=a, tau=t))
summary = pd.DataFrame(summary).set_index("model")
summary.to_csv(OUT / "cv_results.csv")
summary.round(4)

# %% [markdown]
# ## 5. Final predictions
# Best model by nested NMAE; apply post-processing only if it beats the raw predictions out-of-fold.

# %%
best = summary["nmae_pp_nested"].idxmax()
row = summary.loc[best]
use_pp = row["nmae_pp_nested"] > row["nmae_raw"]
a, t = (row["alpha"], row["tau"]) if use_pp else (1.0, 0.0)
test_pred = apply_pp(res[best][1], a, t)
oof_final = apply_pp(res[best][0], a, t)
print(f"best={best}, post-processing={'on' if use_pp else 'off'} (alpha={a}, tau={t:.1f})")
print(f"zero share  OOF {np.mean(oof_final == 0):.3f} | test {np.mean(test_pred == 0):.3f} | true {np.mean(y == 0):.3f}")
print(f"median pred OOF {np.median(oof_final):.1f} | test {np.median(test_pred):.1f}")

# %% [markdown]
# ## 6. Write `outputs/final_pred.csv`

# %%
final = utils.write_final_pred(TASK, test["player_id"], np.round(test_pred, 2))
final.head()
