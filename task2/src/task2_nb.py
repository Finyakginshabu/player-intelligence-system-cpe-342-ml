# %% [markdown]
# # Task 2: Player Segmentation
#
# - **Type:** 4-class classification, target `segment` (0 Casual, 1 Grinder, 2 Social, 3 Whale)
# - **Metric:** F1 **macro**. Post-processing: per-class probability multipliers tuned on OOF (coordinate ascent).
#
# Final notebook: must run top-to-bottom. Source: `src/task2_nb.py` (jupytext percent format).
# Experiments are logged in `experiments.md`.

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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils

sys.path.insert(0, str(ROOT / "task2" / "src"))
from features import ENGAGE, SPEND, add_composites  # latent-axis composites (round 2)

TASK = 2
utils.seed_everything()
DATA = paths.task_data(TASK)
OUT = paths.outputs_dir(TASK)

# %% [markdown]
# ## 1. Data card & load

# %%
print(paths.datacard(TASK)[:600])
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["segment"].astype(int).values
print("train", train.shape, "test", test.shape)
print(pd.Series(y).value_counts(normalize=True).sort_index().round(4))

# %% [markdown]
# ## 2. EDA (details in `experiments.md`)
#
# Two latent axes: **engagement** (playtime, frequency, chat, friends, tournaments: high for Grinder/Whale)
# and **spending** (total/monthly spending, spending frequency, rare items: high for Social/Whale).
# Only `loyalty_tier` and `engagement_level` among the categoricals carry signal.

# %%
CATS = [c for c in test.columns if not pd.api.types.is_numeric_dtype(test[c]) and c != "player_id"]
NOISE = ["random_metric_1", "random_metric_2", "random_metric_3"]  # datacard: pure noise
FEATURES = [c for c in test.columns if c not in ["player_id", *NOISE]]
NUMS = [c for c in FEATURES if c not in CATS]
train.groupby("segment")[["total_playtime_hours", "chat_activity_score", "avg_monthly_spending", "rare_items_count", "friend_count"]].median().round(1)

# %% [markdown]
# ## 3. Features
# Raw features (noise columns dropped). Categoricals -> pandas `category` with levels fixed from train,
# NaN handled natively by LightGBM.
#
# **Latent-axis composites** (`src/features.py`, round 2: +0.0029 macro F1, better in all 5 folds): mean z-score
# of the engagement features (`eng_score`) and of the spending features (`spend_score`), their sum/difference,
# ordinal loyalty/engagement tiers, spend per playtime hour and rare items per THB. The z-score stats are fit on
# the training fold only, inside the CV loop.

# %%
def prep(df):
    X = df[FEATURES].copy()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=sorted(train[c].dropna().unique()))
    return X

X, X_test = prep(train), prep(test)
print(len(ENGAGE), "engagement +", len(SPEND), "spending features in the composites")
X.shape, X_test.shape

# %% [markdown]
# ## 4. Cross-validation: LightGBM multiclass, 5-fold stratified

# %%
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(X, y))
LGB_PARAMS = dict(
    objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    seed=utils.SEED, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1,
)

t0 = time.time()
oof = np.zeros((len(X), 4))
test_prob = np.zeros((len(X_test), 4))
best_iters, importance = [], None
for k, (tr, va) in enumerate(FOLDS):
    Xtr, Xva, Xte = add_composites(X.iloc[tr], [X.iloc[tr], X.iloc[va], X_test])  # stats from the training fold
    dtr = lgb.Dataset(Xtr, y[tr])
    dva = lgb.Dataset(Xva, y[va], reference=dtr)
    m = lgb.train(LGB_PARAMS, dtr, 5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
    test_prob += m.predict(Xte, num_iteration=m.best_iteration) / len(FOLDS)
    best_iters.append(m.best_iteration)
    imp = pd.Series(m.feature_importance("gain"), index=Xtr.columns)
    importance = imp if importance is None else importance + imp
    print(f"fold {k}: macro F1 (argmax) {metrics.macro_f1(y[va], oof[va].argmax(1)):.4f}  iter {m.best_iteration}")
print(f"runtime {time.time() - t0:.0f}s")
utils.save_preds(TASK, "lgbm_comp", oof, test_prob)

# %% [markdown]
# ### Post-processing: per-class probability multipliers (coordinate ascent on OOF)
# Honest estimate: multipliers fit on 4 folds' OOF, scored on the 5th (nested).

# %%
GRID = np.round(np.arange(0.5, 2.01, 0.05), 2)


def fit_multipliers(p, yt, n_rounds=3):
    w = np.ones(p.shape[1])
    best = metrics.macro_f1(yt, (p * w).argmax(1))
    for _ in range(n_rounds):
        for c in range(1, p.shape[1]):  # class 0 fixed as reference
            for g in GRID:
                w2 = w.copy(); w2[c] = g
                s = metrics.macro_f1(yt, (p * w2).argmax(1))
                if s > best + 1e-9:
                    best, w = s, w2
    return w, best


fold_raw, fold_nested = [], []
for k, (tr, va) in enumerate(FOLDS):
    w_k, _ = fit_multipliers(oof[tr], y[tr])
    fold_raw.append(metrics.macro_f1(y[va], oof[va].argmax(1)))
    fold_nested.append(metrics.macro_f1(y[va], (oof[va] * w_k).argmax(1)))
W, full_score = fit_multipliers(oof, y)
raw_score = metrics.macro_f1(y, oof.argmax(1))
print(f"argmax       macro F1 {raw_score:.4f}  (fold std {np.std(fold_raw):.4f})")
print(f"multipliers  nested   {np.mean(fold_nested):.4f} +- {np.std(fold_nested):.4f}; full-OOF {full_score:.4f}; w={W}")
# Keep the multipliers only if they beat plain argmax out-of-fold (nested); otherwise they are OOF overfit.
if np.mean(fold_nested) <= raw_score:
    W = np.ones(4)
    print("nested multipliers do not beat argmax -> using argmax (w = 1)")

# %%
pred_oof = (oof * W).argmax(1)
cm = pd.DataFrame(confusion_matrix(y, pred_oof), index=[f"true_{i}" for i in range(4)], columns=[f"pred_{i}" for i in range(4)])
print(cm)
print("per-class F1:", np.round(f1_score(y, pred_oof, average=None), 4))
(importance / importance.sum()).sort_values(ascending=False).head(15).round(4)

# %% [markdown]
# ## 5. Final predictions
# Fold-averaged test probabilities x OOF-tuned multipliers.

# %%
test_label = (test_prob * W).argmax(1)
print("OOF pred dist ", np.bincount(pred_oof, minlength=4) / len(pred_oof))
print("test pred dist", np.bincount(test_label, minlength=4) / len(test_label))
pd.DataFrame([{"model": "lgbm_comp", "macro_f1_argmax": raw_score, "macro_f1_nested_mult": np.mean(fold_nested),
               "fold_std": np.std(fold_nested), "macro_f1_full_mult": full_score, "weights": W.tolist(),
               "best_iter_mean": np.mean(best_iters)}]).to_csv(OUT / "cv_results.csv", index=False)

# %% [markdown]
# ## 6. Write `outputs/final_pred.csv`

# %%
final = utils.write_final_pred(TASK, test["player_id"], test_label)
final.head()
