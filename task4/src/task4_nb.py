# %% [markdown]
# # Task 4: Game Title Detection (CPU fallback)
#
# - **Type:** 5-class image classification, 230×120 JPEG crops. **Metric:** macro F1. Pretrained models allowed.
# - **This notebook = CPU fallback:** frozen ImageNet ConvNeXt-Tiny embeddings (`src/features.py`, cached)
#   + logistic regression, 5-fold StratifiedGroupKFold (near-duplicate groups by average hash).
#   The GPU fine-tuning version is `task4_kaggle.ipynb`.
# - **Labels:** train.csv holds strings; submitted integers follow the datacard
#   `{7-11k:0, fiveN:1, freefried:2, pube:3, roblock:4}` (= alphabetical; see README "Conflicts").
#
# Source: `src/task4_nb.py` (jupytext percent format). Experiments are logged in `experiments.md`.

# %%
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
while not (ROOT / "common").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import metrics, paths, utils

TASK = 4
BACKBONE = "convnext_tiny"
LABELS = ["7-11k", "fiveN", "freefried", "pube", "roblock"]  # index = submitted integer (datacard order)
utils.seed_everything()
DATA = paths.task_data(TASK)
OUT = paths.outputs_dir(TASK)

# %% [markdown]
# ## 1. Data card & load

# %%
print(paths.datacard(TASK)[:420])
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
assert set(train["class"]) == set(LABELS)
y = train["class"].map({c: i for i, c in enumerate(LABELS)}).to_numpy()
print("train", train.shape, "test", test.shape)
print(train["class"].value_counts(normalize=True).round(4))

# %% [markdown]
# ## 2. Embeddings + image metadata (cached; computed by `src/features.py` if missing, ~15 min on CPU)

# %%
f_tr, f_te = OUT / f"feat_{BACKBONE}_train.npy", OUT / f"feat_{BACKBONE}_test.npy"
if not (f_tr.exists() and f_te.exists()):
    subprocess.run([sys.executable, str(ROOT / "task4/src/features.py"), "--backbone", BACKBONE], check=True)
X, X_test = np.load(f_tr), np.load(f_te)
meta_tr, meta_te = pd.read_csv(OUT / "img_meta_train.csv"), pd.read_csv(OUT / "img_meta_test.csv")
assert (meta_tr["id"].astype(str).values == train["id"].astype(str).values).all()
assert (meta_te["id"].astype(str).values == test["id"].astype(str).values).all()
print(X.shape, X_test.shape)

# %% [markdown]
# ## 3. EDA (details in `experiments.md`)

# %%
print("image sizes train:", meta_tr.groupby(["width", "height"]).size().to_dict(),
      "| test:", meta_te.groupby(["width", "height"]).size().to_dict())
meta_tr["label"] = train["class"].values
print(meta_tr.groupby("label")[["mean_r", "mean_g", "mean_b"]].mean().round(3))
dup = meta_tr["ahash"].duplicated(keep=False)
mixed = (meta_tr[dup].groupby("ahash")["label"].nunique() > 1).sum()
cross = meta_te["ahash"].isin(set(meta_tr["ahash"])).mean()
print(f"train rows in a duplicate-hash group: {dup.mean():.3%} ({meta_tr['ahash'].nunique():,} unique hashes); "
      f"mixed-label groups: {mixed}; test rows whose hash appears in train: {cross:.3%}")

# %% [markdown]
# ## 4. Cross-validation: logistic regression on standardised embeddings
# Groups = identical average hash → near-duplicates stay in the same fold. Scaler fit inside each fold.

# %%
groups = meta_tr["ahash"].factorize()[0]
FOLDS = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(X, y, groups))


def run_cv(name, **lr_kw):
    t0 = time.time()
    oof, pred = np.zeros((len(X), 5)), np.zeros((len(X_test), 5))
    for tr, va in FOLDS:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=utils.SEED, **lr_kw))
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])
        pred += model.predict_proba(X_test) / len(FOLDS)
    folds = [metrics.macro_f1(y[va], oof[va].argmax(1)) for _, va in FOLDS]
    print(f"{name:14s} macro F1 {metrics.macro_f1(y, oof.argmax(1)):.4f} +- {np.std(folds):.4f}  ({time.time() - t0:.0f}s)")
    utils.save_preds(TASK, name, oof, pred)
    return dict(name=name, score=metrics.macro_f1(y, oof.argmax(1)), std=np.std(folds), oof=oof, pred=pred)


runs = [run_cv("lr_C0.1", C=0.1), run_cv("lr_C0.1_bal", C=0.1, class_weight="balanced")]

# %%
best = max(runs, key=lambda r: r["score"])
print("best:", best["name"])
pred_oof = best["oof"].argmax(1)
print(pd.DataFrame(confusion_matrix(y, pred_oof), index=LABELS, columns=LABELS))
print("per-class F1:", dict(zip(LABELS, f1_score(y, pred_oof, average=None).round(4))))
pd.DataFrame([{k: v for k, v in r.items() if k in ("name", "score", "std")} for r in runs]).to_csv(OUT / "cv_results.csv", index=False)

# %% [markdown]
# ## 5. Write `outputs/final_pred.csv` (integers 0–4, datacard order)

# %%
test_label = best["pred"].argmax(1)
print("test pred dist:", dict(zip(LABELS, np.bincount(test_label, minlength=5) / len(test_label))))
final = utils.write_final_pred(TASK, test["id"], test_label)
final.head()
