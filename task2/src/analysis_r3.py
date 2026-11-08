"""Task 2 round 3 analysis (2026-11-08): where is the structure the trees miss?
Run from repo root: python task2/src/analysis_r3.py"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from common import metrics, paths, utils

utils.seed_everything()
D = paths.task_data(2)
train = pd.read_csv(D / "train.csv")
test = pd.read_csv(D / "test.csv")
y = train.segment.to_numpy()
CATS = [c for c in test.columns if train[c].dtype == object or str(train[c].dtype) in ("str", "string") and c != "player_id"]
CATS = [c for c in CATS if c != "player_id"]
NOISE = ["random_metric_1", "random_metric_2", "random_metric_3"]
NUMS = [c for c in test.columns if c not in CATS + NOISE + ["player_id"]]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y))
oof = np.load(paths.outputs_dir(2) / "oof_lgbm_comp.npy")
pred = oof.argmax(1)
NAMES = ["Casual", "Grinder", "Social", "Whale"]
print(f"reference lgbm_comp: macro F1 {metrics.macro_f1(y, pred):.4f}")

print("\n== B1. artefacts")
pid = train.player_id.str[1:].astype(int)
print("  player_id AUC one-vs-rest:", [round(roc_auc_score(y == k, pid), 3) for k in range(4)],
      "| lag-1 same-class rate", round(float((y[1:] == y[:-1]).mean()), 4), "vs chance",
      round(float((np.bincount(y) / len(y)) @ (np.bincount(y) / len(y))), 4))
rows = []
for c in NUMS:
    v = train[c]
    ok = v.notna().to_numpy()
    vv, yy = v[ok], y[ok]
    integ = (vv % 1 == 0).to_numpy()
    dec = vv.astype(str).str.split(".").str[-1].str.len().to_numpy()
    lo, hi = (vv == vv.min()).to_numpy(), (vv == vv.max()).to_numpy()
    rows.append(dict(feat=c, **{f"int{k}": integ[yy == k].mean() for k in range(4)},
                     **{f"lo{k}": lo[yy == k].mean() for k in range(4)}, **{f"hi{k}": hi[yy == k].mean() for k in range(4)},
                     **{f"dec{k}": dec[yy == k].mean() for k in range(4)}))
r = pd.DataFrame(rows).set_index("feat")
spread = lambda p: r[[f"{p}{k}" for k in range(4)]].max(1) - r[[f"{p}{k}" for k in range(4)]].min(1)  # noqa: E731
r["gap"] = np.maximum.reduce([spread("int"), spread("lo"), spread("hi")])
r["dec_gap"] = spread("dec")
show = r[(r.gap > 0.02) | (r.dec_gap > 0.5)]
print(show.round(3).to_string() if len(show) else "  none")

print("\n== B2. confidence regimes and calibration (OOF max prob)")
mx = oof.max(1)
bins = pd.cut(mx, [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
print(pd.DataFrame({"b": bins, "ok": pred == y}).groupby("b", observed=True).agg(n=("ok", "size"), acc=("ok", "mean")).round(3).T.to_string())

print("\n== B3. confusion (rows true, cols pred)")
print(pd.DataFrame(confusion_matrix(y, pred), index=NAMES, columns=NAMES).to_string())

Xn = train[NUMS]
sd = Xn.std()
print("\n== B4. what separates the big confusions? standardised mean diff, misclassified vs correctly classified of the TRUE class")
for t, p in [(1, 0), (2, 0), (0, 1), (3, 2), (3, 1)]:
    a = (y == t) & (pred == p)
    b = (y == t) & (pred == t)
    smd = ((Xn[a].mean() - Xn[b].mean()) / sd).sort_values(key=np.abs, ascending=False).head(6)
    print(f"  true {NAMES[t]} -> pred {NAMES[p]} (n={a.sum()}): " + ", ".join(f"{k} {v:+.2f}" for k, v in smd.items()))

print("\n== B5. within-class feature correlations that differ between classes (top 12 by range)")
sub = train.sample(40000, random_state=42)
cs = [sub[sub.segment == k][NUMS].corr(method="spearman") for k in range(4)]
stack = np.stack([c.to_numpy() for c in cs])
rng_ = stack.max(0) - stack.min(0)
iu = np.triu_indices(len(NUMS), 1)
order = np.argsort(-rng_[iu])[:12]
for i in order:
    a, b = iu[0][i], iu[1][i]
    print(f"  {NUMS[a]:30s} x {NUMS[b]:30s} " + " ".join(f"{NAMES[k][:3]} {stack[k, a, b]:+.2f}" for k in range(4)))

print("\n== B6. linear / quadratic discriminants vs trees (5-fold, in-fold preprocessing)")
pre_num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
models = {
    "LDA": lambda: make_pipeline(ColumnTransformer([("n", pre_num, NUMS)]), LinearDiscriminantAnalysis()),
    "QDA": lambda: make_pipeline(ColumnTransformer([("n", pre_num, NUMS)]), QuadraticDiscriminantAnalysis(reg_param=0.05)),
    "LR+cats": lambda: make_pipeline(ColumnTransformer([
        ("n", pre_num, NUMS),
        ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CATS)]),
        LogisticRegression(max_iter=3000, C=1.0)),
    "splineLR": lambda: make_pipeline(ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=5, degree=3)), NUMS),
        ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CATS)]),
        LogisticRegression(max_iter=3000, C=0.3)),
}
glm_oof = {}
for name, mk in models.items():
    t0 = time.time()
    o = np.zeros((len(y), 4))
    for tr, va in FOLDS:
        m = mk().fit(train.iloc[tr], y[tr])
        o[va] = m.predict_proba(train.iloc[va])
    glm_oof[name] = o
    print(f"  {name:9s} macro F1 {metrics.macro_f1(y, o.argmax(1)):.4f}  ({time.time() - t0:.0f}s)")
    np.save(paths.outputs_dir(2) / f"oof_diag_{name}.npy", o)
for name, o in glm_oof.items():
    for w in (0.2, 0.4):
        print(f"  blend LGBM + {w}·{name}: {metrics.macro_f1(y, ((1 - w) * oof + w * o).argmax(1)):.4f}")

print("\n== B7. error vs position on the latent axes (eng_score x spend_score quartiles), error rate")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_composites  # noqa: E402
comp = add_composites(train, [train])[0]
e = pd.qcut(comp.eng_score, 4, labels=False)
s = pd.qcut(comp.spend_score, 4, labels=False)
print(pd.crosstab(e, s, values=(pred != y).astype(float), aggfunc="mean").round(3).to_string())
print("  majority true class per cell:")
print(pd.crosstab(e, s, values=y, aggfunc=lambda v: NAMES[np.bincount(v).argmax()][:3]).to_string())
