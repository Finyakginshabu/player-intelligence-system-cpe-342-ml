"""Task 3 round 4 analysis (2026-11-08): where is the structure the trees miss?
Run from repo root: python task3/src/analysis_r4.py"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from common import metrics, paths, utils

utils.seed_everything()
D = paths.task_data(3)
train = pd.read_csv(D / "train.csv")
test = pd.read_csv(D / "test.csv")
y = train.spending_30d.to_numpy(float)
FEATS = [c for c in test.columns if c != "player_id"]
X = train[FEATS]
nz = y > 0
bins = np.zeros(len(y), int)
bins[nz] = 1 + pd.qcut(y[nz], 5, labels=False)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, bins))
oof = np.load(paths.outputs_dir(3) / "oof_twostage.npy")
parts = np.load(paths.outputs_dir(3) / "oof_twostage_parts.npy")
p_oof = parts[:, 0]
print(f"reference twostage NMAE {metrics.nmae(y, oof):.4f}")

print("\n== B1. target format")
yp = y[nz]
print(f"  zeros {1 - nz.mean():.4f} | positives min {yp.min():.2f} | at max 300000: {(y == 300000).sum()} "
      f"| 2-decimal share {np.mean(np.isclose(yp * 100, np.round(yp * 100))):.3f} | integer share {np.mean(yp % 1 == 0):.3f}")
for c in ["prev_month_spending", "historical_spending", "avg_transaction_value"]:
    ok = nz & train[c].notna().to_numpy() & (train[c].to_numpy() > 0)
    lr = np.log(y[ok] / train[c].to_numpy()[ok])
    print(f"  log(y / {c:22s}) among spenders: median {np.median(lr):+.3f}  IQR {np.subtract(*np.percentile(lr, [75, 25])):.3f}"
          f"  corr(log y, log x) {np.corrcoef(np.log(y[ok]), np.log(train[c].to_numpy()[ok]))[0, 1]:+.3f}")

print("\n== B2. does any single feature gate spending (hard zero/non-zero rule)?")
rows = []
for c in FEATS:
    v = train[c]
    ok = v.notna().to_numpy()
    if v.nunique() <= 12:
        rates = pd.Series(nz[ok]).groupby(v[ok].to_numpy()).mean()
    else:
        rates = pd.Series(nz[ok]).groupby(pd.qcut(v[ok], 20, duplicates="drop").to_numpy()).mean()
    rows.append((c, rates.min(), rates.max(), roc_auc_score(nz[ok], v[ok])))
g = pd.DataFrame(rows, columns=["feat", "min_rate", "max_rate", "auc"]).set_index("feat")
print(g[(g.min_rate < 0.05) | (g.max_rate > 0.95) | ((g.auc - 0.5).abs() > 0.1)].round(3).to_string())
print(f"  stage-1 OOF AUC(spend>0) {roc_auc_score(nz, p_oof):.4f}")
cal = pd.DataFrame({"b": pd.cut(p_oof, np.linspace(0, 1, 11)), "y": nz}).groupby("b", observed=True).y.agg(["size", "mean"])
print("  calibration (p bin -> true rate):", {str(k): round(v, 3) for k, v in cal["mean"].items()})

print("\n== B3. error budget of the twostage OOF")
err = np.abs(y - oof)
yb = pd.cut(y, [-1, 0, 100, 1000, 10000, 50000, 1e9], labels=["0", "≤100", "≤1k", "≤10k", "≤50k", ">50k"])
eb = pd.DataFrame({"b": yb, "err": err, "y": y, "seg": train.segment.fillna(-1).to_numpy()})
t = eb.groupby("b", observed=True).agg(n=("err", "size"), err_share=("err", "sum"), y_share=("y", "sum"))
t["err_share"] /= err.sum()
t["y_share"] /= y.sum()
print(t.round(3).to_string())
s = eb.groupby("seg").agg(n=("err", "size"), err_share=("err", "sum"))
s["err_share"] /= err.sum()
print("  by segment:", s.round(3).to_dict("index"))

print("\n== B4. spenders: log(y) — linear on log features vs trees (5-fold R² and MAE on raw scale)")
Xs = X[nz].reset_index(drop=True)
ly = np.log(yp)
LOG = ["historical_spending", "prev_month_spending", "avg_transaction_value", "total_transactions", "purchase_frequency",
       "total_playtime_hours", "friend_count", "social_interactions", "achievement_count", "cross_game_activity"]
Xl = Xs.copy()
for c in LOG:
    Xl[c] = np.log1p(Xl[c].clip(lower=0))
seg_cat = ["segment", "vip_status", "primary_game", "platform"]
num = [c for c in FEATS if c not in seg_cat]
res = {}
for name, mk, data in [
    ("linear(log X)", lambda: make_pipeline(ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1), OneHotEncoder(handle_unknown="ignore")), seg_cat)]),
        LinearRegression()), Xl),
    ("spline(log X)", lambda: make_pipeline(ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=6)), num),
        ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1), OneHotEncoder(handle_unknown="ignore")), seg_cat)]),
        LinearRegression()), Xl),
]:
    o = np.zeros(len(ly))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(Xs, pd.qcut(ly, 5, labels=False)):
        o[va] = mk().fit(data.iloc[tr], ly[tr]).predict(data.iloc[va])
    res[name] = o
o = np.zeros(len(ly))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(Xs, pd.qcut(ly, 5, labels=False)):
    m = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1),
                  lgb.Dataset(Xs.iloc[tr], ly[tr]), 4000, valid_sets=[lgb.Dataset(Xs.iloc[va], ly[va])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    o[va] = m.predict(Xs.iloc[va], num_iteration=m.best_iteration)
res["lgbm(raw X)"] = o
for name, o in res.items():
    r2 = 1 - np.mean((ly - o) ** 2) / np.var(ly)
    print(f"  {name:14s} R² log-y {r2:.4f} | resid std {np.std(ly - o):.3f} | MAE raw (exp pred) {np.mean(np.abs(yp - np.exp(o))):.0f}")
resid = ly - res["lgbm(raw X)"]
print(f"  residual skew {pd.Series(resid).skew():+.3f}, kurtosis {pd.Series(resid).kurt():+.3f} (lognormal noise -> ~0, ~0)")

print("\n== B5. within-segment correlations between features differ by segment? (top 8 by range, spenders + non)")
sub = train.dropna(subset=["segment"]).sample(40000, random_state=42)
cs = [sub[sub.segment == k][FEATS[:-1]].corr(method="spearman") for k in range(4)]
st = np.stack([c.to_numpy() for c in cs])
rg = st.max(0) - st.min(0)
iu = np.triu_indices(len(FEATS) - 1, 1)
for i in np.argsort(-rg[iu])[:8]:
    a, b = iu[0][i], iu[1][i]
    print(f"  {FEATS[a]:26s} x {FEATS[b]:26s} " + " ".join(f"s{k} {st[k, a, b]:+.2f}" for k in range(4)))
print("  overall corr log(prev) vs log(hist) among spenders:",
      round(np.corrcoef(np.log1p(Xs.prev_month_spending.fillna(Xs.prev_month_spending.median())),
                        np.log1p(Xs.historical_spending.fillna(Xs.historical_spending.median())))[0, 1], 3))
