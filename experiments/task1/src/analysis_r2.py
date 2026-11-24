"""Task 1 round 2 analysis (2026-11-08): (A) does the train/test missingness shift hurt? (B) where is the structure
the trees miss? Run from repo root: python task1/src/analysis_r2.py [A|B ...]  (default: both)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils

utils.seed_everything()
D = paths.task_data(1)
tr_raw = pd.read_csv(D / "train.csv")
test = pd.read_csv(D / "test.csv")
train = tr_raw[tr_raw.is_cheater.notna()].reset_index(drop=True)
y = train.is_cheater.astype(int).to_numpy()
FEATS = [c for c in test.columns if c != "player_id"]
X, X_test = train[FEATS], test[FEATS]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, deterministic=True, force_row_wise=True,
         n_jobs=-1, verbose=-1)
oof_base = np.load(paths.outputs_dir(1) / "oof_lgbm.npy")
rate_tr, rate_te = X.isna().mean(), X_test.isna().mean()


def remask(df, rng, target=rate_te):
    """Add NaNs so each column's NaN rate reaches the test rate (columns already above it are left alone)."""
    out = df.copy()
    for c in FEATS:
        cur = out[c].isna().mean()
        if target[c] > cur:
            p = (target[c] - cur) / (1 - cur)
            out.loc[out[c].notna() & (rng.random(len(out)) < p), c] = np.nan
    return out


def nested_f2(oof):
    pred = np.zeros(len(y), int)
    for trn, va in FOLDS:
        t, _ = metrics.best_fbeta_threshold(y[trn], oof[trn], 2)
        pred[va] = oof[va] >= t
    return metrics.f2(y, pred)


def part_a():
    print("== A. missingness shift: columns where test has MORE NaN than train")
    d = pd.DataFrame({"train": rate_tr, "test": rate_te}).query("test > train + 0.01").round(3)
    print(d.to_string())
    rng = np.random.default_rng(42)
    oof_clean, oof_masked, oof_aug_masked, oof_aug_clean = (np.zeros(len(y)) for _ in range(4))
    t0 = time.time()
    for trn, va in FOLDS:
        Xva_m = remask(X.iloc[va], rng)
        m = lgb.train(P, lgb.Dataset(X.iloc[trn], y[trn]), 5000, valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_clean[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        oof_masked[va] = m.predict(Xva_m, num_iteration=m.best_iteration)
        # augmentation: train on train-fold rows re-masked to the union of both masks' rates
        Xtr_aug = remask(X.iloc[trn], rng, target=np.maximum(rate_tr, rate_te))
        m2 = lgb.train(P, lgb.Dataset(Xtr_aug, y[trn]), 5000, valid_sets=[lgb.Dataset(Xva_m, y[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_aug_masked[va] = m2.predict(Xva_m, num_iteration=m2.best_iteration)
        oof_aug_clean[va] = m2.predict(X.iloc[va], num_iteration=m2.best_iteration)
    for name, o in [("baseline, clean val", oof_clean), ("baseline, test-masked val", oof_masked),
                    ("mask-augmented, test-masked val", oof_aug_masked), ("mask-augmented, clean val", oof_aug_clean)]:
        print(f"  {name:34s} F2m nested {nested_f2(o):.4f}  AUC {roc_auc_score(y, o):.4f}")
    print(f"  ({time.time() - t0:.0f}s)")


def part_b():
    print("\n== B1. ordering / id artefacts")
    pid = train.player_id.str[1:].astype(int)
    print(f"  AUC(player_id number) {roc_auc_score(y, pid):.4f}; lag-1 autocorr of target in file order "
          f"{np.corrcoef(y[:-1], y[1:])[0, 1]:+.4f}")

    print("\n== B2. value-format artefacts per class (share of integer values, of values at the column min/max)")
    rows = []
    for c in FEATS:
        v = train[c]
        ok = v.notna()
        integ = (v[ok] % 1 == 0)
        at_min, at_max = (v[ok] == v[ok].min()), (v[ok] == v[ok].max())
        dec = v[ok].astype(str).str.split(".").str[-1].str.len()
        rows.append(dict(feat=c, int0=integ[y[ok] == 0].mean(), int1=integ[y[ok] == 1].mean(),
                         min0=at_min[y[ok] == 0].mean(), min1=at_min[y[ok] == 1].mean(),
                         max0=at_max[y[ok] == 0].mean(), max1=at_max[y[ok] == 1].mean(),
                         dec0=dec[y[ok] == 0].mean(), dec1=dec[y[ok] == 1].mean()))
    r = pd.DataFrame(rows).set_index("feat")
    r["gap"] = (r[["int0", "min0", "max0"]].to_numpy() - r[["int1", "min1", "max1"]].to_numpy()).__abs__().max(1)
    r["dec_gap"] = (r.dec0 - r.dec1).abs()
    print(r[(r.gap > 0.01) | (r.dec_gap > 0.3)].round(3).to_string() or "  none")

    print("\n== B3. OOF errors by reports bucket (base LGBM, full-OOF threshold)")
    thr, _ = metrics.best_fbeta_threshold(y, oof_base, 2)
    pred = (oof_base >= thr).astype(int)
    rep = train.reports_received
    bucket = pd.cut(rep, [-1, 3, 4, 15, 19, 1e9], labels=["<4", "4", "5-15", "16-19", "20+"])
    g = pd.DataFrame({"b": bucket.astype(str), "y": y, "p": oof_base, "pred": pred})
    out = g.groupby("b").apply(lambda d: pd.Series(dict(n=len(d), rate=d.y.mean(),
                                                        auc=roc_auc_score(d.y, d.p) if d.y.nunique() > 1 else np.nan,
                                                        err=(d.y != d.pred).mean())))
    print(out.round(3).to_string())
    print(f"  share of all errors inside 5-15: {((g.y != g.pred) & (g.b == '5-15')).sum() / (g.y != g.pred).sum():.3f}")

    print("\n== B4. confident errors vs correct rows (standardised mean difference; top 8 features)")
    sd = X.std()
    fn = (y == 1) & (oof_base < 0.15)
    tp = (y == 1) & (oof_base > 0.6)
    fp = (y == 0) & (oof_base > 0.75)
    tn = (y == 0) & (oof_base < 0.3)
    print(f"  confident FN {fn.sum()}, confident FP {fp.sum()}")
    for a, b, name in [(fn, tp, "conf. FN vs TP"), (fp, tn, "conf. FP vs TN")]:
        smd = ((X[a].mean() - X[b].mean()) / sd).sort_values(key=np.abs, ascending=False).head(8)
        print(f"  {name}: " + ", ".join(f"{k} {v:+.2f}" for k, v in smd.items()))

    print("\n== B5. pairwise Spearman correlation: class 0 vs class 1 (top 15 |diff|)")
    sub = X.sample(30000, random_state=42)
    ys = y[sub.index]
    c0 = sub[ys == 0].corr(method="spearman")
    c1 = sub[ys == 1].corr(method="spearman")
    dif = (c1 - c0).where(np.triu(np.ones(c0.shape, bool), 1)).stack()
    top = dif.reindex(dif.abs().sort_values(ascending=False).index).head(15)
    for (a, b), v in top.items():
        print(f"  {a:26s} x {b:26s} rho0 {c0.loc[a, b]:+.3f} rho1 {c1.loc[a, b]:+.3f} diff {v:+.3f}")

    print("\n== B6. does the residual hold signal? LGBM on OOF residual-sign within 5-15 bucket (AUC of a 2nd model)")
    in515 = (bucket == "5-15").to_numpy()
    res_y = ((y - oof_base) > 0).astype(int)[in515]
    Xs = X[in515].reset_index(drop=True)
    o = np.zeros(len(Xs))
    for trn, va in StratifiedKFold(5, shuffle=True, random_state=42).split(Xs, res_y):
        m = lgb.train({**P, "learning_rate": 0.05}, lgb.Dataset(Xs.iloc[trn], res_y[trn]), 400)
        o[va] = m.predict(Xs.iloc[va])
    print(f"  AUC predicting sign(y - oof) from features: {roc_auc_score(res_y, o):.4f} (0.5 = residual is noise)")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["A", "B"]
    if "A" in parts:
        part_a()
    if "B" in parts:
        part_b()
