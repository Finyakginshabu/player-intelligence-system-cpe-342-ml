"""Task 1 round 2 (2026-11-08): class-conditional Gaussian (QDA-style) log-likelihood-ratio features.

Finding (analysis_r2.py B5): in legit players the features are correlated (reports <-> level speed / age / friends
~0.27, K/D & headshot <-> age / friends ~0.17); in cheaters the same correlations are ~0. A cheater is a row whose
features do not fit together, which a one-feature-at-a-time tree only sees weakly. Feature = log N(z | class 1)
- log N(z | class 0) on rank-gaussianised features, plus each class's Mahalanobis distance.
Everything (quantile maps, means, covariances) is fit on the training fold only; the labels used are training-fold
labels, exactly like a stacked model.
Run from repo root: python task1/src/experiments_r2.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils

utils.seed_everything()
D = paths.task_data(1)
tr_raw = pd.read_csv(D / "train.csv")
test = pd.read_csv(D / "test.csv")
train = tr_raw[tr_raw.is_cheater.notna()].reset_index(drop=True)
y = train.is_cheater.astype(int).to_numpy()
RAW = [c for c in test.columns if c != "player_id"]
NOISE = ["sessions_per_day", "first_blood_rate", "survival_time_avg", "damage_per_round", "device_changes_count",
         "clutch_success_rate", "map_knowledge", "night_play_ratio", "avg_session_length_min", "utility_usage_rate",
         "team_play_score", "buy_decision_score", "communication_rate", "weapon_switch_speed",
         "movement_pattern_score", "input_consistency_score"]
SIGNAL = [c for c in RAW if c not in NOISE]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y))
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, deterministic=True, force_row_wise=True,
         n_jobs=-1, verbose=-1)


class GaussQDA:
    """Rank-gaussianise each column (train-fold empirical CDF), then class-conditional Gaussians.
    NaN -> conditional mean given the observed columns under the pooled Gaussian (fit on the training fold)."""

    def __init__(self, cols, shrink=0.05):
        self.cols, self.shrink = cols, shrink

    def _gauss(self, df):
        Z = np.empty((len(df), len(self.cols)))
        for j, c in enumerate(self.cols):
            s = self.sorted_[j]
            v = df[c].to_numpy(float)
            u = (np.searchsorted(s, v, side="left") + np.searchsorted(s, v, side="right")) / (2 * len(s) + 1)
            Z[:, j] = np.where(np.isnan(v), np.nan, norm.ppf(np.clip(u, 1e-4, 1 - 1e-4)))
        return Z

    def _impute(self, Z):
        Z = Z.copy()
        S = self.pooled_cov_
        miss = np.isnan(Z)
        for pattern in np.unique(miss, axis=0):
            if not pattern.any():
                continue
            rows = (miss == pattern).all(axis=1)
            m, o = pattern, ~pattern
            if not o.any():
                Z[np.ix_(rows, m)] = 0.0
                continue
            coef = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
            Z[np.ix_(rows, m)] = Z[np.ix_(rows, o)] @ coef.T
        return Z

    def _cov(self, Z):
        C = np.cov(Z, rowvar=False)
        return (1 - self.shrink) * C + self.shrink * np.diag(np.diag(C))

    def fit(self, df, yy):
        self.sorted_ = [np.sort(df[c].dropna().to_numpy(float)) for c in self.cols]
        Z = self._gauss(df)
        Zc = np.where(np.isnan(Z), 0.0, Z)  # crude fill only to estimate the pooled covariance for imputation
        self.pooled_cov_ = self._cov(Zc)
        Z = self._impute(Z)
        self.params_ = []
        for k in (0, 1):
            mu, C = Z[yy == k].mean(0), self._cov(Z[yy == k])
            Ci = np.linalg.inv(C)
            self.params_.append((mu, Ci, np.linalg.slogdet(C)[1]))
        return self

    def transform(self, df):
        Z = self._impute(self._gauss(df))
        out = {}
        ll = []
        for k, (mu, Ci, logdet) in enumerate(self.params_):
            d = Z - mu
            maha = np.einsum("ij,jk,ik->i", d, Ci, d)
            out[f"maha{k}"] = maha
            ll.append(-0.5 * (maha + logdet))
        out["llr"] = ll[1] - ll[0]
        return pd.DataFrame(out, index=df.index)


def nested(oof):
    per, pred = [], np.zeros(len(y), int)
    for trn, va in FOLDS:
        t, _ = metrics.best_fbeta_threshold(y[trn], oof[trn], 2)
        pred[va] = oof[va] >= t
        per.append(metrics.f2(y[va], pred[va]))
    return dict(f2_mean=float(np.mean(per)), f2_std=float(np.std(per)), auc=float(roc_auc_score(y, oof)),
                folds=[round(f, 4) for f in per])


def run(name, base_cols, qda_cols, keep=("llr", "maha0", "maha1"), save=True):
    t0 = time.time()
    oof, tst, oof_llr = np.zeros(len(y)), np.zeros(len(test)), np.zeros(len(y))
    for trn, va in FOLDS:
        Xtr, Xva, Xte = train.iloc[trn][base_cols], train.iloc[va][base_cols], test[base_cols]
        if qda_cols:
            q = GaussQDA(qda_cols).fit(train.iloc[trn], y[trn])
            # training-fold QDA features from an inner 5-fold, so the tree never sees in-sample LLR
            inner = np.zeros((len(trn), len(keep)))
            for itr, iva in StratifiedKFold(5, shuffle=True, random_state=7).split(trn, y[trn]):
                qi = GaussQDA(qda_cols).fit(train.iloc[trn[itr]], y[trn[itr]])
                inner[iva] = qi.transform(train.iloc[trn[iva]])[list(keep)].to_numpy()
            Xtr = Xtr.assign(**{k: inner[:, j] for j, k in enumerate(keep)})
            fva, fte = q.transform(train.iloc[va])[list(keep)], q.transform(test)[list(keep)]
            Xva, Xte = Xva.assign(**fva), Xte.assign(**fte)
            oof_llr[va] = fva["llr"].to_numpy() if "llr" in keep else 0
        m = lgb.train(P, lgb.Dataset(Xtr, y[trn]), 5000, valid_sets=[lgb.Dataset(Xva, y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        tst += m.predict(Xte, num_iteration=m.best_iteration) / len(FOLDS)
    s = nested(oof)
    s["runtime_s"] = round(time.time() - t0)
    if qda_cols and "llr" in keep:
        s["auc_llr_alone"] = round(float(roc_auc_score(y, oof_llr)), 4)
    if save:
        utils.save_preds(1, name, oof, tst)
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    return oof, tst


if __name__ == "__main__":
    print("reference lgbm (31 raw)", json.dumps(nested(np.load(paths.outputs_dir(1) / "oof_lgbm.npy"))))
    which = sys.argv[1:] or ["sig", "all"]
    if "sig" in which:
        run("lgbm_qda_sig", RAW, SIGNAL)
    if "all" in which:
        run("lgbm_qda_all", RAW, RAW)
