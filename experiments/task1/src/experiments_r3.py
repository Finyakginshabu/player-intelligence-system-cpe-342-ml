"""Task 1 round 3 (autopilot 2026-11-09): ensembling for BINARY F2 (the LB metric, confirmed 2026-11-09).

Every single model plateaus at OOF binary F2 ~0.837, so the only lever left is variance reduction: multi-seed
averaging and a LightGBM + CatBoost blend, which also smooths the probabilities around the recall-hungry
threshold (~0.16). Scoring is nested: each fold is scored at the threshold (and blend weight) tuned on the other
four folds' OOF. Same 5 folds and features as rounds 1-2.
Run from repo root: python task1/src/experiments_r3.py [existing] [lgbm_seeds] [cat_seeds] [blend]
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import metrics, paths, utils

utils.seed_everything()
D = paths.task_data(1)
OUT = paths.outputs_dir(1)
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
SEEDS = [42, 43, 44, 45, 46]
LGB_P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, deterministic=True, force_row_wise=True,
             n_jobs=-1, verbose=-1)


def nested_f2(p):
    """Binary F2 per fold at the threshold tuned on the other 4 folds (exact search on their OOF)."""
    fs, thrs = [], []
    for k, (_, va) in enumerate(FOLDS):
        inner = np.concatenate([FOLDS[j][1] for j in range(5) if j != k])
        t, _ = metrics.best_fbeta_threshold(y[inner], p[inner], 2, average="binary")
        thrs.append(t)
        fs.append(metrics.f2(y[va], (p[va] >= t).astype(int), average="binary"))
    t_full, f_full = metrics.best_fbeta_threshold(y, p, 2, average="binary")
    return dict(f2_nested=float(np.mean(fs)), std=float(np.std(fs)), folds=np.round(fs, 4).tolist(),
                thr_folds=np.round(thrs, 3).tolist(), f2_full=f_full, thr_full=round(t_full, 4),
                auc=round(roc_auc_score(y, p), 4))


def show(name, p):
    r = nested_f2(p)
    print(f"{name:28s} nested {r['f2_nested']:.4f} ± {r['std']:.4f} | full-OOF {r['f2_full']:.4f} @ {r['thr_full']}"
          f" | AUC {r['auc']} | folds {r['folds']} | thr {r['thr_folds']}", flush=True)
    return r


def run_lgb_seeds(feats, name):
    oof, tst = np.zeros(len(y)), np.zeros(len(test))
    t0 = time.time()
    for s in SEEDS:
        o, t = np.zeros(len(y)), np.zeros(len(test))
        for trn, va in FOLDS:
            m = lgb.train({**LGB_P, "seed": s}, lgb.Dataset(train.iloc[trn][feats], y[trn]), 5000,
                          valid_sets=[lgb.Dataset(train.iloc[va][feats], y[va])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            o[va] = m.predict(train.iloc[va][feats], num_iteration=m.best_iteration)
            t += m.predict(test[feats], num_iteration=m.best_iteration) / len(FOLDS)
        show(f"  {name} seed {s}", o)
        oof += o / len(SEEDS)
        tst += t / len(SEEDS)
    print(f"  ({time.time() - t0:.0f}s)")
    utils.save_preds(1, name, oof, tst)
    return oof, tst


def run_cat_seeds(feats, name):
    from catboost import CatBoostClassifier
    oof, tst = np.zeros(len(y)), np.zeros(len(test))
    t0 = time.time()
    for s in SEEDS:
        o, t = np.zeros(len(y)), np.zeros(len(test))
        for trn, va in FOLDS:
            m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3, eval_metric="Logloss",
                                   random_seed=s, od_type="Iter", od_wait=200, verbose=0, thread_count=-1)
            m.fit(train.iloc[trn][feats], y[trn], eval_set=(train.iloc[va][feats], y[va]), use_best_model=True)
            o[va] = m.predict_proba(train.iloc[va][feats])[:, 1]
            t += m.predict_proba(test[feats])[:, 1] / len(FOLDS)
        show(f"  {name} seed {s}", o)
        oof += o / len(SEEDS)
        tst += t / len(SEEDS)
    print(f"  ({time.time() - t0:.0f}s)")
    utils.save_preds(1, name, oof, tst)
    return oof, tst


def nested_blend(a, b, grid=np.linspace(0, 1, 11)):
    """Weight w on a (1-w on b) and threshold both tuned on 4 folds, scored on the 5th."""
    fs, ws = [], []
    for k, (_, va) in enumerate(FOLDS):
        inner = np.concatenate([FOLDS[j][1] for j in range(5) if j != k])
        best = max(grid, key=lambda w: metrics.best_fbeta_threshold(y[inner], w * a[inner] + (1 - w) * b[inner], 2,
                                                                     average="binary")[1])
        p = best * a + (1 - best) * b
        t, _ = metrics.best_fbeta_threshold(y[inner], p[inner], 2, average="binary")
        fs.append(metrics.f2(y[va], (p[va] >= t).astype(int), average="binary"))
        ws.append(round(float(best), 2))
    return float(np.mean(fs)), float(np.std(fs)), np.round(fs, 4).tolist(), ws


if __name__ == "__main__":
    which = sys.argv[1:] or ["existing", "lgbm_seeds", "cat_seeds", "blend"]
    ref = np.load(OUT / "oof_lgbm.npy")
    print("reference:")
    show("lgbm (31 raw, seed 42)", ref)
    if "existing" in which:
        names = ["lgbm", "lgbm_nonoise", "lgbm_fe", "lgbm_qda_sig", "cat_raw"]
        P = {n: np.load(OUT / f"oof_{n}.npy") for n in names}
        for n in names[1:]:
            show(n, P[n])
        show("mean of 4 LightGBM variants", np.mean([P[n] for n in names[:4]], axis=0))
        show("mean of all 5 tree models", np.mean([P[n] for n in names], axis=0))
    if "lgbm_seeds" in which:
        run_lgb_seeds(SIGNAL, "lgbm_sig_5seed")
    if "cat_seeds" in which:
        run_cat_seeds(SIGNAL, "cat_sig_5seed")
    if "blend" in which:
        L, Lt = utils.load_preds(1, "lgbm_sig_5seed")
        C, Ct = utils.load_preds(1, "cat_sig_5seed")
        show("lgbm_sig_5seed", L)
        show("cat_sig_5seed", C)
        show("mean(lgbm_sig_5seed, cat_sig_5seed)", (L + C) / 2)
        m, s, f, w = nested_blend(L, C)
        print(f"nested weight blend lgbm/cat: {m:.4f} ± {s:.4f} folds {f} | w_lgbm per fold {w}")
