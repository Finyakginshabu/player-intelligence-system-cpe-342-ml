"""Task 1 experiment runner. Usage (from repo root): python task1/src/experiments_r1.py <stage> [args]
stage: e1 (FE pack), e3 (noise ablation on best set), e2 (catboost + blend)."""
import json
import sys
import time

sys.path.insert(0, ".")
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
y = train.is_cheater.astype(int)
RAW = [c for c in test.columns if c != "player_id"]
NOISE = ["sessions_per_day", "first_blood_rate", "survival_time_avg", "damage_per_round", "device_changes_count",
         "clutch_success_rate", "map_knowledge", "night_play_ratio", "avg_session_length_min", "utility_usage_rate",
         "team_play_score", "buy_decision_score", "communication_rate", "weapon_switch_speed",
         "movement_pattern_score", "input_consistency_score"]
SKILL_POS = ["kill_death_ratio", "headshot_percentage", "accuracy_score", "crosshair_placement",
             "game_sense_score", "spray_control_score", "aiming_smoothness"]
FE_ALL = ["reports_bucket", "reports_eq_4", "age_gt_1100", "friends_le_10", "friends_gt_300", "skill_score", "skill_n",
          "skill_per_level", "kd_per_level", "level_per_age", "friends_per_age", "lps_ratio"]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y))
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, deterministic=True, force_row_wise=True,
         n_jobs=-1, verbose=-1)


def add_features(df, ref):
    """ref = training-fold rows; used only for percentile ranks (no target)."""
    out = df.copy()
    r = out.reports_received
    out["reports_bucket"] = np.select([r == 4, r <= 15, r <= 19, r >= 20], [0, 1, 2, 3], default=-1).astype(float)
    out.loc[r.isna(), "reports_bucket"] = np.nan
    out["reports_eq_4"] = (r == 4).astype(float).where(r.notna())
    out["age_gt_1100"] = (out.account_age_days > 1100).astype(float).where(out.account_age_days.notna())
    out["friends_le_10"] = (out.friend_network_size <= 10).astype(float).where(out.friend_network_size.notna())
    out["friends_gt_300"] = (out.friend_network_size > 300).astype(float).where(out.friend_network_size.notna())
    ranks = []
    for c in SKILL_POS:
        s = np.sort(ref[c].dropna().values)
        ranks.append(pd.Series(np.searchsorted(s, out[c].values, side="right") / len(s),
                               index=out.index).where(out[c].notna()))
    rk = pd.concat(ranks, axis=1)
    out["skill_score"] = rk.mean(axis=1)
    out["skill_n"] = rk.notna().sum(axis=1)
    out["skill_per_level"] = out.skill_score / out.level
    out["kd_per_level"] = out.kill_death_ratio / out.level
    out["level_per_age"] = out.level / out.account_age_days
    out["friends_per_age"] = out.friend_network_size / out.account_age_days
    out["lps_ratio"] = np.log(out.level_progression_speed / out.level_per_age)
    return out


def score(oof):
    """Per-fold macro F2, each fold at the threshold tuned on the other 4 folds' OOF."""
    per_fold, pred = [], np.zeros(len(y), int)
    for trn, va in FOLDS:
        t, _ = metrics.best_fbeta_threshold(y.iloc[trn], oof[trn], 2)
        pred[va] = oof[va] >= t
        per_fold.append(metrics.f2(y.iloc[va], pred[va]))
    thr, full = metrics.best_fbeta_threshold(y, oof, 2)
    return dict(f2_mean=float(np.mean(per_fold)), f2_std=float(np.std(per_fold)), f2_pooled=metrics.f2(y, pred),
                f2_fullthr=full, thr=thr, auc=roc_auc_score(y, oof), folds=[round(f, 4) for f in per_fold])


def run_lgb(feats, fe):
    oof, tst = np.zeros(len(y)), np.zeros(len(test))
    for trn, va in FOLDS:
        Xtr, Xva, Xte = train.iloc[trn], train.iloc[va], test
        if fe:
            Xtr, Xva, Xte = add_features(Xtr, Xtr), add_features(Xva, Xtr), add_features(test, Xtr)
        m = lgb.train(P, lgb.Dataset(Xtr[feats], y.iloc[trn]), 5000,
                      valid_sets=[lgb.Dataset(Xva[feats], y.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(Xva[feats], num_iteration=m.best_iteration)
        tst += m.predict(Xte[feats], num_iteration=m.best_iteration) / len(FOLDS)
    return oof, tst


def run_cat(feats, fe):
    from catboost import CatBoostClassifier
    oof, tst = np.zeros(len(y)), np.zeros(len(test))
    for trn, va in FOLDS:
        Xtr, Xva, Xte = train.iloc[trn], train.iloc[va], test
        if fe:
            Xtr, Xva, Xte = add_features(Xtr, Xtr), add_features(Xva, Xtr), add_features(test, Xtr)
        m = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3, eval_metric="Logloss",
                               random_seed=42, od_type="Iter", od_wait=200, verbose=0, thread_count=-1)
        m.fit(Xtr[feats], y.iloc[trn], eval_set=(Xva[feats], y.iloc[va]), use_best_model=True)
        oof[va] = m.predict_proba(Xva[feats])[:, 1]
        tst += m.predict_proba(Xte[feats])[:, 1] / len(FOLDS)
    return oof, tst


def report(name, oof, tst, t0, save=True):
    s = score(oof)
    s["runtime_s"] = round(time.time() - t0, 1)
    if save:
        utils.save_preds(1, name, oof, tst)
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    return s


stage = sys.argv[1]
base_oof = np.load(OUT / "oof_lgbm.npy")
print("baseline lgbm", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in score(base_oof).items()}))

if stage == "e1":
    t0 = time.time(); o, t = run_lgb(RAW + FE_ALL, True); report("lgbm_fe", o, t, t0)
elif stage == "e3":
    keep_fe = sys.argv[2] == "fe"
    feats = [c for c in RAW if c not in NOISE] + (FE_ALL if keep_fe else [])
    t0 = time.time(); o, t = run_lgb(feats, keep_fe); report("lgbm_fe_nonoise" if keep_fe else "lgbm_nonoise", o, t, t0)
elif stage == "e2":
    base_name = sys.argv[2]  # name of the kept LGBM preds, e.g. lgbm_fe
    feats_key = sys.argv[3]  # raw | fe | fe_nonoise | nonoise
    feats = {"raw": RAW, "fe": RAW + FE_ALL, "nonoise": [c for c in RAW if c not in NOISE],
             "fe_nonoise": [c for c in RAW if c not in NOISE] + FE_ALL}[feats_key]
    t0 = time.time(); o, t = run_cat(feats, "fe" in feats_key); report(f"cat_{feats_key}", o, t, t0)
    lo, lt = utils.load_preds(1, base_name)
    for w in (0.3, 0.5, 0.7):
        report(f"blend_{base_name}_cat_w{w}", w * o + (1 - w) * lo, w * t + (1 - w) * lt, t0, save=False)
    # rank average (scale-free)
    from scipy.stats import rankdata
    r = lambda a: rankdata(a) / len(a)
    report(f"rankblend_{base_name}_cat", (r(o) + r(lo)) / 2, (r(t) + r(lt)) / 2, t0, save=False)
