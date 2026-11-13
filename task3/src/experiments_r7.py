"""Task 3 round 7 (2026-11-13): last checks on the two-stage conditional median (current best: twostage_ms_cat, 0.78843).

E1 (bound only, no model): rows with a missing `segment` (5.7%) carry 6.5% of the error at relative error 0.279 vs 0.268
    for the rest; closing that gap completely would add at most +0.0004 NMAE (< fold std 0.0036) -> not built.
E2  Optuna on the stage-1 gate (LightGBM binary, objective = OOF log-loss on the same folds, 40 trials), then
    recombined with the saved 3-seed quantiles; NMAE compared per fold with the current model.
E3  Per-segment gate thresholds (the 0.5 cut is optimal only if the gate is calibrated in every segment), chosen
    nested: thresholds fit on 4 folds' OOF, scored on the 5th.
Run from repo root: python task3/src/experiments_r7.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_r5 as r5  # noqa: E402  (module level only loads data, folds and helpers)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from common import metrics, paths, utils  # noqa: E402

y, nz, X, X_test, FOLDS = r5.y, r5.nz, r5.X, r5.X_test, r5.FOLDS
OUT = paths.outputs_dir(3)
parts, tparts = np.load(OUT / "oof_twostage_ms_parts.npy"), np.load(OUT / "test_twostage_ms_parts.npy")
gate_cur, Q, tgate_cur, TQ = parts[:, 0], parts[:, 2:], tparts[:, 0], tparts[:, 2:]
cur = r5.combine(gate_cur, Q)
f_cur = np.array(r5.fold_scores(cur))
print(f"current twostage_ms_cat: {f_cur.mean():.5f} ± {f_cur.std():.4f} | gate logloss {log_loss(nz, gate_cur):.4f}")


def report(tag, f, t0):
    d = f - f_cur
    print(f"{tag:50s} {f.mean():.5f} ± {f.std():.4f}  Δ {d.mean():+.5f}  folds up {(d > 0).sum()}/5  {time.time() - t0:.0f}s",
          flush=True)


# ---------------- E2: Optuna on the gate ----------------
t0 = time.time()


def gate_oof(params, test_too=False):
    o, t = np.zeros(len(y)), np.zeros(len(X_test))
    for tr, va in FOLDS:
        dtr = lgb.Dataset(X.iloc[tr], nz[tr].astype(int))
        m = lgb.train(params, dtr, 5000, valid_sets=[lgb.Dataset(X.iloc[va], nz[va].astype(int), reference=dtr)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        o[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        if test_too:
            t += m.predict(X_test, num_iteration=m.best_iteration) / len(FOLDS)
    return o, t


def objective(trial):
    p = {**r5.BASE, "objective": "binary", "seed": 42,
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
         "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
         "min_child_samples": trial.suggest_int("min_child_samples", 20, 800, log=True),
         "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
         "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 50, log=True)}
    return log_loss(nz, gate_oof(p)[0])


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40)
best = {**r5.BASE, "objective": "binary", "seed": 42, **study.best_params}
print(f"E2 best gate logloss {study.best_value:.4f} (current {log_loss(nz, gate_cur):.4f}) params {study.best_params}")
g_opt, tg_opt = gate_oof(best, test_too=True)
report("E2 tuned gate alone + 3-seed quantiles", np.array(r5.fold_scores(r5.combine(g_opt, Q))), t0)
g_mix = (gate_cur + g_opt) / 2
report("E2 (current gate + tuned gate)/2 + quantiles", np.array(r5.fold_scores(r5.combine(g_mix, Q))), t0)
utils.save_preds(3, "twostage_tunedgate", r5.combine(g_mix, Q), r5.combine((tgate_cur + tg_opt) / 2, TQ))

# ---------------- E3: per-segment gate thresholds (nested) ----------------
t0 = time.time()
seg = r5.train["segment"].fillna(-1).astype(int).values
GRID = np.round(np.arange(0.40, 0.61, 0.01), 2)


def apply(gate, thr_by_seg):
    out = np.zeros(len(gate))
    for s, thr in thr_by_seg.items():
        m = seg == s
        out[m] = r5.combine(gate[m], Q[m], gate=thr)
    return out


segs = sorted(set(seg))
outer = []
for k, (_, va) in enumerate(FOLDS):
    inner = np.concatenate([FOLDS[j][1] for j in range(5) if j != k])
    thr = {}
    for s in segs:  # NMAE is a sum, so each segment's threshold can be chosen separately (min Σ|y−ŷ| on its rows)
        m = inner[seg[inner] == s]
        errs = [np.abs(y[m] - r5.combine(gate_cur[m], Q[m], gate=g)).sum() for g in GRID]
        thr[s] = GRID[int(np.argmin(errs))]
    o = apply(gate_cur, thr)
    outer.append(metrics.nmae(y[va], o[va]))
    print(f"   fold {k} thresholds {thr}")
report("E3 NESTED per-segment gate thresholds", np.array(outer), t0)
