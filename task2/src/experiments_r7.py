"""Round 7 (2026-11-13): naive-Bayes polish on the refreshed task-2 data. No player_id anywhere.

E1  re-tune SmoothNB smoothing (n_bins, sigma, max_levels): settings were chosen on the old data.
E2  feature selection by drop-one ablation (NB adds noise for every column without signal).
E3  NB + LightGBM-composites blend, weight chosen by nested CV.
Every choice made on the 5 folds is re-checked by NESTED selection (pick on 4 folds' OOF, score on the 5th), because
picking the best of a grid on the same folds is optimistic. Same folds as the notebook (StratifiedKFold 5, seed 42).

Usage: python task2/src/experiments_r7.py [e1|e2|e3 ...]   (default: all three)
"""
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from common import paths, utils
from nb import SmoothNB

TASK = 2
DATA, OUT = paths.task_data(TASK), paths.outputs_dir(TASK)
train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
y = train["segment"].astype(int).values
CATS = [c for c in test.columns if not pd.api.types.is_numeric_dtype(test[c]) and c != "player_id"]
NOISE = ["random_metric_1", "random_metric_2", "random_metric_3"]
FEATURES = [c for c in test.columns if c not in ["player_id", *NOISE]]
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(train, y))
BASE = dict(n_bins=128, sigma=4.0, max_levels=64)


def f1(yy, p):
    return f1_score(yy, p.argmax(1), average="macro")


def cv_nb(feats, test_too=False, **kw):
    """OOF (and fold-averaged test) probabilities of SmoothNB on the given feature list."""
    kw = {**BASE, **kw}
    oof, tp = np.zeros((len(y), 4)), np.zeros((len(test), 4))
    for tr, va in FOLDS:
        m = SmoothNB([c for c in CATS if c in feats], 4, **kw).fit(train.iloc[tr][feats], y[tr])
        oof[va] = m.predict_proba(train.iloc[va][feats])
        if test_too:
            tp += m.predict_proba(test[feats]) / len(FOLDS)
    return oof, tp


def fold_scores(oof):
    return np.array([f1(y[va], oof[va]) for _, va in FOLDS])


def nested(cands):
    """cands: {name: oof}. For each outer fold pick the candidate best on the other 4 folds; return the outer scores."""
    names = list(cands)
    per = np.array([fold_scores(cands[n]) for n in names])  # (cands, folds)
    out, picks = [], []
    for k in range(len(FOLDS)):
        inner = np.delete(per, k, axis=1).mean(1)
        j = int(inner.argmax())
        out.append(per[j, k])
        picks.append(names[j])
    return np.array(out), picks


def report(tag, s, ref=None, t0=None):
    extra = "" if ref is None else f"  Δ {s.mean() - ref.mean():+.4f} (better in {(s > ref).sum()}/5 folds)"
    rt = "" if t0 is None else f"  {time.time() - t0:.0f}s"
    print(f"{tag:58s} {s.mean():.4f} ± {s.std():.4f}{extra}{rt}", flush=True)


t_all = time.time()
oof_base, _ = cv_nb(FEATURES)
s_base = fold_scores(oof_base)
report("baseline SmoothNB (128 bins, σ 4, max_levels 64)", s_base)
todo = sys.argv[1:] or ["e1", "e2", "e3"]

if "e1" in todo:
    t0 = time.time()
    grid = {}
    for nbins, sig, ml in product([64, 128, 256], [2.0, 4.0, 8.0], [16, 64]):
        oof, _ = cv_nb(FEATURES, n_bins=nbins, sigma=sig, max_levels=ml)
        grid[(nbins, sig, ml)] = oof
        report(f"E1 n_bins {nbins:3d}  σ {sig:3.0f}  max_levels {ml:2d}", fold_scores(oof), s_base)
    best = max(grid, key=lambda k: fold_scores(grid[k]).mean())
    s_nest, picks = nested(grid)
    report(f"E1 best on the same folds {best}", fold_scores(grid[best]), s_base)
    report("E1 NESTED (setting picked on 4 folds, scored on the 5th)", s_nest, s_base, t0)
    print("   nested picks:", picks)

if "e2" in todo:
    t0 = time.time()
    drop = {}
    for c in FEATURES:
        oof, _ = cv_nb([f for f in FEATURES if f != c])
        drop[c] = oof
    gain = {c: fold_scores(o).mean() - s_base.mean() for c, o in drop.items()}
    for c, g in sorted(gain.items(), key=lambda t: -t[1])[:8]:
        print(f"   drop {c:32s} Δ {g:+.4f}")
    harmful = [c for c, g in gain.items() if g > 0]
    feats2 = [f for f in FEATURES if f not in harmful]
    oof2, _ = cv_nb(feats2)
    report(f"E2 drop all {len(harmful)} columns whose removal helps", fold_scores(oof2), s_base)
    # nested: choose the drop set on 4 folds (drop-one gains measured on those folds), score on the 5th
    per_drop = {c: fold_scores(o) for c, o in drop.items()}
    outs = []
    for k, (tr, va) in enumerate(FOLDS):
        h = [c for c in FEATURES if np.delete(per_drop[c] - s_base, k).mean() > 0]
        m = SmoothNB([c for c in CATS if c not in h], 4, **BASE)
        trk = np.concatenate([FOLDS[j][1] for j in range(5) if j != k])  # = tr
        fk = [f for f in FEATURES if f not in h]
        m.fit(train.iloc[trk][fk], y[trk])
        outs.append(f1(y[va], m.predict_proba(train.iloc[va][fk])))
    report("E2 NESTED (drop set chosen on 4 folds, scored on the 5th)", np.array(outs), s_base, t0)

if "e3" in todo:
    t0 = time.time()
    oof_lgb, _ = utils.load_preds(TASK, "lgbm_comp")
    cands = {w: (1 - w) * oof_base + w * oof_lgb for w in np.round(np.arange(0, 0.55, 0.05), 2)}
    for w in (0.1, 0.2, 0.3, 0.5):
        report(f"E3 blend NB + {w:.2f} · LightGBM-composites", fold_scores(cands[w]), s_base)
    s_nest, picks = nested(cands)
    report("E3 NESTED (weight picked on 4 folds, scored on the 5th)", s_nest, s_base, t0)
    print("   nested weights:", picks)

print(f"total {time.time() - t_all:.0f}s")
