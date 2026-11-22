"""Task 3 round 8c (2026-11-19): the round-8b model (identity-filled features, stage 2 on the ratio to history) averaged
over 3 LightGBM seeds, as the final task-3 model. Stage-1 probabilities are averaged; stage-2 quantiles are averaged in
log space (as round 5); then the mixture median. Writes outputs/oof|test_twostage_r8_ratio_ms.npy and final candidate
outputs/pred_twostage_r8_ratio_ms.csv (id = player_id).

Usage: python task3/src/experiments_r8c.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_r8 as r8  # noqa: E402
import experiments_r8b as r8b  # noqa: E402
from experiments_r8 import metrics, mixture_median, paths, utils  # noqa: E402

SEEDS = [42, 7, 123]

if __name__ == "__main__":
    data = paths.task_data(3)
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    y = train["spending_30d"].values.astype(float)
    feats = [c for c in test.columns if c != "player_id"]
    bins = np.zeros(len(y), dtype=int)
    bins[y > 0] = 1 + pd.qcut(y[y > 0], 5, labels=False)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED).split(train, bins))
    ftr, _ = r8.fill_identities(train)
    fte, _ = r8.fill_identities(test)
    ratio = float((ftr["prev_month_spending"] / ftr["historical_spending"]).median())
    lr, lr_test = r8b.reference(ftr, ratio), r8b.reference(fte, ratio)
    parts = []
    for s in SEEDS:
        t0 = time.time()
        oof, pred, pr = r8b.twostage_ratio(ftr[feats], y, fte[feats], lr, lr_test, folds, seed=s)
        parts.append(pr)
        print(f"seed {s}: NMAE {metrics.nmae(y, oof):.5f}  ({time.time() - t0:.0f}s)", flush=True)
    p = np.mean([x[0] for x in parts], 0)
    Q = np.expm1(np.mean([np.log1p(x[1]) for x in parts], 0))
    tp = np.mean([x[2] for x in parts], 0)
    TQ = np.expm1(np.mean([np.log1p(x[3]) for x in parts], 0))
    oof, pred = mixture_median(p, Q), mixture_median(tp, TQ)
    f = np.array([metrics.nmae(y[va], oof[va]) for _, va in folds])
    print(f"3-seed filled+ratio NMAE {metrics.nmae(y, oof):.5f} +- {f.std():.4f}  folds {np.round(f, 4).tolist()}")
    utils.save_preds(3, "twostage_r8_ratio_ms", oof, pred)
    pd.DataFrame({"id": test["player_id"], "prediction": np.round(pred, 2)}).to_csv(
        paths.outputs_dir(3) / "pred_twostage_r8_ratio_ms.csv", index=False)
    print(f"test: zero share {np.mean(pred == 0):.3f} (OOF {np.mean(oof == 0):.3f}) | median {np.median(pred):.1f} "
          f"(OOF {np.median(oof):.1f}) | max {pred.max():.0f}")
