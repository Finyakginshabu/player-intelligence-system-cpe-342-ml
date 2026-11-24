"""Task 5 round 1 analysis (2026-11-08): check anomaly-type hypotheses against the task-5 variants ALREADY scored on
the public LB. No new submission is made; this only reuses scores we already have.

Assumptions: equal task weights, t4 = 0, tasks 1-3 at CV, so implied t5 = 5*LB - 2.3131 (submissions/log.md).
Prints: (1) type sizes and pairwise overlap, (2) per-type recall of each scored variant, (3) the fit of every
subset of the single-feature types, (4) the hidden-group size X needed to match the LB, (5) projected t5 of the
typesfirst files. Run from the repo root after task5_nb.py (needs test_iso.npy and pred_typesfirst_*.csv).
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import metrics, paths  # noqa: E402

N = 25_889
df = pd.read_csv(paths.task_data(5) / "test.csv")
OUT = paths.outputs_dir(5)
SUB = ROOT / "submissions"


def per(f):
    return df[[f"{f}_{i}" for i in range(1, 5)]].to_numpy(float)


def oor(f):
    m = per(f)
    return (m[:, 3] > np.nanmax(m[:, :3])) | (m[:, 3] < np.nanmin(m[:, :3]))


def low_any(f, c):
    return (per(f) < c).any(axis=1)


# Same definitions as task5_nb.py section 5.
T = pd.DataFrame({
    "A_cheat": oor("skill_rating") | oor("skill_trajectory") | oor("accuracy_mean") | oor("kda_mean")
               | oor("friend_removes") | oor("reaction_time_mean"),
    "B_takeover": oor("currency_transfers") | oor("play_pattern_consistency"),
    "C_purchase": oor("purchase_frequency") | oor("purchase_amount_mean"),
    "bot": low_any("ip_hash_entropy", 0.25) | low_any("login_time_variance", 0.9) | low_any("session_duration_std", 6),
    "low_activity": low_any("activity_level", 0.32),
    "low_comm": low_any("communication_frequency", 0.15),
    "low_stability": (df["behavior_stability"] < 0.5).to_numpy(),
}).astype(int)
T["loc_only"] = (oor("location_changes") & (T.sum(1) == 0)).astype(int)
U = (T.sum(1) > 0).to_numpy()
ov = T.T.dot(T).to_numpy()
print("(1) type sizes", T.sum().to_dict(), "| union", U.sum(), "| max pairwise overlap", (ov - np.diag(np.diag(ov))).max())

# Scored variants (task5 column, test.csv row order).
LB = {"rate06": 0.59566, "rate10": 0.60044, "rate15": 0.60499, "rule15": 0.59157}
iso = np.load(OUT / "test_iso.npy")
r = pd.Series(iso).rank(method="first").to_numpy()
preds = {"rate06": pd.read_csv(SUB / "sub_v4_task5_rate06.csv")["task5"].to_numpy().astype(bool),
         "rate10": r > N - 2589,  # v1 = IF top 10%
         "rate15": pd.read_csv(SUB / "sub_v4_task5_rate15.csv")["task5"].to_numpy().astype(bool),
         "rule15": pd.read_csv(SUB / "sub_v5_task5_rule15.csv")["task5"].to_numpy().astype(bool)}
lb = np.array(list(LB.values()))
t5 = 5 * lb - 2.3131
print("implied t5 (equal weights):", dict(zip(LB, t5.round(3))))
print("\n(2) per-type recall\n", pd.DataFrame({k: [p[T[c] == 1].mean() for c in T] for k, p in preds.items()},
                                              index=T.columns).round(3))

print("\n(3) subsets of single-feature types (A/B/C always in): F3 vs pseudo-label, free-weight fit")
rows = []
opt = ["bot", "low_activity", "low_comm", "low_stability"]
for n in range(5):
    for sub in itertools.combinations(opt, n):
        lab = (T[["A_cheat", "B_takeover", "C_purchase", "loc_only", *sub]].sum(1) > 0).astype(int).to_numpy()
        s = np.array([metrics.f3(lab, p.astype(int)) for p in preds.values()])
        X = np.c_[np.ones(4), s]
        coef, *_ = np.linalg.lstsq(X, lb, rcond=None)
        rows.append((",".join(sub) or "-", lab.sum(), *s.round(3), round(coef[1], 3),
                     round(float(np.sqrt(np.mean((X @ coef - lb) ** 2))), 5)))
print(pd.DataFrame(rows, columns=["extra", "n", *LB, "w5_fit", "rmse"]).sort_values("rmse").to_string(index=False))
print("-> every subset predicts F3 FALLING with flag rate; the LB rose. Union alone is incomplete.")


def f3c(tp, k, A):
    fp, fn = k - tp, A - tp
    tn = N - A - fp
    return (10 * tp / (10 * tp + 9 * fn + fp) + 10 * tn / (10 * tn + 9 * fp + fn)) / 2


print("\n(4) hidden group X: anomalies of X caught among each variant's non-union flags to match implied t5")
fits = {}
for Xn in (0, 300, 600, 900, 1200, 1800, 2400, 3000):
    A, line, xs = U.sum() + Xn, [], []
    for (k, p), target in zip(preds.items(), t5):
        k1, tpu = p.sum(), (p & U).sum()
        cand = np.arange(0, min(k1 - tpu, Xn) + 1)
        sc = np.array([f3c(tpu + x, k1, A) for x in cand])
        i = int(np.argmin(np.abs(sc - target)))
        xs.append(cand[i])
        line.append(f"{k}: x={cand[i]}/{k1 - tpu} err={sc[i] - target:+.3f}")
    fits[Xn] = xs
    print(f"X={Xn:5d}: " + " | ".join(line))

print("\n(5) projected t5 of typesfirst files (X density in IF's non-union picks interpolated from (4))")
nu = [((r > N - k) & ~U).sum() for k in (1553, 2589, 3883)]
for name in ("rate06", "rate15"):
    p = pd.read_csv(OUT / f"pred_typesfirst_{name}.csv")["prediction"].to_numpy().astype(bool)
    k, tpu, keep = p.sum(), (p & U).sum(), (p & ~U).sum()
    for Xn in (900, 1200, 1800, 2400):
        x6, x10, x15 = fits[Xn][:3]
        xc = np.interp(keep, [0, *nu], [0, x6, x10, x15])
        print(f"typesfirst {name} X={Xn}: t5 ~ {f3c(tpu + xc, k, U.sum() + Xn):.3f}")
