# Autopilot run — 2026-11-09 (5 rounds, no upload)

| Round | Task | Experiment | CV before → after | Kept? |
|---|---|---|---|---|
| 1 | 5 | Structure hunt outside the 7 known types → 8th type "erratic performance" (accuracy_std × kda_std × performance_stability, found by cross-base tail co-occurrence, z = 20.9; 2-component EM, ≈ 229 rows) → **typesfirst8** | no CV (unsupervised); LB-implied t5 0.816 → projected ≈ 0.90 | ✅ → `task5/outputs/final_pred.csv`, pending LB |
| 2 | 2 | Naive Bayes with smoothed per-class densities, NaN skipped (features are conditionally independent given the segment) | 0.7525 ± 0.0022 → **0.7553 ± 0.0036**, every fold up | ✅ → `task2/outputs/final_pred.csv` |
| 3 | 1 | LightGBM, 15 signal features, 5 seeds (binary F2, nested threshold) | 0.8368 ± 0.0010 → **0.8379 ± 0.0014**, every fold up | ✅ (marginal) → `task1/outputs/final_pred.csv`; notebook now uses binary F2 |
| – | 1, 2 | /ensemble (hill climbing, mean, LR stacker; nested) | no blend beats the best single model | – |
| – | 1, 2, 5 | /audit | no leakage / rule violation; 4 minor findings (see log.md) | – |
| 4 | 3 | Two-stage × 3 seeds + CatBoost gate (+ interpolation variants) | 0.7878 ± 0.0037 → 0.7884 ± 0.0036 | ❌ (+0.0006 < std) |
| 5 | 1 | Threshold tuned under test-like missingness | 0.8379 → 0.8371 (re-masked) | ❌ (threshold robust: 0.159 → 0.161) |

`submission.csv` = `submissions/sub_v12_task5_typesfirst8.csv`: v9 (LB 0.67942) with only task 5 replaced.
