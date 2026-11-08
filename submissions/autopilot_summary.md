# Autopilot run — 2026-11-08 (3 rounds; tasks 1/2/3/5 only, task 4 skipped, no upload, no submission build)

| Round | Task | Experiment | CV before → after | Kept? |
|---|---|---|---|---|
| 1 | 5 | **typesfirst**: flag 7 detached-cluster anomaly types first (A cheat, B takeover, C purchase fraud, bot, low activity, low comm, low stability), IsolationForest rank inside tiers and for the rest | no CV (unsupervised). LB-implied t5 0.712 (IF 15%) → projected 0.78–0.82 | pending LB; files prepared |
| 2 | 2 | Latent-axis composites (engagement / spending mean z-scores, tier ordinals, ratios) | 0.7496 ± 0.0027 → **0.7525 ± 0.0022** | ✅ every fold up → `task2/outputs/final_pred.csv` |
| 3 | 3 | Two-stage conditional median (P(spend>0) classifier + spender quantile models, median of the mixture) | 0.7840 ± 0.0040 → **0.7878 ± 0.0037** | ✅ every fold up → `task3/outputs/final_pred.csv` |

- `/ensemble` + `/audit` weren't triggered: only 2 CV-kept improvements, and the trigger is every 3.
- No test rows were used for supervised training. Task 5 fits are unsupervised, on its own features, as allowed.

## Prepared files (nothing uploaded)
| Path | Content | Use |
|---|---|---|
| `task5/outputs/pred_typesfirst_rate15.csv` | id, prediction: task 5, 3,883 flags (15%) | Main bet. Merge into `submissions/sub_v4_task5_rate15.csv`, replacing **only** the task5 column (keep tasks 1–4 = v1) |
| `task5/outputs/pred_typesfirst_rate06.csv` | id, prediction: task 5, 1,553 flags (6%) | Precision test: are the single-feature types anomalies? Same merge |
| `task5/outputs/test_typesfirst.npy` | typesfirst score (tier + IF rank) | Re-cut any other flag rate |
| `task2/outputs/final_pred.csv`, `oof/test_lgbm_comp.npy` | New task-2 model | Separate submission (one task changed vs v1) |
| `task3/outputs/final_pred.csv`, `oof/test_twostage.npy`, `*_twostage_parts.npy` | New task-3 model | Separate submission |

Note: `task5/outputs/final_pred.csv` is still IF top 10% (v1), not the best LB entry (IF 15%). Update it once tomorrow's LB result picks a winner.

## Code added
- `task5/src/task5_nb.py` section 5 (type detection + typesfirst files); `task5/src/lb_consistency.py` (hypothesis tests against past LB scores)
- `task2/src/features.py`, `task2/src/experiments_r2.py`; `task2/src/task2_nb.py` uses the composites
- `task3/src/experiments_r3.py`; `task3/src/task3_nb.py` section 4c (two-stage)
- Notebooks `task2/3/5.ipynb` regenerated with jupytext and executed top to bottom. `pytest`: 13 passed.
