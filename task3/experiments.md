# Task 3 experiments

## Index (one line per experiment; details in the tables below)
| Date | Change | CV NMAE mean ± std (5-fold) | Kept? |
|---|---|---|---|
| 2026-11-08 | Constant median (79 THB) | 0.500 | no (reference) |
| 2026-11-08 | LightGBM L1, raw target | 0.7837 ± 0.0040 | yes (base) |
| 2026-11-08 | LightGBM L1 on log1p target | 0.7797 ± 0.0033 | no |
| 2026-11-08 | L1 + zero-gate τ 346 (nested post-proc) | 0.7840 ± 0.0040 | yes (was `final_pred.csv`, on LB in v1) |
| 2026-11-08 | Two-stage conditional median (classifier + spender quantiles) | **0.7878 ± 0.0037** | **yes → `final_pred.csv`** (not on LB yet) |
| 2026-11-08 | Nested blend two-stage + L1 (w 0.8–0.9) | 0.7879 ± 0.0037 | no (+0.0001, not worth it) |

## Log

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-08 | EDA only | – | – | see below. Constant-median (79 THB) baseline: NMAE **0.500** |
| 2026-11-08 | **LightGBM L1, raw target**, 32 raw features, native NaN, lr 0.05, 63 leaves, ES 200 | NMAE **0.7837** ± 0.0040 | – | 40 s. `oof_lgbm_l1` / `test_lgbm_l1` |
| 2026-11-08 | LightGBM L1 on log1p(target), expm1 back | NMAE 0.7797 ± 0.0033 | – | 9 s (stops early). −0.004 vs raw L1. `oof_lgbm_logl1` |
| 2026-11-08 | Post-proc on raw L1: zero-gate τ = 346, α = 1.00 (nested: fit on 4 folds, scored on 5th) | NMAE **0.7840** ± 0.0040 (nested), 0.7839 full | – | +0.0003 (< fold std, but kept: nested ≥ raw). **Chosen → `final_pred.csv`**. Zero share OOF 0.514 / test 0.518 / true 0.481 |
| 2026-11-08 | **Two-stage conditional median** (autopilot round 3; `src/experiments_r3.py`, now section 4c of the notebook): LGBM binary p = P(y>0) + LGBM quantile regressors on spenders only (log1p target, α ∈ {.01,.05,.1,.2,.3,.4,.5}); prediction = 0 if p ≤ 0.5 else spender quantile at (p−0.5)/p | NMAE **0.7878** ± 0.0037 (folds 0.7839 / 0.7845 / 0.7943 / 0.7885 / 0.7878); with the selected post-proc (τ = 1.2, α = 1.0) 0.7878 nested | – | **+0.0038 vs 0.7840 (> fold std 0.0037); every fold up** (+0.0033 to +0.0057 vs raw L1). 140 s. Nested blend with L1 (w 0.8–0.9) 0.7879, so not worth it. Zero share OOF 0.506 / test 0.506 / true 0.481. **Kept → `final_pred.csv`** (`oof_twostage` / `test_twostage`; stage outputs in `*_twostage_parts.npy`). Not yet on LB |

### Round 4 analysis, 2026-11-08 (`src/analysis_r4.py`): structure hunt → close to the ceiling
- **No ID leak** (AUC(player_id, spend>0) 0.503). The target has 2 decimals and 15 rows sit at the 300,000 cap. No feature
  acts as a hard zero/non-zero rule (the best single feature reaches a 0.93 spend rate).
- **The amount is nearly deterministic; whether a player spends is nearly random.** Among spenders LightGBM explains log y with
  R² 0.989 (corr(log y, log prev_month) 0.981). But stage 1 (spend > 0) has OOF AUC 0.78 overall and only **0.57–0.63
  within each segment**, and it is well calibrated: the segment sets the base rate and little else predicts it.
- **Error budget (twostage OOF):** 66% amount error on rows where both y and ŷ > 0, 32% from y = 0 but ŷ > 0, 3% from
  y > 0 but ŷ = 0. **Whales cause 79%** of the error, and rows with true y > 50k cause 54%. A perfect zero gate (oracle)
  would give NMAE 0.850, but the gate signal isn't there.
- **The amount error is noise:** the log residual is symmetric with MAD 0.178 (about ±18%). Prev + hist alone leave
  0.35, so the other features matter, but a bigger LightGBM (255 leaves, lr 0.02), log/ratio features and L1 all
  stay at MAD 0.177–0.179. Linear and spline models on log features are worse (R² 0.973 / 0.976), so the structure
  isn't log-linear either.
- **Conclusion:** like tasks 1 and 2, task 3 is near its ceiling with legitimate features. The remaining LB gap is
  elsewhere: the metric (binary vs macro), task 4's format, and the `player_id` leak (tasks 2/5).

### Round 3 notes (2026-11-08)
- **Modelling the median of the zero-inflated mixture explicitly beats a single L1 model.** The single model
  must learn the point mass at 0 and the spender distribution with one set of trees. The two-stage model gives
  the zero decision to a classifier (converges in ~90 trees) and the spend level to quantile models fit only on
  spenders.
- The low quantiles (α ≤ 0.1) matter for rows with p just above 0.5, where the median is a low spender
  quantile. The α = 0.5 model needs the most trees (1,000–2,700).
- Next: CatBoost/XGB versions of each stage, tuning the classifier (log-loss drives the gate), and a finer
  α grid near 0.

### Baseline notes (2026-11-08, `task3.ipynb`, source `src/task3_nb.py`)
- 5-fold StratifiedKFold on spend bins (zero + quintiles of non-zero spend), seed 42, 104,000 rows.
- **Raw-target L1 beats log-L1** (0.7837 vs 0.7797) even though both estimate the median. Likely cause:
  early stopping on L1 in log space optimises log-error, which down-weights the big spenders that dominate
  Σ|y−ŷ|. Next: log-L1 with a custom eval metric on the raw scale, or Huber/quantile variants.
- α stays at 1.00: the L1 model is already median-calibrated. The zero-gate helps a little by snapping
  small positive predictions to 0.
- The median prediction is 0 (51% of rows predicted 0), consistent with 48% true zeros.

Next: two-stage model (classifier P(spend > 0) × L1 regressor on spenders, gate tuned on OOF), FE on
prev_month × segment, ratios (historical / account_age, avg_transaction_value × purchase_frequency),
tuning, CatBoost (MAE/Quantile).

## EDA, 2026-11-08

Data: `train.csv` 104,000 × 34 (`player_id`, 32 numeric features, `spending_30d`), `test.csv` 25,889 × 33. No player_id overlap.

### Key insights
1. **Zero-inflated, extremely heavy-tailed target.** 48.1% exact zeros; median 79 THB, mean 10,497,
   p90 37,719, p99 150,032, max 300,000 (looks capped). Under MAE the best constant is the median, which
   scores NMAE 0.50; predicting the mean would be much worse. The few big spenders dominate Σ|y−ŷ|.
2. **`segment` (the Task 2 label, given here as a feature) is the strongest predictor** (Spearman 0.66).
   P(spend > 0) is 0.28 / 0.52 / 0.69 / 0.93 for Casual / Grinder / Social / Whale, and the median spend is
   0 / 264 / 1,838 / 54,760 THB. Whales drive almost all of the error budget.
3. **Past spending is strong but doesn't imply future spending.** `prev_month_spending` (ρ 0.65) and
   `historical_spending` (ρ 0.65) are the next best features. `prev_month_spending` is never 0 (always > 0 or
   NaN), yet 48% of players spend 0 next month. Among players who do spend, y / prev_month has a median of 2.2.
   Other useful features: `event_participation_rate` (0.59), `avg_transaction_value` (0.54), `vip_status` (0.42),
   `is_premium_member` (0.40).
4. **Missingness is uniform and uninformative.** ~6% per column, 1.86 NaN per row in both train and test. The
   zero rate among rows missing `prev_month_spending` / `historical_spending` / `segment` is 0.52–0.53, the same
   as overall. Native NaN handling, no flags.
5. **Several features carry no signal** (|Spearman| < 0.03): `seasonal_spending_pattern`, `guild_membership`,
   `primary_game`, `account_age_days`, `cross_game_activity`, `games_played`, `tournament_participation`,
   `days_since_last_login`. Ablation candidates.
