# Task 1 experiments

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-08 | EDA only (no model) | – | – | see below; scripts in `src/eda_overview.py`, `src/eda_shapes.py` |
| 2026-11-08 | **Baseline: LogisticRegression**, 31 raw features, median impute + scale in-fold | F2 macro **0.6830** (nested), 0.6833 (full-OOF thr 0.381); AUC 0.7649; fold std 0.0012 | – | thr at 0.5 gives 0.6577; binary F2 0.6318. `oof_logreg.npy` / `test_logreg.npy` |
| 2026-11-08 | **LightGBM**, 31 raw features, native NaN, lr 0.03, 31 leaves, early stop 200 | F2 macro **0.7793** (nested), 0.7797 (full-OOF thr 0.402); AUC 0.8912; fold std 0.0035 | – | thr at 0.5 gives 0.7601; binary F2 0.7363; best_iter 575–872. **Chosen → `final_pred.csv`** (test pred positive rate 0.379 vs OOF 0.383) |

### Baseline results, 2026-11-08 (notebook `task1.ipynb`, 5-fold StratifiedKFold, seed 42, 90,816 labeled rows)

| model | AUC | F2 macro (nested thr) | F2 macro (full-OOF thr) | thr | F2 macro @0.5 | F2 binary | fold std |
|---|---|---|---|---|---|---|---|
| logreg | 0.7649 | 0.6830 | 0.6833 | 0.381 | 0.6577 | 0.6318 | 0.0012 |
| lgbm | 0.8912 | **0.7793** | 0.7797 | 0.402 | 0.7601 | 0.7363 | 0.0035 |

What we learned:
- **LightGBM beats logistic regression by +0.096 macro F2.** It confirms the EDA: the signal is
  non-monotonic (step rules, bumps in the middle of ranges), which a linear model on raw features can't
  represent. Binning or FE3/FE4 should narrow the gap for the linear baseline.
- **Threshold tuning adds about +0.02** over 0.5 for both models. The nested and full-OOF thresholds agree
  within 0.0004, so the tuned threshold is stable and not overfit.
- **LightGBM gain matches the EDA ranking:** crosshair 21%, reports 20%, account age 16%, friends 13%,
  level 7%. The top 5 features hold 76% of the gain. The 16 "noise" features each have ≤ 0.7% gain.
- The optimal macro-F2 threshold (~0.40) sits close to the base rate (0.35), and the predicted positive
  rate (38%) is only slightly above the true rate. Macro F2 is much less recall-hungry than binary F2
  would be.
- Test positive rate at the chosen threshold (0.379) ≈ OOF (0.383), consistent with no value drift.

Next: FE3–FE5 (report buckets, rule flags, skill score), FE9 noise-feature ablation, then tuning
(Optuna) and CatBoost/XGBoost for an ensemble.

### Experiment round 1, 2026-11-08 (`src/experiments_r1.py`)

Protocol: same 5 folds; each fold scored at the threshold tuned on the other 4 folds' OOF; report
mean ± std of per-fold macro F2. **Keep only if the gain > the fold std.**
Reference: `lgbm` (31 raw) = **0.7793 ± 0.0033** (folds 0.7800 / 0.7774 / 0.7825 / 0.7740 / 0.7825).

| Exp | Change | F2 macro (mean ± std) | Δ vs ref | AUC | Runtime | Decision | Preds |
|---|---|---|---|---|---|---|---|
| E1 | LGBM + FE pack (12 features: report buckets, rule flags, skill score, skill/level/age ratios, lps_ratio; ranks fit in-fold) | 0.7785 ± 0.0037 | −0.0008 | 0.8905 | 18.6 s | **rejected** | `lgbm_fe` |
| E2 | CatBoost (depth 6, lr 0.05, ES 200) on 31 raw | 0.7774 ± 0.0027 | −0.0019 | 0.8895 | 37.6 s | – | `cat_raw` |
| E2 | Blend 0.7·lgbm + 0.3·cat | 0.7803 ± 0.0034 | +0.0010 | 0.8919 | – | **rejected** (< std; the weight was picked on OOF) | – |
| E2 | Blend 0.5 / 0.7 cat, rank average | 0.7795 / 0.7792 / 0.7793 | ≈ 0 | 0.8918 | – | rejected | – |
| E3 | LGBM, drop 16 noise features (15 left) | 0.7792 ± 0.0029 | −0.0001 | 0.8918 | 9.5 s | **rejected** | `lgbm_nonoise` |

`final_pred.csv` is unchanged (baseline `lgbm`).

What we learned:
- **Hand-made features don't help the tree model.** LightGBM already finds the step thresholds
  (reports, age, friends) on its own, and the combined skill score adds nothing beyond the individual skill
  columns. FE would mostly help the linear baseline.
- **The noise features really are noise.** Dropping them doesn't change the score (−0.0001), AUC goes up
  slightly, and training is 2× faster. Keep them out of future experiments for speed even though the gain
  doesn't pass the bar.
- **CatBoost and LightGBM make similar mistakes.** The 30% CatBoost blend beats LGBM in 4 of 5 folds,
  but only by about 0.001: too correlated to gain much from averaging.
- **Every variant lands at 0.778–0.780.** That suggests a ceiling set by label noise, not by the model.
  The 5–15 report bucket holds 55% of labeled rows (49,814) with a 40–52% cheater rate, and no feature separates them
  well. Further Task 1 gains are likely small (~0.001–0.003). On the leaderboard, team effort may pay off
  more on other tasks.
- Next if continuing: Optuna tuning of LGBM on the 15 kept features, multi-seed averaging, and a
  3-model blend (LGBM + CatBoost + XGBoost) with weights chosen by nested CV.

### Round 2, 2026-11-08: error analysis, looking for structure the trees miss (`src/analysis_r2.py`, `src/experiments_r2.py`)

Reference: `lgbm` (31 raw) = **0.7793 ± 0.0033** nested.

| Exp | Change | F2 macro (nested) | Δ | Decision |
|---|---|---|---|---|
| A1 | Validation folds re-masked to the **test** NaN rates (headshot 2.7→14.1%, win_rate 4.9→13.7%, clutch, device_changes …) | 0.7790 (AUC 0.8904) | −0.0003 | **Missingness shift is harmless** → audit finding closed |
| A2 | Train with random masking up to max(train, test) NaN rates, score on re-masked val | 0.7780 | −0.0013 | rejected |
| E4 | + class-conditional Gaussian features (rank-gaussianised, in-fold; LLR, Mahalanobis to each class; inner folds for the training rows), 15 signal features | 0.7775 ± 0.0036 | −0.0018 | **rejected** (LLR alone AUC 0.784, but adds nothing to trees). `lgbm_qda_sig` |
| D1 | Uncertain middle (0.05 < OOF < 0.95, 57k rows): spline logistic regression vs LGBM | AUC 0.674 vs 0.740 | – | trees win; the middle is not a smooth additive logit |

What we learned:
- **No generator artefacts:** player_id number AUC 0.498, no target autocorrelation in file order, no class-dependent rounding,
  integer shares or clipping (except `reports_received` < 4 ⇒ cheater, already used by the trees).
- **Two regimes.** 37% of rows are decided by hard rules and the model is perfect there (25.0k rows at p ≈ 0 with true rate
  0.000; 8.5k at p ≈ 1 with 0.999). The other 63% are a probabilistic middle where the OOF is **well calibrated** at
  every level (e.g. p 0.3–0.4 → 0.360 true). 66% of all errors sit in the 5–15 report bucket.
- **Feature correlations differ by class** (legit: reports ↔ level speed / age / friends ≈ ±0.27, K/D ↔ age / friends ≈ ±0.17;
  cheaters ≈ 0), but a QDA feature built on that doesn't help. The class-0 correlations are most likely produced by the
  hard rules (reports ≥ 20 or age > 1100 ⇒ legit), which the trees already use.
- **Confident errors look like the other class** (confident FNs have many reports and high skill), so they look like label
  noise, not a missed rule.
- **Conclusion:** with these features, Task 1 is close to its ceiling (~0.78). Every model and feature idea gives 0.777–0.780.
  If teams above 0.70 on the LB really average ≥ 0.83 on tasks 1–3, the gap is more likely in tasks 2/3, or the task weights
  aren't equal (ask the TA).

## EDA, 2026-11-08

Data: `train.csv` 104,000 × 33 (`player_id`, 31 float features, `is_cheater`), `test.csv` 25,889 × 32.
No duplicate feature rows; no `player_id` overlap between train and test.

### Key insights

1. **12.7% of training rows have no label.** 13,184 rows have `is_cheater` = NaN, leaving **90,816 labeled
   rows**. The unlabeled rows look the same as the labeled ones (largest standardized mean difference 0.023;
   same missingness), so dropping them loses nothing. Positive rate among labeled rows: **34.9%**
   (31,725 / 90,816). That's moderately imbalanced, not extreme.

2. **Missing values are everywhere, and their rates differ by column between train and test.** 98.2% of
   train rows and 92.8% of test rows have at least one NaN (3.71 vs 2.48 NaN per row). Per-column rates
   flip: `aiming_smoothness` 20.9% train vs 2.1% test, `account_age_days` 14.4% vs 0.6%,
   `headshot_percentage` 2.7% vs 14.1%, `win_rate` 5.0% vs 13.7%. **Missingness carries no target
   signal**: the cheater rate among rows where a feature is NaN is 0.34–0.36, the same as the base rate.
   So it's random masking with a different mask in each split.

3. **Feature values themselves don't drift.** The train-vs-test KS statistic is ≤ 0.011 for every feature,
   and the `reports_received` distribution differs by at most 0.5 percentage points. Only the missingness
   pattern differs (insight 2).

4. **`reports_received` behaves like a step function**, which suggests synthetic data generated by hard
   rules:

   | reports | 4 | 5–15 | 16–19 | ≥ 20 |
   |---|---|---|---|---|
   | cheater rate | **1.00** (n=1,455) | 0.40–0.52 | 0.16–0.18 | **0.00** (n=7,563) |

   It is the strongest single feature (AUC 0.70). More reports means *less* likely to be a cheater.

5. **Higher skill means less likely to be a cheater.** Cheaters have lower headshot %, K/D, accuracy,
   `game_sense_score` and `crosshair_placement`. A rank-average of five skill stats gives a cheater rate
   falling steadily from **0.59** (lowest-skill decile) to **0.15** (highest). This fits the business
   framing: every row is an already-flagged account, and the false positives are *skilled legitimate
   players* mass-reported by frustrated opponents. A good story for the report.

6. **Several features have hard zero-rate or high-rate regions** (synthetic rules again):
   - Always legitimate: `account_age_days` > 1100 (0.00, n=4,151), `friend_network_size` > 300 (0.00,
     n=2,123) and ≤ 10 (0.04).
   - Mostly legitimate: `level_progression_speed` > 3 (0.06).
   - Bumps in the middle of the range: age 90–180 days (0.53), friends 50–100 (0.55), level ≤ 30 (0.58).
   - `crosshair_placement` ≤ 0.84 gives 0.52 and > 0.92 gives 0.21, nearly flat in between.

   Most of these are **non-monotonic**, so per-feature AUC understates them. Tree models will capture
   them; a logistic-regression baseline needs binning.

7. **About 16 of the 31 features look like pure noise.** Their decile target-rate range is < 0.025 and
   mutual information ≈ 0: `sessions_per_day`, `first_blood_rate`, `survival_time_avg`,
   `damage_per_round`, `device_changes_count`, `clutch_success_rate`, `map_knowledge`,
   `night_play_ratio`, `avg_session_length_min`, `utility_usage_rate`, `team_play_score`,
   `buy_decision_score`, `communication_rate`, `weapon_switch_speed`, `movement_pattern_score`,
   `input_consistency_score`. Notably, the "aimbot-like" features (reaction time, input consistency,
   aiming smoothness) carry little or no signal.

8. **No impossible values; only two heavy tails.** There are no negatives, and the percentage columns stay
   in range (headshot 5–95, win rate 16–100). Most features are clipped to fixed bounds. The tails:
   `level_progression_speed` (p99 4.7, max 80; 8.5% of rows beyond 3×IQR) and `reaction_time_ms`
   (floored at 55 for 2.9% of rows, max 714). The `level_progression_speed` tail is *informative*
   (> 3 → 6% cheaters), so don't clip or remove it.

9. **Features are mostly independent.** The only strong correlations are `account_age_days` ↔
   `level_progression_speed` (Spearman 0.81) and ↔ `friend_network_size` (0.63).
   `level_progression_speed` tracks `level / account_age_days` (ρ 0.88) but isn't exactly derived from
   it (ratio 0.16–1.07). Everything else is below 0.33.

10. **What this means for the metric (F2 macro).** Even within the 5–15 report bucket, `crosshair_placement`
    moves the cheater rate from 0.57 to 0.29, so features combine roughly additively on top of the hard
    rules. With a 35% base rate and macro averaging, the threshold has to balance both classes. Tune it
    on OOF with `metrics.best_fbeta_threshold(..., beta=2)` (macro) rather than assuming 0.5 or a
    recall-heavy cut.

### Proposed feature engineering

| # | Idea | Reasoning |
|---|---|---|
| FE1 | Drop rows with NaN target | Unlabeled; distributionally identical (insight 1). |
| FE2 | **No missing-indicator flags**; use native NaN handling (LightGBM/XGBoost/CatBoost); median-impute inside folds for linear models | Missingness is uninformative (insight 2), and its per-column rates differ between train and test, so flags would add noise *and* shift. |
| FE3 | `reports_bucket` ∈ {4, 5–15, 16–19, 20+}, plus `reports_eq_4` | Encodes the step rule directly (insight 4). Trees find it alone, but the bucket helps the linear baseline and can be used as a stratification check. |
| FE4 | Rule-region flags: `age_gt_1100`, `friends_le_10`, `friends_gt_300`, `lps_gt_3` | Near-certain legitimate regions (insight 6). Also test a **post-processing override** (force 0 in zero-rate regions) on OOF: it's free precision if the rule holds in every fold. |
| FE5 | `skill_score` = mean percentile rank of K/D, headshot %, accuracy, crosshair, game sense, spray control, aiming smoothness (ranks fit on train folds only) | Pools 7 weak, same-direction signals into one strong monotonic feature (decile rate 0.59 → 0.15). Also a more robust fallback when individual skill columns are NaN (test has more NaN in headshot/win rate). |
| FE6 | Skill-vs-experience ratios: `skill_score / level`, `kill_death_ratio / level`, `level / account_age_days`, `friend_network_size / account_age_days` | The business hypothesis is that a flagged player whose skill doesn't fit their experience is suspicious. Ratios to age/level express that directly. Validate by CV; drop if no gain. |
| FE7 | `lps_residual` = `level_progression_speed` − expected from `level / account_age_days` | `lps` correlates with level/age (ρ 0.88) but isn't derived from it; the residual separates abnormal leveling (boosting/smurfing) from normal. |
| FE8 | Model-based imputation of `account_age_days` from `level`, `level_progression_speed`, `friend_network_size` (fit inside folds) | `account_age_days` is a strong feature (has a zero-rate region) and is 14% missing in train. Its correlates are present, so imputing beats a median for linear models; trees may not need it. |
| FE9 | Ablation: drop the ~16 noise features (insight 7) | Reduces variance and fit time. Run it as a CV experiment rather than dropping blindly, since weak interactions may exist. |
| FE10 | `log1p(level_progression_speed)`, `log(reaction_time_ms)` for linear models only | Heavy tails (insight 8); trees are invariant to monotone transforms. |

### Validation notes for modeling
- 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the 90,816 labeled rows; score with
  `metrics.f2` (macro) and log binary F2 alongside.
- Test has a different missingness mask (insight 2). Also check CV on a version of the validation fold
  re-masked with the test per-column NaN rates, to make sure models don't lean on columns that are often
  missing in test (`headshot_percentage`, `win_rate`).
- Baseline to keep for the report: logistic regression on median-imputed, scaled features plus FE3/FE4.
