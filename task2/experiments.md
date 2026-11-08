# Task 2 experiments

## Index (one line per experiment; details in the tables below)
| Date | Change | CV macro F1 mean ± std (5-fold) | Kept? |
|---|---|---|---|
| 2026-11-08 | Baseline LightGBM multiclass, 41 raw features, argmax | 0.7496 ± 0.0027 | yes (base) |
| 2026-11-08 | Per-class probability multipliers (nested) | 0.7490 | no (< argmax) |
| 2026-11-08 | + latent-axis composites (`src/features.py`) | **0.7525 ± 0.0022** | **yes → `final_pred.csv`** (not on LB yet) |
| 2026-11-08 | Multipliers on the composite model (nested) | 0.7522 ± 0.0017 | no (< argmax) |

## Round 3 analysis, 2026-11-08 (`src/analysis_r3.py`): structure hunt

- **`player_id` encodes the label (data leak).** IDs come in contiguous segment blocks of about 97% purity: P000000–P050000
  Casual, ~P057000–P080000 Grinder, ~P086000–P110000 Social, ~P113000–P129999 Whale (one-vs-rest AUC of the ID number:
  0.018 / 0.53 / 0.80 / 0.97). Our ID-free **test** predictions follow the same bands, so test has the same layout. Using
  it would likely give macro F1 ≈ 0.97. **Not used: pending a team/TA decision**.
- **Features are conditionally independent given the class.** No within-class correlation differs between classes by more
  than 0.06 (B5). So the Bayes-optimal model is additive in per-feature log-likelihoods (naive-Bayes form), which explains
  why spline logistic regression (additive logit) **0.7534** matches or beats LightGBM (0.7525). Blend LGBM + 0.4·splineLR
  = 0.7540 (+0.0015 < fold std 0.0022, not kept yet). Legitimate gains left are small (GAM / NB with exact NaN marginalisation).
- LDA 0.6185, QDA 0.6034, LR + one-hot categoricals 0.7505. OOF is well calibrated (max-prob 0.9–0.95 → acc 0.925).
- Errors sit at the axis boundaries: the confused Grinder→Casual rows have low engagement (playtime, streak, friends) and
  the confused Social→Casual rows have low spending, i.e. they look like the predicted class (noise, not a missed rule).

## Log

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-08 | EDA only | – | – | see below |
| 2026-11-08 | **Baseline: LightGBM multiclass**, 41 raw features (3 `random_metric_*` dropped), categoricals as `category`, native NaN, lr 0.05, 31 leaves, ES 200 | macro F1 **0.7496** (argmax), fold std 0.0027; per-class F1 0.820 / 0.740 / 0.709 / 0.730 | – | best_iter 266–328, 37 s. Per-class multipliers: nested 0.7490 < argmax → **not used**. **Chosen → `final_pred.csv`**. Preds `oof_lgbm.npy` / `test_lgbm.npy` (probabilities, 4 cols) |
| 2026-11-08 | **+ latent-axis composites** (autopilot round 2; `src/features.py`, `src/experiments_r2.py`): `eng_score` = mean z of 14 engagement features, `spend_score` = mean z of 7 spending features (z stats fit on the training fold), their sum/difference, ordinal loyalty/engagement tier, spend/hour, rare items/THB | macro F1 **0.7525** ± 0.0022 (folds 0.7516 / 0.7486 / 0.7546 / 0.7540 / 0.7536) | – | **+0.0029 > fold std (0.0027), every fold up** (+0.0020 / +0.0036 / +0.0017 / +0.0049 / +0.0021). Per-class F1 0.822 / 0.742 / 0.709 / 0.737 (Whale +0.007). Fewer trees (best_iter 131–200 vs 266–328). Nested multipliers 0.7522 < argmax → still unused. **Kept → `final_pred.csv`** (`oof_lgbm_comp` / `test_lgbm_comp`). Not yet on LB |

### Round 2 notes (2026-11-08)
- **The gain is mostly Whale** (F1 0.730 → 0.737): averaging the axis features gives a less noisy spending score
  for telling Whale from Grinder. Casual/Social are nearly unchanged; the Social↔Casual boundary stays the hardest.
- **The model converges in about half the trees**, a sign that the composites replace many single-feature splits.
- Next ideas: a weighted axis score (in-fold LDA or logistic projection instead of equal weights), CatBoost on the
  same features for an ensemble, Optuna.

### Baseline notes (2026-11-08, `task2.ipynb`, source `src/task2_nb.py`)
- 5-fold StratifiedKFold (seed 42), 104,000 rows.
- Confusion matrix (OOF): the biggest errors are **Grinder→Casual (4,971)**, **Social→Casual (3,737)**,
  **Casual→Grinder (3,589)**, Whale→Social (2,347), Whale→Grinder (1,947). Casual is over-predicted (42.5% predicted vs 39.4% true).
- Test predicted distribution (43.2 / 24.3 / 18.7 / 13.8%) ≈ OOF, so no obvious drift.
- Multipliers barely move (only Whale ×1.05 on full OOF) and don't generalise: the model is already calibrated for macro F1.

Next: FE for the two axes (spend-per-hour, spending_frequency × avg_monthly_spending, rare_items / playtime,
engagement composite), target-free ratio features, CatBoost on categoricals, multi-seed, Optuna.

## EDA, 2026-11-08

Data: `train.csv` 104,000 × 46 (`player_id`, 34 numeric, 10 categorical, `segment`), `test.csv` 25,889 × 45. No player_id overlap.

### Key insights
1. **No missing labels; mild imbalance.** Casual 39.4%, Grinder 25.0%, Social 20.2%, Whale 15.4%.
   Macro F1 weights the minority Whale/Social classes equally, so their recall matters.
2. **Missing values are uniform and uninformative.** ~3.8 NaN per row in both train and test, ~7.8% per
   numeric column; `payment_method` 31% and `region` 26% missing. The segment distribution is identical for rows
   with and without NaN (0.392/0.253/0.198/0.158 vs 0.394/0.250/0.202/0.154). No missing flags needed.
3. **Two latent axes define the segments (2×2):** *engagement* (play_frequency, session duration, playtime,
   login streak, chat, friends, tournaments, peak hours: high for Grinder & Whale, low for Casual & Social) and
   *spending* (total/monthly spending, spending frequency, rare items, cosmetic preference: high for Social &
   Whale). Casual = low/low, Grinder = high/low, Social = low/mid, Whale = high/high. **Surprise:** Social players
   do *not* have more friends or chat (friend_count median 91 vs Casual 87); they differ from Casual by spending.
   Expected confusions: Casual↔Social and Grinder↔Whale (differ only on the spending axis), confirmed by the CM.
4. **Only 2 of 10 categoricals carry signal:** `loyalty_tier` (mean segment 0.43 Bronze → 2.39 Platinum) and
   `engagement_level` (0.82 Low → 1.74 Veteran). region, platform, device_type, payment_method, language,
   account_status, player_type_tag (!) and skill_tier are flat (mean segment 1.10–1.13 for every level).
5. **Noise columns:** `random_metric_1..3` (datacard says noise; medians identical across classes), plus flat
   `item_type_preference_social`, `account_age_days`, `avg_match_length`, `preferred_game_mode`,
   `responds_to_discounts`, `watches_esports`. Candidates for an ablation.
