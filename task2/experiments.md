# Task 2 experiments

## Index (one line per experiment; details in the tables below)
| Date | Change | CV macro F1 mean ± std (5-fold) | Kept? |
|---|---|---|---|
| 2026-11-08 | Baseline LightGBM multiclass, 41 raw features, argmax | 0.7496 ± 0.0027 | yes (base) |
| 2026-11-08 | Per-class probability multipliers (nested) | 0.7490 | no (< argmax) |
| 2026-11-08 | + latent-axis composites (`src/features.py`) | **0.7525 ± 0.0022** | **yes → `final_pred.csv`** (not on LB yet) |
| 2026-11-08 | Multipliers on the composite model (nested) | 0.7522 ± 0.0017 | no (< argmax) |
| 2026-11-09 | **Naive Bayes, smoothed per-class densities, NaN skipped** (`src/nb.py`, `src/experiments_r4.py`) | **0.7553 ± 0.0036** | **yes → `final_pred.csv`** (refit on all rows; not on LB yet) |
| 2026-11-09 | NB + LightGBM-composites blend (nested, log space) | 0.7556 ± 0.0028 | no (+0.0003 over NB < std) |
| 2026-11-09 | LightGBM stumps (num_leaves 2 = boosted GAM) | 0.7511 ± 0.0039 | no |
| 2026-11-09 | NB + in-fold MI feature filter / + nested per-class log-offsets | 0.7552 / 0.7559 | no (< std) |

## Round 4 (autopilot, 2026-11-09): naive Bayes is the correctly specified model

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-09 | **SmoothNB**: per-class densities, continuous features on 128 equal-mass bins + Gaussian smoother of 4 bins (a KDE on the empirical-CDF scale), discrete features (≤ 64 values) and categoricals as frequency tables (Laplace 1), **missing features skipped**; all fit in-fold. Final = refit on all 104,000 rows | macro F1 **0.7553 ± 0.0036** (folds 0.7541 / 0.7490 / 0.7565 / 0.7573 / 0.7595) | – | **+0.0028 vs 0.7525, every fold up** (+0.0025 / +0.0004 / +0.0019 / +0.0032 / +0.0059; gain > the reference fold std 0.0022). Per-class F1 0.824 / 0.744 / 0.712 / 0.741 (Grinder/Social/Whale up). 3 s for 5 folds vs 40 s for LightGBM. Test: full-fit vs fold-averaged NB agree 99.9%; 1,144 rows (4.4%) differ from the LightGBM final. Preds `oof_snb` / `test_snb` (fold-averaged); `final_pred.csv` = full refit |

Why it works:
- **Within each segment the 41 features are uncorrelated** (max |Spearman| 0.019 / 0.025 / 0.026 / 0.050 per class, mean
  0.004–0.008), while the pooled correlations reach 0.16, i.e. all dependence between features comes through the segment. And
  missingness is independent of the segment (max class-share deviation among NaN rows 1.5 pp ≈ noise). So
  p(x | c) = Π_f p(x_f | c) holds, naive Bayes is the correctly specified model and the Bayes-optimal log-odds is additive.
- With ~3.6 NaN per row, **skipping a missing feature (exact marginalisation)** beats guessing a value. LightGBM sends NaN
  down a learned branch; the spline LR imputed the median.
- The density estimate is what limits it: plain bins 8 / 16 / 32 / 64 give 0.7511 / 0.7538 / 0.7538 / 0.7502 (too coarse vs too
  noisy: 41 noisy log-ratio terms add up). Smoothing fine bins fixes that: 128/σ2 0.7541, **128/σ4 0.7553**, 128/σ8 0.7547,
  256/σ8 0.7550, 256/σ16 0.7539. Picking from this 6-config grid adds at most ≈ 0.001 of optimism.
- Boosted stumps (an additive model fit by trees) only reach 0.7511, so the trees were never limited by depth or interactions.
  What helped was the generative fit plus exact NaN handling.
- Not kept: blend with LightGBM (+0.0003), in-fold mutual-information filter (dropping ~12 flat features changes nothing:
  smoothing already shrinks their terms), per-class log-offsets (+0.0006, every fold up but < std; offsets ≈ +0.1 for Grinder
  and Social).

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

### /ensemble after round 4 (2026-11-09, `python -m common.ensemble 2 ...`, all choices nested)
5 models (snb, nb, lgbm_comp, lgbm, lgbm_stumps), Whale-probability correlation 0.983–0.996. Simple mean 0.7550 ± 0.0030,
hill climbing picks **snb alone in every fold** (0.7553), LR stacker on log-probs 0.7544 ± 0.0025 → best single snb
**0.7553 ± 0.0036** stays; `final_pred.csv` unchanged.

## Round 5 (2026-11-10): `player_id` as a feature, following the TA's reference code (`src/experiments_r5.py`)

The TA gave us last cohort's winning code (`TA/Task2`), which uses the ID number as features. Train and test IDs
interleave over P000000–P129999 in segment blocks (~97% pure), so the ID is a near-direct segment signal. All density /
model fits in-fold, same 5 folds.

| Model | macro F1 (5-fold) | Notes |
|---|---|---|
| snb (clean, round 4) | 0.7553 ± 0.0036 | reference, stays in `final_pred.csv` as the leak-free fallback |
| ID only (NB, smoothed per-class ID density) | 0.9623 (256 bins) → **0.9673** (2048 bins) | the block edges are sharp: fine bins matter |
| NB features + ID inside the same 128-bin NB | 0.9527 | coarse bins blur the blocks |
| NB features × fine ID density | 0.9648–0.9660 | features add nothing on top of the ID |
| **LightGBM + composites + `player_num`** | **0.9665 ± 0.0009** (folds 0.9674 / 0.9653 / 0.9660 / 0.9675 / 0.9662) | **kept as the leak version** → `outputs/pred_lgbm_pid.csv`; same design as the TA's reference |
| blend 0.5·(NB×ID) + 0.5·lgbm_pid | 0.9670 ± 0.0009 | +0.0005 < std, not kept |

- **+0.211 macro F1 over the clean model.** Once the ID is known, the real features add nothing (ID-only ≈ all variants within
  noise): the remaining ~3% errors sit where the ID blocks overlap.
- Test predictions: class mix 0.400 / 0.250 / 0.201 / 0.149 (train 0.394 / 0.250 / 0.202 / 0.154); 5,447 rows differ from the
  clean NB final.
- **LB 0.74153 (2026-11-10), Δ +0.0430 vs v12 as predicted.** Submission `submissions/sub_v15_task2_pid.csv` = v12 (LB 0.69852) with only task 2 replaced. Expected LB ≈ +0.043
  (task 2 in v12 is still the v1 LightGBM, 0.7496). **Must be disclosed in the report** as a data-generation leak used
  following the TA's reference.

## 2026-11-11: player_id model banned by the professor
The professor ruled that the `player_id` leak may not be used. `pred_lgbm_pid.csv` (CV 0.9665, LB 0.74153 in sub_v15) is withdrawn; the final task-2 model is the leak-free naive Bayes in `final_pred.csv` (CV 0.7553). Report: mention the leak only as a finding we did not use. Task 2 is now the task with the most room left (clean CV 0.7553).
- **LB 0.71608 (2026-11-11, `submissions/sub_v20_clean_no_pid.csv`)**: first LB reading of the clean SmoothNB. Δ −0.04500 vs v19, of which −0.00226 is task 5 → task 2 −0.04274 → **NB ≈ 0.752 on the LB (CV 0.7553)**. Against the v1 LightGBM (CV 0.7496) that is only +0.0003 LB (expected +0.0011), within noise.

## Round 6 (2026-11-11): per-class offsets as a final polish (`src/experiments_r6.py`)
| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-11 | SmoothNB + per-class log-offsets (coordinate ascent on macro F1, nested: fit on 4 folds, scored on the 5th) | **0.7559 ± 0.0036** (folds 0.7554 / 0.7495 / 0.7567 / 0.7579 / 0.7600, every fold up vs argmax) | 0.71612 (sub_v21, with task 3 ms_cat; Δ +0.00004 vs v20) | Offsets [0, 0.1, 0.1, 0] in 4/5 folds (fold 4: [0, 0.15, 0.1, 0.05]), full-OOF fit [0, 0.1, 0.1, 0]: a small push towards Grinder and Social. +0.0006 < fold std, kept only as the last-upload polish; `final_pred.csv` unchanged (plain argmax). Test: 413 rows differ, class mix 0.417 / 0.250 / 0.193 / 0.140. File `outputs/pred_snb_offsets.csv` (fold-averaged test probabilities + offsets). Model-implied ceiling from the NB posteriors ≈ 0.761 (NB slightly overconfident: mean max-prob 0.776 vs accuracy 0.770), so ≤ +0.005 remains in principle |

## 2026-11-12: refreshed data (TA re-upload 2026-11-11)
Train + test fully regenerated (same schema and class mix; values per id uncorrelated with the old files). The `player_id` leak is gone (segment purity of 500-id blocks 0.970 → 0.394 = majority share). Old files kept in `data/_old_2026-11-08/`.

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-12 | SmoothNB (128 bins, σ 4), unchanged settings, rerun on new data | **0.7570 ± 0.0022** (folds 0.7536 / 0.7563 / 0.7573 / 0.7574 / 0.7603) | 0.84268 (sub_v22, with t4/t5 also changed) | LightGBM+composites 0.7537. Per-class offsets: nested 0.7569 (no gain) → final = plain NB `final_pred.csv` |
