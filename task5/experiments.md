# Task 5 experiments

## Index (one line per experiment; no labels, so the "CV" column holds the LB-implied t5 = 5·LB − 2.3131)
| Date | Change | Score (no CV possible) | Kept? |
|---|---|---|---|
| 2026-11-08 | IsolationForest top 10% (v1) | LB 0.60044 → t5 ≈ 0.689 | superseded |
| 2026-11-08 | IF top 6% | LB 0.59566 → t5 ≈ 0.665 | no |
| 2026-11-08 | IF top 15% | LB **0.60499** → t5 ≈ 0.712 | yes (best LB so far; `submissions/sub_v4_task5_rate15.csv`). **Notebook now uses `FLAG_RATE = 0.15`, so `final_pred.csv` reproduces this file exactly (0 rows differ)** |
| 2026-11-08 | Rule score top 15% | LB 0.59157 → t5 ≈ 0.645 | no |
| 2026-11-08 | **typesfirst** 15% (7 detached-cluster types first, then IF) | LB **0.66175** → t5 ≈ 0.816 (binary F3) | **yes** (2026-11-09, best task 5) → `outputs/pred_typesfirst_rate15.csv` |
| 2026-11-08 | **typesfirst** 6% | projected t5 0.69–0.80 | dropped (label-probe concern, see `submissions/log.md`) |
| 2026-11-09 | **typesfirst8** 15%: + 8th type "erratic performance" (accuracy_std, kda_std, performance_stability jointly high; 2-component EM, posterior > 0.1 → 303 rows) | no CV. Projected t5 ≈ 0.90 (+0.018 LB) if the group is labelled anomalous, ≈ 0.814 (−0.0004 LB) if not | **yes → `final_pred.csv`** and `outputs/pred_typesfirst8_rate15.csv` (538 rows differ from typesfirst15). **LB 0.69852 (2026-11-10): Δ +0.0191 vs typesfirst15 → t5 ≈ 0.91, as projected** |

| 2026-11-10 | **typesonly8**: flag only the 8 types (2,115 rows, 8.17%), no IsolationForest filler (`outputs/pred_typesonly8.csv`) | LB **0.75705** (Δ +0.0155 vs the same file at 15%) → t5 ≈ 0.99 | **yes (best task 5)**. The filler was almost all false alarms: the 8 detected types are essentially the whole anomaly set |

Analysis behind the projections: `src/lb_consistency.py` (reproduces every number in the round-1 section).
Round-2 structure scans: `src/analysis_r2.py`.

## Round 2 (autopilot, 2026-11-09): an 8th type with no gap, found by tail co-occurrence

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-09 | Structure hunt outside the 7 known types (`src/analysis_r2.py`, `player_id` dropped at load) + **typesfirst8** (section 5b of the notebook) | n/a (no labels) | **0.69852** (+0.0191) | 1 new type, ≈ 229 rows (EM). 303 rows typed erratic (posterior > 0.1), 34 of them already in the typesfirst15 flags; they replace the 269 lowest-ranked IF filler rows at the same 15% rate |

**Key structural fact: in normal rows the 29 periodic bases are mutually independent.** Max |Spearman| between the
per-row means of two different bases is 0.024; within a base, periods persist strongly (r(p1, p2) 0.5–0.99), except
`ip_hash_entropy` (pure per-period noise) and lat/lon (per-player constants). Each base looks like a per-player level +
per-period noise. So an injected type made of *moderate* shifts in several bases leaves no empty gap, but its bases'
tails co-occur far more often than independence allows.

**Scan:** for every cross-base pair of 5% tails (per-row level, within-row sd, period-4 change, period-4 value; 23,332
pairs), observed vs expected count among the non-type rows. Exactly one pair is out of line:

| pair | obs | exp | z |
|---|---|---|---|
| accuracy_std level high × kda_std level high | 222 | 60 | **20.9** |
| (same bases, period-4 values) | 138–176 | 55–57 | 11–16 |
| next best pair (login_count p4 low × session_duration_mean low) | 90 | 57 | 4.3 |

With the accuracy_std/kda_std top-10% corner removed, the largest z is 4.25 and the 99.9th percentile 3.14 over 23,332
pairs: pure noise. The excess in that corner saturates at ≈ 190 rows from the 8% tails on (2%: 74, 5%: 162, 8%: 193,
10%: 191, 15%: 165). AUC profile of the corner vs other non-type rows: accuracy_std / kda_std 0.93–0.99 in **every
period**, performance_stability 0.77–0.81, within-row sd of performance_stability 0.25. Everything else ≈ 0.5.

**What the members look like** (posterior > 0.5): accuracy_std in 8.5–12 (bulk median 6.3), kda_std 0.68–1.0 (bulk 0.49),
performance_stability 0.83–0.96 (bulk 0.78), in all four periods, with no period-4 change and normal accuracy/kda means.
These look like boxes U(9, 12), U(0.7, 1.0) and U(0.85, 0.95): a stable player with wildly inconsistent match results,
e.g. a shared account or a boosting service. Each value on its own is common (bulk tails reach 12 / 1.0 / 1.04), so no
single-feature rule could isolate them. This explains why IF missed them: only 25 of the ≈ 229 expected members are in
the typesfirst15 flags.

**Detector:** normal scores of the 3 per-row means (empirical CDF of the non-type rows), then EM with 2 components:
bulk fixed at N(0, C0) (independent bases), anomaly N(μ, Σ) free. Converges to π = 0.95% (229 rows), μ = (2.12, 2.23,
1.12), sd (0.46, 0.55, 0.22). The posterior is bimodal (non-type rows: 23,659 below 0.01, 242 above 0.5, 127 above 0.9).
**Only 1 of the 1,812 known-type rows has posterior > 0.5**, the same zero-overlap footprint as the other 7 types (one
injected type per row). A 12-D version (each period separately) drifted to a 7% component (the Gaussian bulk is
mis-specified in 12-D) and was discarded.

**Projection (binary F3, flags fixed at 15% = 3,883):** F3 = 10·TP / (9·P + K), so at fixed K each extra true positive
is worth 10/(9P + K) ≈ t5/TP ≈ 0.816/1,850 ≈ +0.00044. If the group is labelled anomalous: ≈ 224 expected members flagged,
≈ 199 of them new, minus the anomalies among the 269 displaced IF filler rows (a few) → t5 ≈ 0.90, **LB ≈ +0.018**. If it
is not: we lose only the few anomalies in those 269 filler rows → LB ≈ −0.0004. A one-sided bet, so it is the next task-5
upload.

**Negative results (also worth a line in the report):**
- *Rows outside the other periods' population range in periods 1/2* (40–140 rows per base, almost none in the union):
  continuous tails with no gap, and the 15 groups barely co-occur (0–2 shared rows). The spread simply differs by period
  (period 1 widest for the std bases, period 4 narrowest). Generator artefact, not a type.
- *Moderate period-4 shifts that stay in range* (|z| > 4 more frequent in period 4 than in period 1 for
  session_duration_mean, login_count, kda_*, communication_frequency, …): symmetric in sign, and rows with ≥ 2 such
  bases (189) match independence (≈ 212 expected). Period 4 just has heavier noise in these bases.
- *Integer bases* follow a simple noise recipe, e.g. device_count: period 1 ∈ {1, 2}, periods 2–3 add U{−1, 0, +1},
  period 4 adds Bernoulli(½). Same for refund_count / guild_changes. All maxima among non-type rows are bounded.
- *Density bumps* (fine histogram vs its smoothed self, 142 views): only sharp peaks at zero period-4 change (a smoothing
  artefact; no structural zeros or constant rows). *NaN pattern:* consistent with independent ~5% masking.
- *Geography:* lat is uniform over ±60° and lon over ±180°. The 69 "bad coordinate" rows outside the union are lon jitter
  just past ±180, evenly spread over the four periods.

## 2026-11-08 (task 2 analysis follow-up): ID leak + metric/weights determined

- **Task 5's `player_id`s run P000000–P025888, and the anomalies are a contiguous block at the end:** 1,811 of the 1,812
  detached-cluster rows have ID ≥ 23,819. The block holds 2,070 rows (8.00%): 1,811 typed + 259 untyped.
- **Already-scored LB results, re-read with no new submission:** taking the block as labels, **binary** F3 explains all
  four task-5 variants exactly: LB = 0.4986 + 0.2001·t5, RMSE 0.00000 (rate06 0.485, rate10 0.509, rate15 0.532,
  rule15 0.465). Macro F3 doesn't fit (RMSE 0.004, wrong ordering). So:
  1. **Kaggle scores task 5 with binary F3**, not macro;
  2. **task weights are equal** (slope 0.2);
  3. tasks 1–4 together contribute 2.493 → **t4 ≈ 0.18–0.22**, roughly chance, not 0.
- The earlier "hidden group of ≥ 1,000 rows" and the 0.78–0.82 projections assumed macro F3, so they are **superseded**.
- Legitimate consequence (no leak needed): binary F3 is far more recall-hungry than macro, so typesfirst 15% (binary F3
  0.82 vs rate15 0.53 under the block) is still the right next upload.
- Using the ID block itself for predictions is a leak: **not used, pending a team/TA decision**. Note that this analysis
  effectively reveals test labels from LB scores. Disclose it in the report whatever the decision.

## Log

No labels → no CV. Each change is judged by sanity checks below and, later, a planned LB sweep that changes
only task5 (flag rates 3% / 6% / 10% / 15%; see the project rules).

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-08 | EDA only | – | – | see below |
| 2026-11-08 | **Baseline: IsolationForest** (500 trees, max_samples 1024) on 93 engineered cross-period features (period-4 robust z, max consecutive change, 4-period std per base; haversine jumps; bad-coordinate flag; static + n_flags), log1p + median/IQR scaling. **Flag rate 10%** | n/a | – | Top 10% captures 99.9% of the 752 rows with ≥3 period-4 shifts; flagged rows: n_flags 0.89 vs 0.21, max geo jump 1,278 km vs 120 km. **Chosen → `final_pred.csv`**. Scores in `test_iso.npy`; `test_rule.npy` = transparent rule score (Spearman 0.52 with IF). OOF files are empty arrays (unsupervised) |

### Baseline notes (2026-11-08, `task5.ipynb`, source `src/task5_nb.py`)
- Recall of the "≥3 shifted features" group by IF top-k: 3% → 0.965, 5% → 0.997, 10% → 0.999.
  If that group is close to the true anomalies (~3–5% of rows), **10% over-flags**. Rough F3-macro maths for a
  3.5% true rate with perfect recall: flagging 10% gives ≈ 0.89, flagging 5% ≈ 0.95. The first flag-rate LB
  sweep should test 4–6%.
- IF and the rule score only correlate 0.52 (Spearman). IF also reacts to static extremes (age, stability)
  that may be normal variation.

Next: rank-average IF + rule score + LOF/ECOD-style stats; sweep flag rate on the LB (task5 only).

### LB-implied score (2026-11-08, see `submissions/log.md` gap analysis)
- Assuming equal weights and t4 = 0, **t5 (IF, top 10%) ≈ 0.69 macro F3**. Random flags score ≈ 0.50, and a perfect
  ranking at 10% would score 0.82–0.98 for a true rate of 2–10%. Solving for recall: IF top-10% catches only about
  **54% of the anomalies at a 3.5% true rate (50% at 5%)**.
- That **contradicts "the ≥3-shift group ≈ the anomalies"**: IF top-10% contains 99.9% of that group (3.1% of
  rows). If the group were the anomalies, t5 would be ≈ 0.89. Either the true rate is higher, or about half the
  anomalies are of a kind our features don't capture.
- So the scorer matters more than the flag rate. The sweep (6% → 3% or 15%) shows where the missed anomalies rank;
  the bigger gain is a better score (new anomaly signatures, rank-averaged detectors).
- Sweep files: `submissions/sub_v4_task5_rate{03,06,15}.csv` (only task5 differs from v1; flags are nested).

### LB sweep results (2026-11-08)
| Submission | t5 ranking / rate | LB | implied t5 |
|---|---|---|---|
| rate06 | IF 6% | 0.59566 | 0.665 |
| v1 | IF 10% | 0.60044 | 0.689 |
| rate15 | IF 15% | **0.60499** | **0.712** |
| rule15 | rule score 15% | 0.59157 | 0.645 |
- **More flags help** (F3 rewards recall), with roughly constant anomaly share per band past 6%, so the IF score is
  weak below the top. Base is now 15%.
- **The rule score is much worse than IF at the same rate.** Our EDA's "period-4 injection" story covers only part of
  the anomalies. IF's unique picks carry the signal. The static extremes (age, behavior_stability, …) that we called
  "may be normal variation" are now suspects.
- Next: profile the IF-only vs rule-only flagged rows, add detectors that see other anomaly types
  (per-feature extremes, ECOD/COPOD-style tail scores, LOF, all-period stats, not just period 4), and rank-average.

### Round 1 (autopilot, 2026-11-08): anomaly types = detached clusters → `typesfirst` score
| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-08 | **typesfirst**: flag the 7 detached-cluster types first (tier A/B/C > bot > low_stability > low_activity/comm), IF rank inside tiers and for the rest | n/a (no labels) | _pending (submit tomorrow)_ | Files `outputs/pred_typesfirst_rate06.csv`, `pred_typesfirst_rate15.csv` (id, prediction; only task 5). Scores `test_typesfirst.npy`. Projected t5 at 15%: **0.78–0.82** vs 0.712 (rate15) if all types are anomalies. `final_pred.csv` unchanged |

**How the types were found.** A histogram scan of every column (200 bins, empty runs ≥ 4 bins) plus a period-4 vs
periods-1–3 *population-range* check:

| type | definition (cut-off sits inside an empty gap) | rows | IF top-6/10/15% recall |
|---|---|---|---|
| A cheat/boost | period 4 outside the periods-1–3 range in skill_rating, skill_trajectory, accuracy, kda, friend_removes, reaction_time (co-occur 80–98%) | 311 | 1.00 / 1.00 / 1.00 |
| B takeover | period-4 currency_transfers / play_pattern_consistency out of range; also location jumps; **password_changed = 1 in 100%**, mass_item_sale 68% | 310 | 1.00 / 1.00 / 1.00 |
| C purchase fraud | period-4 purchase_frequency / purchase_amount_mean out of range (+ location jumps) | 205 | 1.00 / 1.00 / 1.00 |
| bot | ip_hash_entropy < 0.25, login_time_variance < 0.9, session_duration_std < 6, all at once, in all 4 periods | 207 | 0.52 / 0.73 / 0.85 |
| low_activity | activity_level < 0.32 (all periods) | 266 | 0.03 / 0.07 / 0.14 |
| low_comm | communication_frequency < 0.15 (all periods) | 237 | 0.03 / 0.07 / 0.14 |
| low_stability | behavior_stability 0.20–0.40 (bulk 0.68–0.95) | 274 | 0.08 / 0.19 / 0.33 |

Union 1,812 rows (7.0%). **Pairwise overlap between types is exactly 0**: one injected type per anomalous row.

**What the four already-scored LB variants say (no new submission used).** Insights:
- **Every big geographic jump (> 1,000 km) belongs to B or C.** Impossible travel isn't a separate type.
- **Outside type B the four security flags are independent at their base rates**, so they look like noise.
  IF ranks flagged rows high only because the flags are among its inputs. The rule score (flag-heavy) lost to IF.
- **Beyond the out-of-range rows, period-4 deltas show no excess over a control period** (period 3 vs 1–2).
  There's no population of moderate period-4 shifts, and no monotone-trend excess either.
- **Union alone ≠ the label.** It predicts t5 falling with flag rate (0.760 / 0.752 / 0.736 at 6/10/15%), but the LB
  rose. No subset of the single-feature types fits either; the full union fits best. With equal weights, the LB
  needs a further hidden group of ≥ ~1,000 rows that IF ranks 2–4× above random (rough split: it sits in IF's
  no-flag picks, not in flagged rows). It is not found yet.
- **Projection** (hidden group 900–2,400 rows, density in IF's band taken from the fits): typesfirst 15% t5 ≈
  0.78–0.82 (+0.014–0.021 LB), typesfirst 6% ≈ 0.69–0.80. Downside: if low_activity / low_comm / low_stability
  are injected *normal* outliers, forcing them in costs about 770 FPs. The 6% vs 15% pair separates the two cases.
- Data quirks seen along the way (probably generator artefacts, not anomalies): `refund_count_2/3` and
  `guild_changes_2/3` contain impossible −1 values, and `device_count_2/3` contain 0, in ~16% of rows each (integer
  noise ±1 in periods 2–3 only). `device_count_4`, `refund_count_4` and `guild_changes_4` are shifted +0.5 for the
  whole population. Period 4 is slightly *less* noisy than periods 2–3. friend_adds/friend_removes are clipped at 0.
- Next: find the hidden group (ideas: cross-feature consistency such as session_duration_std vs mean, IF without
  flag/NaN artefacts), and once LB confirms the types, drop the IF-only no-signal picks.

## EDA, 2026-11-08

Data: `test.csv` 25,889 × 123 (`player_id`, 29 bases × 4 periods = 116 periodic + 6 static). No labels, no train set.

### Key insights
1. **Anomalies look injected in period 4.** For each base feature, a robust z of period 4 vs periods 1–3
   flags 4–7% of rows (|z| > 5). The same rows shift in many features at once: **3.1% of rows have ≥3
   features with z > 3** in period 4, versus only 0.19% if period 1 is tested against 2–4. So the change
   is concentrated in the latest period (account takeover / sudden behaviour change).
2. **Impossible coordinates.** Latitudes range −105…129 and longitudes −224…275; 0.8% of rows have
   |lat| > 90 or |lon| > 180, and 62% of those also have ≥3 period-4 shifts. Likely a corruption
   injected as an anomaly signature.
3. **Geographic jumps are bimodal.** The max haversine distance between consecutive login points has median
   117 km and p95 216 km, but p99 6,204 km (max 12,666). Rows with 3–4 security flags have median jumps of
   5,275–9,913 km vs ~116 km otherwise ("impossible travel").
4. **Security flags line up with the shifts.** password_changed 6.0%, vpn_usage 14.9%,
   suspicious_login_time 4.7%, mass_item_sale 2.6% (plus ~5% NaN each). Rows with ≥3 period-4 shifts average
   1.01 flags vs 0.26 for the rest. VPN alone (15%) is too common to mean anomaly by itself.
5. **Missingness ~5% per column, ~6 NaN per row; static features look normal** (`behavior_stability` median
   0.82 in both shifted and other rows, so it isn't a useful anomaly signal). Period-1 to 3 medians are stable;
   `device_count` median rises from 1 to 2 in period 4 across the population.

## 2026-11-11: `player_id` block used (user decision)
| Date | Change | CV | LB | Kept? |
|---|---|---|---|---|
| 2026-11-11 | Flag `player_id` ≥ 23,819 (2,070 rows, 8.00%) → `outputs/pred_idblock.csv` | n/a (no labels) | **0.76108** (sub_v19, with t1/t3 polish; t5 → ≈ 1.0 as predicted) | yes, leak version (clean typed-only stays in `final_pred.csv`) |

- The boundary is sharp: typed-flag rate 0.0 on IDs 23,700–23,818 and 1.0 on 23,819–23,899.
- vs typed-only (2,115 flags): 2,051 shared, 19 block-only, 64 typed-only. Taking the block as truth, typed-only scores
  binary F3 0.9887, which matches the LB-implied t5 ≈ 0.988, so the block should score ≈ 1.0 (+0.00226 LB).
- Like task 2's ID feature, this is a data-generation leak: disclose it in the report. The clean method (8 types found
  without the ID) is the real contribution and stays the documented final model.

## 2026-11-11: player_id block banned by the professor
The professor ruled that the `player_id` leak may not be used. `pred_idblock.csv` (in sub_v19) is withdrawn; the final task-5 file is the typed-only detector in `final_pred.csv` (8 types, 2,115 flags, LB t5 ≈ 0.988). The ID boundary may be mentioned in the report only as an observation, not as a method.
- 2026-11-11: typed-only `final_pred.csv` is back in the best clean file (`submissions/sub_v20_clean_no_pid.csv`, LB 0.71608).
