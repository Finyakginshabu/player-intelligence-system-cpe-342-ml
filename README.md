# Player Intelligence System (CPE342 ML, KMUTT 1/2026)

Kaggle competition: https://www.kaggle.com/competitions/player-intelligence-system-cpe-342-ml

Karena Thailand needs five models that together feed a Player LTV system. All five share one
submission file.

| Task | Problem | Type | Target | Metric |
|---|---|---|---|---|
| 1 | Anti-cheat pre-filter | Binary classification | `is_cheater` | F2 (macro) |
| 2 | Player segment | Multi-class (0 Casual, 1 Grinder, 2 Social, 3 Whale) | `segment` | F1 (macro) |
| 3 | 30-day spending | Regression, zero-inflated | `spending_30d` (THB) | NMAE = 1/(1+Σ\|y−ŷ\|/Σy) |
| 4 | Game title from screenshot | Image classification (230x120) | `class` 0–4 | Macro F1 |
| 5 | Account security | Unsupervised anomaly detection (test.csv only) | `is_anomaly` | F3 (macro) |

**Leaderboard score** = (Σ wᵢSᵢ) / Σ wᵢ, range [0, 1]. The weights wᵢ are not published.
Baselines (public LB): ground 0.68 (LogReg / RF / CNN / IsolationForest), challenge 0.75.

**Grading:** Kaggle leaderboard 70%, technical report 20%, source code 10%. Above baseline the grade is at
least α% and rises with Φ(t), where t is the z-score among above-baseline teams (the exact formula is
an image the PDF export did not keep). Below baseline the grade is α·(score/baseline).

**Timeline:** released 2 Nov · Kaggle closes **24 Nov** · report + code due **30 Nov** (LEB2).

**Rules:** use provided data only; train on train+val; never train on test (no pseudo-labeling);
pretrained models for Task 4 only; no AutoML (Optuna etc. are fine); no sharing code or predictions
outside the team.

## Repository layout

```
.
├── README.md
├── requirements.txt / pyproject.toml   # deps; `pip install -e .` makes `common` importable
├── download_data.ps1                   # Kaggle CLI download (alternative to the Google Drive zip)
├── make_submission.py                  # merges taskN/outputs/final_pred.csv -> submission.csv, validates
├── common/                             # shared code, imported by every task
│   ├── paths.py                        #   data/output locations (auto-detects data layout; CPE342_DATA override)
│   ├── metrics.py                      #   exact competition metrics + F-beta threshold search
│   └── utils.py                        #   seed=42, save/load OOF+test preds, write final_pred.csv
├── task1/ … task5/
│   ├── taskN.ipynb                     #   final, clean, runs top-to-bottom
│   ├── experiments.md                  #   date | change | CV | LB | notes
│   ├── src/                            #   optional task-specific helpers
│   └── outputs/                        #   oof_<model>.npy, test_<model>.npy (git-ignored), final_pred.csv
├── scripts/inspect_ids.py              # read-only id-mapping check
├── tests/                              # pytest: metrics + submission validation
├── submissions/log.md                  # every Kaggle upload and its score (CSVs git-ignored)
├── report/                             # technical report outline + figures
├── references/                         # instruction PDF
└── data/                               # git-ignored, see "Data" below
```

## Data

Unzip the course data (`cpe342-kaggle-main`) so `data/` looks like this:

```
data/
├── sample_submission.csv                 # 25,889 rows; only 4 example rows filled, rest NaN
├── 01_binary_classification/             # train.csv 104,000x33 · test.csv 25,889x32 · datacard.txt
├── 02_multiclass_classification/         # train.csv 104,000x46 · test.csv 25,889x45
├── 03_regression/                        # train.csv 104,000x34 · test.csv 25,889x33
├── 04_image_classification/              # datacard.txt + README.md (link to images, separate download)
│   └── data/{train,test}/{id}.jpg        #   expected location of the images per the datacard
└── 05_anomaly_detection/                 # test.csv 25,889x123 only (no train, no labels)
```

Datacards are `.txt`. `common.paths.task_data(n)` and `paths.datacard(n)` resolve the `0N_*` folders.
A plain `data/taskN/` layout from the Kaggle CLI also works.

## Setup

```powershell
pip install -r requirements.txt
pip install -e .                 # optional; notebooks also find `common` on their own
python scripts/inspect_ids.py    # re-checks the id-mapping assumptions below
python -m pytest -q
```

Colab/Kaggle (Task 4 GPU): clone the repo, set `os.environ["CPE342_DATA"] = "/path/to/data"`
before importing `common`.

## Workflow

1. Read `data/taskN/datacard.csv` before writing any code for a task.
2. Every experiment: K-fold CV with the metric from `common.metrics`, all preprocessing fit inside
   folds, then `utils.save_preds(task, name, oof, test)`. Log it in `taskN/experiments.md`.
3. Choose and ensemble models from saved OOF files. Write
   `utils.write_final_pred(task, test["player_id"], preds)`.
4. Run `python make_submission.py` (add `--fill-missing` to put a constant 0 in unfinished tasks),
   upload, and record the result in `submissions/log.md`.

## ID mapping: task test ids -> sample_submission rows

**Rule: row k (0-based) of each task's `test.csv` goes to submission id `ANS{k+1:05d}`.**

Evidence (`python scripts/inspect_ids.py`, 2026-11-08):

| Task | test file | id column | rows | unique | ids found in sample ids |
|---|---|---|---|---|---|
| 1 | 01_binary_classification/test.csv | `player_id` (P######) | 25,889 | 25,889 | 0% |
| 2 | 02_multiclass_classification/test.csv | `player_id` | 25,889 | 25,889 | 0% |
| 3 | 03_regression/test.csv | `player_id` | 25,889 | 25,889 | 0% |
| 4 | 04_image_classification/test.csv | `id` (per datacard) | not downloaded yet | | |
| 5 | 05_anomaly_detection/test.csv | `player_id` | 25,889 | 25,889 | 0% |

- No test id appears in `sample_submission.csv`, so there is no key to join on.
- The tasks score **different players**: any two test sets share only about 5,200 of 25,889
  player_ids, and never in the same row. One row of the submission is therefore not one player.
  It is "row k of each task's test set".
- Every test set has exactly 25,889 rows, the same as the submission. That makes position the only
  consistent mapping.

This is **inferred, not confirmed by the organisers**. Confirm it with a TA, or with an early
submission that has one task filled in. Do not sort, filter or deduplicate `test.csv` before
predicting. Notebooks write `final_pred.csv` keyed by `player_id` in any order; `make_submission.py`
checks the ids exactly match that task's `test.csv` and reorders them by file order.

## Conflicts between sources (checked 2026-11-08)

When sources disagree, follow the actual data files, then the Kaggle page, then the instruction PDF.

| Topic | Kaggle page | Datacard / files | Instruction PDF | We use |
|---|---|---|---|---|
| Task 1/5 metric | F2 / F3 **(Macro)** | – | plain F2 / F3 formula | **macro**. `metrics.f2/f3` default to macro, and binary is kept for logging. Macro also scores class 0, so false positives cost more. |
| Task 2 metric | F1 (Macro) | – | "F1" | macro |
| Task 4 class ids | 0 PUBE, 1 Free Fried, 2 FiveN, 3 Roblock, 4 7-Eleven | datacard: 0 7-11 Knights, 1 FiveN, 2 Free Fried, 3 PUBE, 4 Roblock. **train.csv holds strings** (`7-11k`, `fiveN`, `freefried`, `pube`, `roblock`) | strings (`freefried`…) | **Unresolved: task 4 scores 0 on the LB for every team** (public LB scores above 0.70 are consistent with that). The datacard order (v1, 0.60044) and the page order (v2, 0.59981) score the same. Open questions: integers vs strings (the PDF sample shows `freefried`), and which order. Ignore task 4 until the TA answers; submit the datacard order meanwhile. |
| Separator | "pipe-separated `id\|task1\|…`" | `sample_submission.csv` uses commas (Kaggle-hosted copy too) | – | **comma**. The page text is wrong. |
| Task 4 image size | 200×100 | 230×120 | 230×120 | read from the files |
| ID mapping | "All test samples share the same ID range" (sentence cut off) | 25,889 rows per test.csv, disjoint players | – | row order (see above) |

Confirm with a TA: the Task 4 class encoding and the id mapping.

**Kaggle vs Drive data (checked 2026-11-08 via the Kaggle CLI):** Kaggle hosts the same 15 files as
the `cpe342-kaggle-main` Drive zip, with identical sizes. SHA-256 matches for `sample_submission.csv`,
`01_binary_classification/test.csv` and `04_image_classification/README.md`. Kaggle has no Task 4
images either: they are only at the Google Drive link in `data/04_image_classification/README.md`.

**Leaderboard snapshot (2026-11-08):** 12 teams. Kaggle's "Baseline" entry scores **0.70514** public;
the top team is 0.76297.
