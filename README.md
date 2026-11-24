# Player Intelligence System (CPE342 Machine Learning, KMUTT 1/2026), team ML008

Kaggle competition: https://www.kaggle.com/competitions/player-intelligence-system-cpe-342-ml

Five models for Karena's Player LTV system share one submission file. This repository holds the **final code** that
produced our final Kaggle file (best public LB **0.86586**), one folder per task. Everything we tried along the way is
kept separately in [`experiments/`](experiments/).

| Task | Problem | Metric (as scored by Kaggle) | Final model | CV |
|---|---|---|---|---|
| 1 | Anti-cheat pre-filter (`is_cheater`) | F2 on the cheater class | LightGBM on 15 features × 5 seeds, threshold 0.159 tuned on OOF | 0.8379 |
| 2 | Player segment (4 classes) | macro F1 | Naive Bayes with smoothed per-class densities + per-class weights | 0.7553 |
| 3 | 30-day spending (THB) | NMAE = 1 / (1 + Σ\|y−ŷ\| / Σy) | Identity-filled features + two-stage conditional median on the ratio to the player's history, 3 seeds | 0.7919 |
| 4 | Game title from a frame (5 classes) | macro F1 | 13 fine-tuned pretrained backbones (43 runs), equal-weight blend, balanced-prior decision rule | LB ≈ 0.95 |
| 5 | Account anomalies (no labels) | F3 on the anomaly class | 8 anomaly types found by unsupervised analysis of the test features; only typed rows flagged | LB ≈ 0.99 |

## Repository layout

```
README.md               this file (how to run)
requirements.txt        Python packages
make_submission.py      merges the five taskN/outputs/final_pred.csv into submission.csv and validates it
submission.csv          our final Kaggle file (v61, public LB 0.86586), as rebuilt by make_submission.py
common/                 shared code: data paths, exact competition metrics, seeds, prediction I/O
task1/task1.ipynb       final notebook  -> task1/outputs/final_pred.csv
task2/task2.ipynb       final notebook  -> task2/outputs/final_pred.csv   (helpers: task2/src/nb.py, features.py)
task3/task3.ipynb       final notebook  -> task3/outputs/final_pred.csv
task4/task4.ipynb       final notebook  -> task4/outputs/final_pred.csv   (training script: task4/src/ft.py)
      outputs/saved_probs/   test probabilities of our 43 training runs (used when TRAIN = False)
      outputs/train_logs/    the training log of each of those runs (exact arguments, loss per epoch)
task5/task5.ipynb       final notebook  -> task5/outputs/final_pred.csv
experiments/            everything else: experiment scripts, logs, the Kaggle upload log, old notebooks (see experiments/README.md)
```

## How to run

### 1. Environment
Python 3.12. LightGBM on macOS also needs OpenMP (`brew install libomp`).

```bash
pip install -r requirements.txt
```

### 2. Data
Tasks 1, 2, 3 and 5 come from Kaggle; task 4's images and csv files come from the TA's Google Drive zip (link in the
Kaggle file `04_image_classification/README.md`).

```bash
kaggle competitions download -c player-intelligence-system-cpe-342-ml -p data
unzip data/player-intelligence-system-cpe-342-ml.zip -d data && mv data/cpe342-kaggle-main/* data/
```

Expected layout:
```
data/
├── sample_submission.csv
├── 01_binary_classification/      train.csv  test.csv  datacard.txt
├── 02_multiclass_classification/  train.csv  test.csv  datacard.txt
├── 03_regression/                 train.csv  test.csv  datacard.txt
├── 04_image_classification/       train.csv  test.csv  datacard.txt
│   └── data/{train,test}/*.jpg    images (only needed to retrain task 4)
└── 05_anomaly_detection/          test.csv  datacard.txt     (no train set, no labels)
```

### 3. Run the five notebooks
Open each `taskN/taskN.ipynb` and run all cells (or `jupyter nbconvert --to notebook --execute --inplace taskN/taskN.ipynb`).
Each notebook finds the repository root itself and writes `taskN/outputs/final_pred.csv`.

| Notebook | Hardware | Time (Apple M4 CPU) |
|---|---|---|
| task1 | CPU | ≈ 5 min |
| task2 | CPU | ≈ 2 min |
| task3 | CPU | ≈ 15 min |
| task4, `TRAIN = False` (default) | CPU | seconds: blends the saved probabilities of our 43 runs |
| task4, `TRAIN = True` | NVIDIA GPU ≥ 16 GB + task-4 images | roughly 100 GPU-hours: retrains all 43 runs, then the same blend |
| task5 | CPU | < 1 min |

### 4. Build the submission
```bash
python make_submission.py        # -> submission.csv (25,889 rows: id, task1 … task5), validated
```

**Reproducibility (checked 2026-11-23 on the Mac above):** running the five notebooks and `make_submission.py` rebuilds
our final Kaggle file row for row (tasks 1, 2, 3, 5 retrained from scratch; task 4 from the saved probabilities). A task-4
retrain on a GPU is not bit-deterministic, so expect a few hundred of 25,889 task-4 rows to differ.

## Notes
- **Submission ids:** row k of each task's `test.csv` is submission row `ANS{k+1:05d}`. Each notebook keys
  `final_pred.csv` by the task's own test id; `make_submission.py` checks the ids against `test.csv` and puts them in
  file order. Never sort or filter `test.csv`.
- **Metrics:** Kaggle scores task 1 with binary F2 and task 5 with binary F3 (positive class only), confirmed on the
  leaderboard; `common.metrics.f2/f3` are called with `average="binary"`. Task weights are equal.
- **Course rules:** only the provided data; no training on test rows (task 5 is unsupervised by design: its detectors are
  fit on the test features, which is the task itself); pretrained weights for task 4 only; no AutoML; `player_id` is never
  used as a feature or rule.
