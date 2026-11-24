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
README.md
requirements.txt        Python packages
make_submission.py      merges the five final_pred.csv into submission.csv and validates
submission.csv
common/                 shared code: data paths, exact competition metrics, seeds, prediction I/O
task1/task1.ipynb
task2/task2.ipynb
task3/task3.ipynb
task4/task4.ipynb
      outputs/saved_probs/   test probabilities of our 43 training runs
      outputs/train_logs/    the training log of each of those runs
task5/task5.ipynb       
experiments/            experiment scripts, logs, the Kaggle upload log, old notebooks
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
├── 01_binary_classification/
├── 02_multiclass_classification/
├── 03_regression/
├── 04_image_classification/
│   └── data/{train,test}/*.jpg
└── 05_anomaly_detection/
```

### 3. Run the five notebooks
Open each `taskN/taskN.ipynb` and run all cells (or `jupyter nbconvert --to notebook --execute --inplace taskN/taskN.ipynb`).
Each notebook finds the repository root itself and writes `taskN/outputs/final_pred.csv`.

### 4. Build the submission
```bash
python make_submission.py
```
