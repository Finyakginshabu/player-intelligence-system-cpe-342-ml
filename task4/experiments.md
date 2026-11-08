# Task 4 experiments

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-08 | EDA only | – | – | see below |
| 2026-11-08 | **CPU fallback: frozen ConvNeXt-Tiny (torchvision IN-1k) embeddings @ native 120×230 + LogisticRegression C=0.1**, StandardScaler in-fold, 5-fold StratifiedGroupKFold (groups = identical 16×8 aHash) | macro F1 **0.8413** ± 0.0021 | – | per-class F1 7-11k 0.915 / fiveN 0.840 / freefried 0.821 / pube 0.807 / roblock 0.824. Extraction ~16 min CPU (`src/features.py`, cached), LR 151 s. **Chosen → `final_pred.csv`**. `oof_lr_C0.1` / `test_lr_C0.1` |
| 2026-11-08 | Same + `class_weight="balanced"` | 0.8332 ± 0.0043 | – | −0.008, rejected |
| 2026-11-08 | LB check: v1 (datacard encoding) 0.60044 vs v2 (Kaggle-page encoding) 0.59981, only task 4 changed | – | Δ −0.0006 | Neither encoding is the scoring one, or task-4 rows are misaligned: task 4 ≈ chance on LB either way. See `submissions/log.md`. Ask TA |
| 2026-11-08 | Adversarial validation train vs test (ConvNeXt embeddings, LR) | AUC **0.768** | – | Real domain shift: test darker (mean 0.41 vs 0.45), more very dark/noisy frames, model confidence 0.81 vs 0.89 OOF. Both splits heavily augmented (rotation + black borders, colour inversion, noise). Fine-tuning with strong photometric/noise aug should help |
| 2026-11-08 | GPU notebook `task4_kaggle.ipynb` written (timm ConvNeXt-Tiny IN-22k, 224×448, 8 epochs, label smoothing, class-weighted CE, TTA ×3) | not run yet | – | needs a CUDA runtime (Kaggle/Colab, or local RTX 5070 after installing a CUDA build of torch) |

| 2026-11-08 | **GPU fine-tune** `task4_kaggle.ipynb` on the local RTX 5070 (`.venv-gpu`, torch 2.14.1+cu130, timm ConvNeXt-Tiny IN-22k→1k, 224×448, 8 epochs, batch 48, AdamW + one-cycle, label smoothing 0.1, class-weighted CE; augmentation: RRC, rotation ±8° with black fill, colour jitter skewed darker, 5% inversion, Gaussian noise, blur; TTA ×3; same StratifiedGroupKFold) | macro F1 **0.9835 ± 0.0013** (folds 0.9853 / 0.9832 / 0.9827 / 0.9847 / 0.9816, OOF with TTA) | – (not uploaded; the task 4 format question is open) | **+0.142 vs the CPU LR.** Per class 7-11k 0.991 / fiveN 0.983 / freefried 0.987 / pube 0.984 / roblock 0.973. 80 s/epoch, 6.2 GB, 67 min total. Outputs `oof_convnext_ft.npy`, `test_convnext_ft.npy`, **`final_pred_convnext_ft.csv`** (`final_pred.csv` is untouched; promote it with `make_submission.py --replace task4=task4/outputs/final_pred_convnext_ft.csv` once the TA confirms the format). Best epoch chosen on the val fold, so CV is slightly optimistic |

### GPU fine-tune notes (2026-11-08)
- **The test set looks different from train, and brightness is not the cause.** Mean max-prob is 0.80 on test vs 0.89 OOF, and it is
  lower in *every* brightness band (0.74–0.82 vs 0.87–0.90). 15.7% of test rows have max-prob < 0.6.
- **The predicted test class mix differs from train**: 7-11k 18.6% (train 39.4%), fiveN 24.0% (14.0%), freefried 14.3% (17.7%),
  pube 27.2% (18.8%), roblock 15.9% (10.1%). This holds in every brightness band. Either the test priors really differ
  (closer to balanced) or the test images differ in some other way. Expect the LB score below CV 0.98.
- Fine-tuned and CPU predictions agree on only 67% of test rows, so the CPU baseline was unreliable on test.
- Fitting anything on test images (pseudo-labels, domain adaptation) is forbidden (rule 2). Options left: stronger or
  different augmentation, a second backbone for an ensemble, or checking what makes test frames different (sources/videos).
- Local GPU run: call the venv's own `python -m nbconvert --ExecutePreprocessor.kernel_name=t4gpu`. `python -m jupyter`
  dispatches to the system nbconvert, which starts the CPU-only system kernel.

### Baseline notes (2026-11-08, `task4.ipynb`, source `src/task4_nb.py`)
- Main confusions: freefried↔pube (444 / 466), 7-11k→freefried/pube (284 / 293), pube→7-11k (362).
- **Test prediction distribution ≠ train label distribution:** predicted 7-11k 25.8% (train 39.4%),
  pube 27.2% (18.8%), fiveN 17.9% (14.0%), roblock 14.1% (10.1%), freefried 15.0% (17.7%). Either the test class
  priors really differ, or the frozen features suffer a domain shift on test images. Check: adversarial validation
  on embeddings (train vs test), compare image stats of test vs train. If priors differ, the first task-4 LB score
  will be well below CV.
- **Label mapping is assumed** (datacard = alphabetical): `{7-11k:0, fiveN:1, freefried:2, pube:3, roblock:4}`.
  The first LB score confirms it: a wrong mapping would put task 4 near chance level.

Next: run `task4_kaggle.ipynb` on a GPU (expected large gain from fine-tuning), then ensemble with the
embedding LR; investigate the test distribution shift.

## EDA, 2026-11-08

Data: `train.csv` 31,546 × 2 (`id`, `class` as **string**), `test.csv` 25,889 (`id`, empty `class`).
Images `data/04_image_classification/data/{train,test}/{id}.jpg`, from the course Drive zip (607 MB).

### Key insights
1. **Labels are strings, not integers.** The submission needs integers 0–4; the datacard order equals alphabetical
   order (`LabelEncoder`), so we use it. The Kaggle page lists a different order (see README "Conflicts").
2. **Imbalanced classes:** 7-11k 39.4%, pube 18.8%, freefried 17.7%, fiveN 14.0%, roblock 10.1%. Macro F1 weights
   roblock equally with 7-11k.
3. **All images are exactly 230×120** in both splits (the Kaggle page's 200×100 is wrong).
4. **Few near-duplicates:** 0.68% of train rows share an exact 16×8 average hash with another (31,411 unique hashes
   / 31,546), 9 duplicate groups carry mixed labels (label noise or generic frames like loading screens), and only
   0.05% of test hashes appear in train. Group folds matter little for CV inflation here; keep them anyway.
5. **Colour statistics differ by title:** 7-11k and roblock are brighter (mean RGB ≈ 0.47–0.48) than fiveN (0.41),
   pube and freefried (0.43). Global colour alone is a weak but real cue; the CNN features capture far more.
