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

## 2026-11-10: LB proof that the task-4 key is broken
| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-10 | GPU ConvNeXt fine-tune (`final_pred_convnext_ft.csv`), **test.csv row order** (confirmed by the TA's mapping: ANS00001 = scyd5lz.jpg), **data-card ints** (confirmed visually from 50 train images per class) | 0.9835 | 0.75820 (Δ +0.0012 vs v16) | Task 4 ≈ 0.22 = chance. Earlier: the frozen model in the same order scored the same under the data-card (v1) and page (v2) numberings, which no working key allows (one of them should have been ≈ 0.6 higher on task 4). The task-4 answer key / scorer on Kaggle does not match these test images → TA |

## 2026-11-11: professor's ruling — task 4 counts as 0
The professor did not accept that the Kaggle task-4 key is broken, so task 4 counts as 0 for us. No further work on task 4; keep the ConvNeXt fine-tune column (CV 0.9835) in every file only so the submission stays valid. Report the CV result and the chance-level LB factually.

## 2026-11-12: test.csv fixed by the TA → task 4 scores
The TA's refreshed zip (Drive, 2026-11-11) has identical images but a new `test.csv` = `id` (ANS00001..), `file_name`, `label`, in a **different order** from the old test.csv (ANS00001 = sjqyfwc.jpg, not scyd5lz.jpg). Train labels are integers 0=7-11k … 4=roblock (same as our encoding). So the old chance-level LB was the wrong row mapping in the old test.csv, not our model. `make_submission.py` now maps task 4 via `file_name`.

| Date | Change | CV score | LB score | Notes |
|---|---|---|---|---|
| 2026-11-12 | ConvNeXt FT predictions (unchanged) remapped through the new test.csv | 0.9835 | 0.84268 (sub_v22, +0.12656 vs v21 with t2/t5 also rerun) | implied t4+t5 ≈ 1.84 → t4 roughly 0.85–0.98. `task4_nb.py` still expects the old `id,class` csv format (needs a small update to rerun) |
| 2026-11-12 | Diagnostic (sub_v23, task 5 zeroed) separates t4 from t5 | 0.9835 | – | **t4 ≈ 0.85 on the LB** (t5 = 0.989) → CV 0.98 vs LB 0.85: the train/test domain shift (adv. AUC 0.77, darker/noisier test) costs ~0.13. Next: shift-robust training (dark/noise/JPEG augmentation, more backbones, TTA), validated on degraded val images |

## Round R (2026-11-12): robustness to unseen test scenes (`src/t4data.py`, `src/scene_cv.py`, `src/ft.py`)
Goal: close the CV 0.98 → LB ≈ 0.85 gap. Rule 2 holds throughout: test images only pass through trained models.

**Why random CV misses the gap.** Adversarial check on frozen CLIP ViT-L/14 embeddings: train-vs-test AUC 0.796; 12.6% of
test rows look test-only (p > 0.9) vs 0.4% of train. The most test-like frames are a scene type barely in train (large
crowds of uniformed characters at night). Model confidence drops as test-likeness rises (0.82 → 0.76). The distortions
(rotation, corner borders, noise level) are the same in both splits (measured on 6,000 images each), so it is
content shift, not a distortion shift.

**New validation.** Scenes = k-means clusters of CLIP embeddings inside each class. `scene` folds (k = 10, grouped) and
`hard` folds (k = 5, fold i holds out the i-th largest region of every class). Frozen CLIP-L linear probe: random 0.881,
scene 0.744, hard 0.667. CLIP zero-shot with the real game names is useless (macro F1 0.13). Near-black frames
(max-channel mean < 25: 1.6% of train, 89% 7-11k; 0.8% of test) carry no game content: holding out the cluster that
contains them makes 7-11k recall collapse (fold 4: 0.61), which is not what test does, so models are compared on
**non-dark** held-out rows.

| Date | Model (6 epochs, scene/hard fold, no test preds) | held-out macro F1 all / non-dark | Notes |
|---|---|---|---|
| 2026-11-12 | ConvNeXt-T in22k, current recipe, scene folds 0, 1 | 0.964 / 0.965, 0.904 / 0.947 | baseline |
| 2026-11-12 | ConvNeXt-T, hard folds 2, 4 | 0.965 / 0.966, 0.844 / 0.926 | fold 4 = benchmark |
| 2026-11-12 | ConvNeXt-T + heavy aug (mixup/cutmix 0.5, erase 0.3, gray 0.2, hue 0.5, invert 0.2, scale ≥ 0.5), hard fold 4 | 0.844 / 0.924 | no gain → heavier augmentation is not the lever |
| 2026-11-12 | **ConvNeXt-S in22k**, hard fold 4 | 0.862 / **0.949** | +0.023 non-dark vs T: capacity helps |
| 2026-11-12 | CLIP ViT-B/16 openai (lr 5e-5, layer decay 0.7), hard fold 4 | 0.830 / 0.912 | worse |
| 2026-11-12 | DINOv3 ViT-B/16 (lr 5e-5, layer decay 0.75), hard fold 4 | 0.844 / 0.925 | = T |
| 2026-11-12 | mean of the 5 fold-4 models above | 0.858 / – | averaging < best single (S) |
| 2026-11-12 | Class-prior correction p / prior^τ on held-out OOF (equal-mix re-weighting) | τ 0 best or flat | no gain → not used |
| 2026-11-12 | ConvNeXt-B in22k (lr 1e-4), hard fold 4 | 0.844 / 0.9295 | bigger than S but worse |
| 2026-11-12 | ConvNeXt-B CLIP LAION-2B ft in12k/in1k (lr 1e-4), hard fold 4 | 0.849 / 0.9315 | 2nd best; blend with S 0.9485 = S alone |

**Decision:** ConvNeXt-S in22k is the best on held-out scenes (also scene fold 1: 0.962 non-dark vs T 0.947). Final =
ConvNeXt-S trained on **all** 31,546 images (`--cv full`, 6 epochs, batch 48, lr 2e-4, drop-path 0.2), **4 seeds**
(42, 7, 123, 2026), probabilities averaged. Second candidate = that average 50/50 with one ConvNeXt-B CLIP model (seed 42).
Notebook: `task4_final.ipynb` (source `src/task4_final_nb.py`).

**Final runs (2026-11-12/13, RTX 5070, ≈ 22 min per Small seed, 47 min for Base CLIP).** Last-epoch train loss 0.401–0.403 for
all Small seeds, 0.406 for Base CLIP. On test: the 4 seeds agree pairwise on only 88.6–88.9% of rows (all 4 unanimous on
81.0%), with disagreements spread over every class pair → large seed variance on the unseen test scenes, which the
average reduces. Ensemble vs round-1 ConvNeXt-T (LB ≈ 0.85): 90.9% agreement, 2,350 rows changed. Mean max-prob 0.839
(round 1: 0.801). Test class mix (final): 7-11k 0.177, fiveN 0.254, freefried 0.145, pube 0.262, roblock 0.162 (train:
0.394 / 0.140 / 0.177 / 0.188 / 0.101; the round-1 model gives almost the same test mix, so this is the test set, not
the model). The blend with Base CLIP changes 2,241 rows vs the Small-only file.

| Date | File | Model | held-out (hard 4, non-dark) | LB |
|---|---|---|---|---|
| 2026-11-13 | `outputs/final_pred.csv` → `submissions/sub_v24_t4_small4.csv` | ConvNeXt-S × 4 seeds, all data | 0.949 (single seed) | **0.84731** (Δ +0.00463 vs v22 → t4 ≈ 0.873, +0.023 = the held-out gain) |
| 2026-11-13 | `outputs/pred_small4_baseclip.csv` → `submissions/sub_v25_t4_small4_clip.csv` | 0.5 · S × 4 + 0.5 · B-CLIP | 0.9485 (single seeds) | not uploaded yet |

## Round R2 (2026-11-13): push ConvNeXt-S further, judged on hard fold 4 non-dark
v24 confirmed the benchmark: held-out +0.023 (T → S) gave LB t4 +0.023. Queue `src/queue_r2.sh`, scorer `src/bench.py`.
Runs: second seed of the S recipe (seed noise of the benchmark), 10 epochs, DINOv3-distilled ConvNeXt-S (LVD-1689M),
ConvNeXt-S in12k, ConvNeXt-S 384-weights at 288×576 input, ConvNeXt-V2-B (FCMAE in22k). A change is kept only if it
beats S by more than the seed noise.

**R2 results (hard fold 4, non-dark; seed noise ±0.006: S seed 42 0.9492, seed 7 0.9435, 2-seed avg 0.9500):**

| Run | non-dark | vs S |
|---|---|---|
| S, 10 epochs | 0.9484 | = |
| S DINOv3 (LVD-1689M) | 0.9141 | worse |
| S in12k | 0.9286 | worse |
| S 384-weights at 288×576 | 0.9409 | worse |
| **ConvNeXt-V2-B FCMAE in22k** | **0.9502** | = alone, but **0.5·S(2 seeds) + 0.5·V2-B = 0.9577** (+0.008 > noise; agreement 96%), + S 10-ep 0.9587 |

Ensembling now helps (unlike S + B-CLIP / B-in22k): V2's FCMAE pretraining makes different errors. → R3 (`src/queue_r3.sh`):
V2-B on all images × 2 seeds, then a ConvNeXt-V2-L benchmark run.

**R3 results:** V2-B full-data seeds 42 and 7 (≈ 67 min each; test argmax agreement between seeds 0.893). ConvNeXt-V2-L
hard fold 4: 0.9442 alone; blends with it are lower (S + V2-L 0.9502, S + V2-B + V2-L 0.9545) → not used. Candidate
`outputs/pred_s4_v2b2.csv` = 0.5·S×4 + 0.5·V2-B×2 → `submissions/sub_v26_t4_s4_v2b2.csv` (1,524 rows differ from v24).

**2026-11-13: user rule — no upload until a file is expected to beat #1 (ML003 0.85787 ⇒ t4 ≈ 0.92).** v26 (expected ≈ 0.849)
is kept but not uploaded. R4 (`src/queue_r4.sh`): V2-B seed 7 on hard fold 4 (is the blend gain real?), EfficientNetV2-M,
CAFormer-S36. R5 (`src/queue_r5.sh`): new `ft.py --scene-balance α --scene-k 40` = sample train rows with weight
(scene size)^−α inside each class (200 CLIP scenes, sizes 7–611), aimed at the test's scene shift; train data only.
- 2026-11-13: idea "HUD/corner model" (game UI is scene-invariant) **dropped after looking at the frames**: every frame is a
  random crop of the game screen with rotation up to ≈ 30°, black corner borders, heavy noise and inverted/shifted colours,
  so UI elements have no fixed position and are mostly absent. Added an EMA run (decay 0.998) to R5 instead.

### Test-set EDA (2026-11-13; looking only: nothing fitted on test, no test label used or set by hand)
`outputs/eda_test_likeness.npy` = train-vs-test classifier on CLIP-L embeddings (EDA only, never a training weight).
The 2,500 most test-like test frames, k-means into 6 groups, nearest train images by CLIP similarity (typical test→train
top-1 similarity 0.979):

| Group | n | top-1 sim | labels of the 10 nearest train images (7-11k/fiveN/freefried/pube/roblock) | our preds | mean conf |
|---|---|---|---|---|---|
| 0 crowds in matching white uniforms, night | 410 | **0.940** | 0.16 / 0.43 / 0.28 / 0.10 / 0.04 | 0.24 / 0.75 / 0 / 0.01 / 0 | **0.69** |
| 2 same crowd scene | 357 | 0.966 | 0.04 / 0.44 / 0.33 / 0.16 / 0.02 | 0.13 / 0.76 / 0.03 / 0.04 / 0.03 | 0.78 |
| 3 same crowd scene | 465 | 0.965 | 0.18 / 0.18 / 0.47 / 0.13 / 0.05 | 0.35 / 0.55 / 0.03 / 0.06 / 0 | **0.65** |
| 1 | 102 | 0.961 | 0.85 / 0.11 / 0.03 / 0 / 0 | 0.74 / 0.25 / … | 0.92 |
| 4 blocky scenes, saturated single colours | 870 | 0.969 | 0.04 / 0.08 / 0.08 / 0.44 / 0.37 | … / 0.86 roblock | 0.85 |
| 5 same | 296 | 0.949 | 0.05 / 0.02 / 0.02 / 0.07 / 0.85 | … / 0.75 roblock | 0.82 |

**Finding:** ≈ 1,200 test frames (≈ 5% of test) show one event-like scene (large crowds in matching uniforms) that has no
close match in train: the nearest train images are ordinary street/fight scenes spread over 3 games, and the models split
these frames between two classes with low confidence. This is the main source of the CV 0.95 → LB 0.87 gap and is
largely out of reach for any train-only method; we must not label or steer these frames by hand. Other uncertain test
frames are very dark (near-black) frames. Consequence: t4 ≈ 0.92 (needed for #1) is unlikely; methods that push the
model towards game style rather than scene content (scene-balanced sampling, diverse backbones) are the only legal levers.

**R4 partial / audit (2026-11-13 11:30):** V2-B seed 7 on hard fold 4 = 0.9504 (seed 42: 0.9502). Blend gain replicates but
is smaller than first measured: S(2) + V2-B(s7) 0.9552, S(2) + V2-B(both) **0.9558** (+0.0058 over S(2) 0.9500); on all
rows (exact competition metric) 0.8640 → 0.8692 (+0.005). Note: V2-B(2 seeds) alone is also 0.9558 (all rows 0.8701), so
the gain comes from V2-B; mixing it with S is neutral on this fold but kept for diversity. Expected LB for v26 ≈ +0.001.
`task4_final.ipynb` now builds this blend as `outputs/final_pred.csv` (= v26 column, checked) and keeps the S-only column
(v24) as `outputs/pred_small4.csv`. Still to do before it is final: confirmation on an unused fold (selection bias:
≈ 20 variants have been compared on hard fold 4).
- R4 (2026-11-13): EfficientNetV2-M in21k 0.931 (worse; S+V+E 0.9524 < S+V 0.9558). CAFormer-S36 in22k **0.9506** alone (= V2-B),
  S+V+C 0.9567 (+0.0009 over S+V, within noise) → not adopted for now; a candidate if a later round needs more diversity.
  R5 started 11:40: scene-balance 0.5 / 1.0, EMA, then the unused-fold check (hard fold 3: S vs V2-B).
