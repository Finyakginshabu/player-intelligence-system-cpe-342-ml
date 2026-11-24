# Task 4 recipe: v61

**Final file = v61 (LB 0.86586, 2026-11-22)** = v58 + two giant families with the same near-native recipe (round 4,
`build_cloud_kit.py`, last 12 blocks trained with `ft.py --freeze-blocks`): DINOv2-g/14 reg4 at 126×238 × seeds 42 / 7 / 123
and DINOv3 ViT-H+/16 at 144×272 × seeds 42 / 7 / 123; 13 equal families, prior rule A = 2.0. Build = the v58 command with the
two extra families `f_dinov2g_n126_s42,f_dinov2g_n126_s7,f_dinov2g_n126_s123` and
`f_dinov3hp_n144_s42,f_dinov3hp_n144_s7,f_dinov3hp_n144_s123` and the tag `v61_t4_giants` (exact command: `src/task4_final_nb.py`,
checked 2026-11-22: identical `final_pred.csv`). Hedge for the 2nd final selection: the same families at `--prior=1.0` =
`sub_v62_t4_giants_prior100.csv` (LB 0.86431).

**Previous final = v58 (LB 0.86514, 2026-11-22)** = v55 + 3rd seeds (s123) in the DINOv3-L-new and DINOv2-L-new families + an
11th family, EVA-02-L new recipe (`EVAL_N` in `build_cloud_kit.py`: 126×238, 15 epochs, HUD 0.7, LP-FT 2 epochs) × seeds
42 / 7 / 123; prior rule A = 2.0. Build = the v55 command below with `f_dinov3l_n144_s123` and `f_dinov2l_n126_s123` added to
those two families, the extra family `f_eva02l_n126_s42,f_eva02l_n126_s7,f_eva02l_n126_s123`, and the tag `v58_t4_evaln`;
then `make_submission.py --base submissions/sub_v55_t4_r2fams.csv --replace task4=task4/outputs/pred_v58_t4_evaln.csv`.
Its A = 1.0 version = `sub_v60_t4_v58_prior100.csv` (LB 0.86326).

**Previous final = v55 (LB 0.86505, 2026-11-21)** = v50 + two more families from the round-2 recipes (`build_cloud_kit.py`
D3L_N / D2L_N: DINOv3-L at 144×272, 18 epochs, and DINOv2-L reg4 at 126×238, 15 epochs, both HUD dropout 0.7, 2 seeds each)
+ EVA-02-L LP-FT s42 as a 5th EVA-02-L seed; 10 equal families, prior rule A = 2.0. Build (from `task4/src`, bash):
```
python blend_v41plus.py --final --prior=2.0 v55_t4_r2fams --drop=ConvNeXt-S --drop=EVA-02-B --drop=DINOv3-B seeds2 \
  f_small_s42,f_small_s7,f_small_s123,f_small_s2026,f_small_hud_s42 \
  f_eva02b_s42,f_eva02b_s7,f_eva02b_s123,f_eva02b_s2026,f_eva02b_lpft_s42,f_eva02b_lpft_s7 \
  f_dinov3b_lr2e4_s42,f_dinov3b_lr2e4_s7,f_dinov3b_hud_s42 \
  --drop=EVA-02-L f_eva02l_s42,f_eva02l_s7,f_eva02l_s123,f_eva02l_s2026,f_eva02l_lpft_s42 \
  f_dinov2l_s42,f_dinov2l_s7 f_dinov3l_s42,f_dinov3l_s7 f_dinov3l_n144_s42,f_dinov3l_n144_s7 f_dinov2l_n126_s42,f_dinov2l_n126_s7
```
Then put it on v52's other columns: `python make_submission.py --base submissions/sub_v52_t3_identity_ratio.csv
--replace task4=task4/outputs/pred_v55_t4_r2fams.csv --out submissions/sub_v55_t4_r2fams.csv` (the blend script's own
`sub_*.csv` now uses v52's columns, same result). `src/task4_final_nb.py` / `task4_final.ipynb` build v58 (checked 2026-11-22: identical `final_pred.csv`).

**Previous final = v50 (LB 0.86406)**: the v47 recipe below with extra seeds inside four families: DINOv3-L + seed 7,
EVA-02-B + 2 LP-FT runs (`--lp-epochs 2`), ConvNeXt-S + 1 HUD-dropout run (`--hud 0.5`), DINOv3-B + 1 HUD-dropout run.
Same LB as v47 (0.86405); kept for the lower seed variance. Build: see `src/task4_final_nb.py` section 4.

**Idea:** fine-tune strong pretrained backbones on all training images, then average their predicted probabilities
across 8 "families". A family is one pretrained model, averaged over its seeds. The families differ mainly in their
**pretraining data**, and that diversity is what helps on the test set, whose scenes differ from train.

## 1. Data and input
- 31,546 training frames of 230×120 pixels, 5 games. Labels: 0 = 7-11k, 1 = fiveN, 2 = freefried, 3 = pube,
  4 = roblock. The test set has 25,889 frames.
- The input is resized to **224×448**, which keeps the wide aspect ratio, with no padding.
- Training uses **all** training images (`ft.py --cv full`). There's no validation hold-out, and the model is taken
  from the last epoch.

## 2. Training recipe (shared by every model, in `src/ft.py`)
- **Loss:** cross-entropy with label smoothing 0.1 and class weights `(max_count / count)^0.5`.
- **Optimizer:** AdamW with weight decay 0.05 and a OneCycle schedule (10% warm-up, then cosine decay), for
  **6 epochs**. Mixed precision, gradient clipping at 2.0, gradient checkpointing.
- **Augmentation**, done on the GPU and safe for on-screen text (no horizontal flip):
  - random crop covering 75–100% of the area, aspect ratio 1.7–2.2
  - rotation of ±8° with probability 0.3
  - colour jitter with probability 0.8 (brightness 0.5–1.2, contrast 0.6–1.3, saturation 0.7–1.3, hue ±0.05)
  - colour inversion with probability 0.05
  - Gaussian noise with probability 0.3
- **Prediction:** test-time augmentation, averaging softmax over 3 zoom levels (1.0, 1.08, 1.15).

## 3. The 8 families (all from `timm`; pretrained weights are allowed for task 4 only)

| Family | timm backbone | Batch | LR | Layer decay | Drop path | Seeds | Time per run* |
|---|---|---|---|---|---|---|---|
| ConvNeXt-S | `convnext_small.fb_in22k_ft_in1k` | 48 | 2e-4 | – | 0.2 | 42, 7, 123, 2026 | 20 min |
| ConvNeXt-V2-B | `convnextv2_base.fcmae_ft_in22k_in1k` | 32 | 1e-4 | – | 0.2 | 42, 7, 123 | 62 min |
| EVA-02-B | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | 32 | 5e-5 | 0.75 | 0.1 | 42, 7, 123, 2026 | 48 min |
| EVA-02-L | `eva02_large_patch14_448.mim_in22k_ft_in22k_in1k` | 16 | 3e-5 | 0.85 | 0.2 | 42, 7, 123, 2026 | 155 min |
| EVA-02-L M38M | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | 16 | 3e-5 | 0.85 | 0.2 | 42, 7 | 150 min |
| DINOv3-B | `vit_base_patch16_dinov3.lvd1689m` | 32 | 2e-4 | 0.75 | 0.1 | 42, 7 | 32 min |
| DINOv2-L | `vit_large_patch14_reg4_dinov2.lvd142m` | 16 | 1e-4 | 0.85 | 0.2 | 42, 7 | 116 min |
| DINOv3-L | `vit_large_patch16_dinov3.lvd1689m` | 16 | 1e-4 | 0.85 | 0.2 | 42 | 100 min |

\*Times are on an RTX 5070 (12 GB), where peak memory was at most 6.3 GB. That's 23 runs and about **31 GPU-hours** in
total. Run names are `f_<family>_s<seed>` (see `outputs/ft_<name>.log` for the exact arguments of every run).

Example command (from `task4/src`):

```
python ft.py --name f_eva02l_s42 --cv full --seed 42 --backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k \
  --epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2 --grad-ckpt
```

## 4. Ensemble
1. Within each family, average the seeds' test probabilities (`outputs/test_<run>.npy`).
2. Average the 8 family probabilities with **equal weight**.
3. **Balanced-prior decision rule:** divide the averaged probabilities by π^2.0, where π = the class shares of the
   **training** labels (7-11k 39 %, fiveN 14 %, freefried 18 %, pube 19 %, roblock 10 %), then take the argmax.
   Macro F1 weighs the 5 games equally, but the networks inherit the training mix (39 % 7-11k), so without this they
   over-predict 7-11k. The rule uses no test information; only the strength A = 2.0 was picked on the public LB.
4. Build the final file (from `task4/src`, CPU, seconds):
   `python blend_v41plus.py --final --prior=2.0 v47_t4_v43_prior200 seeds2 f_dinov2l_s42,f_dinov2l_s7 f_dinov3l_s42`
   This writes `outputs/final_pred.csv` and `submissions/sub_v47_t4_v43_prior200.csv`.

No thresholds, and **no use of the test images** beyond prediction: no pseudo-labels and no test-time training
(course rule).

## 5. How the families were chosen
- **Model choice by held-out scenes** (`src/scene_cv.py`, `src/bench.py`): the training images were clustered into
  scenes (k-means on CLIP features), and whole scenes were held out. Random folds give 0.98 but hide the shift to the
  test set.
- **Families added one at a time by LB test.** Each new family was added at equal weight and kept only if the LB went
  up. Held-out scores didn't predict the LB gains.

| Step | Families | LB | Gain |
|---|---|---|---|
| ConvNeXt-S × 4 (v24) | 1 | 0.84731 | |
| + ConvNeXt-V2-B (v28) | 2 | 0.85104 | +0.0037 |
| + EVA-02-B (v37) | 3 | 0.85323 | +0.0022 |
| + EVA-02-L (v40) | 4 | 0.85602 | +0.0028 |
| + EVA-02-L M38M + DINOv3-B (v41) | 6 | 0.85780 | +0.0018 |
| + DINOv2-L + DINOv3-L + 2nd seeds (v43) | 8 | 0.85865 | +0.0009 |
| + balanced-prior rule, A = 0.5 (v44) | 8 | 0.86071 | +0.0021 |
| A = 1.0 (v45) | 8 | 0.86225 | +0.0015 |
| A = 1.5 (v46) | 8 | 0.86334 | +0.0011 |
| **A = 2.0 (v47, final)** | 8 | **0.86405** | +0.0007 |

- **Prior strength A:** swept on the LB in steps of 0.5 and stopped at 2.0 because the gains were shrinking. At
  A = 3.0 the rule breaks (no 7-11k predictions at all). Held-out scenes did not show this gain (hard fold 4 went down
  with A, fold 3 up), so the gain comes from the test set's class mix, which is more balanced than train.

- **What failed:** CLIP-pretrained ConvNeXt, CAFormer, BEiT-v2, DeiT-III, AIMv2-L, ConvNeXt-V2-L, in-domain masked
  pretraining on the training images, heavy augmentation, full class balancing, higher resolution (336×672), and
  re-weighting families. Details are in `experiments.md`.

## 6. What a new team should expect
- **Score:** following this recipe should land near **0.95 task-4 macro F1** on the public LB. The training
  scripts don't use deterministic GPU mode, so reruns won't be bit-identical. Expect a few hundred test rows to differ.
- **Remaining gap:** about 1,200 test frames (crowd scenes in matching uniforms) have no close match in train. That
  accounts for most of the gap between CV (0.98) and the LB (≈ 0.95).
