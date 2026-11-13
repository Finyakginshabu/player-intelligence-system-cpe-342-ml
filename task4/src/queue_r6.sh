#!/usr/bin/env bash
# Round R6 (2026-11-13): TARGETED at the test scene shift (test EDA: ~1,200 test frames of a scene type absent from train).
# Small random crops force the model onto local game style (character models, textures, rendering) instead of scene
# layout; prediction averages the full frame + a 3x2 grid of training-size crops. Benchmark: hard fold 4 (non-dark),
# then the unused hard fold 3 for the winners.
cd "$(dirname "$0")"
PY=../../.venv-gpu/Scripts/python.exe
run() { f=$1; shift; echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv hard --folds $f --no-test --grad-ckpt --drop-path 0.2 > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
S="--backbone convnext_small.fb_in22k_ft_in1k --epochs 6 --batch 48 --lr 2e-4"
run 4 h_small_crop14 $S --scale-min 0.10 --scale-max 0.40 --tta-grid
run 4 h_small_crop26 $S --scale-min 0.20 --scale-max 0.60 --tta-grid
run 4 h_small_sb10_s7 $S --scene-k 40 --scene-balance 1.0 --seed 7
run 3 h3_small  $S
run 3 h3_small_sb10 $S --scene-k 40 --scene-balance 1.0
run 3 h3_v2base --backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4
run 4 h_small_ema $S --ema 0.998
