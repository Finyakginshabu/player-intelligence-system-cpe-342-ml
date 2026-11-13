#!/usr/bin/env bash
# Round R2 (2026-11-13): improvements over ConvNeXt-S, all on hard fold 4 (the benchmark that matched the LB gain 1:1).
cd "$(dirname "$0")"
PY=../../.venv-gpu/Scripts/python.exe
S="--cv hard --folds 4 --no-test --grad-ckpt --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" $S > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run h_small_s7     --backbone convnext_small.fb_in22k_ft_in1k --epochs 6  --batch 48 --lr 2e-4 --seed 7
run h_small_e10    --backbone convnext_small.fb_in22k_ft_in1k --epochs 10 --batch 48 --lr 2e-4
run h_small_dinov3 --backbone convnext_small.dinov3_lvd1689m  --epochs 6  --batch 48 --lr 2e-4
run h_small_in12k  --backbone convnext_small.in12k_ft_in1k    --epochs 6  --batch 48 --lr 2e-4
run h_small384_hr  --backbone convnext_small.fb_in22k_ft_in1k_384 --epochs 6 --batch 32 --lr 1.5e-4 --h 288 --w 576
run h_v2base       --backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4
