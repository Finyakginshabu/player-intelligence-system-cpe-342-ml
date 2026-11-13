#!/usr/bin/env bash
# Round R5 (2026-11-13): scene-balanced sampling (rare train scenes drawn more often; k = 40 CLIP scenes per class).
cd "$(dirname "$0")"
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv hard --folds 4 --no-test --grad-ckpt --drop-path 0.2 > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
S="--backbone convnext_small.fb_in22k_ft_in1k --epochs 6 --batch 48 --lr 2e-4 --scene-k 40"
run h_small_sb05 $S --scene-balance 0.5
run h_small_sb10 $S --scene-balance 1.0
run h_small_ema   --backbone convnext_small.fb_in22k_ft_in1k --epochs 6 --batch 48 --lr 2e-4 --ema 0.998
# Unused-fold confirmation (audit item 2): hard fold 3 was never used for any choice.
run3() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv hard --folds 3 --no-test --grad-ckpt --drop-path 0.2 > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run3 h3_small  --backbone convnext_small.fb_in22k_ft_in1k --epochs 6 --batch 48 --lr 2e-4
run3 h3_v2base --backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4
