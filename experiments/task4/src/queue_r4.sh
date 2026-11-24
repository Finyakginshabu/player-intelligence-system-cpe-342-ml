#!/usr/bin/env bash
# Round R4 (2026-11-13): confirm the S + V2-B blend gain with a 2nd V2-B seed; try more diverse families for the blend.
cd "$(dirname "$0")"
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv hard --folds 4 --no-test --grad-ckpt --drop-path 0.2 > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run h_v2base_s7 --backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4 --seed 7
run h_effv2m    --backbone tf_efficientnetv2_m.in21k_ft_in1k --epochs 6 --batch 32 --lr 3e-4
run h_caformer  --backbone caformer_s36.sail_in22k_ft_in1k --epochs 6 --batch 32 --lr 1e-4
