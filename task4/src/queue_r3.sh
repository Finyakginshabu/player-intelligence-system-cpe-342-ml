#!/usr/bin/env bash
# Round R3 (2026-11-13): ConvNeXt-V2-B blends with ConvNeXt-S (+0.008 on hard fold 4) -> full-data seeds; then V2-L benchmark.
cd "$(dirname "$0")"
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt --drop-path 0.2 > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
V2B="--backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4"
run f_v2base_s42 --cv full --seed 42 $V2B
run f_v2base_s7  --cv full --seed 7  $V2B
run h_v2large --cv hard --folds 4 --no-test --backbone convnextv2_large.fcmae_ft_in22k_in1k --epochs 6 --batch 16 --lr 6e-5
