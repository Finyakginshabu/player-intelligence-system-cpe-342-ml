#!/bin/bash
# R20 (2026-11-15): one more family with its own pretraining data: DINOv2-L (LVD-142M, register tokens, patch 14).
# Same recipe as DINOv3-L (lr 1e-4, ld 0.85, dp 0.2). Starts after R19 prints QUEUE DONE.
cd "$(dirname "$0")"
until grep -q "QUEUE DONE" ../outputs/queue_r19.out 2>/dev/null; do sleep 120; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_dinov2l_s42 --cv full --seed 42 --backbone vit_large_patch14_reg4_dinov2.lvd142m --epochs 6 --batch 16 --lr 1e-4 --layer-decay 0.85 --drop-path 0.2
echo "### $(date +%H:%M) QUEUE DONE"
