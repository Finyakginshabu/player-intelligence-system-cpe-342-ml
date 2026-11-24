#!/bin/bash
# R19 (2026-11-15): v41 showed that different PRETRAINING DATA helps (EVA-02-L M38M +0.0014, DINOv3-B LVD +0.0007, together
# +0.0018). Next: DINOv3-L (LVD-1689M, patch 16 → higher lr than EVA), then 2nd seeds of the two v41 newcomers.
# Starts after R18 (EVA-02-B 336x672) prints QUEUE DONE.
cd "$(dirname "$0")"
until grep -q "QUEUE DONE" ../outputs/queue_r18.out 2>/dev/null; do sleep 120; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
LARGE="--epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_dinov3l_s42 --cv full --seed 42 --backbone vit_large_patch16_dinov3.lvd1689m --epochs 6 --batch 16 --lr 1e-4 --layer-decay 0.85 --drop-path 0.2
run f_dinov3b_lr2e4_s7 --cv full --seed 7 --backbone vit_base_patch16_dinov3.lvd1689m --epochs 6 --batch 32 --lr 2e-4 --layer-decay 0.75 --drop-path 0.1
run f_eva02lm38_s7 --cv full --seed 7 --backbone eva02_large_patch14_448.mim_m38m_ft_in22k_in1k $LARGE
echo "### $(date +%H:%M) QUEUE DONE"
