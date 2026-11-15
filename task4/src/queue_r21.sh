#!/bin/bash
# R21 (2026-11-15 evening → night): GPU idle since R20. (1) AIMv2-L = new pretraining data + objective (DFN-2B, autoregressive),
# (2) 2nd seeds of DINOv2-L / DINOv3-L so a winning family can be averaged at once, (3) EVA-02-L at 336x672 (speculative:
# used only if tomorrow's EVA-B 336x672 LB test helps).
cd "$(dirname "$0")"
PY=../../.venv-gpu/Scripts/python.exe
DL="--epochs 6 --batch 16 --lr 1e-4 --layer-decay 0.85 --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_aimv2l_s42 --cv full --seed 42 --backbone aimv2_large_patch14_224.apple_pt --epochs 6 --batch 16 --lr 5e-5 --layer-decay 0.85 --drop-path 0.2
run f_dinov2l_s7 --cv full --seed 7 --backbone vit_large_patch14_reg4_dinov2.lvd142m $DL
run f_dinov3l_s7 --cv full --seed 7 --backbone vit_large_patch16_dinov3.lvd1689m $DL
run f_eva02l_hr_s42 --cv full --seed 42 --backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k --h 336 --w 672 --epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2
echo "### $(date +%H:%M) QUEUE DONE"
