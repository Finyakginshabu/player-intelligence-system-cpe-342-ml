#!/bin/bash
# R15 (2026-11-14): replaces the R13/R14 shells (killed in a session restart; the EVA-02-L s42 training itself survived).
# Runs detached (Start-Process) so it survives session restarts. Status lines go to ../outputs/queue_r15.out.
# Order: the cheap ViT-B family tests first (1 h each), EVA-02-L seed 7 last (only useful if EVA-L passes its LB test).
cd "$(dirname "$0")"
until [ -f ../outputs/test_f_eva02l_s42.npy ]; do sleep 60; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
VIT="--epochs 6 --batch 32 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1"
EVAL="--backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_beitv2b_s42 --cv full --seed 42 --backbone beitv2_base_patch16_224.in1k_ft_in22k_in1k $VIT
run f_dinov3b_s42 --cv full --seed 42 --backbone vit_base_patch16_dinov3.lvd1689m $VIT
run f_deit3b_s42  --cv full --seed 42 --backbone deit3_base_patch16_224.fb_in22k_ft_in1k $VIT
run f_eva02l_s7   --cv full --seed 7 $EVAL
echo "### $(date +%H:%M) QUEUE DONE"
