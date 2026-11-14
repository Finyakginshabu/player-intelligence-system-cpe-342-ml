#!/bin/bash
# R14 (2026-11-14): more ViT families for the CNN + ViT ensemble (EVA-02-B gave +0.011 LB t4). Full data, 1 seed each,
# each judged by an LB family test. Starts when R13 (EVA-02-L) has finished.
R13OUT="$1"
until grep -q "end f_eva02l_s7" "$R13OUT" 2>/dev/null; do sleep 60; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
VIT="--epochs 6 --batch 32 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_beitv2b_s42 --cv full --seed 42 --backbone beitv2_base_patch16_224.in1k_ft_in22k_in1k $VIT
run f_dinov3b_s42 --cv full --seed 42 --backbone vit_base_patch16_dinov3.lvd1689m $VIT
run f_deit3b_s42  --cv full --seed 42 --backbone deit3_base_patch16_224.fb_in22k_ft_in1k $VIT
