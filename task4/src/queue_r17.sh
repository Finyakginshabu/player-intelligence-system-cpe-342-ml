#!/bin/bash
# R17 (2026-11-14 night): after R16. lr 2e-4 retries for the patch-16 ViT-Bs (BEiT-v2-B at 2e-4: loss 0.459 -> 0.408),
# then a 4th EVA-02-L seed. Starts when R16 prints QUEUE DONE.
cd "$(dirname "$0")"
until grep -q "QUEUE DONE" ../outputs/queue_r16.out 2>/dev/null; do sleep 120; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
VIT2="--epochs 6 --batch 32 --lr 2e-4 --layer-decay 0.75 --drop-path 0.1"
LARGE="--epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_dinov3b_lr2e4_s42 --cv full --seed 42 --backbone vit_base_patch16_dinov3.lvd1689m $VIT2
run f_deit3b_lr2e4_s42  --cv full --seed 42 --backbone deit3_base_patch16_224.fb_in22k_ft_in1k $VIT2
run f_eva02l_s2026 --cv full --seed 2026 --backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k $LARGE
echo "### $(date +%H:%M) QUEUE DONE"
