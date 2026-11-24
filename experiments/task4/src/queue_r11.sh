#!/bin/bash
# R11 (2026-11-14): the TA's advice — heavy augmentation, class imbalance, CNN + ViT ensemble. Train images only.
# Then R10 (in-domain masked pretraining) as before.
until [ -f ../outputs/test_f_v2bssl_s42.npy ]; do sleep 60; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
SMALL="--backbone convnext_small.fb_in22k_ft_in1k --batch 48 --lr 2e-4 --drop-path 0.2"
HEAVY="--epochs 8 --rot-max 30 --rot-p 0.6 --chperm 0.5 --invert 0.3 --hue 0.3 --gray 0.1 --erase 0.25 --mix 0.3 --scale-min 0.6"
EVA="--backbone eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 32 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run h_small_heavy   --cv hard --folds 4 --no-test --seed 42 $SMALL $HEAVY
run h_small_bal1    --cv hard --folds 4 --no-test --seed 42 $SMALL --epochs 6 --balance 1.0
run f_small_heavy_s42 --cv full --seed 42 $SMALL $HEAVY
run f_small_bal1_s42  --cv full --seed 42 $SMALL --epochs 6 --balance 1.0
run f_eva02b_s42    --cv full --seed 42 $EVA
run h_eva02b        --cv hard --folds 4 --no-test --seed 42 $EVA
./queue_r10.sh
