#!/bin/bash
# K1 (KAGGLE-ONLY, 2026-11-14): pseudo-labelling round 1. Teacher = v28 test probabilities (S x4 + V2-B x3), conf >= 0.9.
# First a simulation on hard fold 4 (held-out rows pseudo-labelled by the hard-fold teacher) to check the mechanism.
until [ -f ../outputs/test_f_v2bssl_s42.npy ]; do sleep 60; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
SMALL="--backbone convnext_small.fb_in22k_ft_in1k --epochs 6 --batch 48 --lr 2e-4 --drop-path 0.2"
V2B="--backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4 --drop-path 0.2"
T=../outputs/test_blend_s4_v2b3.npy
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run h_small_pl     --cv hard --folds 4 --no-test --seed 42 $SMALL --pl ../outputs/oof_h_teacher4.npy
run f_small_pl_s42 --cv full --seed 42 $SMALL --pl $T
run f_v2base_pl_s42 --cv full --seed 42 $V2B --pl $T
run f_small_pl_s7  --cv full --seed 7 $SMALL --pl $T
run f_v2base_pl_s7 --cv full --seed 7 $V2B --pl $T
