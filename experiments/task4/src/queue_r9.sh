#!/bin/bash
# R9 (2026-11-13 night): the FCMAE family is the only pretraining that helped on the test (v26 +0.017 t4).
# Test whether LESS supervised fine-tuning before ours helps: FCMAE + in1k ft, and pure self-supervised FCMAE.
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv full --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_v2bin1k_s42  --backbone convnextv2_base.fcmae_ft_in1k --epochs 6 --batch 32 --lr 1e-4 --drop-path 0.2 --seed 42
run f_v2bssl_s42   --backbone convnextv2_base.fcmae --epochs 8 --batch 32 --lr 2e-4 --layer-decay 0.8 --drop-path 0.1 --seed 42
