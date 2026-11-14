#!/bin/bash
# R12 (2026-11-14): EVA-02-B seeds to confirm v34 (LB +0.00183 with 1 seed). Same recipe as f_eva02b_s42.
PY=../../.venv-gpu/Scripts/python.exe
EVA="--backbone eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 32 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
for s in 7 123 2026; do run f_eva02b_s$s --cv full --seed $s $EVA; done
