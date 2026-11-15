#!/bin/bash
# R18 (2026-11-15): higher input resolution. Frames are 230x120; at 224x448 each 14-px patch sees ~7 source pixels, so small
# HUD text / icons that identify the game are blurred into few tokens. EVA-02-B at 336x672 (2.25x tokens) as one LB test.
cd "$(dirname "$0")"
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_eva02b_hr_s42 --cv full --seed 42 --backbone eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --h 336 --w 672 --epochs 6 --batch 16 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1
echo "### $(date +%H:%M) QUEUE DONE"
