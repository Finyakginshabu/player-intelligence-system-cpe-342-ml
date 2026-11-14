#!/bin/bash
# R13 (2026-11-14 night): EVA-02-Large (MIM in22k -> in22k -> in1k), 2 seeds, full data. ~2.8 h per seed on the RTX 5070.
# Why: the EVA-02-B family gave +0.011 on LB task 4 (v37); the large model is the next step in the only family still gaining.
PY=../../.venv-gpu/Scripts/python.exe
EVAL="--backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
for s in 42 7; do run f_eva02l_s$s --cv full --seed $s $EVAL; done
