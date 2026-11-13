#!/usr/bin/env bash
# Round R7 (2026-11-13): v26 showed family diversity pays ~3x more on the shifted test than on held-out -> train more
# families on ALL images for the blend. Waits for R6 to finish (single GPU).
cd "$(dirname "$0")"
R6="${R6_OUT:?set R6_OUT to the output file of the R6 queue}"
until grep -q "end h_small_ema" "$R6" 2>/dev/null; do sleep 30; done
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv full --grad-ckpt --drop-path 0.2 > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_caformer_s42 --backbone caformer_s36.sail_in22k_ft_in1k --epochs 6 --batch 32 --lr 1e-4 --seed 42
run f_baseclip_s7  --backbone convnext_base.clip_laion2b_augreg_ft_in12k_in1k --epochs 6 --batch 32 --lr 1e-4 --seed 7
run f_caformer_s7  --backbone caformer_s36.sail_in22k_ft_in1k --epochs 6 --batch 32 --lr 1e-4 --seed 7
