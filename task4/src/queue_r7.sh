#!/usr/bin/env bash
# Round R7 (2026-11-13): v26 (S + V2-B) gained 3x its held-out gain on the LB, but v27 (+ B-CLIP) LOST on the LB, so the
# gain is specific to V2-B -> more V2-B seeds on ALL images; CAFormer as one separate family test. Waits for R6 to finish (single GPU).
cd "$(dirname "$0")"
R6="${R6_OUT:?set R6_OUT to the output file of the R6 queue}"
until grep -q "end h_small_ema" "$R6" 2>/dev/null; do sleep 30; done
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv full --grad-ckpt --drop-path 0.2 > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
V2B="--backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4"
run f_v2base_s123  $V2B --seed 123
run f_v2base_s2026 $V2B --seed 2026
run f_caformer_s42 --backbone caformer_s36.sail_in22k_ft_in1k --epochs 6 --batch 32 --lr 1e-4 --seed 42
