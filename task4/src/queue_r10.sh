#!/bin/bash
# R10 (2026-11-14 night): in-domain masked pretraining on TRAIN images (mim.py), then the usual fine-tune.
# A (pure FCMAE) dropped 2026-11-14: pure FCMAE fine-tunes far too slowly (f_v2bssl_s42 final loss 0.668 vs 0.395).
# B: FCMAE+in22k-ft (the V2-B used in v28) -> in-domain MIM -> V2-B recipe; compare with f_v2base_s42.
until [ -f ../outputs/test_f_v2bssl_s42.npy ]; do sleep 60; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
step() { echo "### $(date +%H:%M) start $1"; "${@:2}" > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
step mim_v2b22k   $PY mim.py --name v2b22k --backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 5e-5
step f_v2b22kmim_s42 $PY ft.py --name f_v2b22kmim_s42 --cv full --grad-ckpt --backbone convnextv2_base.fcmae_ft_in22k_in1k --init ../outputs/mim_v2b22k.pt --epochs 6 --batch 32 --lr 1e-4 --drop-path 0.2 --seed 42
