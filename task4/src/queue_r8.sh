#!/usr/bin/env bash
# Round R8 (2026-11-13): pretraining search. Breakthrough analysis: masked-image pretraining (V2 FCMAE) helped on the test
# (+0.017 t4), image-text pretraining (CLIP) hurt (-0.012), and held-out cannot see it -> train masked-pretrained
# families on ALL images, judged one at a time on the LB. Waits for the running V2-B seed-123 job (single GPU).
cd "$(dirname "$0")"
until [ -f ../outputs/test_f_v2base_s123.npy ]; do sleep 30; done
sleep 60
PY=../../.venv-gpu/Scripts/python.exe
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --cv full --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_v2base_s2026 --backbone convnextv2_base.fcmae_ft_in22k_in1k --epochs 6 --batch 32 --lr 1e-4 --drop-path 0.2 --seed 2026
run f_v2large_s42  --backbone convnextv2_large.fcmae_ft_in22k_in1k --epochs 6 --batch 16 --lr 6e-5 --drop-path 0.2 --seed 42
run f_eva02b_s42   --backbone eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 32 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1 --seed 42
run f_hierab_s42   --backbone hiera_base_224.mae_in1k_ft_in1k --epochs 6 --batch 48 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1 --h 224 --w 224 --seed 42
run f_caformer_s42 --backbone caformer_s36.sail_in22k_ft_in1k --epochs 6 --batch 32 --lr 1e-4 --drop-path 0.2 --seed 42
