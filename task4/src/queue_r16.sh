#!/bin/bash
# R16 (2026-11-14 night): large ViTs after v39 (EVA-02-L: LB t4 +0.013). Starts when R15 ends (EVA-02-L s7 done or R15 gone).
# BEiT-v2-L first (masked-image pretraining like EVA), then a 3rd EVA-02-L seed, then DeiT-III-L.
cd "$(dirname "$0")"
r15_alive() { powershell -c "\$p='queue_'+'r15'; @(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like \"*\$p.sh*\" }).Count" | tr -d '\r'; }
until [ -f ../outputs/test_f_eva02l_s7.npy ] || [ "$(r15_alive)" = "0" ]; do sleep 120; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
LARGE="--epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_beitv2l_s42 --cv full --seed 42 --backbone beitv2_large_patch16_224.in1k_ft_in22k_in1k $LARGE
run f_eva02l_s123 --cv full --seed 123 --backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k $LARGE
run f_deit3l_s42  --cv full --seed 42 --backbone deit3_large_patch16_224.fb_in22k_ft_in1k $LARGE
echo "### $(date +%H:%M) QUEUE DONE"
