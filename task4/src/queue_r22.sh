#!/bin/bash
# R22 (2026-11-15 evening) replaces the rest of R21 after AIMv2-L. Goal: a bigger t4 jump than one more family.
# LP-FT (head-only stage on the frozen backbone, then full fine-tuning; Kumar et al. 2022) keeps pretrained features less
# distorted and is reported to be clearly more robust under distribution shift, which is exactly our CV 0.98 → LB 0.90 gap.
# (1) benchmark on hard fold 4 vs h_eva02b (non-dark 0.9371), (2) EVA-02-B and EVA-02-L LP-FT on all images for LB tests,
# (3) 2nd seeds of DINOv2-L / DINOv3-L, (4) EVA-02-L 336x672 (speculative).
cd "$(dirname "$0")"
# relaunched 20:45: h_eva02b_lpft and the orphan f_dinov2l_s7 (from R21) are already running → wait until no ft.py runs
ft_alive() { powershell -c "\$p='ft'+'.py'; @(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like \"*\$p*\" }).Count" | tr -d '\r'; }
until [ "$(ft_alive)" = "0" ]; do sleep 120; done
sleep 30
PY=../../.venv-gpu/Scripts/python.exe
EVAB="--backbone eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 32 --lr 5e-5 --layer-decay 0.75 --drop-path 0.1"
EVAL="--backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k --epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2"
DL="--epochs 6 --batch 16 --lr 1e-4 --layer-decay 0.85 --drop-path 0.2"
run() { echo "### $(date +%H:%M) start $1"; $PY ft.py --name "$@" --grad-ckpt > /dev/null 2>&1; echo "### $(date +%H:%M) end $1 (exit $?)"; }
run f_eva02b_lpft_s42 --cv full --seed 42 $EVAB --lp-epochs 2
run f_eva02l_lpft_s42 --cv full --seed 42 $EVAL --lp-epochs 2
run f_dinov3l_s7 --cv full --seed 7 --backbone vit_large_patch16_dinov3.lvd1689m $DL
run f_eva02l_hr_s42 --cv full --seed 42 --backbone eva02_large_patch14_448.mim_in22k_ft_in22k_in1k --h 336 --w 672 --epochs 6 --batch 16 --lr 3e-5 --layer-decay 0.85 --drop-path 0.2
echo "### $(date +%H:%M) QUEUE DONE"
