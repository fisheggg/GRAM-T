#!/bin/bash
# One GRAM-T pretraining run, then two SELD runs (format fix off / on) that start when it succeeds.
# All settings are in the yaml files (archived Stage 3 A0/B0: 200k steps, batch 80, seed 42):
#   configs/final_pt_A0.yaml         -> job and wandb run final-pt-A0
#   configs/seld/final_SELD_A0.yaml  -> final-SELD-A0 (no ambisonics format fix)
#   configs/seld/final_SELD_A1.yaml  -> final-SELD-A1 (with the fix)
# wandb: ritmo/ambi-ssl, group final-A0. Run from the repo root: bash scripts/submit_pretrain_seld.sh
set -euo pipefail
mkdir -p slurm_log

pre=$(sbatch --parsable --job-name=final-pt-A0 --output="slurm_log/pretrain-%j.out" scripts/train.sh \
    --config-name final_pt_A0)
a0=$(sbatch --parsable --dependency=afterok:$pre --job-name=final-SELD-A0 scripts/seld.sh configs/seld/final_SELD_A0.yaml)
a1=$(sbatch --parsable --dependency=afterok:$pre --job-name=final-SELD-A1 scripts/seld.sh configs/seld/final_SELD_A1.yaml)
echo "final-pt-A0:   $pre"
echo "final-SELD-A0: $a0 (after $pre)"
echo "final-SELD-A1: $a1 (after $pre)"
