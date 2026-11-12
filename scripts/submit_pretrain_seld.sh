#!/bin/bash
# One GRAM-T pretraining run, then two SELD runs (format fix off / on) that start when it succeeds.
# Settings follow archived Stage 3 A0/B0: 200k steps, batch 80, seed 42, frame patching, mask 160.
# Run from the repo root: bash scripts/submit_pretrain_seld.sh [experiment name]
# Job names: final-pt-A0 (pretraining), final-SELD-A0 (fix off), final-SELD-A1 (fix on).
# The same names are the wandb run names (ritmo/ambi-ssl), grouped under the experiment name.
set -euo pipefail
NAME=${1:-final-A0}
EXP_DIR=/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/$NAME
mkdir -p slurm_log

pre=$(sbatch --parsable --job-name=final-pt-A0 --output="slurm_log/pretrain-%j.out" scripts/train.sh \
    trainer.steps=200000 trainer.batch_size=80 seed=42 "save_dir=$EXP_DIR/pretrain" \
    wandb.enabled=true wandb.name=final-pt-A0 "wandb.group=$NAME")
fix0=$(sbatch --parsable --dependency=afterok:$pre --job-name=final-SELD-A0 scripts/seld.sh "$EXP_DIR" 0)
fix1=$(sbatch --parsable --dependency=afterok:$pre --job-name=final-SELD-A1 scripts/seld.sh "$EXP_DIR" 1)
echo "experiment: $EXP_DIR"
echo "pretrain:   $pre"
echo "seld fix=0: $fix0 (after $pre)"
echo "seld fix=1: $fix1 (after $pre)"
