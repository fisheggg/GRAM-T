#!/bin/bash
#SBATCH --account=nn8104k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=seld
#SBATCH --output=slurm_log/seld-%j.out

# Downstream SELD (STARSS23 + synth, task id 20) from one GRAM pretraining run.
# Submit from the repo root: sbatch scripts/seld.sh <experiment dir> <fix 0|1>
#   fix=0: STARSS as is (SN3D, Z on ch2), as upstream GRAM-T
#   fix=1: sn3d_to_n3d + swap_vertical_to_ch1 (match the pretraining RIR convention)
# The checkpoint is <experiment dir>/pretrain/**/step=${STEP:-200000}.ckpt; GRAM_CKPT overrides it.
set -euo pipefail
EXP_DIR=$1
FIX=$2
[[ $FIX == 0 || $FIX == 1 ]] || { echo "fix must be 0 or 1, got '$FIX'" >&2; exit 1; }

if [[ -z ${GRAM_CKPT:-} ]]; then
    mapfile -t ckpts < <(find "$EXP_DIR/pretrain" -name "step=${STEP:-200000}.ckpt")
    [[ ${#ckpts[@]} -eq 1 ]] || { echo "expected 1 checkpoint, found ${#ckpts[@]}: ${ckpts[*]:-}" >&2; exit 1; }
    GRAM_CKPT=${ckpts[0]}
fi
export GRAM_CKPT SELD_FOA_FIX=$FIX SELD_RUN_DIR="$EXP_DIR/seld_fix$FIX"
# wandb run = Slurm job name (e.g. final-SELD-A0), group = experiment folder name; SELD_WANDB_NAME= disables it
export SELD_WANDB_NAME=${SELD_WANDB_NAME-${SLURM_JOB_NAME:-}} SELD_WANDB_GROUP=${SELD_WANDB_GROUP:-$(basename "$EXP_DIR")}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MPLBACKEND=agg
echo "checkpoint: $GRAM_CKPT"
echo "output:     $SELD_RUN_DIR"

source /cluster/projects/nn8104k/homes/jinyueg/conda/activate-gram-t-gpu.sh
cd seld-dcase-2022-GRAM
# Each run has its own feature cache (foa_dev or foa_dev_n3d) under $SELD_RUN_DIR/features.
python3 -u batch_feature_extraction.py 20
python3 -u train_seldnet.py 20 "fix${FIX}_${SLURM_JOB_ID:-local}"
