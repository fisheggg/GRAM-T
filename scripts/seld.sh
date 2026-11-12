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

# Downstream SELD from one GRAM pretraining run, configured by a run yaml in configs/seld/.
# Submit from the repo root: sbatch --job-name=final-SELD-A0 scripts/seld.sh configs/seld/final_SELD_A0.yaml
# The yaml gives the experiment folder, pretraining step, output folder, wandb run and all SELD
# parameters. The encoder is <experiment_dir>/pretrain/**/step=<pretrain_step>.ckpt; GRAM_CKPT overrides it.
set -euo pipefail
export SELD_CONFIG=$(realpath "$1")

source /cluster/projects/nn8104k/homes/jinyueg/conda/activate-gram-t-gpu.sh
read -r TASK_ID EXP_DIR STEP < <(python3 -c "
import sys, yaml; c = yaml.safe_load(open(sys.argv[1]))
print(c['task_id'], c['experiment_dir'], c['pretrain_step'])" "$SELD_CONFIG")

if [[ -z ${GRAM_CKPT:-} ]]; then
    mapfile -t ckpts < <(find "$EXP_DIR/pretrain" -name "step=${STEP}.ckpt")
    [[ ${#ckpts[@]} -eq 1 ]] || { echo "expected 1 checkpoint, found ${#ckpts[@]}: ${ckpts[*]:-}" >&2; exit 1; }
    GRAM_CKPT=${ckpts[0]}
fi
export GRAM_CKPT TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MPLBACKEND=agg
echo "config:     $SELD_CONFIG"
echo "checkpoint: $GRAM_CKPT"

cd seld-dcase-2022-GRAM
# Features come from the shared x86 cache (scripts/seld_features_x86.sh); this step only checks it exists.
python3 -u batch_feature_extraction.py "$TASK_ID"
python3 -u train_seldnet.py "$TASK_ID" "$(basename "$SELD_CONFIG" .yaml)_${SLURM_JOB_ID:-local}"
