#!/bin/bash
# SELD from intermediate pretraining checkpoints, to see how many pretraining steps are needed.
# Usage (repo root of the branch that ran the pretraining: fix/ambisonics-format for A, feature/spr for B-E):
#   bash scripts/submit_seld_steps.sh <group letter> <step> [<step> ...]     e.g.  bash scripts/submit_seld_steps.sh B 50000 100000
# For each step it submits both SELD variants, configured like the 200k runs (configs/seld/final_SELD_<G>{0,1}.yaml)
# except for the encoder checkpoint, the output folder and the run name:
#   final-SELD-<G>0-<N>k   format fix off   -> runs/final-<G>0/seld_fix0_<N>k
#   final-SELD-<G>1-<N>k   format fix on    -> runs/final-<G>0/seld_fix1_<N>k
# If step=<step>.ckpt does not exist yet, a 1-CPU job on 'small' waits for it (up to 2 days) and the SELD jobs
# start after it (afterok), so they never run on a missing or half-written checkpoint.
set -euo pipefail
G=$1; shift
EXP=/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/final-${G}0
mkdir -p slurm_log
# All checkpoints of a run share one folder; take it from any saved checkpoint.
CKPT_DIR=$(dirname "$(find "$EXP/pretrain" -name 'step=*.ckpt' | head -1)")
[[ -d $CKPT_DIR ]] || { echo "no checkpoint folder under $EXP/pretrain" >&2; exit 1; }

for STEP in "$@"; do
    K=$((STEP / 1000))k
    CKPT="$CKPT_DIR/step=${STEP}.ckpt"
    dep=()
    if [[ ! -f $CKPT ]]; then
        # Wait until the file exists and its size has stopped changing for a minute (torch.save has finished).
        w=$(sbatch --parsable --account=nn8104k --partition=small --ntasks=1 --cpus-per-task=1 --mem=1G \
            --time=2-00:00:00 --job-name="wait-${G}0-${K}" --output="slurm_log/wait-%j.out" \
            --wrap="f='$CKPT'; until [ -f \"\$f\" ] && s=\$(stat -c %s \"\$f\") && sleep 60 && [ \"\$s\" = \"\$(stat -c %s \"\$f\")\" ]; do sleep 120; done; echo ready: \$f")
        dep=(--dependency=afterok:$w)
        echo "wait-${G}0-${K}: $w (waits for $CKPT)"
    fi
    for FIX in 0 1; do
        NAME=final-SELD-${G}${FIX}-${K}
        # Passed through the environment (sbatch exports it); the checkpoint path contains '='.
        j=$(GRAM_CKPT="$CKPT" SELD_RUN_DIR="$EXP/seld_fix${FIX}_${K}" SELD_WANDB_NAME="$NAME" \
            sbatch --parsable ${dep[@]+"${dep[@]}"} --job-name="$NAME" scripts/seld.sh "configs/seld/final_SELD_${G}${FIX}.yaml")
        echo "$NAME: $j"
    done
done
