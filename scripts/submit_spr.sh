#!/bin/bash
# One SPR pretraining run, then two SELD runs that start when it succeeds:
#   final-pt-<G>     configs/final_pt_<G>.yaml          GRAM-T + SPR pretraining
#   final-SELD-<L>0  configs/seld/final_SELD_<L>0.yaml  correct_ambi_format: false
#   final-SELD-<L>1  configs/seld/final_SELD_<L>1.yaml  correct_ambi_format: true
# Groups: B0 pair, C0 orthogonal, D0 w_to_xyz, E0 foa_to_iv. wandb: ritmo/ambi-ssl, group final-<G>; outputs in runs/final-<G>/.
# Run from the repo root: bash scripts/submit_spr.sh B0
set -euo pipefail
G=${1:?usage: bash scripts/submit_spr.sh <B0|C0|D0|E0>}
L=${G:0:1}
for f in configs/final_pt_$G.yaml configs/seld/final_SELD_${L}0.yaml configs/seld/final_SELD_${L}1.yaml; do
    [[ -f $f ]] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p slurm_log

pre=$(sbatch --parsable --job-name=final-pt-$G --output="slurm_log/pretrain-%j.out" scripts/train.sh --config-name final_pt_$G)
s0=$(sbatch --parsable --dependency=afterok:$pre --job-name=final-SELD-${L}0 scripts/seld.sh configs/seld/final_SELD_${L}0.yaml)
s1=$(sbatch --parsable --dependency=afterok:$pre --job-name=final-SELD-${L}1 scripts/seld.sh configs/seld/final_SELD_${L}1.yaml)
echo "final-pt-$G:     $pre"
echo "final-SELD-${L}0: $s0 (after $pre)"
echo "final-SELD-${L}1: $s1 (after $pre)"
