#!/bin/bash
#SBATCH --account=nn8104k
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=seld-features-x86
#SBATCH --output=slurm_log/seld-features-x86-%j.out

# Build the shared SELD feature cache once, on x86 with torch 2.5.1 (the Fox environment), so all SELD runs
# use features that reproduce the Fox features. Submit from the repo root: sbatch scripts/seld_features_x86.sh
set -euo pipefail
PY=/cluster/projects/nn8104k/homes/jinyueg/conda/envs/gram-t/bin/python   # x86 env, torch 2.5.1
cd seld-dcase-2022-GRAM
REUSE_COMPLETE_NATIVE=1 "$PY" -u build_feature_cache.py
