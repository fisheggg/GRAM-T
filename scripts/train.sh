#!/bin/bash
#SBATCH --account=nn8104k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=168:00:00
#SBATCH --job-name=GRAMTi
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

# Submit from the repo root: sbatch scripts/train.sh
export HYDRA_FULL_ERROR=1
# torch>=2.6 defaults to weights_only loading, which rejects Lightning checkpoints on resume
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

source /cluster/projects/nn8104k/homes/jinyueg/conda/activate-gram-t-gpu.sh

# Binaural RIRs (data=audioset) are not on Olivia; only the ambisonic setup has data here.
python3 train.py data=audioset_ambisonics data.clean_data_ratio=0.0 patching=frame data.mask_patch=160
