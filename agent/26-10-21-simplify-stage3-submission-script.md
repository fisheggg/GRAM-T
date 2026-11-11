# 26-10-21 Simplify Stage 3 submission to match train.sh

Replaced the array launcher with a 31-line Slurm script, `submit_stage3.sh`, following `train.sh`. Edit the single `CONFIG=configs/stage3/C0.yaml` line to choose a run and submit with `sbatch submit_stage3.sh`. Retained Olivia resource settings, W&B activation, and `HYDRA_FULL_ERROR=1`; removed argument parsing, array selection, dry runs, manifests, and submission bookkeeping. Updated the submission and configuration guides.

The combined Stage 3 Python runner still needs implementation; the script reports its absence and exits before environment activation. Verified Bash syntax and whitespace checks. No jobs were submitted or training changes made.
