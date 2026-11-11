# 26-10-21 Stage 3 submission launcher

Created executable `submit_stage3.sh` in the repository root. It selects whole groups or individual runs from `stage3_training_plan.csv`, validates the existing 24 YAMLs, and previews an Olivia Slurm array by default. `--submit` enables actual submission; `--smoke` selects a short allocation and isolated smoke outputs. It supports a concurrency limit, resource overrides, the W&B GPU environment, config-hash checks, and per-run submission records.

The combined Stage 3 training entry point remains unimplemented. Actual submission fails before contacting Slurm when the default `scripts/stage3/run.py` is missing. A compatible Python entry point can be supplied with `--runner`; its required interface is documented in `scripts/stage3/README.md`. The existing legacy training scripts cannot consume these combined YAMLs directly.

Verified Bash syntax, the 24-run config validator, selections/deduplication, invocation outside the repository, invalid arguments, missing-runner handling, and worker rejection outside Slurm. A mocked `sbatch` verified resource arguments, array indices, and CSV records; its temporary files were removed. No Torch was imported and **no real jobs were submitted**. Updated the Stage 3 configuration guide with launcher links; training configs and existing result values were retained.
