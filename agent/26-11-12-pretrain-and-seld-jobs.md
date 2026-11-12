# 26-11-12 One GRAM-T pretraining run plus two downstream SELD jobs (branch `fix/ambisonics-format`)

## What was submitted
`bash scripts/submit_pretrain_seld.sh` submits three Slurm jobs. Two earlier submissions were cancelled before they started: 2845897–2845899 for renaming, and 2846079–2846081 to add wandb. The current jobs:

| Slurm name (ID) | What it runs | Starts |
|---|---|---|
| `final-pt-A0` (2846118) | upstream GRAM-T, `data=audioset_ambisonics`, 200k steps, batch 80, seed 42, frame patching, mask 160 | now |
| `final-SELD-A0` (2846119) | STARSS as is (SN3D, Z on ch2), same as upstream GRAM-T | after pretrain succeeds (`afterok`) |
| `final-SELD-A1` (2846120) | `sn3d_to_n3d` + `swap_vertical_to_ch1` | after pretrain succeeds (`afterok`) |

The settings follow the archived Stage 3 runs A0 and B0. They share the same GRAM pretraining, and their SELD differs only in the STARSS format. Outputs go to `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/final-A0/` with the subfolders `pretrain/`, `seld_fix0/` and `seld_fix1/`. Each SELD run has its own feature cache, model and predictions. Slurm logs are in `slurm_log/`.

## wandb
- **Runs**: the runs log to `ritmo/ambi-ssl` under their Slurm job names, grouped as `final-A0`.
- **Pretraining**: logs to wandb in addition to TensorBoard, at every step, along with the full Hydra config. This is the new `wandb` block in `configs/base.yaml`. It is off by default, so plain `train.py` behaves as before; the submit script turns it on.
- **SELD**: logs `train/loss` each epoch and `val/{F,ER,LE,LR,SELD,loss}` every third epoch. The run summary holds `test/{F,ER,LE,LR}` with the jackknife `_ci_low`/`_ci_high` bounds and `best_val_epoch`. It is enabled when `SELD_WANDB_NAME` is set; `scripts/seld.sh` sets it to the job name.
- **Env**: wandb 0.31.0 was added to the `gram-t-gpu` tykky env by job 2846103, using `conda/build/update-gpu-wandb.sbatch` and `install-wandb.sh`. Existing packages were pinned. `pip freeze` before and after shows only new packages (wandb and its dependencies), and `pip check` passes. Package lists: `conda/build/gram-t-gpu-pip-freeze{-before-wandb,}.txt`.
- **Check**: an offline test on a GPU node ran a Lightning fit with TensorBoard + WandbLogger and the SELD logging calls. Both passed. Compute nodes reach api.wandb.ai (HTTP 200), and the login is in `~/.netrc`.

## Code changes
- `train.py`, `configs/base.yaml`: the optional WandbLogger described above.
- `scripts/train.sh`: passes extra Hydra overrides through (`"$@"`). Without arguments it runs the same command as before.
- `scripts/seld.sh` (new): finds `step=200000.ckpt` in the pretraining folder, extracts features, then trains SELD with task id 20.
- `scripts/submit_pretrain_seld.sh` (new): submits the three jobs with the dependency.
- `seld-dcase-2022-GRAM/parameters.py`:
  - new task id `20`: Olivia `SELD_combined` (STARSS23 + 1,200 synthetic recordings), 13 classes, effective batch 512 built from micro-batches of 32, lr 1e-4, 100 epochs;
  - optional environment variables `SELD_RUN_DIR`, `SELD_FOA_FIX`, `GRAM_CKPT` and `SELD_QUICK_TEST`;
  - dataset splits and classes now come from `dataset_year` rather than a `2022` substring in the path.
- `seld-dcase-2022-GRAM/train_seldnet.py`:
  - reads the checkpoint from the parameters instead of a hard-coded Snellius path;
  - accumulates gradients over micro-batches, which gives the same update as one 512 batch because the loss is a mean and GRAM has no BatchNorm.
- `seld-dcase-2022-GRAM/seldnet_model.py`: fails if most encoder weights are missing from the checkpoint. The upstream code uses `strict=False`, which hides such mismatches.

- Removed the unused `from IPython import embed` line from the SELD files, because the Olivia GPU env has no IPython and every SELD job would have crashed at import.

**GPU smoke test** (job 2845914, fix=1):
- all SELD imports are present;
- feature extraction on one STARSS file works;
- a checkpoint written from a fresh GRAM-T loads all 352/352 keys;
- one 512-sample update split into micro-batches of 32 takes about 1.6 s and uses 6.2 GB of GPU memory. That is about 75 updates per epoch, so roughly 3–5 h for 100 epochs, well inside the 48 h limit.

Pretraining jobs, real checkpoints and full feature extraction were not tested.

The upstream SELD protocol already matches the archived `stage2_legacy` protocol: train on folds 1–3, validate on fold 4 every 3 epochs, test on fold 4 with jackknife. Pretraining code is unchanged.

## Differences from archived Stage 3 that remain
- SELD uses the upstream GRAM readout (mean over frequency), not Stage 3's `adjacent` readout.
- The best epoch is still chosen by upstream's validation SELD score. Compare results using F1 first, then ER, LE and LR.
- Stage 3's `legacy_last_loo` point estimate, which omits `fold4_room15_mix010`, is not reproduced. The upstream jackknife over all 78 test files is used instead.

## Notes
- Do not switch branches or edit the SELD files until both SELD jobs have started, because they run the working tree.
- If pretraining fails or hits the 168 h limit, both SELD jobs stay pending with `DependencyNeverSatisfied`.
