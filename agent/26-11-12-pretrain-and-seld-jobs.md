# 26-11-12 One GRAM-T pretraining run plus two downstream SELD jobs (branch `fix/ambisonics-format`)

## What was submitted
`bash scripts/submit_pretrain_seld.sh` submits three Slurm jobs, configured by the yaml files below. Two earlier submissions were cancelled before they started: 2845897–2845899 for renaming, and 2846079–2846081 to add wandb. The current jobs:

| Slurm name (ID) | What it runs | Starts |
|---|---|---|
| `final-pt-A0` (2846118) | upstream GRAM-T, `data=audioset_ambisonics`, 200k steps, batch 80, seed 42, frame patching, mask 160 | now |
| `final-SELD-A0` (2846119) | STARSS as is (SN3D, Z on ch2), same as upstream GRAM-T | after pretrain succeeds (`afterok`) |
| `final-SELD-A1` (2846120) | `sn3d_to_n3d` + `swap_vertical_to_ch1` | after pretrain succeeds (`afterok`) |

The settings follow the archived Stage 3 runs A0 and B0. They share the same GRAM pretraining, and their SELD differs only in the STARSS format. Outputs go to `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/final-A0/` with the subfolders `pretrain/`, `seld_fix0/` and `seld_fix1/`. Each SELD run has its own feature cache, model and predictions. Slurm logs are in `slurm_log/`.

## Config files
All run settings are in yaml:
- [`configs/final_pt_A0.yaml`](../configs/final_pt_A0.yaml): standalone Hydra config, used as `sbatch scripts/train.sh --config-name final_pt_A0`.
- [`configs/seld/final_SELD_A0.yaml`](../configs/seld/final_SELD_A0.yaml) and [`final_SELD_A1.yaml`](../configs/seld/final_SELD_A1.yaml): differ only in `sn3d_to_n3d`/`swap_vertical_to_ch1`, the wandb name and the run folder. Each sets the task id, experiment folder, pretraining step, run folder, wandb run and SELD parameters. `parameters.py` loads it through `$SELD_CONFIG`, and its `params` block replaces the task-id defaults. Usage: `sbatch --job-name=final-SELD-A0 scripts/seld.sh configs/seld/final_SELD_A0.yaml`.

`scripts/seld.sh` now takes the yaml path, and `scripts/submit_pretrain_seld.sh` uses the three files.

The queued jobs (2846118–2846120) were submitted before the yaml files existed, so they use the old command line and environment variables. Checked against those jobs:
- the composed Hydra config of `final_pt_A0.yaml` is identical to job 2846118's;
- each SELD yaml gives a parameter dictionary identical to its queued job's.

## wandb
- **Runs**: the runs log to `ritmo/ambi-ssl` under their Slurm job names, grouped as `final-A0`.
- **Pretraining**: logs to wandb in addition to TensorBoard, at every step, along with the full Hydra config. This is the new `wandb` block in `configs/base.yaml`. It is off by default, so plain `train.py` behaves as before; the submit script turns it on.
- **SELD**: logs `train/loss` each epoch and `val/{F,ER,LE,LR,SELD,loss}` every third epoch. The run summary holds `test/{F,ER,LE,LR}` with the jackknife `_ci_low`/`_ci_high` bounds and `best_val_epoch`. It is enabled when `SELD_WANDB_NAME` is set; `scripts/seld.sh` sets it to the job name.
- **Env**: wandb 0.31.0 was added to the `gram-t-gpu` tykky env by job 2846103, using `conda/build/update-gpu-wandb.sbatch` and `install-wandb.sh`. Existing packages were pinned. `pip freeze` before and after shows only new packages (wandb and its dependencies), and `pip check` passes. Package lists: `conda/build/gram-t-gpu-pip-freeze{-before-wandb,}.txt`.
- **Check**: an offline test on a GPU node ran a Lightning fit with TensorBoard + WandbLogger and the SELD logging calls. Both passed. Compute nodes reach api.wandb.ai (HTTP 200), and the login is in `~/.netrc`.

## Code changes
- `train.py`, `configs/base.yaml`: the optional WandbLogger described above.
- `scripts/train.sh`: passes extra Hydra overrides through (`"$@"`). Without arguments it runs the same command as before.
- `scripts/seld.sh` (new): reads a `configs/seld/*.yaml`, finds the pretraining checkpoint, extracts features, then trains SELD.
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

## Scoring fix for STARSS23 metadata
Real STARSS23 metadata rows have six columns (frame, class, source, azimuth, elevation, distance). The fork's scoring read six-column rows as Cartesian, so ground truth became [az, el, dist] against 2-D predictions. That crashes `least_distance_between_gt_pred` at the first validation (epoch 3); a GPU test with perfect predictions reproduced it. The archived `with_distance=True` fix is ported to `cls_feature_class_gram.load_output_format_file` and used for ground truth in `cls_compute_seld_results.py`. Afterwards the same perfect predictions score ER 0, with F = 9/13 and LE = 4/13 × 180°, which is the macro average when 4 of the 13 classes are absent from the two test files. Label extraction was not affected, because it reads az/el by index. The queued jobs final-SELD-A0/A1 pick up the fix because they run the working tree when they start.

## Resampling fix for SELD features
`SELD_combined/foa_dev` holds the 24 kHz STARSS23/DCASE audio, both real and synthetic. `cls_feature_class_gram._load_audio` used to treat it as 32 kHz without resampling. A comparison on 12 recordings against the Fox features (fox-replication) showed the result: 0.75× as many feature and label frames (4552 instead of 6070 for `fold4_room23_mix001`), spectrograms on the wrong frequency scale, and labels that were too short.

The old Olivia cache, made by the Stage 3 code (load → resample → normalize), matches the Fox features to about 1e-7. `_load_audio` now does the same: it resamples to `fs` with the default settings of `torchaudio.functional.resample`, as in `data/convert_starss_sr.py`, before the SN3D→N3D scaling and the RMS normalization. The queued final-SELD-A0/A1 jobs pick this up when they start. The GPU feature comparison of the fixed code (job 2848101) had not run at commit time.

## Pretraining data loader on Olivia
Real pretraining stalled before its first batch, with every process idle. fox-replication's own code did the same today, although it ran on 29 October. The data, audio decoding and the noise/RIR loader processes all work, as does the data module with 16 workers on its own; training also works with 0 workers. The deadlock comes from forking DataLoader workers inside the running training process.
- **Loader:** `src/data_modules/WebAudioDataModule.py` is now fox-replication's version (the archived freeze fix, commit `190efd8`): single-threaded noise/RIR loader processes, loader errors reported, and a stall timeout. It is picklable: a worker gets only the queue handles, never the loader processes or the Lightning trainer.
- **Worker start:** `configs/data/audioset_ambisonics.yaml` sets `num_workers: 16` and `mp_context: forkserver`. `train.py` passes both to the data module. With forkserver (or spawn), 16 workers train at about 0.5–1 s per step; fox-replication ran at 0.74 s per step.
- **Time limit:** pretraining requests 4 days (`scripts/train.sh`). fox-replication's 200k-step runs took 41–45 h.

## Shared SELD feature cache
- **Cache:** all SELD runs read one cache, built once on an x86 CPU node with torch 2.5.1 (`scripts/seld_features_x86.sh` → `seld-dcase-2022-GRAM/build_feature_cache.py`), in `data/SELD_combined/features_x86/`: `foa_dev`, `foa_dev_n3d` and `foa_dev_label`.
- **Agreement with Fox:** the native cache is bit-identical to the Fox features for 1278 of 1368 recordings. 90 recordings (85 synthetic) differ by up to 6.7e-2, which re-extraction on Olivia cannot avoid; labels are identical. Both formats come from this one extraction (your choice), so each fix-off/fix-on pair differs only in the format conversion.
- **Safety:** `batch_feature_extraction.py` refuses to run without the cache's completion marker, so no job can extract ARM features into it. On the ARM GPU env (torch 2.7.1), the same code differs from Fox by up to 1e-3 relative on every file.

## Scoring fix (jackknife point values)
`get_SELD_Results(..., is_jackknife=True)` reused the all-file variables inside its leave-one-out loop. It therefore returned the estimate without the last listed file instead of the 78-file score; this is where Fox's "historical77" came from. It now returns the all-file values, with unchanged confidence intervals. `train_seldnet.py` also reports the 77-file score without `fold4_room15_mix010.csv` (`test77/*`). On Fox's B2 predictions, both conventions reproduce fox-replication's metrics exactly (seeds 42/43/44).

## Checks (GPU, 2026-11-12)
- **Determinism:** two runs with the same seed give identical weights for pretraining (`torch.compile`, bf16, deterministic algorithms) and for SELD on the shared cache.
- **SPR:** all four pretexts run forward and backward (feature/spr).
- **End-to-end SELD:** a 3-epoch run through `scripts/seld.sh` on the N3D cache loaded all 352/352 encoder weights, then trained, validated, tested and logged to wandb. One epoch takes about 5 min.

## Seeding and determinism
- **Pretraining** (`train.py`): `seed_everything(cfg.seed, workers=True)` is now the first line of `main()`. It used to run after the model and data module were built, so model initialization and the noise/RIR loader processes were unseeded. The new `trainer.deterministic: True` (`configs/trainer/default_trainer.yaml`) makes Lightning use deterministic algorithms with no cuDNN benchmark. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA starts.
- **SELD** (`train_seldnet.set_seed_and_determinism`): seeds `random`/`numpy`/`torch` with `seed=42`, once, before data and model, as fox-replication did. With `deterministic=True` it also enables deterministic algorithms and cuDNN deterministic mode, with no benchmark. Both are new defaults in `parameters.py`.
- **Not deterministic:** pretraining still draws noise and RIRs from one shared queue read by 16 DataLoader workers, so the pairing of clips with RIRs and noise depends on worker timing.
- **Not yet verified:** the GPU check (repeated runs, weight hashes; job 2847606) had not run at commit time. final-pt-A0 (2846118) is held until it passes.

## Differences from archived Stage 3 that remain
- SELD uses the upstream GRAM readout (mean over frequency), not Stage 3's `adjacent` readout.
- The best epoch is still chosen by upstream's validation SELD score. Compare results using F1 first, then ER, LE and LR.
- Stage 3's `legacy_last_loo` point estimate, which omits `fold4_room15_mix010`, is not reproduced. The upstream jackknife over all 78 test files is used instead.

## Notes
- Do not switch branches or edit the SELD files until both SELD jobs have started, because they run the working tree.
- If pretraining fails or hits the 96 h limit, both SELD jobs stay pending with `DependencyNeverSatisfied`.

## 26-11-13 Out-of-memory crash and resubmission
- **Crash:** final-pt-A0 (2846118) was killed for exceeding its 128 GB memory limit after 2 h, around step 5500. Slurm cancelled final-SELD-A0 and A1 (2846119/2846120). The log shows a DataLoader worker bus error (no shared memory left) and two oom_kill events.
- **Cause:** memory is not leaking; it sits at the limit. Cgroup samples of the running SPR jobs held steady over 6 minutes:
  - about 54–65 GB of shared memory: the batches the 16 workers keep ready (prefetch 2), each holding 80 multichannel RIRs;
  - about 55 GB of process memory.
  - The `forkserver` workers no longer share the main process's memory, which adds about 40 GB over fox-replication's fork workers (fox peaked at 70–80 GB).
  - Batch contents move usage by about 10 GB, so A0 died on a spike.
- **Resubmitted** with `sbatch --mem=192G`, same code (`261b638`) and configs:
  - final-pt-A0 **2857390**;
  - final-SELD-A0 **2857391** and final-SELD-A1 **2857392**, both after 2857390.
  - The node has 808 GB for 4 GPUs, and the accel partition does not bill for memory.
  - The crashed run is still in wandb under the same name (run `pcwld9b6`).
- **B0–E0 (2852514/17/20/23)** still run with 128 GB and peak at 123–127 GB, so they may hit the same crash. If one does, resume it from its last 25k-step checkpoint with 192 GB and resubmit its SELD jobs.
- `scripts/train.sh` still requests 128 GB. Raise it before submitting new pretraining runs.

## 26-11-14 SELD from intermediate checkpoints
- Added `scripts/submit_seld_steps.sh` (also on `feature/spr`). It runs SELD from the 50k and 100k checkpoints as `final-SELD-A{0,1}-50k` and `-100k` in `runs/final-A0/seld_fix{0,1}_<N>k`.
- Jobs:
  - 50k: 2868912 / 2868913.
  - 100k: 2868915 / 2868916, after `wait-A0-100k` 2868914, which waits for the 100k checkpoint of 2857390.
- See `agent/26-11-12-spr-port.md` on `feature/spr` for all groups.
