# 26-11-12 Add SPR pretraining from `archived/GRAM-T` (branch `feature/spr`)

`feature/spr` is one SPR commit on top of `fix/ambisonics-format` (`261b638`). It adds the simplest SPR setup: GRAM-T pretraining with the SPR (spatial pseudo-label regression) objective. Downstream evaluation uses the existing SELD pipeline. The branch is checked out as a worktree at `.worktrees/feature-spr` (git-excluded), because the main checkout must stay on `fix/ambisonics-format` for the queued final-A0 SELD jobs.

## How to run
```bash
sbatch scripts/train.sh --config-name spr
```
`configs/spr.yaml` follows the archived Stage 3 GRAM+SPR runs: pair pretext, linear SPR head, reconstruction and SPR weights 1.0, 200k steps, batch 80, seed 42, wandb run `spr`. Checkpoints go to `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/spr/pretrain/saved_models_spr/PretextType=pair/MultiTaskType=linear/…/step=*.ckpt`. The other pretexts are `w_to_xyz`, `orthogonal` and `foa_to_iv`; the other head is `decoder`.

The fork's SELD (`scripts/seld.sh` with a `configs/seld/*.yaml` whose `experiment_dir` is `runs/spr`) loads an SPR checkpoint like a GRAM-T one. The extra SPR-head weights are ignored.

## SPR experiment groups B0–E0
Each group is one SPR pretraining run and two SELD runs, mirroring final-A0. Submit a group with `bash scripts/submit_spr.sh <G>`; the SELD jobs depend on the pretraining job (`afterok`).

Submitted on 2026-11-12 from this worktree at `48496aa`, after the smoke, determinism and end-to-end SELD checks:

| Group | Pretraining | SELD fix off | SELD fix on |
|---|---|---|---|
| B0 | final-pt-B0 2852514 | final-SELD-B0 2852515 | final-SELD-B1 2852516 |
| C0 | final-pt-C0 2852517 | final-SELD-C0 2852518 | final-SELD-C1 2852519 |
| D0 | final-pt-D0 2852520 | final-SELD-D0 2852521 | final-SELD-D1 2852522 |
| E0 | final-pt-E0 2852523 | final-SELD-E0 2852524 | final-SELD-E1 2852525 |

The jobs run the code in `.worktrees/feature-spr` when they start, so leave that worktree on `48496aa` until all of them have started.

| Group | Pretext | Pretraining (job / wandb run) | SELD, `correct_ambi_format: false` | SELD, `correct_ambi_format: true` |
|---|---|---|---|---|
| B0 | pair | `configs/final_pt_B0.yaml` → `final-pt-B0` | `configs/seld/final_SELD_B0.yaml` → `final-SELD-B0` | `final_SELD_B1.yaml` → `final-SELD-B1` |
| C0 | orthogonal | `final_pt_C0.yaml` → `final-pt-C0` | `final_SELD_C0.yaml` → `final-SELD-C0` | `final_SELD_C1.yaml` → `final-SELD-C1` |
| D0 | w_to_xyz | `final_pt_D0.yaml` → `final-pt-D0` | `final_SELD_D0.yaml` → `final-SELD-D0` | `final_SELD_D1.yaml` → `final-SELD-D1` |
| E0 | foa_to_iv | `final_pt_E0.yaml` → `final-pt-E0` | `final_SELD_E0.yaml` → `final-SELD-E0` | `final_SELD_E1.yaml` → `final-SELD-E1` |

- **Pretraining configs:** `final_pt_<G>.yaml` extends `configs/spr.yaml`. The shared settings are: linear SPR head, reconstruction and SPR weights 1.0, 200k steps, batch 80, seed 42, frame patching, mask 160. The pretraining configs differ from `final_pt_A0` only in the SPR fields, `save_dir` and the wandb names; data, trainer, optimizer and patching are identical.
- **Outputs:** `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/<G>/` with `pretrain/`, `seld_fix0/` and `seld_fix1/`. wandb group: `final-<G>`.
- **SELD configs:** they use the new top-level switch `correct_ambi_format`. `parameters.py` maps it to both `sn3d_to_n3d` and `swap_vertical_to_ch1`, and refuses configs that also set those flags. Loaded through `parameters.py`, each `final_SELD_<L>0/1` gives the same parameters as `final_SELD_A0/A1` except for its own run folder and wandb names.

## Changes
- `src/model/GRAMT_SPR.py`: copied from archived. Its two spectrogram-figure calls use `self.logger.experiment.add_figure` like `GRAMT.py`, instead of the archived `src/experiment_logging.py` helper.
- `src/model/__init__.py`: registers `GRAMT_SPR`.
- `train.py`: adds `GRAMT_SPR` to `networks`. For it only, passes `pretext_type`, `multitask_type` and the two loss weights, and saves under `saved_models_spr/`.
- `utils.py`: `GRAMT_SPR` uses the archived checkpoint identity (`PretextType=…_MultiTaskType=…_ReconLossW=…_SPRLossW=…_Model=GRAMT_SPR…`).
- The GRAM-T path is unchanged.

## Left out on purpose
- SPR-DOA (`GRAMT_SPR_DOA`).
- The Stage 3 runner.
- The STARSS SPR SSL phase (`train_ssl_seldnet.py`).
- li's standalone `SPR/` code.
- All historical run configs and CSVs.
- The archived data-loader freeze fix (commit `190efd8`). It isn't SPR code.

## Scoring fix
The STARSS23 distance-column scoring fix found during the port was committed on `fix/ambisonics-format` (`32709f3`), and this branch is based on it.

## Checks
- `configs/spr.yaml` composes with Hydra (also with the fixed overrides in `scripts/train.sh`) and resolves to the checkpoint path above.
- GPU smoke test: see the result below.

## 26-11-14 Out-of-memory crashes and resumed runs
- **What stopped:** final-pt-B0 (2852514) and final-pt-D0 (2852520) were killed for exceeding their 128 GB memory limit at about 05:12 on Nov 14. Slurm cancelled their SELD jobs.
  - **D0:** an oom_kill at step ~98k.
  - **B0:** the out-of-memory killer took the RIR loader's manager process, so the queue connection broke (`EOFError`) at step ~105k.
  - **Same cause as the A0 crash:** memory sits at about 115–127 GB, with no leak, and spikes push it over the limit (see the A0 report). All three crashes happened around 05:11–05:12, so something on the nodes at that hour may add the final push.
- **Fix:** `scripts/train.sh` now requests 192 GB on both branches. Each node has 808 GB for 4 GPUs, and memory is not billed.
- **Resumed** with `+ckpt_path=<checkpoint>` and `+wandb.id=<run id>`. `train.py` now passes `wandb.id` with `resume="allow"`, so the wandb run continues.

| Run | Resumed from | New job | SELD fix off / on |
|---|---|---|---|
| final-pt-B0 | step 100000 (wandb f5gcf4ek) | 2868860 | 2868862 / 2868863 |
| final-pt-D0 | step 75000 (wandb afwcctzg) | 2868861 | 2868864 / 2868865 |

- **Effect on comparability:**
  - Weights, optimizer and LR schedule continue from the checkpoint.
  - B0 redoes steps 100k–105k and D0 redoes 75k–98k.
  - `seed_everything` runs again at startup, so the data and augmentation streams restart from their beginning, not from where the first job stopped.
- **C0 (2852517) and E0 (2852523)** are still running with 128 GB and are at the same risk.

## 26-11-14 SELD from intermediate checkpoints (50k and 100k steps)
- **Purpose:** see how many pretraining steps SELD needs. Both format variants run for each group at 50k and 100k steps, on top of the 200k runs.
- **Script:** `scripts/submit_seld_steps.sh <G> <step>...`, on both branches. Each job uses the same yaml as the 200k run (`configs/seld/final_SELD_<G>{0,1}.yaml`) and overrides only three things:
  - `GRAM_CKPT`: the encoder checkpoint;
  - `SELD_RUN_DIR`: `runs/final-<G>0/seld_fix{0,1}_<N>k`;
  - `SELD_WANDB_NAME`: `final-SELD-<G>{0,1}-<N>k`, in wandb group `final-<G>0`.
- **Missing checkpoints:** a 1-CPU `wait-<G>0-<N>k` job waits on `small` until the checkpoint exists and its size has stopped changing; the SELD jobs start after it.
- **Jobs submitted:**

| Group | 50k (fix0 / fix1) | 100k (fix0 / fix1) |
|---|---|---|
| A0 | 2868912 / 2868913 | 2868915 / 2868916 (after wait-A0-100k 2868914) |
| B0 | 2868917 / 2868918 | 2868919 / 2868920 |
| C0 | 2868921 / 2868922 | 2868923 / 2868924 |
| D0 | 2868925 / 2868926 | 2868928 / 2868929 (after wait-D0-100k 2868927) |
| E0 | 2868930 / 2868931 | 2868932 / 2868933 |

- **Caveats:**
  - B0's 100k checkpoint is from the first job, just before it crashed.
  - D0's 100k checkpoint will come from the resumed job (2868861).
  - If a pretraining run dies before 100k, its waiter times out after 2 days and the SELD jobs for that step are cancelled.
