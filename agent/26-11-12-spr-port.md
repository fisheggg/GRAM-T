# 26-11-12 Add SPR pretraining from `archived/GRAM-T` (branch `feature/spr`)

`feature/spr` branches from `fix/ambisonics-format` and was fast-forwarded to `32709f3` (the scoring fix). It adds the simplest SPR setup: GRAM-T pretraining with the SPR (spatial pseudo-label regression) objective. Downstream evaluation uses the existing SELD pipeline. The branch is checked out as a worktree at `.worktrees/feature-spr` (git-excluded), because the main checkout must stay on `fix/ambisonics-format` for the queued final-A0 SELD jobs.

## How to run
```bash
sbatch scripts/train.sh --config-name spr
```
`configs/spr.yaml` follows the archived Stage 3 GRAM+SPR runs: pair pretext, linear SPR head, reconstruction and SPR weights 1.0, 200k steps, batch 80, seed 42, wandb run `spr`. Checkpoints go to `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/spr/pretrain/saved_models_spr/PretextType=pair/MultiTaskType=linear/…/step=*.ckpt`. The other pretexts are `w_to_xyz`, `orthogonal` and `foa_to_iv`; the other head is `decoder`.

The fork's SELD (`scripts/seld.sh` with a `configs/seld/*.yaml` whose `experiment_dir` is `runs/spr`) loads an SPR checkpoint like a GRAM-T one. The extra SPR-head weights are ignored.

## SPR experiment groups B0–E0
Each group is one SPR pretraining run and two SELD runs, mirroring final-A0. Submit a group with `bash scripts/submit_spr.sh <G>`; the SELD jobs depend on the pretraining job (`afterok`). Nothing has been submitted yet.

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
