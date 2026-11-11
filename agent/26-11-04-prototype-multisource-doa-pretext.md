# 26-11-04 — Prototype: multi-source scenes + direct-path direction-map pretext

Implements suggestions 1 + 2 of [26-11-04-ssl-task-design-suggestions.md](26-11-04-ssl-task-design-suggestions.md) as a new model, `GRAMT_SPR_DOA`. `GRAMT_SPR` itself is unchanged.

## What changed
- `src/model/GRAMT_SPR_DOA.py` (new): a subclass of `GRAMT_SPR`.
  - **Scenes.** 0–2 extra AudioSet clips are added per scene. They are other clips of the same batch, played from the scene's non-empty noise-RIR positions in the same room. Each is gated to a random segment of at least 1 s with 20 ms fades, at a level of −10 to +10 dB relative to the main clip. WHAM noise is mixed as before.
  - **Target.** The SPR head becomes `Linear(512 → 256)`: a distribution over a Fibonacci sphere grid for each patch. The target is the activity-weighted mean of vMF kernels (κ = 25) centred on the direct-path directions of the sources active in that patch's 8-frame time column. Directions come from a torch port of li_review's plane-wave fit, with the same reliability tests.
  - **Loss.** KL divergence, computed only on masked patches whose frames are all reliable. Each patch uses its own time column, which fixes C2 and C3. The loss is logged as `spr_loss`. Diagnostics are logged as `doa_used_frac`, `doa_mode_err_deg` and `doa_sources_per_crop`.
- `src/model/__init__.py`, `train.py`: register `GRAMT_SPR_DOA`. `train.py` passes `cfg.doa` as extra keyword arguments; configs without a `doa` key behave as before.
- `configs/spr_doa_P0.yaml`: the prototype config (Olivia paths, batch 80, 200k steps). Not submitted.
- `tests/test_gramt_spr_doa.py`: self-test.

## Verification
GPU job 2716086 (log `slurm_log/spr-doa-test-2716086.out`) passed. The first attempt, 2716083, failed on a wrong assertion in the test; the model code was not at fault.

- **Direction fit.**
  - Synthetic plane waves are recovered to < 0.01°. Empty, two-arrival and wrongly normalised RIRs are all flagged unreliable.
  - On 40 real scenes from shard 0, the torch port matches li_review's numpy reference exactly: 0.0000°, and no disagreement on reliability.
  - Reliable: 85% of source RIRs and 88% of non-empty noise slots.
  - Non-empty noise slots per scene: 25 scenes have 1, 4 have 3, 6 have 4 and 5 have 5. So most scenes can take at most one extra source.
- **Patch layout.** Patch p covers time column p % 25.
- **Targets.**
  - Single source (K = 0): 53% of columns are valid, and each column's mode is within 5.5° of the source.
  - K = 2: 78% of columns are valid, holding 1, 2 or 3 active sources in 222, 202 and 46 columns. Each column's mode is within 8.5° of an active source.
- **Learning.** Encoder, patch embedding and head all receive gradients. Overfitting one batch for 60 steps lowers the direction KL from 3.10 to 1.40.
- **`train.py` smoke run** (30 steps, real dataloader, batch 8): runs end to end. `spr_loss` goes from 3.50 to 3.34, `doa_used_frac` is 0.71 and `doa_sources_per_crop` is 1.53. No learning is expected at step 30, because the learning rate is still in its 10k-step warm-up. The `[NOISE] Loader failed` traceback at exit is the loader's existing shutdown noise; it also appears in older logs (`C-2567654_*`).
- **W&B.** `WANDB_MODE=offline` is overridden by `src/experiment_logging.py`, which passes `mode="online"` by default. The smoke run therefore synced to `ritmo/ambi-ssl` as run `hxvelw1s`.

## Submission: P0, seed 42, pretraining then SELD
- **Stage 3 runner support.**
  - `scripts/stage3/models.py`: `build_model` now selects `GRAMT_SPR_DOA` when `pretrain.model: GRAMT_SPR_DOA`, and passes `pretrain.doa` through as keyword arguments. Other configs are unaffected.
  - `GRAMT_SPR_DOA.pretrain_forward` now consumes the prepared targets in consecutive slices, because Stage 3 feeds the 1,280 crops in 256-crop microbatches. It asserts if the slices are misaligned. Diagnostics are logged from the first microbatch only.
  - The self-test gained a check for microbatch slicing.
- **Config.** `configs/stage3_doa/P0-seed42.yaml` is `configs/stage3/A0.yaml` with only `pretrain` changed, plus its own output and metrics paths and W&B group. That makes A0 and C0 the matched baselines: seed 42, original STARSS format, stage2_legacy SELD protocol, legacy_last_loo scoring. The file is kept outside `configs/stage3/`, and `validate_plan.py` still passes.
- **Frozen source.** `run_snapshots/stage3-doa-20261104` holds 186 files, with `SHA256SUMS`. `.wandb-deps` is a symlink to the repository's copy.
- **Smoke job 2716116 passed** (`slurm_log/doa-P0-smoke-2716116.out`).
  - The self-test passed, including microbatch slicing; overfitting one batch lowered the direction KL from 3.10 to 1.33.
  - The Stage 3 smoke run made 2 pretraining updates (`doa_used_frac` 0.77, 1.6 sources per crop), loaded 150/150 encoder tensors and made 1 SELD update. Status: COMPLETE.
  - It created W&B runs `P0-smoke-pretrain` (b5yi7de1) and `P0-smoke-seld-stage2_legacy` (8daqpoyf) in `ritmo/ambi-ssl`, tagged `smoke`.
- **Full run.** Job 2716117 depends on the smoke job (afterok), uses `STAGE3_CODE_DIR` = the snapshot, and has a 7-day limit. Output goes to `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/2026-11-04-spr-doa/P0-seed42` and the log to `slurm_log/doa-P0-2716117.out`. Metrics will be appended to `results/stage3_doa_stage2_legacy_metrics.csv`. Expected duration is about 40 h of pretraining plus about 5 h of SELD, based on A0.
- **Baselines to compare against** (same protocol, seed 42): A0 F1 41.85%, C0 42.56%. A single seed cannot separate differences of about 1 point; seeds 43 and 44 are still to do.

## Not done / caveats
- The FOA conventions are not converted (TODO #9). Directions are expressed in the RIR's own channel order (ch1 vertical, N3D), which matches the IV input channels. `plane_wave_norm` must change if the RIRs are converted to SN3D.
- WHAM noise still plays from the positions used by the extra sources. Activity is measured on reverberant energy, so reverb tails count as active. Both are marked `shortcut:` in the code.
- No full pretraining, downstream SELD fine-tune or frozen probe has been run, so it is unknown whether this improves STARSS23 F1. Run at least 3 pretraining seeds and compare against reconstruction-only (A).
