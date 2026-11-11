# 26-10-18 Start Stage 2 group C training and Fox–Olivia comparison

## Scheduling adjustment

Slurm reservation `nb_testing` covers all Olivia GPU nodes from 19 October
08:00 until 22 October 00:00 (Europe/Oslo). The original seven-day array 2567729
could not start before it. Replaced that still-pending Olivia array with
17-hour first-phase array **2567739**: C0/C2 target their remaining 50k; C1/C3
stop at 50k with checkpointing enabled and the original 200k learning-rate
horizon. C1/C3 continuations **2567740_1** and **2567741_3** have successful-job
dependencies and separate restart YAMLs.
The run driver validates the first checkpoint before resuming and distinguishes
an intentional checkpointed chunk from completed 200k training. Fox jobs remain
untouched by this scheduling adjustment.

The user requested the Fox–Olivia experiment and Stage 2 group C training.
Preserve historical SPR losses, FOA conventions, feature caches, SELD folds,
readout, thresholds, and scorer. No corrected SPR or review-branch objectives
are enabled by this work. No commits or pushes are requested.

## Changes

- Restored only `GRAMT_SPR.pass_through_decoder` from historical commit
  `3931c60`, whose MWMAE decoder path produced the existing C0/C2 checkpoints.
  The current main checkout tried to access the linear-only `decoder_embed`
  even in separate-decoder mode. The latent return now selects the SPR branch
  in decoder mode. No review imports, objective changes, or decoder rejection
  guard were copied.
- `train.py` now respects the configured full-data shard path instead of
  overwriting it with a Fox path. Added opt-in optimizer-progress JSON and a
  smoke-run stop callback; smoke runs can disable checkpointing and the progress
  bar while keeping the full 200k-step learning-rate schedule. Historical
  defaults, optimizer, seed ordering, and loss computation are unchanged.
- Added dedicated Olivia Slurm launch/recording helpers under `scripts/olivia/`.
  Stage 2 runs require a passed comparability gate, smoke jobs keep the 200k LR
  horizon while stopping after 100 real optimizer steps, and concurrent CSV
  status updates use a file lock and atomic replacement. Stage 2 group results
  remain in `results/stage2.csv`; smoke/comparison results use `results/stage3.csv`.
- Added a fixed-checkpoint GPU replay for both platforms: identical C2 state,
  cached feature/mask fixture, a strict FP32 first forward/backward/update, and
  100 BF16 training steps with the historical optimizer and scheduler states.
  Replay records parameter-gradient/update samples, losses, LR sequence, runtime,
  GPU, checkpoint step, and fixture hash. Fox runs use an isolated work-directory
  code snapshot, not modifications to its live training checkout.

## Preparation

- Dataset transfer completed with 870 AudioSet, 20 noise, and 70 RIR shards.
- No Olivia training jobs were active at the initial Slurm check.
- Fox SFTP access is available. A shell-capable Fox connection is pending for
  fresh reference-job submission.
- Fox retains intact-looking C0/C2 step-150000 and last checkpoints and the
  A0/B2 step-200000 checkpoints. Files will be validated before resuming.
- Runtime outputs will use project work storage, because the project home
  filesystem has only about 25 GB free.
- Fox reference recovery passed a full `rclone check --download`: 247 matching
  files and zero differences. C0/C2 checkpoint ZIP structures are valid.
- Smoke array 2567622 stalled before its first optimizer step with parent CPU
  threads set to eight. Process wait states showed sleeping/futex-waiting
  workers and an inherited compiler-worker pool. Cancelled only this test
  array and changed the dedicated Olivia launcher to one OpenMP/MKL thread and
  one Inductor compiler worker; the existing loader single-thread fix remains.
- Added the Olivia comparison job: replay A0/B2/D2 historical predictions with
  the recorded final jackknife omission, verify the saved D2 model with a strict
  full-state load over all 78 test recordings, then run the shared C2 replay.
  The historical A0/B2 supervised model files are no longer present on Fox, so
  D2 is the available complete model/prediction pair for inference parity.
- The user explicitly requested keeping Fox restart jobs 4320073 (C2) and
  4320083 (C0) while also running C0/C2 on Olivia. Neither Fox job will be cancelled.

- Created four dedicated `configs/stage2_C*_olivia.yaml` files, portable full/smoke data groups, and the Slurm output directory. C0/C2 point to their recovered 150k checkpoints; C1/C3 start fresh. All four use batch size 50, seed label 42, and 200k total steps. Original SPR configs and `train.sh` remain unchanged. Added ignore exceptions for data configs and the Stage 2/3 CSVs.

- CSV update stage2 `C0`: Pretrain status=PREPARING.

- CSV update stage2 `C1`: Pretrain status=PREPARING.

- CSV update stage2 `C2`: Pretrain status=PREPARING.

- CSV update stage2 `C3`: Pretrain status=PREPARING.

- CSV update stage3 `Fox-artifact-recovery`: Status=COPIED_VERIFYING.

- CSV update stage3 `C-smoke-C1`: Status=RUNNING, Job_ID=2567624.

- CSV update stage3 `C-smoke-C0`: Status=RUNNING, Job_ID=2567623.

- CSV update stage3 `C-smoke-C0`: Status=RUNNING, Job_ID=2567655.

- CSV update stage3 `C-smoke-C1`: Status=RUNNING, Job_ID=2567656.

- CSV update stage3 `C-smoke-C1`: Status=PASSED, Job_ID=2567656.

- CSV update stage3 `C-smoke-C0`: Status=PASSED, Job_ID=2567655.

- CSV update stage3 `C-smoke-C3`: Status=RUNNING, Job_ID=2567654.

- CSV update stage3 `C-smoke-C2`: Status=RUNNING, Job_ID=2567657.

- CSV update stage3 `C-smoke-C2`: Status=PASSED, Job_ID=2567657.

- CSV update stage3 `C-smoke-C3`: Status=PASSED, Job_ID=2567654.

- All four retry smoke jobs reached 100 optimizer steps and returned success.
  Their custom loader children printed cleanup EOF/broken-pipe messages after
  the deliberate stop; training itself completed the required steps.
- Submitted Fox reference job **4325603** in an isolated work directory and
  Olivia comparison job **2567683**. The shared fixture and code snapshot were
  uploaded to Fox without touching its live checkout or existing training jobs.
  Missing evaluation-only IPython dependencies were installed into the isolated
  run's `runtime-deps` directory, leaving both shared Conda environments intact.
- Added `scripts/olivia/check_gate.py`: it requires four successful smoke runs,
  scorer and frozen-inference validation, 100 finite replay steps, identical LR
  sequences, and measured FP32 prediction/gradient/update agreement. Tolerances
  are explicit in the script and no long run can pass the gate without results.
- Renamed the two new resume configurations to `stage2_C0_olivia_restart.yaml`
  and `stage2_C2_olivia_restart.yaml` and updated the run driver accordingly,
  following the separate restart-config convention.

- CSV update stage3 `Olivia-comparison`: Status=RUNNING, Job_ID=2567683.

- Limited the data-config ignore exception to the two new Olivia YAMLs, so
  unrelated existing scratch files in that directory stay outside this change.
  Work storage has about 194 TB available; Fox's project mount has 31 GB and its
  work mount 13 TB. Existing Fox output roots remain unchanged.

- Comparison job 2567683 passed A0/B2/D2 scorer replay and strict D2 model
  inference over all 78 recordings. D2 SELD was 0.3971157420763225 versus
  0.39711574654842463 on Fox (difference -4.47e-9); other metric differences
  were also within the declared limits. Its subsequent replay subprocess could
  not acquire CUDA while the evaluation parent retained its GPU context.
  Changed the handoff to call replay in the same process. Added a replay-only
  retry that requires the completed, passed evaluation records and rechecks
  their measured inference tolerances, avoiding repeated jackknife scoring.

- CSV update stage3 `Fox-artifact-recovery`: Status=VERIFIED.

- CSV update stage3 `Fox-reference`: Status=SUBMITTED, Job_ID=4325603.

- CSV update stage3 `Olivia-comparison`: Status=SUBMITTED, Job_ID=2567683.

- CSV update stage2 `C0`: .

- CSV update stage2 `C2`: .

- CSV update stage3 `Fox-reference`: Status=COMPLETED, Job_ID=4325603.

- CSV update stage3 `scorer-A0`: Status=PASSED, Job_ID=2567683.

- CSV update stage2 `C0`: .

- CSV update stage2 `C1`: .

- CSV update stage2 `C2`: .

- CSV update stage2 `C3`: .

- Restored only `pass_through_decoder` in Fox’s live checkout so its pending C0/C2 jobs can execute. Saved the original file outside the checkout and wrote a dated Fox report. Both jobs remain queued with batch 50 and 200k total steps; neither was cancelled or resubmitted. Both restart archives are readable ZIPs and record global_step=150000. Method SHA256: `405e058397bf569079f7afff963d72243b265178f0fd98491ff8997c5470a44a`.

- CSV update stage3 `scorer-B2`: Status=PASSED, Job_ID=2567683.

- CSV update stage3 `Olivia-comparison`: Status=RUNNING, Job_ID=2567683.

- SHA256 verification of Fox last-versus-step-150000 checkpoints and loader source is recorded in `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/2026-10-18-stage2-and-comparability/stage3/fox-resume-checks.json`.

- CSV update stage3 `scorer-D2`: Status=PASSED, Job_ID=2567683.

- CSV update stage3 `D2-inference`: Status=PASSED, Job_ID=2567683.

- CSV update stage3 `Olivia-comparison`: Status=FAILED, Job_ID=2567683.

- CSV update stage3 `Olivia-comparison`: Status=RUNNING, Job_ID=2567726.

- CSV update stage3 `Olivia-comparison`: Status=AWAITING_FOX_REPLAY, Job_ID=2567726.

- CSV update stage3 `Olivia-evaluation-2567683`: Status=EVALUATION_PASSED_REPLAY_HANDOFF_FAILED, Job_ID=2567683.

- CSV update stage3 `Fox-Olivia-gate`: Status=PASSED.

- CSV update stage2 `C0`: Pretrain status=SUBMITTED.

- CSV update stage2 `C1`: Pretrain status=SUBMITTED.

- CSV update stage2 `C2`: Pretrain status=SUBMITTED.

- CSV update stage2 `C3`: Pretrain status=SUBMITTED.

- CSV update stage3 `Olivia-comparison`: Status=PASSED, Job_ID=2567726.

- Focused replay retry **2567726** completed successfully. The measured Fox–Olivia gate passed: prediction relative L2 4.29e-7, gradient relative L2 1.05e-6, update relative L2 3.53e-5, gradient cosine 0.99999999999948, and zero LR difference. Both runtimes completed 100 finite BF16 replay steps. These validate migration within the declared checks; independent stochastic training trajectories may differ.

- Submitted long Group C production array **2567729** (four tasks, one GPU each, maximum 168 hours). C0/C2 resume their verified 150k states; C1/C3 start fresh. All target 200k steps at batch 50. Runtime data/checkpoints stay on work storage. Historical seed ordering is preserved: seed label 42 is set after model initialization, as in the old code.

- Added `stage2_C1_olivia_restart.yaml` and `stage2_C3_olivia_restart.yaml`, retaining all identity fields and the full 200k LR schedule while resuming their planned first-phase checkpoints.

- CSV update stage3 `Olivia-array-2567729`: Status=RESCHEDULED_BEFORE_START, Job_ID=2567729.

- CSV update stage2 `C0`: Pretrain status=SUBMITTED.

- CSV update stage2 `C1`: Pretrain status=SUBMITTED.

- CSV update stage2 `C2`: Pretrain status=SUBMITTED.

- CSV update stage2 `C3`: Pretrain status=SUBMITTED.

- CSV update stage2 `C1`: .

- CSV update stage2 `C3`: .

- Confirmed all tasks of Olivia array 2567729 were still PENDING and cancelled only that unused array. Submitted first-phase array **2567739** (17 hours) and dependent continuations **{'C1': '2567740_1', 'C3': '2567741_3'}**. Each continuation depends on its own matching first-phase task, rather than on unrelated groups. No Fox job was cancelled.

- CSV update stage2 `C0`: Pretrain status=RUNNING, Job_ID=2567742.

- CSV update stage2 `C1`: Pretrain status=RUNNING, Job_ID=2567743.

- CSV update stage2 `C3`: Pretrain status=RUNNING, Job_ID=2567739.

- CSV update stage2 `C2`: Pretrain status=RUNNING, Job_ID=2567744.

- CSV update stage3 `Fox-reference`: Status=PASSED.

- CSV update stage3 `Olivia-comparison`: Status=PASSED.

- CSV update stage2 `C0`: .

- CSV update stage2 `C1`: .

- CSV update stage2 `C2`: .

- CSV update stage2 `C3`: .

## Verified launch state

All four tasks of first-phase array **2567739** are RUNNING and have completed at least 100 real optimizer steps. C0/C2 restored all checkpoint states and reached at least **150100**; C1/C3 reached at least **100**. Losses are finite. C1/C3 continuation jobs **2567740_1** and **2567741_3** are pending their matching first-phase success; their checkpoint paths were checked against the actual identity function without importing Torch on the login node. Fox jobs **4320083** (C0) and **4320073** (C2) remain pending for priority.

Results: `results/stage3.csv` contains all measured migration/smoke checks and replay diagnostics; `results/stage2.csv` contains Group C job IDs, continuation IDs, and verified training progress. Current models/checkpoints and detailed comparison arrays are under `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/2026-10-18-stage2-and-comparability`. The run helpers update completion status and measured final losses when each phase exits.

Historical seed ordering is retained: the seed call follows model initialization, so label 42 does not guarantee identical fresh initial weights. The migration gate checks frozen-state behavior and finite short replays; it does not establish statistical equivalence of independently shuffled full training runs. No final 200k or new SELD outcome is claimed at launch.

- CSV update stage2 `C0`: Pretrain status=Finished, Job_ID=2567742.

- CSV update stage2 `C2`: Pretrain status=Finished, Job_ID=2567744.

- CSV update stage2 `C1`: Pretrain status=CHECKPOINTED, Job_ID=2567743.

- CSV update stage2 `C3`: Pretrain status=CHECKPOINTED, Job_ID=2567739.

- CSV update stage2 `C0`: Pretrain status=Finished.

- CSV update stage3 `C0-completion-check-20261019`: Status=COMPLETED_200K, Job_ID=2567739_0.

- CSV update stage2 `C1`: Pretrain status=CHECKPOINTED.

- CSV update stage3 `C1-completion-check-20261019`: Status=CHECKPOINTED_50K_CONTINUATION_PENDING, Job_ID=2567739_1.

- CSV update stage2 `C2`: Pretrain status=Finished.

- CSV update stage3 `C2-completion-check-20261019`: Status=COMPLETED_200K, Job_ID=2567739_2.

- CSV update stage2 `C3`: Pretrain status=CHECKPOINTED.

- CSV update stage3 `C3-completion-check-20261019`: Status=CHECKPOINTED_50K_CONTINUATION_PENDING, Job_ID=2567739_3.
