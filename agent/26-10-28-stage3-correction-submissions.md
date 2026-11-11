# Stage 3 correction: 24 finetuning jobs submitted on Olivia

Two fixed pretrained checkpoints are used: pair-SPR (C0) and vanilla GRAM (A0), both from the completed 200k-step Fox-source Olivia reruns. Each has four preprocessing groups and three finetuning seeds (42/43/44, suffixes 0/1/2). No new pretraining is submitted.

| Groups | Preprocessing | Pair-SPR job IDs | GRAM job IDs |
|---|---|---|---|
| A0/A1/A2 | Native | 2665576–2665578 | 2665588–2665590 |
| B0/B1/B2 | Swap only | 2665579–2665581 | 2665591–2665593 |
| C0/C1/C2 | Gain only | 2665582–2665584 | 2665594–2665596 |
| D0/D1/D2 | Both | 2665585–2665587 | 2665597–2665599 |

Names are `stage3-correction-{spr,gram}-{A,B,C,D}{0,1,2}`. Within each seed, the saved head initialization and 100 epoch data orders are shared across conditions and checkpoint families. Each family uses its one fixed encoder checkpoint. Training uses FP32, deterministic algorithms, full batch 512, Adam at 1e-4, and 100 epochs.

The latest user instruction supersedes the earlier independent-validation request. The original Fox protocol is retained: train folds 1–3 (1,290 recordings), validation and test both fold 4 (78 recordings), validation every third epoch, and the original minimum validation composite criterion for selecting the final checkpoint. F1 is the primary reported metric; ER/LE/LR are also saved. Full-78 metrics and historical-77 metrics (omitting `fold4_room15_mix010.csv`) are both saved for comparison. Original random training trajectories are not reproduced; paired control applies to the new runs.

Verified before submission: 24 configurations, source hashes, checkpoint paths, protocol settings, and the original Fox validation/checkpoint-selection source. Slurm accepted all 24 jobs, each with `afterok:2665501` and cancellation on an invalid dependency. Each training process additionally requires the successful preflight record and matching source/checkpoint/control hashes. Resources per run: one GPU, eight CPU cores, 128 GB RAM, 12-hour limit.

GPU preflight 2665501 will regenerate native/gain caches together, verify exact paired encoder/head initialization, test full-batch updates in all eight groups, check deterministic replay, and exercise validation/scoring and W&B. At the final queue check, preflight was PENDING (Priority), and all training jobs were PENDING (Dependency). These runtime checks and actual training progress have therefore not yet been verified; no new F1 results are available.

Outputs: `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/2026-10-28-stage3-correction`.

Frozen code/configs: `/cluster/projects/nn8104k/homes/jinyueg/ambi_ssl/GRAM-T/run_snapshots/stage3-correction-20261028`.

Job records: `jobs.csv` and `submissions.json` under the output directory and repository `results/stage3_correction_2026-10-28/`.

Limitations: using the same validation/test fold preserves Fox's test-set selection bias. Training has no automatic optimizer-state resume; failed or timed-out runs require explicit recovery.
