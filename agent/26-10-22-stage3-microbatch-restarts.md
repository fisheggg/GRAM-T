# 26-10-22 Stage 3 microbatch change and replacement jobs

Applied `runtime.pretraining_feature_microbatch` in all 24 Stage 3 YAMLs: 256 crops for A–G, 64 for H. The model and execution metadata now consume the same validated setting. The effective pretraining batch remains 80 audios × 16 crops; the 200,000-step target, objectives, optimizer, seeds, and SELD batch/microbatch settings are unchanged.

Submitted the nine requested replacements and confirmed all are RUNNING. Cancelled the nine corresponding original Olivia jobs after successful restore checks. No Fox jobs were touched.

| Run | Cancelled job | New running job | Microbatch | Restored step |
|---|---:|---:|---:|---:|
| A0 | 2595273 | 2597889 | 256 | 11000 |
| B0 | 2595275 | 2597890 | 256 | 12000 |
| C0 | 2595277 | 2597891 | 256 | 12000 |
| C1 | 2595279 | 2597892 | 256 | 13000 |
| C2 | 2595281 | 2597893 | 256 | 12000 |
| D0 | 2595283 | 2597894 | 256 | 11000 |
| E0 | 2595285 | 2597895 | 256 | 11000 |
| F0 | 2595287 | 2597896 | 256 | 9000 |
| H0 | 2595289 | 2597897 | 64 | 9000 |

GPU verification job **2597887** passed two actual real-data updates for each preserved checkpoint, verifying restored weights, optimizer step/LR, scheduler epoch, requested microbatch, and finite losses. Preserved checkpoint copies have recorded SHA-256 checksums. Steps performed after those saved checkpoints are replayed; exact counts are retained in the CSV.

All nine production replacements subsequently restored their recorded steps and logged resumed updates successfully. Verified their new W&B runs in `ritmo/ambi-ssl`: microbatch, seed, batch size, 200k-step budget, finite loss history, and provenance linking the prior run and checkpoint. Each continuation retains previous config/W&B metadata in `pretrain/resume_history/` and writes `training_started.json`. Microbatch changes and dataloader restart do not reproduce a bit-identical trajectory.

Source is frozen at `run_snapshots/stage3-microbatch-20261022T105212Z` (82 files); the original snapshot is retained. Manifest SHA-256: `dfd232f25c5f73c800f233a99da507c4a61c86ac183e82c84287f906e89830b5`. Plan validation, all nine config checks, syntax checks, and snapshot checksums passed. Shared Conda environments were untouched. The short submission script remains unchanged; it accepts `train --resume --resume-checkpoint PATH`.

H0 still needs a later checkpoint continuation if it exceeds its 168-hour allocation. Performance/accuracy conclusions are not inferred from these short restore checks.

[Replacement job records](../results/stage3_microbatch_restarts_2026-10-22.csv), [restore checks and online verification](../results/stage3_microbatch_restart_2026-10-22/), and [runtime instructions](../scripts/stage3/README.md).
