# 26-10-19 — Check completed Stage 2 training on Olivia

C0 and C2 completed their 200,000-step pretraining targets. C1 and C3 completed the intentional 50,000-step first phase and have queued continuation jobs for their remaining 150,000 steps. All four first-phase Slurm jobs finished with `COMPLETED` and exit code `0:0`.

| Group | Pretext | Slurm job | Verified checkpoint step | Latest reconstruction loss | Latest SPR loss | Status |
|---|---|---|---:|---:|---:|---|
| C0 | w_to_xyz | 2567739_0 | 200,000 | 0.7298107 | 0.0853040 | Full pretraining complete |
| C1 | orthogonal | 2567739_1 | 50,000 | 0.6298370 | 0.0614607 | First phase complete; continuation 2567740_1 pending |
| C2 | pair | 2567739_2 | 200,000 | 0.6808656 | 0.0028780 | Full pretraining complete |
| C3 | foa_to_iv | 2567739_3 | 50,000 | 0.6875676 | 0.0102035 | First phase complete; continuation 2567741_3 pending |

The loss values are the final recorded training-batch values, not validation metrics or epoch averages. They do not establish which checkpoint will perform best on SELD, and SPR losses across different pretext definitions/scales are not directly comparable.

## Completion evidence

- Slurm accounting confirms successful exits after about 7 hours 43–50 minutes. On 18 October, C0 finished at 21:55:09, C2 at 21:59:23, C1 at 22:00:25, and C3 at 22:02:13, in Europe/Oslo time.
- C0/C2 logs state that `max_steps=200000` was reached. C1/C3 reached the configured 50k chunk stop. All logged progress samples have finite reconstruction, SPR, and total losses, with nondecreasing step counters.
- Every group has both `last.ckpt` and its numbered final/chunk checkpoint, each approximately 1.65 GB. ZIP directories and the complete `data.pkl` opcode stream were readable, with 1,660 tensor-storage entries each. The serialized `global_step` is 200,000 for C0/C2 and 50,000 for C1/C3 in both files.
- These checks did not deserialize model tensors with Torch, recompute full-file CRCs, or perform a new inference run. They establish structural/metadata consistency, not a complete numerical reload test. No Torch imports were run on the login node.

Checkpoint roots are under `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/runs/2026-10-18-stage2-and-comparability/stage2/<group>/train_logs/saved_models_spr/`. Exact paths are recorded in the updated experiment CSVs. The inspected logs are [C0](../slurm_log/C-2567739_0.out), [C1](../slurm_log/C-2567739_1.out), [C2](../slurm_log/C-2567739_2.out), and [C3](../slurm_log/C-2567739_3.out).

## Continuations and reservation

The C1/C3 continuation dependencies have been satisfied; `scontrol` reports `Dependency=(null)`. Both jobs remain pending for unavailable GPU capacity. The active `nb_testing` reservation covers all 112 Olivia GPU nodes from 19 October 08:00 through 22 October 00:00, Europe/Oslo. Slurm currently estimates both continuations can start on **22 October at 00:00:01**; this is a scheduling estimate, not a guaranteed start time.

The existing C1/C3 restart YAMLs point at their respective `last.ckpt` and preserve the 200k-step learning-rate horizon. No jobs were cancelled, resubmitted, or reconfigured during this check.

## Shutdown diagnostics

All four logs contain EOF/broken-pipe/file-not-found exceptions from noise/RIR multiprocessing workers and manager cleanup. The exceptions begin only after the final recorded training progress; C0/C2 have already reported their normal max-step stop. Slurm still records exit `0:0`, and checkpoint metadata matches the intended stop. The evidence points to a teardown/IPC-cleanup problem rather than an interrupted training phase. A future cleanup fix should be tested separately and should not change historical objectives or overwrite these artifacts.

## Other recent jobs and limits

The Fox–Olivia retry job 2567726 completed successfully. The earlier 2567683 attempt remains marked failed in Slurm because of its previously documented CUDA process-handoff issue; its completed evaluation checks were reused by the successful retry. No additional Stage 3 training jobs appear in the checked Olivia accounting history.

Fox's interactive control sockets are no longer connected, so Fox C0/C2 completion was not rechecked. Historical Fox-status cells are therefore not refreshed as though they were current measurements. The new results in this report refer to Olivia.

## Recorded changes

Refreshed C0–C3 verification fields in [results/stage2.csv](../results/stage2.csv), including exact checkpoint paths and current continuation status. Added four completion-check rows to [results/stage3.csv](../results/stage3.csv), retaining historical comparison rows. This report records the scheduler, log, and checkpoint checks; no model code, configuration, or running job was changed.
