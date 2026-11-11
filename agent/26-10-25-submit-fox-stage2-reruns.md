# 26-10-25 — Submit fresh Fox-code A0/B0/C0/D0 reruns

Continued the requested fresh pretraining-plus-finetuning reruns after the previous approval-review usage failure. The earlier attempt had not submitted a job. Retrieved source remains preserved at `run_snapshots/fox-stage2-rerun-20261024/original/`: 103 files verified byte-for-byte against Fox, repository `190efd81bf8748dff98946ecb353c350fc571283`, including its local SPR edit. Four sampled native feature/label pairs matched Fox in shape, dtype and sampled content hashes.

Completed the isolated launcher, frozen configs and progress/result hooks. The four existing-source modifications are portable paths, a progress callback, result recording and B0/D0's downstream feature permutation. The original encoder/decoder/loss code and full-batch optimization remain in use. Pretraining is 200k updates at 80×16 crops; finetuning is 100 epochs at batch 512. Original late seeding, fold-4 checkpoint selection and scoring remain. Reports use F1 with ER/LE/LR. B0/D0 regenerate a separate matched cache with dipole gain before RMS normalization.

Both pretraining smoke jobs passed two real full-batch updates: A0 **2615308**, C0 **2615309**. Native supervised smoke jobs A0 **2615318** and C0 **2615312** passed one full-batch update and verified 150 encoder tensors. Peak allocation was about 65 GB for reconstruction pretraining, 72 GB for SPR pretraining, and 68 GB for supervised training. Reconstruction compilation emitted graph-break warnings but completed successfully. Original background loader shutdown emitted warnings after smoke completion; saved checkpoints and explicit success markers were verified.

| Group | Fresh pretraining | Dependent finetuning |
|---|---:|---:|
| A0 | 2615316 | 2615320 |
| B0 | 2615317 | 2615321 |
| C0 | 2615310 | 2615314 |
| D0 | 2615311 | 2615315 |

All four pretraining jobs were confirmed RUNNING. C0/D0 had logged finite optimizer updates. All four finetuning jobs were PENDING on successful pretraining and supervised checks. Matched feature extraction **2615306** was RUNNING; B0/D0 supervised checks **2615319/2615313** wait for it. The CPU job's initial four-hour reservation exceeded remaining account billing capacity; reducing its limit to one hour allowed it to start. Extraction had passed 251/1,368 recordings at the recorded check.

Evidence and current submission state: [job CSV](../results/fox_stage2_reruns_2026-10-24/jobs.csv), [Slurm snapshot](../results/fox_stage2_reruns_2026-10-24/slurm-status.txt), [source diff](../results/fox_stage2_reruns_2026-10-24/source-diff.patch), [launch hashes](../results/fox_stage2_reruns_2026-10-24/launch-manifest.json), and [launcher notes](../scripts/fox_stage2_rerun/README.md). Python/shell syntax, config mapping, source hashes and dependency chains were checked. Existing production outputs, running jobs and shared environments were preserved. No Torch was imported on the login node.

The long runs and matched-cache checks have not finished. These reruns retain historical test-set checkpoint selection and historical nondeterministic initialization; they are protocol-reproduction runs, not unbiased final-test comparisons.
