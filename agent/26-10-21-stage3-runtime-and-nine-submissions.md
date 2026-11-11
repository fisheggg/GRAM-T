# 26-10-21 Stage 3 runtime and nine requested submissions

Submitted A0, B0, C0, C1, C2, D0, E0, F0, and H0 on Olivia. Each full training job depends on a separate GPU smoke test, which depends on successful shared data preparation and CPU runtime checks. At the final queue check, data preparation was running and all GPU jobs were pending dependencies. No GPU smoke or full training result is claimed yet; the `nb_testing` GPU reservation ends at **00:00 on 22 October, Europe/Oslo**.

| Run | Seed | GPU smoke | Full training |
|---|---:|---:|---:|
| A0 | 42 | 2595272 | 2595273 |
| B0 | 42 | 2595274 | 2595275 |
| C0 | 42 | 2595276 | 2595277 |
| C1 | 43 | 2595278 | 2595279 |
| C2 | 44 | 2595280 | 2595281 |
| D0 | 42 | 2595282 | 2595283 |
| E0 | 42 | 2595284 | 2595285 |
| F0 | 42 | 2595286 | 2595287 |
| H0 | 42 | 2595288 | 2595289 |

All full jobs request one GPU, 16 CPUs, 128 GB host memory, and 168 hours under `nn8104k/accel`. Smoke jobs have a 30-minute limit, use the configured batch sizes, perform two pretraining optimizer steps and one supervised update, and write into separate smoke directories. Failed prerequisites cancel dependent jobs rather than starting long training. Submission records and exact commands are in [stage3_submissions_2026-10-21.csv](../results/stage3_submissions_2026-10-21.csv) and [submission journal](../results/stage3_submission_journal_2026-10-21.jsonl).

## Branch inspection and implementation

Read `li_review` at **d57e15b**, including its SPR model and change report, without switching branches, merging, or popping the stash. Its channel-separated projection is in the separate SPR pipeline, not the GRAM backbone. Its optional cross-channel block mixes before masking, matching the review's leakage finding. H adapts the shared per-channel projection idea to the planned GRAM 16x8 mel/time patches; cross-channel attention follows visible-token selection. H uses per-channel LayerNorm as part of isolation, 1,400 tokens, channel-local legacy frame-index targets, and the same patch layout downstream. This is an adaptation, not an unchanged copy of that pipeline. Full details and scope are in [li_review_changes.md](../li_review_changes.md).

Added the missing combined runtime in `scripts/stage3/run.py`, model adapters in `models.py`, and manifest-based SELD/data preparation in `seld.py`. The requested groups execute fresh pretraining followed by SELD. Group G's additional STARSS SSL orchestration is explicitly unsupported and was not submitted. The runtime retains the CSV's legacy SPR alignment, all-row/all-slot loss, adjacent readout, original pretraining RIR convention, and `legacy_last_loo` final scoring. Complete-data point scoring is available as a switch. Encoder checkpoint loading checks shapes and the minimum loaded fraction before training. Both phases seed before creating their models and data.

Batch sizes remain **80 audios × 16 crops** for pretraining and **512 sequences** for SELD. Internal feature microbatches of 32 accumulate sample-weighted gradients into each configured optimizer update. The pretraining optimizer, 200k-step warmup/cosine horizon, and gradient clipping at 5 are retained. Milestone checkpoints remain every 50k steps; a resumable `last.ckpt` is additionally saved every 1k steps to limit lost progress on long runs, especially H. Explicit `--resume` checks unchanged seed/encoder/pretraining settings before restoring a continuation. Completion within a single 168-hour allocation has not been established on GPU.

The short `submit_stage3.sh` keeps its `train.sh` layout. It now accepts an optional YAML argument, optional smoke mode, and a frozen-code directory through the submission environment. Each full run writes its own CSV/JSON results and appends final metrics to `results/stage3_metrics.csv`; F and LR are fractions. Existing measured results and the original source CSV are retained.

## Data preparation and validation

The configured STARSS audio, metadata, and feature directories were empty. The original local archives contain 168 real recordings and 1,200 synthetic recordings at 24 kHz. Preparation job **2595248** recovers these without needing Fox, resamples to 32 kHz, and generates native and separate N3D feature caches. Matched dipoles are scaled after waveform RMS normalization and before spectral/IV extraction; vertical adaptation occurs in the downstream model. At the last check, at least **650/1,368** recordings had been prepared. Shared CPU runtime check **2595254** depends on preparation completing.

CPU check **2595239** verified real model forward/backward for reconstruction, pair SPR, orthogonal SPR, separate decoders, and H; encoder tensors loaded 150/150, and every SELD output shape was `[1,20,39]`. H's local projection isolation and patch reconstruction checks passed, as did repeated-seed initialization. The overall job then failed at the manifest stage because data were absent. It is not recorded as a complete runtime pass.

CPU optimizer check **2595252** subsequently **passed** two real Lightning manual optimizer/scheduler steps with the full SPR model and synthetic feature crops. It did not exercise the audio pipeline or GPU. Evidence is in [optimizer verification](../results/stage3_runtime_checks/2595252-optimizer.json) and its Slurm log. Full CPU checks and real-data GPU smoke tests remain dependencies of the submitted training jobs.

The first feature preparation attempt, **2595246**, exposed a label-parser argument error. Corrected it to load and convert the metadata dictionary, and corrected use of the already separate matched cache configured by B0/D0/F0. Recovered the correctly scaled first file into that cache before restarting as 2595248. No historical cache was overwritten. Missing CPU dependencies were installed only in `.venv-wandb`; existing pure-Python IPython dependencies were copied into the isolated GPU overlay. Shared Conda environments were not changed.

## Frozen source

Jobs use `run_snapshots/stage3-20261021T212132Z`, containing **80 files**, with a verified `SHA256SUMS` manifest. Its SHA-256 is `dc8eda44dff9e1ed3ebae166c3d48281a8ab20ae4050635a15f40b1eb6b60d05`. The isolated GPU dependency overlay is referenced by symlink; source/config files are copied. This keeps later working-tree edits from changing queued job code. Syntax checks, the 24-config plan validator, nine requested runtime config checks, snapshot checksums, and Slurm dependency/resource inspection passed.
