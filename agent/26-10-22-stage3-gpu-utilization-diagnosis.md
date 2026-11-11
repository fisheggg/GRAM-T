# 26-10-22 Stage 3 GPU utilization diagnosis

All nine original trainings remain running and were not modified or restarted. Their low GPU utilization is primarily caused by CPU dispatch/autograd overhead from **40 eager microbatches of 32 crops** per optimizer update. The configured batch remains 80 audios × 16 crops = 1,280 crops.

Live samples showed **24–35% GPU utilization** in A0/B0/C0/C1/C2/D0/E0/F0 and **93% in H0**. No loader crashes, stall errors or OOMs were found. FLAC decoding warnings are present; Python deduplication means displayed counts do not measure all skipped records.

Submitted two isolated diagnostic batch jobs, **2597811** and **2597838**, both completed successfully. They used frozen source, fresh weights, no W&B and no experiment checkpoints. C0's real-data median update fell from **3.85 s to 0.73 s** at microbatch 256, a **5.27×** improvement, with data waits below **0.25 ms** after warmup. Cached-feature model updates improved about 6× in C0/F0; H0 improved about 20%. A PyTorch trace confirmed thousands of small kernels. Batching finite-loss checks did not materially improve speed.

Recommend making the feature microbatch configurable and recorded, initially 256 for validated C0/F0 and 64 for H0. Other groups still need a check at the larger setting. No training configs or runtime source were changed. Microbatch changes can alter numerical/RNG trajectories; the diagnostics do not establish convergence or final SELD accuracy.

Current speeds project 198–276 hours for pretraining alone, beyond the 168-hour allocations. All nine runs have resumable last checkpoints. Rechecked all 80 frozen-source file checksums successfully; shared Conda environments were untouched.

[Detailed report and methodology](../results/stage3_performance_2026-10-22/report.md), [nine-run measurements CSV](../results/stage3_performance_2026-10-22.csv), [microbatch benchmarks CSV](../results/stage3_microbatch_benchmarks_2026-10-22.csv), and [pipeline timing CSV](../results/stage3_pipeline_timings_2026-10-22.csv). Added reusable [Python diagnostic](../scripts/stage3/profile_performance.py) and [Slurm wrapper](../scripts/stage3/profile_performance.sbatch).
