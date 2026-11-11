# 26-10-24 — Analyze the Stage 2 / Stage 3 performance gap

Verified that seven Stage 3 runs completed pretraining and SELD successfully; F0/H0 remain in pretraining. Matched historical methods to the renamed groups: A→A0, B2→C0/C1/C2, B1→E0. Recorded the comparison in [a new CSV](../results/stage3_vs_stage2_2026-10-24.csv), with a [detailed report](../results/stage3_gap_analysis_2026-10-24/report.md), raw evidence and plots.

The strongest explanation is the changed downstream protocol: historical validation and test were the same fold; current validation removes 20 of 90 real training recordings, shares all its real rooms with training, and poorly predicts unseen-room test rankings. This is a confirmed confound, not a measured causal decomposition. Current pair runs average SELD 0.41666 versus historical B2 0.38868; the reconstruction baseline also declines by 0.02587.

Independently rescored all seven prediction sets, reproduced recorded legacy metrics to 1e-12 tolerance and computed full-set scores. The evaluation correction leaves most of the between-stage gap. W&B confirms complete encoder loading and historically similar pretraining loss averages. Calculated paired recording intervals for current-run contrasts; all four inspected contrasts include zero. Historical raw predictions/models are missing from their recovered directories, so cross-stage paired uncertainty was not recomputed.

Recommended controlled finetuning of old/new encoders under one protocol before further pretraining changes. Added only analysis scripts, evidence, reports and a comparison CSV; production code/configs, experiment metrics, checkpoints and jobs were unchanged. No Torch was imported on the login node and no jobs were submitted or cancelled.
