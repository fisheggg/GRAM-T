# 26-10-28 — Verified Fox–Olivia run comparison

Read source, configs, Slurm state, saved predictions and checkpoints on both clusters. Independently rescored 22 prediction sets (1,716 CSVs) on all 78 test recordings, reproducing recorded metrics to 1e-12 while accounting for the legacy scorer's differing omitted recording.

Restoring the historical downstream protocol recovers 3.41 F1 points for reconstruction and 2.72 for pair-SPR on the existing seed-42 Olivia encoders. Fresh runs through recovered Fox source score 42.89% / 43.41% versus historical Fox 42.24% / 45.46%. The residual pair deficit is not causally isolated. Late pretraining seeding and unseeded historical supervised initialization prevent exact run reproduction.

Latest corrected reconstruction/pair runs score 42.13% / 41.85%, changes -0.76 / -1.56 pp; recording-paired intervals include zero. Separate pretrained weights confound the correction comparisons. GPU job 2665218 verified 150 loaded encoder tensors per run, exact application of the swap, close regeneration of five matched caches, and the retained SPR target-alignment defect (8/200 patches).

Full native cache checks: all 1,368 labels match exactly; 1,346/1,368 feature-file hashes differ. Five complete numerical comparisons found only 1–6 differing IV values per file, maximum 2.4e-7. Fixed-checkpoint replay arrays independently reproduce prediction relative error 4.3e-7 and gradient error 1.0e-6. These checks do not establish full stochastic training equivalence.

Artifacts were saved in the local Ambi-SSL workspace under investigation_2026-10-28/report.md and verification.md, with scripts and JSON/CSV evidence. Remote audit outputs are in each cluster's 2026-10-28-investigation directory. Fox 4363535 and Olivia 2665218 completed; the earlier Olivia 2665119 audit stopped at a strict regeneration tolerance after completing cache hashes, and its discrepancy was quantified in the successful follow-up. Pending redundant jobs 2665142/2665162 were cancelled. No production code, cache, checkpoint or existing training job changed.

Not run: a same-checkpoint, same-initialization native/swap/gain/both finetuning ablation. Remaining causal attribution is unresolved; historical protocol reproduction retains test-set checkpoint selection.
