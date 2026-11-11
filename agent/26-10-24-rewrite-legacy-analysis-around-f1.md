# 26-10-24 — Rewrite legacy gap and conversion analysis around F1

Rewrote the [report](../results/stage3_legacy_gap_analysis_2026-10-24/report.md) with macro location-dependent F1 at 20° as the primary metric. Reran saved-prediction scoring for all seven legacy runs, reproduced recorded metrics to 1e-12, and regenerated the JSON/CSV with F1 recovery, seed spread and paired recording jackknife intervals for both legacy and full-set scoring. Added a PNG/PDF comparison and reproducible plotting script.

The pair mean improves 40.19%→42.73%, recovering 47.7% of its historical gap to 45.51%. Its observed seed SD is only 0.143 percentage points, so the earlier SELD-based variability explanation is weak for F1. Reconstruction and orthogonal residual deficits are 0.50 and 0.44 points.

Matching changes legacy F1 by +0.56 points for reconstruction and −2.68 for pair SPR. Both F1-specific paired intervals include zero. Added exact scorer-counter decomposition with/without the 20° gate: on full-set scoring, D0−C0 is −2.08 points without the gate and −0.08 additional points from its gate penalty. Per-class counters identify missed detections in class 5 and spatial failures in class 11; class 12 has sparse support. These are descriptive diagnostics, not causal attribution.

Retained the normalization-order discrepancy and conversion confounds. Explicitly documented that existing checkpoints were selected by SELD, not F1; recommended F1-based development selection for controlled follow-ups. No production changes, training jobs, threshold searches, or Torch imports on the login node.
