# 26-10-24 — Investigate completed legacy reruns and FOA matching

Verified all seven legacy SELD runs completed 100 epochs / 8,400 updates and independently reproduced all saved metrics from their 78 prediction files to 1e-12. Pair mean improves 0.41666 → 0.40148 versus historical 0.38868, recovering 54.3% of the prior gap. The residual cause remains unisolated: encoders, initialization, data order, execution and possibly feature caches differ from historical training.

Matched FOA improves reconstruction (A0→B0, −0.00635 legacy SELD) but worsens pair SPR (C0→D0, +0.01825). Both full-set paired recording intervals include zero and exclude seed/selection uncertainty. Separate pretrained checkpoints and bundled gain/swap changes prevent causal attribution.

Found a concrete provenance discrepancy: Stage 3 applies dipole gain after global RMS normalization; reviewed branch d57e15b applies gain before normalization. These are not equivalent under the joint log-mel/IV LayerNorm. Sampled caches confirm the conversion was applied. A waveform-energy/cached-feature analytical check quantifies the order difference, without claiming it causes the performance change. This cannot explain the unconverted C versus historical B2 residual.

Added [report](../results/stage3_legacy_gap_analysis_2026-10-24/report.md), two reproducible CPU analysis scripts, JSON evidence and comparison CSV. Recommended fixed-encoder gain/swap/order controls and old/new encoder finetuning under identical supervised conditions. No production changes or jobs; no Torch imported on the login node. Existing worktree edits were preserved.
