# 26-11-04 — Suggestions for new ambisonics SSL pretext tasks

Advice only. No code, configs or jobs changed.

## Diagnosis from the project's own evidence
- All current SPR targets (pair / orthogonal / foa_to_iv / w_to_xyz) are functions of the IV channels already in the input. Frozen probe: best encoder readout 22.4° vs raw-IV linear readout 22.8° (li_review report 2 §4). Fine-tune differences sit inside paired CIs; Oct-29 replicas give B2 42.3–44.7 F1 and D1 42.9–44.8 F1.
- Pretraining scenes are mostly single source plus diffuse WHAM noise. STARSS23 is polyphonic with moving sources.
- IV pseudo-labels are 22.3° (median) away from the direct-path truth, and the simulator knows that truth for 84% of scenes.
- Supervised fine-tuning in `seld-dcase-2022-GRAM` has no spatial augmentation. `spr_starss_srir_acs_prob` is used only in the SSL stage, and it is 0.

Design rule: a target is useful only if the visible input does not already give it away. That means it is withheld (a masked modality or time span), privileged (simulator truth) or taken from another view (rotation, another source, video).

## Ranked suggestions
1. Polyphonic spatial mixtures (SSLAM-style): convolve extra AudioSet clips with the scene's other RIR positions. Then add mixture→component latent prediction and direction-queried source completion.
2. Privileged dense spatial targets: a sphere-grid heatmap built from direct-path directions of the active sources, plus DRR/T60 from the RIR, replacing IV pseudo-labels.
3. Split invariant/equivariant rotation objective (SIE): waveform-domain rotations with an R-conditioned predictor on the equivariant half. Do not use rotation prediction, because the mean IV gives it away.
4. Modality-split tokens (spectral W vs spatial dipole+IV) with block masking of the spatial stream. This generalises TODO #14 (xview).
5. Asymmetric temporal masking (Siamese-MAE): past visible, future 90–95% masked, for moving sources.
6. Real unlabeled FOA (STARSS eval audio, YouTube 360 FOA) for continued pretraining with objectives 1–5. Optionally add 360° video for DOA-aware audio-visual contrastive learning.
7. Semi-supervised option: a teacher trained on synthetic and STARSS labels pseudo-labels real FOA, then a student trains on those labels (Depth Anything V2 / Noisy Student).

## Prerequisites
- Fix the FOA conventions (TODO #9). Add 16-pattern ACS to SELD fine-tuning as the baseline.
- Select checkpoints on a non-test split. Use ≥3 pretraining seeds.
- Screen candidates with the frozen linear SELD probe and a low-label fine-tune (10–25% of STARSS) before running full fine-tunes.
- Report F1 stratified by polyphony.

Not checked: the actual size of the gains on this codebase. The literature numbers come from other setups.
