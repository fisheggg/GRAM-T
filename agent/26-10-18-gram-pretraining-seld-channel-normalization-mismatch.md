# 26-10-18 — GRAM pretraining and SELD: channel and normalization mismatch

The audited GRAM pretraining RIRs use N3D scaling with the physical vertical component in raw channel 1, whereas the SELD data use SN3D with vertical in raw channel 2, the Z slot of ACN order. The current pipeline passes both conventions into the same seven feature channels without conversion, creating a dipole-to-W level shift and a change in the spatial meaning of channel-specific weights that RMS normalization and clip LayerNorm do not generally remove. The mismatch is well supported by the existing `li_review` measurements, but its effect on end-to-end SELD has not been isolated, so convention corrections need controlled experiments and must be recorded separately from the historical runs.

## 1. Scope and evidence

This report combines the parent directory's [review index](../../li_review/README.md), [report 1, §4](../../li_review/report1_spr_pair/report.md), [report 2, §4.5](../../li_review/report2_pseudolabel_ceiling/report.md), [report 3, §2.4](../../li_review/report3_li_review_branch/report.md), and their saved audit outputs with a source inspection of the current GRAM-T checkout on 18 October 2026. Measurements below are **previously recorded review results**, not newly run experiments.

The pretraining evidence concerns `GRAM-Naturalistic-Scenes-Ambisonics`, the RIR collection used by the local GRAM reconstruction and SPR experiments. It does not independently establish the conventions of every externally distributed GRAM checkpoint. The review's author-checkpoint probes are useful empirical comparisons, but they do not substitute for an audit of that checkpoint's original training data.

Three distinctions matter throughout:

- **Channel order** describes storage positions; **coordinate convention** describes their physical directions. A slot called “Y” in code need not represent left in the RIR generator's frame.
- **N3D/SN3D normalization** changes dipole gains relative to W. Waveform RMS normalization, mel-filter normalization, and neural-network LayerNorm are different operations.
- Evidence that the input conventions differ is much stronger than evidence that fixing them improves the final SELD score.

## 2. What the channels actually contain

All indices here are zero-based. Standard downstream FOA is stored in ACN order **[W, Y, Z, X]**, with X forward, Y left, and Z up. Its ideal steering vector is

\[
[1,\;\sin\phi\cos\theta,\;\sin\theta,\;\cos\phi\cos\theta],
\]

where azimuth increases toward the left. These are the SN3D first-order responses in the [official DCASE 2023 recording-format and label specification](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#recording-formats). The same source notes that the measured array's encoded responses depart gradually from the ideal model above approximately 9 kHz.

| Property | Audited GRAM pretraining RIRs | STARSS/DCASE downstream input |
|---|---|---|
| Raw channel 0 | W, omnidirectional | W, omnidirectional |
| Raw channel 1 | Physical vertical; positive is up | Y, left/right |
| Raw channel 2 | Horizontal; front/back versus left/right unresolved | Z, up/down |
| Raw channel 3 | Horizontal; front/back versus left/right unresolved | X, front/back |
| First-order normalization | N3D | SN3D |
| Ideal plane-wave dipole-vector magnitude relative to W | √3 | 1 |
| Conversion in the inspected pipeline | None | None |

The pretraining convention is often abbreviated “ACN Y-up.” That means the slot conventionally named Y, raw channel 1, carries vertical information; it does **not** establish the complete world-coordinate interpretation of the other slots.

### The seven-channel representation

Both extractors concatenate four log-mel power channels and three mel-filtered intensity-vector channels. Their actual order is

\[
[L_W,L_{C_1},L_{C_2},L_{C_3},J_{C_1},J_{C_2},J_{C_3}],
\]

where \(C_a\) means a raw channel slot and \(J\) denotes the stored mel-filtered intensity feature.

| Feature index | Calculation | Meaning in pretraining | Meaning downstream |
|---|---|---|---|
| 0 | Log-mel power of W | Omnidirectional | Omnidirectional |
| 1 | Log-mel power of raw channel 1 | Vertical dipole power | Y dipole power |
| 2 | Log-mel power of raw channel 2 | Horizontal dipole power | Z dipole power |
| 3 | Log-mel power of raw channel 3 | Horizontal dipole power | X dipole power |
| 4 | Intensity feature from W and channel 1 | Vertical component | Y component |
| 5 | Intensity feature from W and channel 2 | Horizontal component | Z component |
| 6 | Intensity feature from W and channel 3 | Horizontal component | X component |

Thus vertical information moves from feature slots **1 and 4** during pretraining to **2 and 5** downstream. The correct downstream shorthand is `[W,Y,Z,X,J_Y,J_Z,J_X]`.

The extractor's `[W,X,Y,Z]` docstring and `XYZ` variable name are misleading: executable code takes raw channels `1:` without rearranging them. Likewise, SPR comments call feature slots 4–6 `I_x,I_y,I_z`, although their standard downstream order is Y, Z, X. These names must not be used as evidence of a conversion. See [the pretraining extractor](../src/model/ambisonic_feature_extractor.py), [the downstream extractor](../seld-dcase-2022-GRAM/cls_feature_class_gram.py), and `get_channel_pair_pseudolabels` in [GRAMT_SPR.py](../src/model/GRAMT_SPR.py).

## 3. Where the mismatch passes through the code

| Stage | Inspected behavior | Consequence |
|---|---|---|
| RIR selection | [AudioSet FOA configuration](../configs/data/audioset_ambisonics.yaml) selects the naturalistic RIR shards | Local pretraining uses the audited collection |
| RIR decoding | [WebAudioDataModule.py](../src/data_modules/WebAudioDataModule.py) converts NumPy arrays to float tensors; selects source and noise RIRs | No axis permutation or dipole gain correction |
| Scene construction | [generate_scenes_batch.py](../src/data_modules/scene_module/generate_scenes_batch.py) convolves per channel and applies common noise gains | The RIR convention survives into the rendered waveform |
| Pretraining features | [ambisonic_feature_extractor.py](../src/model/ambisonic_feature_extractor.py) computes powers and W–dipole products in raw order | Both mismatches enter the seven-channel tensor |
| Downstream audio loading | [cls_feature_class_gram.py](../seld-dcase-2022-GRAM/cls_feature_class_gram.py), `_load_audio` and `_normalize_audio`, retain the first four channels and apply one −14 dBFS RMS gain | Neither ACN slots nor dipole/W ratios change |
| Encoder input | [GRAMT.py](../src/model/GRAMT.py) and [GRAMT_SPR.py](../src/model/GRAMT_SPR.py) normalize the whole clip and use channel-specific patch projections | Learned weights encounter different physical axes and relative levels |
| SPR targets | `pretrain_forward` computes pseudolabels before `spectrogram_normalize` | Targets also inherit the original convention and intensity scaling |

The relevant frontends use 32 kHz configuration, 1,024-point FFTs, 128 mel bands, a 50 Hz lower cutoff, and power spectra. The searched current source and configurations do not contain the review-branch conversion controls `foa_match_finetune`, `sn3d_to_n3d`, or `swap_vertical_to_ch1`. Descriptions of those controls in `li_review` concern a separate reviewed implementation, not a conversion already active here.

The downstream methods named `get_normalized_feat_dir` and `get_unnormalized_feat_dir` return the same feature directory in this implementation. Their names are not evidence that FOA normalization has been reconciled.

## 4. Normalization mismatch: size, evidence, and feature effects

### 4.1 What N3D versus SN3D changes

For first order, converting N3D to SN3D leaves W unchanged and divides all three dipole waveforms by √3. The inverse multiplies the dipoles by √3. This is an amplitude factor of approximately 1.732, a power factor of 3, and a relative power-level difference of **4.771 dB**.

With power-based log-mel features, a dipole gain of √3 adds approximately

\[
\log 3 = 1.098612
\]

to each dipole log-mel channel, while leaving W unchanged. This approximation assumes powers comfortably above the additive numerical floor and no additional global gain adjustment. It describes the effect of a convention change on the same signal; it is not a claim that every observed pretraining/downstream feature difference has this value.

### 4.2 Recorded normalization measurements

The strongest direct evidence is the isolated direct arrival. The review fitted

\[
\mathbf r = \frac{\sum_t W_t\mathbf V_t}{\sum_t W_t^2}
\]

around direct paths in 100 source RIRs from shards 0, 17, 34, 51, and 68. The median \(\|\mathbf r\|\) was **1.732**, and 89 of 100 exceeded 1.70, supporting N3D. Weak direct paths and overlapping reflections account for the less clean cases. Sources: [report 1, §4.1](../../li_review/report1_spr_pair/report.md) and [direct-path audit results](../../li_review/common/results/rir_pseudolabel_vs_truth.md).

An independent feature-space audit used

\[
D=\log\!\left(e^{L_{C_1}}+e^{L_{C_2}}+e^{L_{C_3}}\right)-L_W.
\]

It took a median over valid time-frequency bins per file, then a median over files. The implementation excluded bins where any of the four log-mel values was at or below −9. Its results were:

| Dataset | Files/scenes | All bands | 50–500 Hz | 0.5–2 kHz | 2–8 kHz | 8–16 kHz |
|---|---:|---:|---:|---:|---:|---:|
| STARSS23, folds 3 and 4 | 40 | **0.032** | 0.042 | 0.154 | 0.067 | −0.139 |
| DCASE synthetic, folds 1 and 2 | 40 | **0.107** | 0.254 | 0.247 | 0.067 | −0.039 |
| Pink noise convolved with pretraining source RIRs | 40 | **1.103** | 1.152 | 1.154 | 1.096 | 1.056 |

The last audit sampled RIR shards 0, 23, 46, and 69, which is a different sample from the 100-scene direct-path audit. The downstream values are close to the SN3D plane-wave benchmark of zero; the pretraining value is close to the N3D benchmark of log 3. See [saved normalization results](../../li_review/common/results/foa_normalisation_check.md) and [the generating script](../../li_review/common/audit/foa_normalisation_check.py).

Whole-RIR dipole/W energy ratios were less clean: median **2.448**, versus **2.077** for the tail after 50 ms. Frequency-band medians above 50 Hz were 2.733, 2.798, 2.809, and 2.763; below 50 Hz the median was 0.524. About 11.7% of W's RIR energy was below 50 Hz, outside the configured mel range. These measurements help explain why a whole-waveform ratio need not equal three.

**Qualification to the original review:** its description of the ratio as independent of the sound field is too strong. For a single ideal plane wave, \(\sum_a|V_a|^2/|W|^2\) is exactly one or three; analogous ratios of expected energies can hold under incoherent-source assumptions. An arbitrary coherent mixture need not obey this: two equal, in-phase waves from opposite directions can reinforce W while cancelling the dipole vector. An expectation of a ratio also need not equal a ratio of expectations. Therefore the D table is supporting empirical evidence, not a universal per-bin identity; the clean direct-path fit is the more direct normalization check.

### 4.3 Effect on the intensity-vector features

Let \(P=|W|^2\), \(Q=\sum_a|V_a|^2\), and \(\epsilon\) be the implementation's stabilizer. The **executed** formula is

\[
\mathbf i=\frac{2\operatorname{Re}(W^*\mathbf V)}{\epsilon+P+Q}.
\]

The comment immediately above it shows \(Q/3\), but the code does **not** divide Q by three. Analysis must follow the executable formula.

For a single plane wave \(\mathbf V=\kappa\mathbf u W\), with unit direction \(\mathbf u\), and ignoring the numerical floor,

\[
\mathbf i=\frac{2\kappa}{1+\kappa^2}\mathbf u.
\]

Consequently, SN3D gives \(\mathbf i=\mathbf u\), while N3D gives \(\mathbf i=(\sqrt3/2)\mathbf u\approx0.866\mathbf u\). The N3D magnitude is **13.4% smaller in this ideal case**. A positive common dipole rescaling preserves the pre-mel vector's direction wherever it is nonzero, but changes its magnitude; it does not repair the separate axis mismatch.

The stored feature is a mel-weighted sum,

\[
\mathbf J_m=\sum_k H_{mk}\mathbf i_k.
\]

The mel filterbank uses `norm=None`, so its filters are not normalized to unit area. Even for a constant-direction plane wave, \(\mathbf J_m\) has the extra factor \(\sum_kH_{mk}\). **The cached IV channels are not automatically unit vectors**, and 1 versus 0.866 should not be interpreted as their universal observed norms.

For a general dipole rescaling \(\mathbf V'=\alpha\mathbf V\), the exact pre-mel relation is

\[
\mathbf i'_k=
\alpha\frac{\epsilon+P_k+Q_k}{\epsilon+P_k+\alpha^2Q_k}\mathbf i_k.
\]

The factor varies by frequency bin and sound field. Mel summation loses the information needed to apply this factor retrospectively; it can also change the aggregate direction when bins have different directions. A constant multiplier on cached IV features is therefore not a general N3D/SN3D conversion. Recompute the full feature tensor from the waveform or complex STFT. A simple log-mel offset can approximate the power-channel part above the floor, but it cannot repair the complete seven-channel representation.

## 5. Vertical-axis mismatch: how it was established

Normalization measurements alone cannot determine channel order. The review instead used the geometry of direct arrivals and early reflections to identify the floor.

For a direct-source vector \(\mathbf D=L_0\mathbf u_0\) and an image-source vector \(\mathbf D'=L_1\mathbf v\), path lengths came from arrival times using 343 m/s. Under the single planar-reflection model,

\[
\mathbf n=-\frac{\mathbf D'-\mathbf D}{\|\mathbf D'-\mathbf D\|},\qquad
h_s=\frac{\|\mathbf D'-\mathbf D\|}{2},\qquad
h_r=h_s-\mathbf D\cdot\mathbf n.
\]

Here \(\mathbf n\) is the inward plane normal and \(h_r\) is the receiver-to-plane distance. Floor reflections should share an upward normal and a consistent receiver height across the RIRs of a scene.

The audit inspected 100 scenes from five shards: 340 source/noise RIRs, 311 clean direct paths, 2,459 candidate early peaks within 30 ms, and 582 retained single-plane-wave reflections. For normals within 10° of each pole:

| Inward-normal pole | Reflections | Scenes | Receiver-to-plane distance, median [IQR], m | Largest share within one 0.2 m interval |
|---|---:|---:|---|---:|
| **+channel 1** | **96** | **32** | **1.75 [1.70, 1.86]** | **0.67** |
| −channel 1 | 51 | 34 | 1.21 [0.85, 1.52] | 0.24 |
| +channel 2 | 27 | 15 | 0.93 [0.36, 1.12] | 0.37 |
| −channel 2 | 30 | 18 | 0.79 [0.54, 1.16] | 0.40 |
| +channel 3 | 23 | 11 | 1.03 [0.92, 1.23] | 0.48 |
| −channel 3 | 27 | 19 | 1.25 [1.14, 1.51] | 0.37 |

Only +channel 1 combines many reflections with a tight receiver-height cluster. Aggregating within scenes first gives 1.73 m [1.72, 1.84]. Tightening the pole tolerance to 5° gives a 0.2 m cluster share of 0.81; relaxing it to 15° gives 0.61. This supports **+channel 1 as up**, with the opposite pole corresponding to ceilings. The direct-path distribution provides supporting evidence: 66% of clean paths have a negative channel-1 component, consistent with many sources below the receiver. See [reflection results](../../li_review/common/results/rir_reflection_geometry.md) and [audit implementation](../../li_review/common/audit/rir_reflection_geometry.py).

### What remains unresolved

The evidence does not identify which horizontal channel is forward versus left, their signs, or the complete coordinate-system handedness. Horizontal direction statistics are approximately symmetric, so these alternatives are not identifiable from those statistics alone.

Habitat's [official geometry constants](https://aihabitat.org/docs/habitat-sim/habitat_sim.geo.html) specify +Y up and −Z front. A Habitat-based explanation of the RIRs is therefore plausible, but the review did not verify the acoustic encoder's complete channel convention. It should remain a named hypothesis, not a fact inferred solely from the simulation framework.

## 6. Why the existing normalizations do not resolve it

**Waveform RMS normalization:** the downstream loader calculates one RMS over the multichannel waveform and applies one scalar gain. This preserves channel ratios and axis positions. It cannot turn N3D into SN3D or move vertical from channel 2 to channel 1. If conversion is introduced before RMS normalization, the resulting common gain can change, but the intended relative dipole/W conversion remains.

**Clip LayerNorm:** the GRAM models use `LayerNorm([in_channels, num_mel_bins, target_length], elementwise_affine=False)`, sharing the clip's mean and variance across the seven-channel tensor. A dipole-only log-power offset is not a uniform shift of all entries. LayerNorm changes its numerical expression but does not generally remove it. The original review's wording that the offset enters the model “unchanged” is too literal; **the relative feature distribution remains different**, rather than retaining exactly the same numerical offset after normalization.

A pure channel permutation also survives LayerNorm as a permutation. The subsequent patch projection has distinct weights for each input channel, so that permutation changes which learned weights act on vertical information. Likewise, arbitrary global waveform gain is not guaranteed to disappear completely: it shifts the four log-power channels while leaving the three normalized-intensity channels approximately invariant, rather than shifting all seven channels equally.

**SPR supervision:** pseudolabel extraction precedes LayerNorm. Intensity magnitudes and component positions therefore affect targets as well as inputs. The physical meaning of a `pair` output slot cannot be identified with an ACCDOA X/Y/Z output simply from its comment. The review's separate finding that the current `pair` objective reduces to a scaled `foa_to_iv` loss is an objective-implementation issue; convention matching alone does not repair it.

## 7. What has actually been measured about the impact

The review completed **axis-remapping frozen ridge probes**, not a full causal test of normalization plus axis matching in end-to-end SELD. They fit on 1,848 stable single-source windows from fold 3 and tested on 1,900 windows from fold 4, using three readouts and eight encoders, including random initialization.

The input variants were:

| Variant | Seven features supplied to the encoder | Interpretation |
|---|---|---|
| `none` | `[W,Y,Z,X,J_Y,J_Z,J_X]` | Original downstream convention |
| `swap_yz` | `[W,Z,Y,X,J_Z,J_Y,J_X]` | Minimal vertical-slot swap |
| `habitat` | `[W,Z,X,Y,J_Z,-J_X,-J_Y]` | Vertical alignment plus hypothesized horizontal mapping |

W/X/Y/Z in this table denote log-power channels, so sign flips appear only in the intensity features. Both remaps move vertical into feature slots 1 and 4. They leave the N3D/SN3D discrepancy unresolved.

Selected results with the historical `seld` readout:

| Encoder | Original median angle error | `swap_yz` | `habitat` | `swap_yz` minus original, paired 95% CI |
|---|---:|---:|---:|---|
| A | 30.8° | 34.2° | 36.8° | +3.4° [+0.3, +6.9] |
| B1, orthogonal | 36.3° | 31.9° | 32.9° | −4.4° [−8.6, −0.7] |
| B2, pair | 34.3° | 34.0° | 35.9° | −0.3° [−2.6, +2.3] |
| B3, foa_to_iv | 28.0° | 28.9° | 26.6° | +0.9° [−2.4, +3.5] |
| Random initialization | 34.0° | 33.8° | 31.6° | −0.2° [−3.9, +2.7] |

The intervals use 2,000 paired bootstrap resamples over recordings. Among 42 comparisons against the original input across seven pretrained encoders, three readouts, and two remaps, seven nominal 95% intervals excluded zero: **six favored the original input and one favored remapping**. These are exploratory comparisons without a multiple-comparison claim. There is no general angular-error improvement from the tested axis adapters. Source: [full axis-remapping probe results](../../li_review/common/results/frozen_probe_axis_remap.md).

Vertical information nevertheless responds to the mapping. B2's `seld` vertical-output R² changes from 0.154 to 0.242/0.292, while B1 changes from 0.256 to 0.147/0.223. Across readouts, all six B2 point estimates improve and all six B1 point estimates worsen. This checkpoint dependence argues against treating the mismatch as harmless or treating its correction as an automatic improvement.

Fold 4's elevation distribution is narrow: median −16°, with 5th–95th percentiles −42° to +14°. A constant-elevation reference already achieves an 8.0° median absolute elevation error, comparable to the probes' approximately 8–12°. Overall angular error can therefore obscure changes in vertical information; future evaluations should include vertical-output and elevation-stratified diagnostics.

### Consequences for historical comparisons and Fox–Olivia parity

The local historical groups share exposure to the same RIR convention. Their results remain comparisons under that historical protocol, but this does not prove that convention matching would affect every objective or checkpoint equally, or preserve the ranking.

The [18 October Stage 2 and Fox–Olivia launch report](26-10-18-stage2-c-and-fox-olivia-runs.md) explicitly preserves historical FOA conventions. A successful platform-comparability check establishes reproducibility of the same computation; it does not establish physical agreement between pretraining and downstream coordinates. Changing conventions inside those resumed runs would introduce a new experimental condition. The shared readout and other historical choices should likewise remain fixed while isolating this question.

## 8. Controlled remedies and the experiments they answer

These are proposed follow-up experiments, not changes applied by this report.

### 8.1 Existing checkpoints: adapt downstream inputs

For an old checkpoint, a four-condition experiment separates gain matching from vertical-slot matching:

| Condition | Downstream dipole normalization | Axes supplied to encoder | Question |
|---|---|---|---|
| Baseline | Native SN3D | Native Z in channel 2 | Historical behavior |
| Normalization only | Convert to N3D | Native Z in channel 2 | Effect of matching relative gains |
| Vertical only | Native SN3D | Swap channels 1 and 2 | Effect of matching the known vertical slot |
| Both | Convert to N3D | Swap channels 1 and 2 | Combined effect and interaction |

Start with one fixed reconstruction checkpoint, such as A, and at least three matched seeds per condition: 12 fine-tuning runs. Keep checkpoint loading, readout, optimizer schedule, augmentation, masking, labels, and scoring identical. Use a genuinely held-out validation split for checkpoint selection and reserve the test fold for final evaluation; the historical `val_splits = test_splits = [[4]]` arrangement should not define the new causal experiment. Report ER, F, LE, LR, SELD score, recording-paired intervals, seed variability, and elevation diagnostics. Treat the Habitat mapping as a separate hypothesis if subsequently tested.

Normalization matching requires re-extracting features from all affected real and synthetic audio with dipoles multiplied by √3. Use separate, versioned feature caches so native and converted tensors cannot be mixed. The old review reported missing extracted audio directories and an available STARSS archive; that was a historical filesystem observation, so current Olivia audio availability must be checked before implementation.

The vertical swap can be implemented exactly on existing features with index order **`[0,2,1,3,5,4,6]`**, before encoder normalization and after spatial augmentation. Both the log-power and corresponding IV channels must move. Labels remain in the standard downstream frame because this is an input adapter for a legacy encoder, not a new physical rotation of the labeled scene. Physical scene augmentation still requires corresponding label transformations.

Signed permutations are exact feature-domain operations: log powers are permuted and IV components receive the corresponding signs. Arbitrary rotations are different; they mix waveform channels and cross terms that are absent from cached channel powers. The minimal swap is an orthogonal reflection, whereas the hypothesized Habitat mapping is a proper rotation under that coordinate interpretation. Neither should be described as proving the unknown horizontal convention.

### 8.2 New pretraining: convert RIRs before rendering

For a new pretraining round, choose a documented target convention and convert **all source and noise RIRs** at the loading boundary. Mono AudioSet and WHAM source signals themselves are not FOA dipoles and should not receive channel-dependent scaling.

Writing original raw RIR slots as \(C_0,C_1,C_2,C_3\), the minimal transform into downstream slot names is

\[
(W',Y',Z',X')=
(C_0,\;C_2/\sqrt3,\;C_1/\sqrt3,\;C_3/\sqrt3).
\]

This matches the verified gain and vertical-axis differences. It assigns the unresolved horizontal axes by convention rather than proving their physical correspondence. If independent evidence supports the Habitat interpretation, its alternative is

\[
(W',Y',Z',X')=
(C_0,\;-C_3/\sqrt3,\;C_1/\sqrt3,\;-C_2/\sqrt3).
\]

Any geometry-derived targets must be calculated in the converted frame or transformed consistently. Train a new reconstruction baseline alongside the new SPR conditions; comparing a corrected objective and convention against only an old baseline would confound the two changes. New checkpoint identities and manifests should record the convention, and old checkpoints should retain their historical interpretation.

### 8.3 Review-branch pitfalls to avoid

[Report 3, §2.4](../../li_review/report3_li_review_branch/report.md) found two relevant integration problems in the reviewed branch:

1. New pretraining configurations already enabled `data.foa_match_finetune`, producing SN3D with vertical in Z, while the fine-tuning configuration still enabled `swap_vertical_to_ch1`. Applying the legacy adapter to a corrected checkpoint creates a fresh mismatch. Select adapters from checkpoint convention metadata and reject contradictory settings.
2. In a reviewed SSL fine-tuning path, the online branch received swapped features but the EMA target branch received the original tensor. Both sides of a representation loss must use the same intended coordinate representation.

The report also warns against bundling axis matching, a readout change, encoder freezing, and checkpoint selection into one experiment. Those switches need independent controls to make an observed change interpretable. These are findings about the reviewed branch snapshot, not claims that its converter or these exact bugs are active in the current checkout.

### 8.4 Validation before any conversion run

A future implementation should record raw channel order, harmonic normalization, positive vertical axis, horizontal-transform assumption, feature-cache version, and checkpoint convention. Small synthetic checks should cover front/left/up plane waves, the expected √3 gain ratio, conversion round trips, exact waveform-versus-feature signed-permutation agreement, all source/noise RIR paths, consistent online/target inputs, and rejection of accidental double conversion. Mel-area factors and the numerical floor must be included when checking IV magnitudes. Torch-based checks belong in Slurm jobs under this repository's execution rules.

## 9. Conclusions supported by the evidence

- **Established for the audited data:** N3D versus SN3D scaling, positive vertical in pretraining raw channel 1 versus downstream channel 2, and no conversion in the inspected historical pipeline.
- **Mechanistically established:** relative dipole levels and the numerical intensity features change; vertical information reaches different learned channel weights; global RMS and clip LayerNorm are not substitutes for convention matching.
- **Measured but mixed:** axis-only frozen probes change spatial information in checkpoint-dependent ways, without a consistent improvement in angular error.
- **Still unresolved:** the complete horizontal mapping and handedness of the RIRs, and the end-to-end SELD benefit of normalization matching, axis matching, or both.

## 10. Work performed for this report

Read the parent `li_review` reports and the normalization, reflection-geometry, direct-path, and axis-probe artifacts; traced the current RIR loader, scene renderer, feature extractors, GRAM/SPR normalization, and pseudolabel paths; checked the official downstream steering convention and Habitat coordinate constants; and documented the findings here. This task adds this report only: it does not change training code, configurations, feature caches, checkpoints, or running jobs, and it does not claim new experimental results.
