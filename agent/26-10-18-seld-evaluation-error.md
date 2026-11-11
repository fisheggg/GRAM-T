# 26-10-18 — SELD evaluation error: jackknife point values and validity of comparisons

The current SELD scorer overwrites its full-dataset metrics during jackknife resampling, so with `is_jackknife=True` it returns the final leave-one-recording-out point estimates—77 recordings in these experiments—instead of the intended 78-recording values. The confidence intervals still use the full-dataset value and all leave-one-out estimates, making the reported point/interval pair inconsistent, while directory enumeration determines which recording is omitted. Rescoring saved predictions fixes this reporting error, but trustworthy method comparisons also require independent validation, verified checkpoint loading, and uncertainty across both recordings and training seeds.

## 1. Scope and evidence

This report examines the evaluation error identified in the parent [li_review report 1, §3](../../li_review/report1_spr_pair/report.md), checks it against the current source, and explains its consequences for the historical results and the Fox–Olivia comparison. It also separates the scorer defect from other established problems that affect interpretation of the same results.

Evidence consists of:

- The current [scoring wrapper](../seld-dcase-2022-GRAM/cls_compute_seld_results.py) and [metric implementation](../seld-dcase-2022-GRAM/SELD_evaluation_metrics.py).
- The [supervised training entry point](../seld-dcase-2022-GRAM/train_seldnet.py), [SSL-plus-supervised entry point](../seld-dcase-2022-GRAM/train_ssl_seldnet.py), and [model loader](../seld-dcase-2022-GRAM/seldnet_model.py).
- The review's [rescoring script](../../li_review/common/audit/seld_rescore.py), [human-readable results](../../li_review/common/results/seld_rescore.md), and [full-precision JSON](../../li_review/common/results/seld_rescore.json).
- The [Olivia comparison driver](../scripts/olivia/compare_olivia.py), [Stage 3 CSV](../results/stage3.csv), and [launch/comparison report](26-10-18-stage2-c-and-fox-olivia-runs.md).

The numerical rescoring and bootstrap results below were produced by the existing review, not by a new training or inference run for this report. Their values and the Stage 3 correspondence were checked directly against the saved JSON and CSV. Source inspection on 18 October 2026 confirms that the jackknife point-value fix is **not present in the current checkout**.

## 2. What the SELD numbers mean

The configured protocol uses `average: macro`, 13 classes, and a 20° localization threshold. The four reported metrics are:

| Metric | Interpretation | Preferred direction | Reporting unit |
|---|---|---|---|
| ER | Location-sensitive detection error rate | Lower | Ratio |
| F | Location-sensitive F-score | Higher | Fraction internally; often percent in tables |
| LE | Class-dependent localization error of matched predictions | Lower | Degrees |
| LR | Class-dependent localization recall | Higher | Fraction internally; often percent in tables |

The [official DCASE 2022 metric description](https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#metrics) specifies one-second evaluation segments, a 20° threshold for location-sensitive detection, and class-dependent localization metrics without that angular threshold. A prediction can therefore contribute a localization error even when it fails the detection-angle criterion.

The repository combines its metrics using

\[
S_{\mathrm{SELD}}=\frac{ER+(1-F)+LE/180+(1-LR)}{4}.
\]

F and LR must be fractions in this formula: 45.5% means approximately 0.455, not 45.5. The implementation computes a global ER from accumulated substitutions, deletions, insertions, and reference counts; in macro mode it averages classwise F, LE, LR, and the corresponding composite scores. The “classwise ER” row repeats the global ER rather than providing separate per-class error rates. See `compute_seld_scores` and `early_stopping_metric` in [SELD_evaluation_metrics.py](../seld-dcase-2022-GRAM/SELD_evaluation_metrics.py).

This composite is the repository's model-selection score. It should not be confused with a probability of failure or a percentage of incorrectly localized recordings. Nor is it the official DCASE 2022 competition ranking rule: that challenge used cumulative ranks across the four metrics. [Official ranking specification](https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#ranking).

## 3. The jackknife defect

### 3.1 Intended behavior

For a prediction set \(\mathcal D\) containing N recordings:

1. Accumulate all N recordings and calculate the full-set point estimate \(\theta=S(\mathcal D)\).
2. Calculate N leave-one-out estimates \(\theta_{(-i)}=S(\mathcal D\setminus\{i\})\), each using N−1 recordings.
3. Use the collection of leave-one-out estimates to calculate uncertainty.
4. Return the declared point estimate together with its interval.

In this API, requesting uncertainty should not silently change the reported point estimate from the full-set score to one arbitrary subset score. For the historical fold-4 predictions, N is 78.

### 3.2 Actual control flow

The current `get_SELD_Results` contains the following sequence:

| Source location in `cls_compute_seld_results.py` | What happens |
|---|---|
| Line 102 | `pred_files = os.listdir(pred_files_path)` obtains an unsorted directory listing |
| Lines 105–116 | All listed prediction files are scored; the full-set metrics are assigned to `ER, F, LE, LR, seld_scr, classwise_results` |
| Lines 119–121 | Full-set values are copied into `global_values` for interval calculation |
| Lines 124–137 | The leave-one-out loop repeatedly reassigns those same metric variables |
| Lines 141–146 | Intervals are calculated using `global_values` and all partial estimates |
| Line 147 | The return statement uses the overwritten variables, which now hold the final leave-one-out result |

The effective behavior is:

```python
scores = score(all_files)
global_values = flatten(scores)
for omitted_file in all_files:
    scores = score(all_files_except(omitted_file))
    partial_estimates.append(flatten(scores))
intervals = jackknife(global_values, partial_estimates)
return scores, intervals  # scores now describes the final N-1 subset
```

This is a variable-overwrite/return-value bug. The missing recording is not deleted, and the initial full-set computation does include it. The error occurs when choosing which already-computed values to return.

**Affected outputs:** ER, F, LE, LR, the composite SELD score, and the classwise point-value matrix. It is not confined to the headline SELD number.

**Unaffected by this particular overwrite:** calls with `is_jackknife=False`, the original prediction files, the model weights, and the full-set values retained for the interval calculation. This statement concerns the overwrite defect, not a claim that every aspect of the evaluation protocol is correct.

The fetched [DCASE 2022 baseline scoring wrapper](https://raw.githubusercontent.com/sharathadavanne/seld-dcase2022/master/cls_compute_seld_results.py) contains the same assignment-and-return pattern, supporting the review's attribution to inherited baseline code. The local defect is independently visible in the current source.

### 3.3 Why copying results between machines can change the apparent score

The last omitted recording is `os.listdir(pred_dir)[-1]`. Directory enumeration order is not an evaluation protocol and need not remain the same after copying files, rebuilding a directory, or moving to another filesystem.

Identical prediction CSVs can therefore produce different returned jackknife point estimates if a different recording is listed last. The full-set metric remains the same apart from floating-point summation differences; the collection of leave-one-out estimates also remains the same, merely reordered.

For all eleven runs present in the saved review JSON, the recorded final omission is:

```text
fold4_room15_mix010.csv
```

That is an observation about those saved evaluations, not a guarantee about a fresh `os.listdir` call. Sorting file names would make the erroneous omission deterministic, but **would not repair the point estimate**.

### 3.4 What the interval actually describes

The current jackknife helper calculates, for each metric,

\[
\bar\theta_{(-)}=\frac1N\sum_i\theta_{(-i)},\qquad
b=(N-1)(\bar\theta_{(-)}-\theta),
\]

\[
\widehat{SE}=\sqrt{\frac{N-1}{N}\sum_i(\theta_{(-i)}-\bar\theta_{(-)})^2},
\]

and constructs a t-based interval around the bias-corrected estimate \(\theta-b\). Thus the interval computation retains the full-set statistic and all N leave-one-out estimates; it does not accidentally use only the final 77-recording subset.

There are three distinct quantities: the raw full-set estimate \(\theta\), the bias-corrected jackknife estimate \(\theta-b\), and an individual leave-one-out estimate \(\theta_{(-i)}\). The erroneous return value is the third. The existing proposed patch returns the first while retaining the existing interval method; consequently, an interval need not be exactly centered on the displayed point even after the bug is fixed. That is a documented estimator choice, separate from returning an arbitrary subset by mistake.

## 4. Measured effect on the historical results

The review reconstructed per-recording metric accumulators from saved fold-4 predictions. It verified the historical leave-one-out values and then recomputed scores using all 78 recordings. The following values are rounded from its [full-precision JSON](../../li_review/common/results/seld_rescore.json); “error” means historical returned score minus full-set score.

| Run | Historical 77-recording score | Full 78-recording score | Point-value error |
|---|---:|---:|---:|
| A, GRAM reconstruction pretraining | 0.397541 | **0.396264** | +0.001276 |
| A0, SPR code with SPR weight zero | 0.408419 | **0.405216** | +0.003202 |
| B0, w_to_xyz | 0.402203 | **0.399752** | +0.002451 |
| B1, orthogonal | 0.405265 | **0.403280** | +0.001985 |
| B2, pair | 0.388684 | **0.386785** | +0.001898 |
| B3, foa_to_iv | 0.403235 | **0.400561** | +0.002674 |
| D0, B0 followed by STARSS SSL | 0.405473 | **0.403166** | +0.002306 |
| D2, B2 followed by STARSS SSL | 0.399228 | **0.397116** | +0.002112 |
| D3, B3 followed by STARSS SSL | 0.395137 | **0.392033** | +0.003105 |
| Historical “author checkpoint,” actually random initialization | 0.546379 | **0.546560** | −0.000181 |

For the nine validly identified A/A0/B/D entries above, the omission makes the score worse by approximately 0.0013–0.0032. The author-checkpoint entry illustrates that the direction is not guaranteed: omitting a recording can either increase or decrease the score. These are absolute score differences, not percentage-point changes in F or LR.

The error is small relative to the approximately 0.03 half-width of a typical single-run interval, but it matters for reproducibility and close comparisons. It is not a common constant that can be subtracted from every row: the same omitted recording affects different models differently. The supplied set still has B2 as its lowest-scoring A/A0/B checkpoint after rescoring, but that does not establish a general method advantage.

For context, corrected full-set component metrics for three key runs are:

| Run | ER | F (%) | LE (°) | LR (%) | SELD | Existing single-run jackknife 95% interval |
|---|---:|---:|---:|---:|---:|---|
| A | 0.525 | 42.2 | 17.5 | 61.5 | 0.3963 | [0.368, 0.426] |
| B2 | 0.510 | 45.5 | 17.4 | 60.5 | 0.3868 | [0.360, 0.414] |
| D2 | 0.499 | 44.5 | 16.7 | 55.9 | 0.3971 | [0.368, 0.430] |

These component values are rounded for readability; the composite was calculated from unrounded values. Reconstructing a precise SELD score from rounded log entries introduces additional rounding error.

The old review could not rescore D1, F0, and F2 because their prediction directories were unreadable to the reviewing account. It reported log-derived values only. This report does not treat that historical access problem as proof of their current accessibility, and it does not invent full-set scores or paired intervals for them.

## 5. What the Fox–Olivia evaluation checked

The current comparison deliberately contains two different checks:

1. **Historical scorer replay:** `fixed_order_score` places the recorded omitted file last, then calls `get_SELD_Results(..., is_jackknife=True)`. It compares the returned points with the saved `as_logged_77` values and checks the saved interval. This reproduces the historical bug as part of platform parity.
2. **D2 checkpoint inference:** the recovered complete D2 model is loaded strictly, predictions are generated for 78 recordings, and the scorer is called with `is_jackknife=False`. This compares the full-set metric against the review's full-set reference.

The [Stage 3 CSV](../results/stage3.csv) therefore contains:

| Run ID | Phase | Recorded SELD score | Recordings contributing to the point value |
|---|---|---:|---:|
| `scorer-A0` | `scorer_replay` | 0.408418584894 | **77** |
| `scorer-B2` | `scorer_replay` | 0.388683552562 | **77** |
| `scorer-D2` | `scorer_replay` | 0.399227831435 | **77** |
| `D2-inference` | `frozen_checkpoint_inference` | 0.397115742076 | **78** |

All four rows currently contain `Recording_count=78`. In the replay rows, that is the size of the input prediction set, not the support of the returned point value. This distinction is visible in [compare_olivia.py](../scripts/olivia/compare_olivia.py), which asserts 78 files before reproducing the saved 77-recording points.

The full-set D2 reference is 0.397115746548; the Olivia inference difference is approximately **−4.47×10⁻⁹**, as recorded in the CSV. The approximately 0.002112 difference between `scorer-D2` and `D2-inference` is therefore primarily the historical omission, not evidence of a Fox–Olivia numerical discrepancy.

`PASSED` on a historical replay row means “matched the declared historical reference.” It does not mean that its score is the preferred full-set scientific result. Future records should distinguish `input_recording_count`, `point_recording_count`, `point_estimator`, `omitted_recording`, and `scorer_version`, with full-set metrics stored separately from legacy replay values. This report leaves the existing comparison records intact.

## 6. Other established evaluation problems

### 6.1 Validation and test use the same recordings

In both training entry points, the 2022, STARSS23, and `SELD_combined` development branches select:

```python
train_splits = [[1, 2, 3]]
val_splits   = [[4]]
test_splits  = [[4]]
```

These assignments are in the Python entry points, not merely a YAML annotation. The supervised stage evaluates validation predictions and saves the checkpoint when `val_seld_scr` improves. Its final test then reuses fold 4. Selection has therefore already used information from the reported test set.

An important distinction: validation calls the scorer with its default `is_jackknife=False`, while the final test explicitly sets it to true. Consequently, the jackknife overwrite is a **final-reporting error**, not the mechanism by which this code chooses its best supervised checkpoint. Nevertheless, that checkpoint is selected on the same fold later called test. A final test value can also differ from its corresponding validation score purely because the two calls use different point-value paths.

Correcting the scorer cannot undo model-selection bias. Existing recording-bootstrap or jackknife intervals hold the selected checkpoint fixed; they do not account for repeatedly choosing it using these test labels. The original review suggests that groups may be similarly affected, but equal bias across groups has not been demonstrated and should not be assumed when interpreting their ranking.

This does not imply that the project violated a challenge rule: DCASE distinguished development testing from the separate challenge evaluation set. It does mean that these local fold-4 numbers cannot simultaneously be presented as an untouched test estimate after using fold 4 for checkpoint selection. [Official development/evaluation distinction](https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#task-setup).

**A repair must preserve the intended training domain.** [Review report 3, §2.3](../../li_review/report3_li_review_branch/report.md) identified a proposed split that used folds 1–2 for training and all of fold 3 for validation. In the audited collection, folds 1–2 contain 1,200 synthetic recordings, fold 3 contains all 90 real training recordings, and fold 4 contains 78 real development-test recordings. Moving all of fold 3 out of training silently makes the task synthetic-only training. Better-controlled options include a fixed recording/session-aware subset of fold 3 for validation, retaining real data for training, or an explicitly declared synthetic validation split. All groups must use the same manifest.

### 6.2 The historical “author checkpoint” did not load

The review's [checkpoint-key audit](../../li_review/common/results/ckpt_key_audit.json) found that the author safetensors file has a `model.` prefix while the receiving model expects unprefixed names. With the historical `strict=False` loader, **0 of 352 model tensors and 0 of 149 encoder tensors matched**. Removing the prefix allows 350 of 352 model tensors to match; the remaining two are feature-extractor buffers.

The corresponding 0.546560 full-set score is therefore a fine-tuned random-initialization result, not an evaluation of the pretrained author encoder. Rescoring repairs its numeric reporting but cannot change what model was trained. Relabel that historical row and rerun the intended baseline after verifying weight coverage.

The current `seldnet_model.GRAM` still loads the supplied state dictionary with `strict=False` and does not implement the reviewed prefix normalization or coverage check. A successful call is insufficient evidence that the intended encoder loaded. Future evaluation should record checkpoint identity/hash, matched encoder tensors and parameter fraction, shape mismatches, and missing/unexpected keys, and fail when encoder coverage is inadequate. The review's [loader patch](../../li_review/report1_spr_pair/code/patches/01_load_pretrained_gram.patch) provides its proposed approach.

### 6.3 One historical batch produced identical predictions for different labels

[Report 1, §3.4](../../li_review/report1_spr_pair/report.md) found seven September 6 test-output directories, each with 78 files, whose predictions were byte-identical even though several jobs named different intended pretraining checkpoints. Those runs do not demonstrate different checkpoint behavior or provide independent replication. The review's A0/B0–B3 rescoring uses the distinct September 7 reruns instead. Identical outputs establish the comparison failure; they do not by themselves identify every cause in the earlier launch/loading path.

### 6.4 Readout and input conventions also limit comparability

The legacy SELD readout averages adjacent embedding dimensions rather than the frequency rows described in its comment. All historical groups sharing it can be compared under that implemented architecture, but changing it for only selected groups introduces a new condition. Likewise, the FOA normalization and vertical-axis mismatch is a transfer issue, not the jackknife return-value bug. Its evidence and proposed experiments are documented in the [separate channel/normalization report](26-10-18-gram-pretraining-seld-channel-normalization-mismatch.md).

These problems should not be bundled into a single unlabeled “evaluation fix”: score-return repair, correct checkpoint loading, validation redesign, readout changes, and FOA adaptation answer different questions and have different rerun requirements.

## 7. What the paired comparisons support

Single-run confidence intervals do not directly answer whether two models differ, because the models are evaluated on the same recordings and their errors are correlated. The review used recording-paired leave-one-out intervals and 5,000 paired bootstrap resamples. Each bootstrap sample uses the same recording multiplicities for both models, sums their metric accumulators, recomputes each complete SELD score, and then subtracts them.

This matters because dataset-level SELD is nonlinear in the counts: an average of independently calculated per-recording SELD scores is generally a different statistic. The reviewed script resamples the additive TP/FP/FN, localization, and substitution/deletion/insertion accumulators instead.

| Comparison, X−Y | Full-set ΔSELD | Paired jackknife 95% interval | Paired bootstrap 95% interval |
|---|---:|---|---|
| B2 − A | −0.0095 | [−0.0259, +0.0059] | [−0.0234, +0.0054] |
| A − A0 | −0.0090 | [−0.0234, +0.0052] | [−0.0221, +0.0046] |
| B2 − A0 | −0.0184 | [−0.0305, −0.0077] | [−0.0285, −0.0076] |
| B2 − B3 | −0.0138 | [−0.0269, −0.0028] | [−0.0243, −0.0014] |
| B2 − B0 | −0.0130 | [−0.0274, −0.0009] | [−0.0236, +0.0003] |
| D2 − B2 | +0.0103 | [−0.0063, +0.0310] | [−0.0092, +0.0241] |

Negative Δ favors X. These values support the following limited conclusions:

- B2's numerical advantage over A is not resolved by either reported paired interval.
- B2 differs from A0 and B3 in these fixed-checkpoint comparisons, but a single checkpoint per method cannot establish an advantage over training randomness.
- B2 versus B0 is borderline and sensitive to the interval method; one interval excludes zero while the other barely includes it.
- D2 versus B2 does not demonstrate that adding STARSS SSL improves performance; its point estimate is worse and both intervals include zero.

The A–A0 difference of approximately 0.009 is comparable to B2–A. It is a warning about differences between the observed baseline runs, **not a quantitative estimate or lower bound on seed variance from two observations**. The bootstrap conditions on the existing checkpoints and does not resample pretraining, fine-tuning, or checkpoint selection. Recordings may also share rooms/sessions, so recording-level intervals should be described with their resampling unit; stronger generalization claims may require additional grouping.

New method comparisons should include multiple independently initialized pretraining runs and multiple fine-tuning seeds as appropriate, with paired test recordings and a fixed held-out selection protocol. Three seeds per arm is the review's minimum starting recommendation, not a guarantee of adequate statistical precision. A hierarchical analysis can separate training variation from recording variation. Exploratory comparisons across many groups should not be presented as a single pre-specified significance test.

## 8. Repair and verification plan

### 8.1 Correct the return values without changing the metric definition

The existing [jackknife patch](../../li_review/report1_spr_pair/code/patches/02_jackknife_point_values.patch) saves the full-set tuple before entering the loop and restores it before return. A refactor can instead keep full-set and leave-one-out variables separate. In either case, preserve the declared metric definition, matching, threshold, and interval method while repairing this defect, and restore the **classwise values as well as the five aggregate values**.

Keep legacy scoring available only as an explicitly named historical-replay mode if it is needed for migration validation. Record the omitted file when reproducing it. Full-set scoring should be the normal scientific reporting path; sorting alone is not a fix.

### 8.2 Rescore recoverable predictions and preserve provenance

For each existing run, verify the exact expected recording names against a manifest, then rescore all recordings and write a separate corrected result artifact. Preserve the original logged values alongside an explanation of their support rather than silently overwriting the historical record. Include model/run identity, prediction hashes, recording count, scorer revision, averaging mode, angular threshold, point-estimator definition, and interval/resampling settings.

No retraining is needed to fix the point values when valid predictions exist. Missing predictions may require inference from a saved model. A checkpoint that failed to load as intended requires a new correctly initialized fine-tuning run; test-set selection bias requires a new selection protocol and appropriate reruns. Those cannot be repaired by changing one number in a CSV.

The current scorer loops over the files present in the prediction directory and does not enforce equality with an expected test-file manifest. A missing prediction file can therefore exclude a recording without being detected by this function. This is an additional validation gap, **not evidence that the audited 78-file runs actually lacked files**. Check exact filenames, not only counts, and reject unexpected non-CSV files and duplicate logical recording identities.

### 8.3 Behavioral checks required for a scorer fix

The review's later branch audit notes that its jackknife test checked source-statement order rather than behavior. Before adopting a fix, verify observable outputs:

1. With identical predictions, point values from jackknife and non-jackknife calls agree for all five metrics and every classwise value.
2. Reversing or permuting the file enumeration does not change full-set points or intervals beyond floating-point tolerance.
3. A deliberately influential recording appears in the full-set point and is omitted once in the jackknife samples; the final omitted file cannot determine the returned point.
4. Full-set A0, B2, and D2 values agree with the existing corrected references, while an explicitly requested legacy replay reproduces the saved 77-recording references.
5. Manifest mismatches fail visibly, and the interval API handles unsupported small-N or class-count cases explicitly. The present classwise interval reshape hardcodes 13 classes; that matches this task but should not silently masquerade as a generic scorer.
6. Validation and test manifests are disjoint in the new protocol, and loading the intended pretrained model passes encoder-coverage checks.

This report does not apply the patch or run these proposed regression checks. It documents the live control flow and verifies existing evidence; Torch-dependent model checks should run through Slurm under the repository rules.

## 9. What can be reported now

The review's 78-recording rescored values are the appropriate full-set point values for the supplied historical prediction artifacts. They should be labeled as scores from the historical training/selection protocol, with paired uncertainty conditional on the evaluated checkpoints. The old 77-recording values remain useful only for reproducing previous logs and the declared Fox–Olivia compatibility checks.

The evidence does not establish a general pretext-method improvement, a successfully evaluated historical author checkpoint, or an untouched-test estimate for models selected using fold 4. Those limitations remain after the jackknife point-value error is repaired.

## 10. Work performed for this report

Read the parent review's evaluation findings, rescoring implementation, saved metrics, checkpoint audit, and reviewed fixes; traced the current scoring and training call sites; checked the upstream baseline and official metric definitions; calculated the table's point-value differences from the saved JSON; and cross-checked the Stage 3 replay/inference rows. This task adds this report only and does not modify scorers, training configurations, result CSVs, checkpoints, or running jobs.

Report validation checked the three-sentence opening, local source-link targets, the composite-score formula against all eleven saved runs, and agreement of the three Stage 3 scorer-replay rows with their historical subset references. These are document/data consistency checks, not new end-to-end evaluations.
