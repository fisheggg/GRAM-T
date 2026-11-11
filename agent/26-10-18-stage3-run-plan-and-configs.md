# 26-10-18 — Prepare Stage 3 run plan, configurations, and group tables

Prepared the requested plan for official GRAM, the validation-selected best Stage 2 B/C model, and the validation-selected best corrected SPR model, each reported with legacy and corrected SELD scoring. The user confirmed official GRAM plus local A control and a historical search pool of B0–B3/C0–C3. No jobs were submitted.

The main deliverable is [results/stage3_plan.md](../results/stage3_plan.md), with six headline rows in [stage3_planned_groups.csv](../results/stage3_planned_groups.csv), 21 candidate groups in [stage3_candidate_groups.csv](../results/stage3_candidate_groups.csv), a detailed dependency/run table in [stage3_planned_runs.csv](../results/stage3_planned_runs.csv), and 17 implementation work packages in [stage3_fix_checklist.csv](../results/stage3_fix_checklist.csv).

Created individual versioned YAML specifications for 33 fresh pretraining runs, 63 supervised SELD fine-tuning runs, 63 test-inference passes, and 126 scoring passes, plus two feature preparations, selection barriers, and headline reports. Seeds are 42/43/44, with one paired pretraining/fine-tuning seed per new replica. The two scorer modes share trained checkpoints and predictions.

Created deterministic training/validation/test recording manifests from available metadata: 1,270 training recordings (70 real + 1,200 synthetic), 20 real validation recordings (ten Sony and ten TAU), and 78 test recordings. All 13 classes remain represented in every split. A whole-room holdout was rejected after class-coverage checks; these are recording-level manifests and do not claim room-independent validation. The manifest algorithm uses original training labels and stable filename hashes, not model/test performance.

The files explicitly describe prerequisites rather than claiming that review fixes already exist in the live code. Corrected separate-decoder support, the actual scorer switch, manifest-aware dispatch, loader/convention validation, and other review fixes still require implementation and runtime checks. Missing checkpoints and incomplete C groups are tracked in the artifact registry. Existing Stage 2 runs and historical CSVs remain intact.

Added [scripts/stage3/validate_plan.py](../scripts/stage3/validate_plan.py), which checks plan consistency without Torch or scheduler calls. Added narrow `.gitignore` exceptions for the planned result tables and split manifests. The existing `results/` directory already has setgid enabled, so no ownership or permission changes were needed.

Validation checks YAML/reference/dependency consistency, unique output identities, paired scorer support and predictions, validation-only winner selection, manifest hashes/disjointness/class coverage, and empty job/result fields. This is configuration validation, not GPU execution or certification of the pending code fixes.

The final consistency check passed for 296 referenced YAML run/selection/report configurations, 96 unique training output directories, and 287 planned execution-table rows. Local Markdown links and the Git ignore exceptions were checked. The official model's feature-cache and axis-adapter choices remain explicitly unresolved until its convention audit, rather than assuming the local RIR convention applies to it.
