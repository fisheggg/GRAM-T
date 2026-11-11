# 26-10-21 — Simplify Stage 3 to one YAML per CSV run

Replaced the earlier Stage 3 proposal with the user's [stage3_training_plan.csv](../stage3_training_plan.csv): eight groups A–H, each with seeds 42/43/44, for 24 runs. Created exactly one combined pretraining-and-SELD YAML per run in [configs/stage3](../configs/stage3/README.md). The unseeded official-GRAM row is retained as a reference and receives no YAML.

The user confirmed fresh pretraining plus SELD per seed. The user also confirmed that H adds channel isolation and minor implementation fixes only; H therefore retains original STARSS format and legacy SPR target alignment, all-patch loss, and reduction. The CSV's “Linear probe” field is interpreted as the pretraining SPR head; downstream encoders are fine-tuned in every group.

Retained the existing held-out recording manifests and common legacy final-scoring setting. Pretraining RIRs retain their original N3D/channel-1-up convention; B/D/F/G alone match STARSS features to it, as specified by the CSV. The common readout remains historical `adjacent`. All output roots include the new Stage 3 run ID and seed, and every pretraining run starts with a null resume checkpoint.

Archived the earlier configuration trees, plan, result tables, and validator in [archive/stage3_plan_2026-10-18.tar.gz](../archive/stage3_plan_2026-10-18.tar.gz). SHA-256 checks verified all 309 source files against the archive before removing their active copies. The 24 active run files replace the prior 299 YAML planning files; measured `results/stage2.csv` and `results/stage3.csv` remain intact.

Updated [results/stage3_plan.md](../results/stage3_plan.md), the 24-row [planned-runs index](../results/stage3_planned_runs.csv), and [scripts/stage3/validate_plan.py](../scripts/stage3/validate_plan.py) to check exact correspondence with the CSV. Added an ignore exception for the user's source CSV. The source CSV itself was not modified; its SHA-256 is `b2a15f76872e43feb196573105134d32c5ecd8ea79d5bc197d16f9674503d217`.

Validation covers run counts, source-row values, seeds, head/pretext choices, format adapters, fresh initialization, H's limited scope, patch/mask counts, shared split membership, isolated outputs, and the intended matched comparisons. These are configuration checks. The combined runner, adaptation, and channel-isolated model support still require implementation and GPU verification; this task did not change training code or submit jobs.
