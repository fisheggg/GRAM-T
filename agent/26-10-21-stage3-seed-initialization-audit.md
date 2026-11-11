# 26-10-21 Stage 3 seed-initialization audit

Checked the pretraining and both SELD training entry points. The seed-order error is not fixed: `train.py` calls `seed_everything` after creating its model and data module. Neither SELD entry point explicitly initializes Python, NumPy, or Torch RNGs from its run seed.

All 24 Stage 3 YAMLs declare seeds 42/43/44 and `runtime.seed_before_model_and_data: true`, but the combined runner that would enforce this remains unimplemented. Configuration validation does not establish correct runtime initialization. No code was changed and no jobs were submitted by this audit.
