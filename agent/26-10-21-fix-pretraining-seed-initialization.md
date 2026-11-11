# 26-10-21 Fix pretraining seed initialization

Moved `seed_everything(cfg.seed, workers=True)` to the first statement of `train.py`'s `main()`, before logger, trainer, model, masking, and data-module construction. Removed the late call after data-module creation. This applies to both GRAM reconstruction and SPR through their shared pretraining entry point.

Verified Python syntax and the seed call's position, configured seed argument, worker flag, and uniqueness using AST inspection without importing Torch. Whitespace checks passed. No jobs were submitted. SELD entry points and the missing combined Stage 3 runner were outside this change.
