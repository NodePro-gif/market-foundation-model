# Changelog

Notable repository changes are documented here. Historical experiment results remain in [PROGRESS.md](PROGRESS.md).

## Unreleased

- Enforced chronological preprocessing, pretraining, fine-tuning, calibration and final-test boundaries.
- Added token-span-aware alignment and purging for forward-label horizons.
- Added long-span tokenizer and independent-series BPE boundary regression coverage.
- Added balanced multi-asset supervised fine-tuning, AMP, gradient accumulation and a Colab T4 notebook.
- Bound pretrained checkpoints to tokenizer fingerprints and removed final-test data from checkpoint verification.
- Locked final backtesting to threshold and horizon selected on the 2022 calibration period.
- Expanded continuous integration to Python 3.10–3.12.
- Clarified historical-result limitations and repository security/citation guidance.
