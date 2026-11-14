# Reproducibility and evaluation protocol

## Data and chronology

The corrected pipeline fits tokenizer bins, regime clusters and BPE merges on non-held-out assets from 2015–2020 only; held-out assets are encoded with that frozen vocabulary but never shape its statistics. It pretrains on non-held-out assets from 2015–2020 and selects the language-model checkpoint on 2021; fine-tunes shared heads on non-held-out assets from 2015–2020 with equal per-asset sampling and selects the supervised checkpoint on XAUUSD 2021; reserves 2022 for calibration/threshold selection; and keeps 2023 as the final test. Earlier committed checkpoints and metrics were produced under a different protocol and must not be presented as results from this corrected split.

**Important:** README/PROGRESS metrics were produced by different runs and configurations (including CPU-reduced strides and GPU-oriented defaults). They are not interchangeable. Record the exact configuration when reproducing a result.

The BPE training corpus uses contiguous excerpts from the training period and explicit sequence-boundary sentinels, preventing artificial adjacent pairs between different assets/timeframes. Fine-tuning samples non-held-out assets with equal per-asset probability; checkpoint selection is evaluated on XAUUSD 2021. Pretraining checkpoints include best_manifest.json, which fingerprints all tokenizer and calibration artifacts. If the tokenizer is regenerated, discard the old resume.pt and rerun pretraining; fine-tuning rejects mismatched or legacy checkpoints.

## Reproduction checklist

Record all of the following with each experiment:

- Git commit SHA.
- Python, PyTorch, CUDA and GPU versions.
- Dataset provider, symbols, timeframes, available years and missing files.
- Data filters, deduplication rules and chronological split boundaries.
- Tokenizer vocabulary, merge table and checkpoint identity.
- Random seed, batch size, learning rate, epochs, context length and stride.
- Validation/calibration period and the untouched test period.
- Number of windows/trades and whether windows overlap.
- Spread, commission, slippage, execution assumptions and holding horizon.

## Recommended sequence

```bash
python download_local.py
python test_tokenizer.py
python stage2_tokenizer.py
python stage3_pretrain.py
python verify_ckpt.py
python stage4_finetune.py
python validate_stage4.py
python backtest_stage4.py
```

Some scripts expect artifacts in a specific local or Colab directory. Prepare those paths before running the sequence; see the script constants and setup instructions in the README. Model checkpoints are generated locally and are not distributed in Git. A new `best_manifest.json` binds each newly selected pretraining checkpoint to SHA-256 fingerprints of `calib.npz`, `kmeans.pkl`, `vocab.json` and `bpe.npz`; `stage4_finetune.py` refuses a checkpoint without a matching manifest. `verify_ckpt.py` checks this fingerprint and measures loss on 2021 validation data only, never on the 2023 final test. Regenerate tokenizer artifacts and pretraining checkpoints under the corrected chronology before interpreting new fine-tuning metrics. `walkforward_stage4.py` is a legacy exploratory script whose folds overlap the former training period. Its historical output is retained for auditability, not as a current clean benchmark. Do not use it to report out-of-sample performance for the corrected protocol.

## Evaluation safeguards

1. Split chronologically. Never randomly shuffle time-series observations across train and test.
2. Fit normalization, bins, tokenizer statistics and model parameters on training data only. Purge samples whose longest forward label crosses a split boundary. Fit calibration and thresholds on a period separate from checkpoint selection.
3. Keep the final test set untouched until model and strategy choices are fixed.
4. Use purging/embargo or non-overlapping evaluation where label horizons and windows overlap.
5. Compare against simple baselines on the same dates and with the same costs.
6. Report gross and net returns separately. Include realistic spread, fees, slippage and execution assumptions.
7. Treat AUC, accuracy and calibration metrics as predictive diagnostics, not proof of profitability.

## Existing result files

The `results/` directory contains selected outputs from prior experiments. `PROGRESS.md` records experiment chronology and caveats. Preserve those historical records; append new runs rather than silently replacing old metrics.
