# Reproducibility and evaluation protocol

## Data and chronology

The current repository scripts and experiment notes describe a historical multi-asset corpus spanning 2015 onward. The documented pretraining protocol uses data through 2022 and reserves 2023 for evaluation; the tokenizer also records held-out assets for cross-asset experiments. Always inspect the actual year and symbol filters in the script version used for a run.

For the fine-tuning validation workflow, calibration and strategy choices should be made on the validation period (documented as 2021–2022) and then evaluated on the untouched 2023 test period. The walk-forward script repeats this process across chronological folds.

**Important:** README/PROGRESS metrics were produced by different runs and configurations (including CPU-reduced strides and GPU-oriented defaults). They are not interchangeable. Record the exact configuration when reproducing a result.

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
python verify_ckpt.py
python stage3_pretrain.py
python stage4_finetune.py
python validate_stage4.py
python walkforward_stage4.py
python backtest_stage4.py
```

Some scripts expect artifacts in a specific local or Colab directory. Prepare those paths before running the sequence; see the script constants and the setup instructions in the README.

## Evaluation safeguards

1. Split chronologically. Never randomly shuffle time-series observations across train and test.
2. Fit normalization, bins, tokenizer statistics, calibration and decision thresholds using training/validation data only.
3. Keep the final test set untouched until model and strategy choices are fixed.
4. Use purging/embargo or non-overlapping evaluation where label horizons and windows overlap.
5. Compare against simple baselines on the same dates and with the same costs.
6. Report gross and net returns separately. Include realistic spread, fees, slippage and execution assumptions.
7. Treat AUC, accuracy and calibration metrics as predictive diagnostics, not proof of profitability.

## Existing result files

The `results/` directory contains selected outputs from prior experiments. `PROGRESS.md` records experiment chronology and caveats. Preserve those historical records; append new runs rather than silently replacing old metrics.
