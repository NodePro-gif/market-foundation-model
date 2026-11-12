# Adaptive Market Foundation — Colab T4 upgrade

This branch adds multi-asset fine-tuning and a transparent decision layer.
It does not guarantee accuracy or positive returns. Acceptance requires
chronological validation, untouched test evaluation and cost-aware walk-forward.

## Changes
- Fixed stage4_finetune.py: FinetuneDS(sym) previously loaded XAUUSD tokens
  and symbol context even when another instrument was requested.
- Added train_multisymbol.py: each asset uses its own token stream and ID.
- Chronological split: train 2015-2021, validation 2022, final test 2023.
- NIKKEI, BRENT and WTI are held out from supervised head fitting and measured
  separately as zero-shot transfer diagnostics.
- AMP fp16, gradient scaling/clipping, AdamW, early stopping and best checkpoint.

## Colab steps
1. Open colab/Adaptive_Market_Foundation_T4.ipynb in Google Colab.
2. Runtime > Change runtime type > T4 GPU.
3. Mount Drive and run all cells in order.
4. Ensure corpus is in /content/corpus. If tokenized files are missing, run
   Stage 1 and Stage 2. Stage 3 can reuse the verified checkpoint.
5. Fine-tune with train_multisymbol.py, then inspect metrics.json.
6. Run existing validation and walk-forward tools before any deployment.

## Suggested T4 starting settings
Stage 3: CTX=512, batch 32-48, AMP fp16. Fine-tune: batch 32-48,
stride 8, 6-10 epochs, early stopping. If CUDA OOM, lower batch to 16.
Mount Drive and save checkpoints frequently because Colab sessions disconnect.

## Initial acceptance gates (targets, not promised outcomes)
- Data integrity: monotonic unique timestamps, valid OHLC, no invalid labels,
  no target leakage and purged split boundaries.
- Validation AUC >= 0.60, Brier better than prevalence-only baseline,
  ECE <= 0.08 after calibration fitted on validation only.
- Final test AUC >= 0.58 and positive expectancy after realistic spread,
  commission and slippage; drawdown within a risk limit set before testing.
- Walk-forward positive net expectancy in at least 2/3 folds with adequate
  sample size; no single fold may account for all returns.
- Any failed gate means RESEARCH_ONLY. Do not relax gates after seeing test data.
Accuracy alone is not sufficient. 2023 is a historical holdout, not a guarantee
for future conditions. Add later unseen market data before live use.

## Decision layer
market_intelligence.py combines probabilities, quantiles, regime context and
costs to return BUY/SELL/WAIT. It abstains when conservative edge fails costs
or volatility is high. It is research decision support, not execution software.

## Limits and safety
Zero-shot held-out metrics may be poorly calibrated because heads are not fit
on those symbols. Treat them as transfer diagnostics, not proof of transfer.
Colab T4 sessions can disconnect. Never store broker credentials in GitHub.
No live trading is enabled.
