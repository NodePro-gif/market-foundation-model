# Project structure

This document describes the current repository layout. Training scripts intentionally remain at the repository root because several scripts import one another directly (for example, fine-tuning imports the pretraining module).

## Main pipeline

- `notebooks/Market_Foundation_Model_T4.ipynb`: guided Google Colab T4 workflow with Google Drive persistence.

| File | Purpose |
|---|---|
| `stage1_corpus.py` | Download and verify the multi-asset corpus in the Colab layout |
| `download_local.py` | Download corpus files for a local workstation |
| `stage2_tokenizer.py` | Parse bars, derive market attributes, train BPE and write token/label artifacts |
| `test_tokenizer.py` | Self-test BPE merge and encoding behavior |
| `stage3_pretrain.py` | Causal language-model pretraining |
| `verify_ckpt.py` | Check compatibility between a checkpoint and regenerated tokenizer |
| `stage4_finetune.py` | Fine-tune prediction heads |
| `validate_stage4.py` | Validation-based calibration and threshold selection |
| `walkforward_stage4.py` | Legacy exploratory walk-forward script; historical output is not a valid benchmark for the corrected split |
| `backtest_stage4.py` | Cost-aware strategy evaluation |

## Supporting and diagnostic scripts

- `train_xauusd.py`: historical GRU/MLP baseline.
- `diag.py`, `check_flip.py`, `probe.py`, `probe2.py`, `probe_net.py`: diagnostics and exploratory probes.
- `probe_transfer.py`: frozen-backbone cross-asset transfer experiment.
- `monitor.py`: local training and result monitor.
- `assemble.py`, `prep_stage4.py`, `pack_artifacts.py`, `split_parts.py`: artifact preparation and transfer helpers.
- `move_year.py`: small data-layout utility.

## Generated artifacts

| Directory | Contents | Version-control policy |
|---|---|---|
| `ckpt/` | Pretrained checkpoint and compact training log | Selected checkpoint retained |
| `results/` | Metrics, plots, prediction exports and baseline artifacts | Selected research outputs retained |
| `corpus/`, `npz/`, `mkt/`, `parts/` | Downloaded data and regenerated intermediate artifacts | Ignored by Git |
| `results/finetune/` | Fine-tuning outputs | Metrics retained; large model/array/CSV artifacts ignored |

Do not move or rename pipeline scripts without updating their imports, path assumptions, documentation and Colab commands in the same change.
