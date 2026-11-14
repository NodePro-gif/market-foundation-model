# Contributing

Thank you for helping improve Market Foundation Model.

## Before making changes

1. Open an issue for substantial changes so the intended behavior can be discussed.
2. Keep existing experiment entry points and artifact formats compatible unless a migration is included.
3. Never commit raw datasets, API credentials, access tokens, private account data or local environment files.
4. Do not rewrite historical metrics. New experiments should create a new, clearly described result record.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python test_tokenizer.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Pull requests

A pull request should include:

- A concise description of the problem and the change.
- The commands used to validate the change.
- Any relevant data split, random seed, model configuration and hardware details for experiment changes.
- Updated documentation when commands, paths, configuration or outputs change.

Keep pull requests focused. Avoid mixing refactors, model changes and result updates unless they are inseparable.

## Research integrity

Use chronological splits. Fit preprocessing, calibration and threshold selection on training/validation data only. Do not tune against the final test set. Report transaction costs, sample counts, time ranges and limitations alongside performance metrics.
