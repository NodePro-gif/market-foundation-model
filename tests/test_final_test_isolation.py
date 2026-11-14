"""Regression guards for the chronological final-test boundary."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _year_2023_finetune_calls(source):
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_finetune_ds = (
            isinstance(func, ast.Name) and func.id == "FinetuneDS"
        ) or (
            isinstance(func, ast.Attribute) and func.attr == "FinetuneDS"
        )
        if not is_finetune_ds:
            continue
        for kw in node.keywords:
            if kw.arg == "years" and isinstance(kw.value, ast.List):
                values = [elt.value for elt in kw.value.elts
                          if isinstance(elt, ast.Constant)]
                if values == [2023]:
                    found.append(node.lineno)
    return found


def test_finetuning_stage_never_reads_final_test():
    source = (ROOT / "stage4_finetune.py").read_text(encoding="utf-8")
    assert _year_2023_finetune_calls(source) == []
    assert "test_pred.npy" not in source
    assert "test_y.npy" not in source


def test_final_test_is_accessed_only_after_2022_policy_selection():
    source = (ROOT / "validate_stage4.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    test_lines = _year_2023_finetune_calls(source)
    assert len(test_lines) == 1

    threshold_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "best_threshold"
    ]
    assert threshold_lines and max(threshold_lines) < test_lines[0]

    horizon_loops = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "HORIZONS"
    ]
    assert horizon_loops and max(horizon_loops) < test_lines[0]

    # The test phase may evaluate only the already-selected horizon.
    test_phase = source.split("locked final test", 1)[1]
    assert "for k in HORIZONS" not in test_phase
    assert "k_best" in test_phase


def test_2022_calibration_labels_are_purged_before_july_policy_period():
    source = (ROOT / "validate_stage4.py").read_text(encoding="utf-8")
    assert "split_2022_periods(sample_ts)" in source
    assert "cal_mask &= (bars_all + 288) < july_bar" in source
    assert "pick is None" in source
    assert "2023 final test was NOT accessed" in source


def test_final_test_has_explicit_repeat_access_lock():
    source = (ROOT / "validate_stage4.py").read_text(encoding="utf-8")
    assert "FINAL_TEST_2023_ACCESSED.lock" in source
    assert "MFM_ALLOW_FINAL_TEST_RERUN" in source
    assert 'np.save(f"{OUT}/test_bars.npy", bt)' in source


def test_backtest_reuses_saved_final_test_bars_instead_of_rebuilding_2023_dataset():
    source = (ROOT / "backtest_stage4.py").read_text(encoding="utf-8")
    assert _year_2023_finetune_calls(source) == []
    assert 'test_bars.npy' in source
    assert 'FINAL_TEST_2023_ACCESSED.lock' in source
