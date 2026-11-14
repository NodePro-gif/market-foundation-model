import numpy as np
import pytest

from portfolio import simulate_non_overlapping


def test_non_overlapping_positions_skip_signals_until_exit():
    close = np.array([100, 101, 102, 103, 104, 105, 106], dtype=float)
    bars = np.array([0, 1, 2, 3], dtype=np.int64)
    signals = np.array([1, -1, -1, 1], dtype=np.int8)
    result = simulate_non_overlapping(close, bars, signals, horizon=2, cost_per_side=0.0)
    assert result["trade_count"] == 2
    assert result["trades"][0]["entry_bar"] == 0
    assert result["trades"][0]["exit_bar"] == 2
    assert result["trades"][1]["entry_bar"] == 3
    assert result["trades"][1]["exit_bar"] == 5


def test_costs_are_applied_on_both_sides():
    close = np.array([100.0, 101.0])
    result = simulate_non_overlapping(
        close, np.array([0]), np.array([1]), horizon=1, cost_per_side=0.001
    )
    expected = (1.01) * (0.999 ** 2) - 1.0
    assert result["net_compounded"] == pytest.approx(expected)


def test_rejects_unsorted_or_duplicate_bars():
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_non_overlapping(
            np.arange(10) + 100, np.array([1, 1]), np.array([1, -1]), 1
        )


def test_rejects_incomplete_horizon():
    with pytest.raises(ValueError, match="complete holding-horizon"):
        simulate_non_overlapping(
            np.arange(5) + 100, np.array([4]), np.array([1]), 1
        )
