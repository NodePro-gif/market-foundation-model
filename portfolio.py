"""Non-overlapping, sequential portfolio simulation for fixed-horizon signals."""
from __future__ import annotations

import numpy as np


def simulate_non_overlapping(close, bars, signals, horizon, cost_per_side=0.00015):
    """Enter at close[bar], exit at close[bar+horizon], and ignore signals while open.

    Costs are applied multiplicatively on entry and exit. Signals must be aligned
    with strictly increasing bar indices. No overlapping capital is counted.
    """
    close = np.asarray(close, dtype=np.float64)
    bars = np.asarray(bars, dtype=np.int64)
    signals = np.asarray(signals, dtype=np.int8)
    if bars.ndim != 1 or signals.ndim != 1 or len(bars) != len(signals):
        raise ValueError("bars and signals must be 1-D arrays with equal lengths")
    if horizon < 1 or cost_per_side < 0 or cost_per_side >= 1:
        raise ValueError("horizon must be positive and cost_per_side must be in [0, 1)")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and strictly positive")
    if not np.isin(signals, (-1, 0, 1)).all():
        raise ValueError("signals must contain only -1, 0, or 1")
    if len(bars) and (np.diff(bars) <= 0).any():
        raise ValueError("bars must be strictly increasing")
    if len(bars) and (bars.min() < 0 or bars.max() + horizon >= len(close)):
        raise ValueError("bar indices do not have complete holding-horizon prices")

    trades = []
    next_available_bar = -1
    gross_equity = 1.0
    net_equity = 1.0
    for bar, signal in zip(bars, signals):
        if signal == 0 or bar < next_available_bar:
            continue
        if signal not in (-1, 1):
            raise ValueError("signals must contain only -1, 0, or 1")
        exit_bar = int(bar + horizon)
        underlying = close[exit_bar] / close[bar] - 1.0
        gross_return = float(signal * underlying)
        net_return = float((1.0 + gross_return) * (1.0 - cost_per_side) ** 2 - 1.0)
        gross_equity *= 1.0 + gross_return
        net_equity *= 1.0 + net_return
        trades.append({
            "entry_bar": int(bar), "exit_bar": exit_bar,
            "direction": int(signal), "gross_return": gross_return,
            "net_return": net_return,
        })
        # Require at least one new bar after exit before opening another trade.
        next_available_bar = exit_bar + 1

    returns = np.asarray([t["net_return"] for t in trades], dtype=np.float64)
    gross = np.asarray([t["gross_return"] for t in trades], dtype=np.float64)
    if len(returns):
        curve = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
        drawdown = curve / np.maximum.accumulate(curve) - 1.0
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0
    return {
        "trades": trades,
        "trade_count": int(len(trades)),
        "gross_compounded": float(gross_equity - 1.0),
        "net_compounded": float(net_equity - 1.0),
        "mean_gross": float(gross.mean()) if len(gross) else 0.0,
        "mean_net": float(returns.mean()) if len(returns) else 0.0,
        "win_rate_after_cost": float((returns > 0).mean()) if len(returns) else 0.0,
        "max_drawdown": max_drawdown,
    }
