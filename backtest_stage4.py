#!/usr/bin/env python
"""Evaluate the validation-locked stage-4 policy on the final 2023 test set.

The threshold and holding horizon must come from validation.json (2022).
This script deliberately does not sweep thresholds or choose a horizon on 2023.
"""

import json
import os

import numpy as np

import stage4_finetune as s4

COST_PER_SIDE = float(os.environ.get("BT_COST_PER_SIDE", "0.00015"))
OUT = s4.OUT


def main():
    validation_path = os.path.join(OUT, "validation.json")
    pred_path = os.path.join(OUT, "test_pred_cal.npy")
    if not os.path.isfile(validation_path) or not os.path.isfile(pred_path):
        raise SystemExit(
            "missing validation.json or test_pred_cal.npy; run validate_stage4.py first"
        )
    with open(validation_path, encoding="utf-8") as fh:
        validation = json.load(fh)
    threshold = float(validation["threshold"])
    horizon = validation.get("chosen_horizon")
    if horizon is None:
        raise SystemExit(
            "validation did not select a holding horizon; do not choose one on the test set"
        )
    horizon = int(horizon)

    pred = np.load(pred_path)
    y = np.load(os.path.join(OUT, "test_y.npy"))
    ds = s4.FinetuneDS(s4.TARGET_SYM, years=[2023], stride=s4.TEST_STRIDE)
    if len(ds.bars) != len(pred) or len(y) != len(pred):
        raise SystemExit(
            f"window mismatch: rebuilt={len(ds.bars)} calibrated={len(pred)} labels={len(y)}; "
            "rerun stage4 and validation with the same stride settings"
        )

    lab = np.load(os.path.join(s4.MKT, f"lab_{s4.TARGET_SYM}.npz"))
    bars = np.asarray(ds.bars, dtype=np.int64)
    close = lab["c"].astype(np.float64)
    ts = lab["ts"]

    # Only keep trades whose full selected holding horizon exists.
    valid = bars + horizon < len(close)
    bars, pred, y = bars[valid], pred[valid], y[valid]
    signal = np.where(pred >= threshold, 1,
                      np.where(pred <= 1.0 - threshold, -1, 0))
    active = signal != 0
    if not active.any():
        raise SystemExit("the validation-locked policy produced no test trades")

    trade_bars = bars[active]
    trade_signal = signal[active]
    gross_returns = trade_signal * (
        (close[trade_bars + horizon] - close[trade_bars]) / close[trade_bars]
    )
    net_returns = gross_returns - 2.0 * COST_PER_SIDE
    n = int(len(net_returns))
    gross_total = float(gross_returns.sum())
    net_total = float(net_returns.sum())
    result = dict(
        protocol=dict(validation_year=2022, final_test_year=2023),
        policy=dict(threshold=threshold, horizon_bars=horizon,
                    cost_per_side=COST_PER_SIDE),
        opportunities=int(len(bars)),
        trades=n,
        head=dict(
            accuracy=float((((pred >= 0.5).astype(int)) == y).mean()),
            brier=float(np.mean((pred - y) ** 2)),
        ),
        strategy=dict(
            gross_sum=gross_total,
            net_sum=net_total,
            mean_gross_per_trade=float(gross_returns.mean()),
            mean_net_per_trade=float(net_returns.mean()),
            win_rate_after_cost=float((net_returns > 0).mean()),
            breakeven_cost_per_side=float(gross_returns.mean() / 2.0),
        ),
        first_trade_ts=int(ts[trade_bars[0]]),
        last_trade_ts=int(ts[trade_bars[-1]]),
        note=("Trade returns are summed, not compounded: positions may overlap "
              "when the selected holding horizon exceeds the test-window stride."),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "backtest.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"VALIDATION-LOCKED TEST POLICY: threshold={threshold:.3f}, "
          f"horizon={horizon} bars, cost={COST_PER_SIDE * 1e4:.2f} bps/side")
    print(f"trades={n}/{len(bars)} gross_sum={gross_total:+.4%} "
          f"net_sum={net_total:+.4%} win_after_cost={(net_returns > 0).mean():.3f}")
    print(f"mean_net_per_trade={net_returns.mean() * 1e4:+.3f} bps "
          f"breakeven_cost={gross_returns.mean() * 0.5 * 1e4:.3f} bps/side")
    print("No threshold or horizon sweep was performed on the final test set.")
    print("BACKTEST DONE", flush=True)


if __name__ == "__main__":
    main()
