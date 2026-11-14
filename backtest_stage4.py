#!/usr/bin/env python
"""Run the validation-locked policy as a sequential, non-overlapping portfolio."""
import json
import os

import numpy as np

import stage4_finetune as s4
from portfolio import simulate_non_overlapping

COST_PER_SIDE = float(os.environ.get("BT_COST_PER_SIDE", "0.00015"))
OUT = s4.OUT


def main():
    validation_path = os.path.join(OUT, "validation.json")
    pred_path = os.path.join(OUT, "test_pred_cal.npy")
    if not os.path.isfile(validation_path) or not os.path.isfile(pred_path):
        raise SystemExit("missing validation.json or test_pred_cal.npy; run validate_stage4.py first")
    with open(validation_path, encoding="utf-8") as fh:
        validation = json.load(fh)
    threshold = float(validation["threshold"])
    horizon = validation.get("chosen_horizon")
    if horizon is None:
        raise SystemExit("validation did not select a holding horizon; refusing test-set tuning")
    horizon = int(horizon)

    pred = np.load(pred_path)
    y = np.load(os.path.join(OUT, "test_y.npy"))
    ds = s4.FinetuneDS(s4.TARGET_SYM, years=[2023], stride=s4.TEST_STRIDE)
    if len(ds.bars) != len(pred) or len(y) != len(pred):
        raise SystemExit("prediction/window mismatch; rerun stage4 and validation with identical settings")

    lab = np.load(os.path.join(s4.MKT, f"lab_{s4.TARGET_SYM}.npz"))
    bars = np.asarray(ds.bars, dtype=np.int64)
    close = lab["c"].astype(np.float64)
    ts = lab["ts"]
    valid = bars + horizon < len(close)
    bars, pred, y = bars[valid], pred[valid], y[valid]
    signal = np.where(pred >= threshold, 1, np.where(pred <= 1.0 - threshold, -1, 0))
    sim = simulate_non_overlapping(close, bars, signal, horizon, COST_PER_SIDE)
    if sim["trade_count"] == 0:
        raise SystemExit("the validation-locked policy produced no complete test trades")

    trade_bars = np.asarray([t["entry_bar"] for t in sim["trades"]], dtype=np.int64)
    result = dict(
        protocol=dict(validation_year=2022, final_test_year=2023,
                      overlap_policy="one position at a time; ignore signals until exit"),
        policy=dict(threshold=threshold, horizon_bars=horizon,
                    cost_per_side=COST_PER_SIDE, costs_applied="multiplicative entry and exit"),
        opportunities=int(len(bars)),
        trades=sim["trade_count"],
        head=dict(
            accuracy=float((((pred >= 0.5).astype(int)) == y).mean()),
            brier=float(np.mean((pred - y) ** 2)),
        ),
        strategy=dict(
            gross_compounded=sim["gross_compounded"],
            net_compounded=sim["net_compounded"],
            mean_gross_per_trade=sim["mean_gross"],
            mean_net_per_trade=sim["mean_net"],
            win_rate_after_cost=sim["win_rate_after_cost"],
            max_drawdown=sim["max_drawdown"],
            breakeven_cost_per_side=float(
                np.mean([t["gross_return"] for t in sim["trades"]]) / 2.0),
        ),
        first_trade_ts=int(ts[trade_bars[0]]),
        last_trade_ts=int(ts[trade_bars[-1]]),
        trades_detail=sim["trades"],
        note="Sequential compounding; no overlapping positions or double-counted capital.",
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "backtest.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"VALIDATION-LOCKED TEST: threshold={threshold:.3f}, horizon={horizon} bars, "
          f"cost={COST_PER_SIDE * 1e4:.2f} bps/side")
    print(f"trades={sim['trade_count']}/{len(bars)} "
          f"gross_compounded={sim['gross_compounded']:+.4%} "
          f"net_compounded={sim['net_compounded']:+.4%}")
    print(f"win_after_cost={sim['win_rate_after_cost']:.3f} "
          f"max_drawdown={sim['max_drawdown']:.3%}")
    print("No threshold or horizon sweep was performed on the final test set.")
    print("BACKTEST DONE", flush=True)


if __name__ == "__main__":
    main()
