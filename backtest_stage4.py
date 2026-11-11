#!/usr/bin/env python
"""Cost-aware backtest of the stage-4 direction head.

Takes the probabilities written by stage4_finetune.py (results/finetune/
test_pred.npy), aligns each window with the bar it ended on, and bets on the
very next M5 bar - the exact horizon the head was trained on (label `nxt`).

Bets are independent: consecutive test windows are `ST4_TEST_STRIDE` bars
apart, so every bet is entered and closed (cost = 2 sides).

Usage (same env as the stage4 run that produced test_pred.npy):
    $env:ST4_TEST_STRIDE="64"
    python backtest_stage4.py
"""

import json
import os

import numpy as np

import stage4_finetune as s4

COST_PER_SIDE = 0.00015     # same as check_flip.py / train_xauusd.py
OUT = s4.OUT


def stats(name, ret, cost_sides=0):
    """ret = per-bet simple returns (already net of cost if cost applied).

    win_net = fraction of bets that end POSITIVE AFTER cost (not accuracy).
    """
    n = len(ret)
    gross = float(ret.sum())
    eq = float(np.prod(1 + ret))
    hit = float((ret > 0).mean()) if n else float("nan")
    sd = float(ret.std()) if n > 1 else 0.0
    sharpe = float(ret.mean() / (sd + 1e-12) * np.sqrt(n)) if n > 1 else 0.0
    return dict(name=name, n=n, total=gross, equity=eq, win_net=hit,
                sharpe=sharpe, mean_bps=float(ret.mean() * 1e4),
                cost_sides=cost_sides)


def main():
    pred = np.load(f"{OUT}/test_pred.npy")
    y = np.load(f"{OUT}/test_y.npy")

    ds = s4.FinetuneDS(s4.TARGET_SYM, years=[2023], stride=s4.TEST_STRIDE)
    if len(ds.bars) != len(pred):
        raise SystemExit(
            f"window mismatch: backtest rebuilt {len(ds.bars)} windows but "
            f"test_pred.npy has {len(pred)} - rerun with the SAME "
            f"ST4_TEST_STRIDE/ST4_STRIDE env as the stage4 run")

    lab = np.load(f"{s4.MKT}/lab_{s4.TARGET_SYM}.npz")
    b = np.asarray(ds.bars, dtype=np.int64)
    c = lab["c"].astype(np.float64)
    ts = lab["ts"]
    nxt = lab["nxt"]

    ok = b + 1 < len(c)
    b, pred, y = b[ok], pred[ok], y[ok]
    fwd = (c[b + 1] - c[b]) / c[b]           # next-bar return, the bet horizon
    up = nxt[b].astype(int)                  # same info as y (sanity)

    long_cost = 2 * COST_PER_SIDE            # enter + exit per independent bet
    rows = []

    sig = np.where(pred >= 0.5, 1, -1)
    rows.append(stats("model (p>=0.5)", sig * fwd - long_cost, 2))
    rows.append(stats("model gross (no cost)", sig * fwd, 0))
    rows.append(stats("inverse model", -sig * fwd - long_cost, 2))
    rows.append(stats("always long", fwd - long_cost, 2))
    rows.append(stats("always short", -fwd - long_cost, 2))
    rows.append(stats("persistence (repeat last bar)",
                      np.where(up == 1, 1, -1) * fwd - long_cost, 2))

    # accuracy of the head itself (against label `nxt`), as stage4 reports it
    hit = float((np.where(pred >= 0.5, 1, 0) == y).mean())
    brier = float(((pred - y) ** 2).mean())

    # breakeven round-trip cost: where net model return = 0
    model_gross_mean = rows[1]["mean_bps"] / 1e4          # per bet, no cost
    breakeven_side_bps = model_gross_mean / 2 * 1e4       # two sides per bet

    equity = np.cumprod(1 + (sig * fwd - long_cost))
    buyhold = np.cumprod(1 + fwd)

    res = dict(
        n_bets=int(len(b)), stride=int(s4.TEST_STRIDE),
        cost_per_side=COST_PER_SIDE,
        breakeven_cost_per_side_bps=float(breakeven_side_bps),
        first_ts=int(ts[b[0]]), last_ts=int(ts[b[-1]]),
        head=dict(acc=hit, brier=brier),
        strategies=rows,
        benchmark_buynhold=dict(
            total=float(buyhold[-1] - 1) if len(buyhold) else 0.0),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/backtest.json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.save(f"{OUT}/equity_model.npy", equity)
    np.save(f"{OUT}/equity_buyhold.npy", buyhold)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4.5))
        plt.plot(equity, label=f"model p>=0.5 net ({rows[0]['equity']:.3f})")
        plt.plot(buyhold, label=f"buy & hold ({buyhold[-1]:.3f})", alpha=.8)
        plt.axhline(1.0, color="gray", lw=.7)
        plt.title(f"Stage4 XAUUSD 2023 - {len(b)} one-bar bets, "
                  f"stride {s4.TEST_STRIDE}, cost {COST_PER_SIDE}/side")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT}/equity.png", dpi=120)
        plt.close()
    except Exception as e:                      # plotting is optional
        print("plot skipped:", e)

    print(f"bets={len(b)} stride={s4.TEST_STRIDE} "
          f"head_acc={hit:.4f} brier={brier:.4f} "
          f"breakeven_cost={breakeven_side_bps:.3f} bps/side")
    for r in rows:
        print(f"  {r['name']:28s} equity={r['equity']:+.4f} "
              f"total={r['total']:+.4%} win_net={r['win_net']:.4f} "
              f"mean_bps={r['mean_bps']:+.2f} sharpe={r['sharpe']:+.2f}")
    print(f"  buy & hold                 equity={buyhold[-1]:+.4f} "
          f"total={buyhold[-1]-1:+.4%}")
    print("BACKTEST DONE", flush=True)


if __name__ == "__main__":
    main()
