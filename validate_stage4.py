#!/usr/bin/env python
"""Honest validation pass for the stage-4 direction head.

Fixes three problems at once:

1. Calibration      - fit Platt scaling on the reserved 2022 period (not on the test set),
                      report final-test ECE before/after.
2. Threshold choice - pick the betting threshold on the reserved 2022 period, lock it,
                      then touch 2023 exactly once (no post-hoc selection).
3. Profitability    - the round-trip cost is paid once per position, so
                      holding k bars amortises it. Sweep the holding horizon
                      and report where net > 0 at real cost levels.

Usage (same env as the stage4 run):
    $env:ST4_STRIDE="256"; $env:ST4_TEST_STRIDE="64"
    python validate_stage4.py
"""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from portfolio import simulate_non_overlapping
from training_utils import split_2022_periods

import stage4_finetune as s4

VAL_YEARS = (2022,)  # reserved for calibration/threshold selection
HORIZONS = (1, 3, 6, 12, 24, 48)          # holding horizons evaluated on validation only
# 0.5 / 1.5 / 3.0 bps PER SIDE (1 bps = 1e-4); round trip = 2 sides
COSTS_SIDE = (0.00005, 0.00015, 0.00030)
MIN_BETS = 100
THR_GRID = np.round(np.arange(0.50, 0.801, 0.025), 3)

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def brier_ece(p, y, n_bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    ece, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return brier, float(ece)


def auc_rank(y, s):
    return s4.auc_rank(y, s)


def predict(ds):
    """Run the fine-tuned heads on the configured device."""
    ckpt = os.path.join(s4.OUT, "best.pt")
    base_ckpt = os.environ.get("CKPT") or os.path.join(s4.s3.CKPT, "best.pt")
    base = s4.load_base(base_ckpt)
    model = s4.FinetuneGPT(base).to(s4.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yd, *_ in DataLoader(ds, batch_size=64, shuffle=False):
            xb = xb.to(s4.DEVICE)
            with torch.autocast(device_type=s4.DEVICE.type, dtype=torch.float16,
                                enabled=s4.DEVICE.type == "cuda" and s4.USE_AMP):
                pd, _, _ = model(xb)
            ps.append(torch.sigmoid(pd).float().cpu().numpy())
            ys.append(yd.numpy() if torch.is_tensor(yd) else np.asarray(yd))
    return np.concatenate(ps), np.concatenate(ys)


def next_returns(b, c, k):
    """Simple return of holding k bars from each window end bar b."""
    b = np.asarray(b, dtype=np.int64)
    ok = b + k < len(c)
    out = np.full(len(b), np.nan)
    out[ok] = (c[b[ok] + k] - c[b[ok]]) / c[b[ok]]
    return out


def best_threshold(p, y, b, c, cost):
    """Pick threshold on THIS split: net return after round-trip cost."""
    best = None
    for t in THR_GRID:
        sig = np.where(p >= t, 1, np.where(p <= 1 - t, -1, 0))
        m = sig != 0
        if m.sum() < MIN_BETS:
            continue
        eligible = m & (b + 1 < len(c))
        sim = simulate_non_overlapping(c, b[eligible], sig[eligible], 1, cost)
        net = sim["net_compounded"]
        if sim["trade_count"] < MIN_BETS:
            continue
        if best is None or net > best[1]:
            best = (float(t), net, sim["trade_count"])
    return best


def main():
    OUT = s4.OUT
    # Invalidate prior final-test artifacts before starting a fresh calibration run.
    for stale in ("test_pred.npy", "test_y.npy", "test_pred_cal.npy", "validation.json", "backtest.json"):
        stale_path = os.path.join(OUT, stale)
        if os.path.isfile(stale_path):
            os.remove(stale_path)
    # Split 2022 chronologically: Jan-Jun fits Platt calibration; Jul-Dec
    # selects threshold and holding horizon. No observation serves both roles.
    val_ds = s4.FinetuneDS(s4.TARGET_SYM, years=VAL_YEARS,
                           stride=s4.TEST_STRIDE)
    if len(val_ds) < 200:
        raise SystemExit("2022 dataset too small for calibration and policy selection")
    lab = np.load(f"{s4.MKT}/lab_{s4.TARGET_SYM}.npz")
    c = lab["c"].astype(np.float64)
    bars_all = np.asarray(val_ds.bars, dtype=np.int64)
    sample_ts = np.asarray(lab["ts"], dtype=np.int64)[bars_all]
    cal_mask, policy_mask = [np.asarray(mask, dtype=bool)
                             for mask in split_2022_periods(sample_ts)]
    if cal_mask.sum() < 100 or policy_mask.sum() < 100:
        raise SystemExit("insufficient samples in one 2022 half; reduce ST4_TEST_STRIDE")
    log(f"2022 calibration samples={int(cal_mask.sum())}, policy samples={int(policy_mask.sum())}")
    p_all, y_all = predict(val_ds)
    if len(p_all) != len(bars_all):
        raise SystemExit("2022 prediction/window mismatch")
    np.save(f"{OUT}/val_pred.npy", p_all)
    np.save(f"{OUT}/val_y.npy", y_all)
    p_cal, y_cal = p_all[cal_mask], y_all[cal_mask]
    p_policy, y_policy = p_all[policy_mask], y_all[policy_mask]
    b_policy = bars_all[policy_mask]
    if len(np.unique(y_cal)) != 2 or len(np.unique(y_policy)) != 2:
        raise SystemExit("both target classes are required in each 2022 half")

    # Probability calibration is fit only on Jan-Jun 2022.
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    lcal = np.log(np.clip(p_cal, eps, 1 - eps) /
                  (1 - np.clip(p_cal, eps, 1 - eps)))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(
        lcal.reshape(-1, 1), y_cal)

    def cal(p):
        logits = np.log(np.clip(p, eps, 1 - eps) /
                        (1 - np.clip(p, eps, 1 - eps)))
        return platt.predict_proba(logits.reshape(-1, 1))[:, 1]

    pcc, ppc = cal(p_cal), cal(p_policy)
    log(f"Platt calibration Jan-Jun: a={platt.coef_[0][0]:.4f}, b={platt.intercept_[0]:.4f}")
    log(f"calibration Brier/ECE raw={brier_ece(p_cal, y_cal)} calibrated={brier_ece(pcc, y_cal)}")
    log(f"policy Brier/ECE raw={brier_ece(p_policy, y_policy)} calibrated={brier_ece(ppc, y_policy)}")

    # Threshold and holding horizon are selected only on Jul-Dec 2022.
    COST = COSTS_SIDE[1]
    pick = best_threshold(ppc, y_policy, b_policy, c, COST)
    if pick is None:
        raise SystemExit("no valid threshold on 2022 policy period; 2023 final test was NOT accessed")
    thr = pick[0]
    sigp = np.where(ppc >= thr, 1, np.where(ppc <= 1 - thr, -1, 0))
    policy_horizons = []
    for k in HORIZONS:
        eligible = (sigp != 0) & (b_policy + k < len(c))
        if eligible.sum() < MIN_BETS:
            continue
        sims = {co: simulate_non_overlapping(c, b_policy[eligible], sigp[eligible], k, co)
                for co in COSTS_SIDE}
        ref = sims[COSTS_SIDE[1]]
        if ref["trade_count"] < MIN_BETS:
            continue
        policy_horizons.append(dict(
            horizon=k, trades=ref["trade_count"],
            gross=ref["gross_compounded"],
            net={str(co): sims[co]["net_compounded"] for co in COSTS_SIDE}))
    if not policy_horizons:
        raise SystemExit("no valid holding horizon on 2022 policy period; 2023 final test was NOT accessed")
    chosen = max(policy_horizons, key=lambda h: h["net"][str(COSTS_SIDE[1])])
    k_best = int(chosen["horizon"])
    log(f"locked 2022 policy: threshold={thr:.3f}, horizon={k_best}, "
        f"trades={chosen['trades']}, net={chosen['net'][str(COSTS_SIDE[1])]:+.3%}")

    # ------------------------------------------------------------ locked final test
    # Do not instantiate or infer on 2023 until calibration, threshold and
    # holding horizon have all been selected using 2022 only.
    test_ds = s4.FinetuneDS(s4.TARGET_SYM, years=[2023],
                            stride=s4.TEST_STRIDE)
    bt = np.asarray(test_ds.bars)
    pred, y = predict(test_ds)
    if len(bt) != len(pred):
        raise SystemExit(f"window mismatch: rebuilt {len(bt)} vs predicted {len(pred)}")
    pc = cal(pred)
    np.save(f"{OUT}/test_pred.npy", pred)
    np.save(f"{OUT}/test_y.npy", y)

    res = dict(
        calibration_period="2022-01-01/2022-06-30",
        policy_selection_period="2022-07-01/2022-12-31",
        calibration=dict(n=int(len(p_cal)), brier_ece_raw=list(brier_ece(p_cal, y_cal)),
                         brier_ece_calibrated=list(brier_ece(pcc, y_cal))),
        policy_selection=dict(n=int(len(p_policy)), brier_ece=list(brier_ece(ppc, y_policy))),
        platt=dict(a=float(platt.coef_[0][0]), b=float(platt.intercept_[0])),
        threshold=float(thr), chosen_horizon=k_best,
        policy_horizons=policy_horizons,
        test=dict(acc=float(((pred >= .5) == y).mean()),
                  brier_ece=list(brier_ece(pred, y)),
                  auc=auc_rank(y, pred),
                  brier_ece_calibrated=list(brier_ece(pc, y))),
        horizons=[],
    )

    sig = np.where(pc >= thr, 1, np.where(pc <= 1 - thr, -1, 0))
    m = sig != 0
    log(f"test bets with locked threshold {thr}: {int(m.sum())} "
        f"of {len(sig)} "
        f"(acc {float(((pred[m] >= .5) == y[m]).mean()):.4f})")

    # Evaluate only the policy already selected on 2022. Never sweep
    # thresholds or holding horizons on the final test.
    test_policy = None
    if k_best is not None:
        eligible = m & (bt + k_best < len(c))
        sim = simulate_non_overlapping(c, bt[eligible], sig[eligible],
                                       k_best, COSTS_SIDE[1])
        test_policy = dict(horizon=k_best, cost_per_side=COSTS_SIDE[1],
                           trades=sim["trade_count"],
                           gross_compounded=sim["gross_compounded"],
                           net_compounded=sim["net_compounded"],
                           max_drawdown=sim["max_drawdown"])
        log(f"locked 2023 policy: horizon={k_best}, trades={sim['trade_count']}, "
            f"net={sim['net_compounded']:+.3%}, maxDD={sim['max_drawdown']:.3%}")
    res["test_policy"] = test_policy

    with open(f"{OUT}/validation.json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.save(f"{OUT}/val_pred.npy", p_all)
    np.save(f"{OUT}/val_pred_cal.npy", cal(p_all))
    np.save(f"{OUT}/test_pred_cal.npy", pc)
    log(f"wrote {OUT}/validation.json")
    print("VALIDATE DONE", flush=True)


if __name__ == "__main__":
    main()
