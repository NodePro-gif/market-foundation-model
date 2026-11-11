#!/usr/bin/env python
"""Honest validation pass for the stage-4 direction head.

Fixes three problems at once:

1. Calibration      - fit Platt scaling on 2021-2022 (not on the test set),
                      report test ECE before/after.
2. Threshold choice - pick the betting threshold on 2021-2022, lock it,
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

import stage4_finetune as s4

VAL_YEARS = (2021, 2022)
HORIZONS = (1, 3, 6, 12, 24, 48)          # bars held (must be <= stride grid)
COSTS_SIDE = (0.0005, 0.0015, 0.0030)     # 0.5 / 1.5 / 3.0 bps per side
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
    """Run the fine-tuned heads over a dataset -> (prob, label)."""
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "finetune", "best.pt")
    base = s4.load_base(os.environ.get("CKPT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ckpt", "best.pt"))
    model = s4.FinetuneGPT(base)
    model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yd, *_ in DataLoader(ds, batch_size=64, shuffle=False):
            pd, _, _ = model(xb)
            ps.append(torch.sigmoid(pd).numpy())
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
        r = sig[m] * next_returns(b[m], c, 1)
        net = float(np.nansum(r) - m.sum() * 2 * cost)
        if best is None or net > best[1]:
            best = (float(t), net, int(m.sum()))
    return best


def main():
    OUT = s4.OUT
    # ---------------------------------------------------------- validation
    log(f"validation years {VAL_YEARS}")
    val_ds = s4.FinetuneDS(s4.TARGET_SYM, years=VAL_YEARS,
                           stride=s4.TEST_STRIDE)
    log("running model on validation windows ...")
    pv, yv = predict(val_ds)
    bv = np.asarray(val_ds.bars)
    log(f"validation n={len(pv):,} acc={float((pv >= .5) == yv).mean():.4f} "
        f"brier/ece={brier_ece(pv, yv)} auc={auc_rank(yv, pv):.4f}")

    # ---------------------------------------------------------- calibration
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    lv = np.log(np.clip(pv, eps, 1 - eps) / (1 - np.clip(pv, eps, 1 - eps)))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(lv.reshape(-1, 1), yv)
    log(f"Platt: a={platt.coef_[0][0]:.4f} b={platt.intercept_[0]:.4f}")

    def cal(p):
        e = np.log(np.clip(p, eps, 1 - eps) /
                   (1 - np.clip(p, eps, 1 - eps)))
        return platt.predict_proba(e.reshape(-1, 1))[:, 1]

    pvc = cal(pv)
    log(f"validation calibrated: brier/ece={brier_ece(pvc, yv)}")

    # --------------------------------------------- threshold on validation
    lab = np.load(f"{s4.MKT}/lab_{s4.TARGET_SYM}.npz")
    c = lab["c"].astype(np.float64)
    COST = 0.00015
    pick = best_threshold(pvc, yv, bv, c, COST)
    thr = pick[0] if pick else 0.50
    log(f"threshold chosen on validation @1.5bps/side: {thr} "
        f"(val net {pick[1]:+.4%}, n={pick[2]})" if pick
        else "no valid threshold on validation -> using 0.50")

    # ------------------------------------------------------------ test
    pred = np.load(f"{OUT}/test_pred.npy")
    y = np.load(f"{OUT}/test_y.npy")
    test_ds = s4.FinetuneDS(s4.TARGET_SYM, years=[2023],
                            stride=s4.TEST_STRIDE)
    bt = np.asarray(test_ds.bars)
    if len(bt) != len(pred):
        raise SystemExit(f"window mismatch: rebuilt {len(bt)} vs "
                         f"test_pred {len(pred)} - rerun with the same "
                         f"ST4_STRIDE/ST4_TEST_STRIDE env")
    pc = cal(pred)

    res = dict(
        val_years=list(VAL_YEARS), val_n=int(len(pv)),
        val=dict(acc=float((pv >= .5) == yv).mean(),
                 brier_ece=list(brier_ece(pv, yv)), auc=auc_rank(yv, pv)),
        val_calibrated=dict(brier_ece=list(brier_ece(pvc, yv))),
        platt=dict(a=float(platt.coef_[0][0]), b=float(platt.intercept_[0])),
        threshold=float(thr),
        test=dict(acc=float((pred >= .5) == y).mean(),
                  brier_ece=list(brier_ece(pred, y)),
                  auc=auc_rank(y, pred),
                  brier_ece_calibrated=list(brier_ece(pc, y))),
        horizons=[],
    )

    sig = np.where(pc >= thr, 1, np.where(pc <= 1 - thr, -1, 0))
    m = sig != 0
    log(f"test bets with locked threshold {thr}: {int(m.sum())} "
        f"of {len(sig)} (acc {float((pred[m] >= .5) == y[m]).mean():.4f})")

    print(f"  {'k bars':>7} {'n':>5} {'gross':>9} "
          + " ".join(f"net@{co*1e4:4.1f}bps".rjust(11) for co in COSTS_SIDE))
    for k in HORIZONS:
        r = sig[m] * next_returns(bt[m], c, k)
        n = int(np.isfinite(r).sum())
        gross = float(np.nansum(r))
        row = dict(horizon=k, n=n, gross_total=gross, nets={})
        cells = []
        for co in COSTS_SIDE:
            net = gross - n * 2 * co
            row["nets"][f"{co}"] = net
            cells.append(f"{net*100:+10.2f}%")
        res["horizons"].append(row)
        print(f"  {k:7d} {n:5d} {gross*100:+8.2f}% " + " ".join(cells))

    with open(f"{OUT}/validation.json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.save(f"{OUT}/val_pred.npy", pv)
    np.save(f"{OUT}/val_pred_cal.npy", pvc)
    np.save(f"{OUT}/test_pred_cal.npy", pc)
    log(f"wrote {OUT}/validation.json")
    print("VALIDATE DONE", flush=True)


if __name__ == "__main__":
    main()
