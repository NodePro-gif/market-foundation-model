#!/usr/bin/env python
"""Walk-forward evaluation of the stage-4 direction head.

The single 2023 result rests on only 87 bets. This runs three honest folds:

    fold 1: choose Platt + threshold + horizon on 2019-2020 -> test 2021
    fold 2: choose everything on 2020-2021                 -> test 2022
    fold 3: choose everything on 2021-2022                 -> test 2023

Every decision is made on years strictly before the test year, so the
aggregate is an out-of-sample estimate with ~3x the sample size.

Probabilities per year are cached in results/finetune/prob_<year>.npy
(only the first run pays the forward-pass cost).

Usage (same env as the stage4 run):
    $env:ST4_STRIDE="256"; $env:ST4_TEST_STRIDE="64"
    python walkforward_stage4.py
"""

import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import stage4_finetune as s4

YEARS = (2019, 2020, 2021, 2022, 2023)
FOLDS = (((2019, 2020), 2021), ((2020, 2021), 2022), ((2021, 2022), 2023))
HORIZONS = (1, 3, 6, 12, 24, 48)
COSTS_SIDE = (0.00005, 0.00015, 0.00030)     # 0.5 / 1.5 / 3.0 bps per side
MAIN_COST = COSTS_SIDE[1]
THR_GRID = np.round(np.arange(0.50, 0.851, 0.025), 3)
MIN_BETS = 40
EPS = 1e-6

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


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1),
                                                      y)
    return lambda q: lg.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def year_probs(year):
    """(prob, label, bars) for one year - forward pass only on first call."""
    OUT = s4.OUT
    cp, cy, cb = (f"{OUT}/prob_{year}.npy", f"{OUT}/prob_y_{year}.npy",
                  f"{OUT}/prob_b_{year}.npy")
    if all(os.path.exists(p) for p in (cp, cy, cb)):
        return np.load(cp), np.load(cy), np.load(cb)

    ds = s4.FinetuneDS(s4.TARGET_SYM, years=[year], stride=s4.TEST_STRIDE)
    base = s4.load_base(os.environ.get("CKPT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ckpt", "best.pt"))
    model = s4.FinetuneGPT(base)
    model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yd, *_ in DataLoader(ds, batch_size=64, shuffle=False):
            pd, _, _ = model(xb)
            ps.append(torch.sigmoid(pd).numpy())
            ys.append(yd.numpy() if torch.is_tensor(yd) else np.asarray(yd))
    p, y = np.concatenate(ps), np.concatenate(ys)
    b = np.asarray(ds.bars)
    np.save(cp, p)
    np.save(cy, y)
    np.save(cb, b)
    log(f"  year {year}: {len(p):,} windows -> cached")
    return p, y, b


def next_returns(b, c, k):
    b = np.asarray(b, np.int64)
    out = np.full(len(b), np.nan)
    ok = b + k < len(c)
    out[ok] = (c[b[ok] + k] - c[b[ok]]) / c[b[ok]]
    return out


def choose(p, y, b, c):
    """Lock Platt, threshold and horizon on these years (max net @MAIN_COST)."""
    cal = fit_platt(p, y)
    pc = cal(p)
    best_t, best_net = 0.50, -np.inf
    for t in THR_GRID:
        sig = np.where(pc >= t, 1, np.where(pc <= 1 - t, -1, 0))
        m = sig != 0
        if m.sum() < MIN_BETS:
            continue
        r = sig[m] * next_returns(b[m], c, 1)
        net = float(np.nansum(r) - m.sum() * 2 * MAIN_COST)
        if net > best_net:
            best_t, best_net = float(t), net
    sig = np.where(pc >= best_t, 1, np.where(pc <= 1 - best_t, -1, 0))
    m = sig != 0
    best_k, best_knet = 1, -np.inf
    for k in HORIZONS:
        r = sig[m] * next_returns(b[m], c, k)
        n = int(np.isfinite(r).sum())
        if n < 10:
            continue
        net = float(np.nansum(r) - n * 2 * MAIN_COST)
        if net > best_knet:
            best_k, best_knet = k, net
    return cal, best_t, best_k


def main():
    OUT = s4.OUT
    lab = np.load(f"{s4.MKT}/lab_{s4.TARGET_SYM}.npz")
    c = lab["c"].astype(np.float64)

    log("loading per-year probabilities (forward only for unseen years)...")
    data = {yr: year_probs(yr) for yr in YEARS}
    for yr in YEARS:
        p, y, b = data[yr]
        log(f"  {yr}: n={len(p):,} acc={float(((p >= .5) == y).mean()):.4f} "
            f"auc={s4.auc_rank(y, p):.4f}")

    folds, agg = [], dict(n=0, gross=0.0, wins=0, correct=0, total=0)
    for sel_years, test_year in FOLDS:
        ps = np.concatenate([data[yr][0] for yr in sel_years])
        ys = np.concatenate([data[yr][1] for yr in sel_years])
        bs = np.concatenate([data[yr][2] for yr in sel_years])
        cal, thr, k = choose(ps, ys, bs, c)

        pt, yt, bt = data[test_year]
        pc = cal(pt)
        sig = np.where(pc >= thr, 1, np.where(pc <= 1 - thr, -1, 0))
        m = sig != 0
        r = sig[m] * next_returns(bt[m], c, k)
        n = int(np.isfinite(r).sum())
        gross = float(np.nansum(r))
        acc = float(((pc[m] >= .5) == yt[m]).mean())
        nets = {f"{co}": gross - n * 2 * co for co in COSTS_SIDE}
        row = dict(select=list(sel_years), test=test_year, threshold=thr,
                   horizon=k, n=n, acc=acc, gross=gross, nets=nets,
                   brier_ece=list(brier_ece(pc, yt)))
        folds.append(row)
        agg["n"] += n
        agg["gross"] += gross
        agg["correct"] += acc * n
        agg["total"] += n
        log(f"fold {sel_years} -> {test_year}: thr={thr} k={k} n={n} "
            f"acc={acc:.4f} gross={gross*100:+.2f}% "
            f"net@1.5bps={nets[f'{MAIN_COST}']*100:+.2f}%")

    # aggregate: sign test on per-bet returns (H0: mean <= 0 is not tested
    # here; we test hit-rate vs 0.5 which is the honest, cheap check)
    acc_agg = agg["correct"] / max(agg["total"], 1)
    n = agg["total"]
    z = (acc_agg - 0.5) / math.sqrt(0.25 / max(n, 1))
    nets_agg = {f"{co}": agg["gross"] - agg["n"] * 2 * co
                for co in COSTS_SIDE}
    print("\n=== AGGREGATE (3 walk-forward folds) ===")
    print(f"  bets={agg['n']}  acc={acc_agg:.4f}  z_vs_0.5={z:+.2f}  "
          f"gross={agg['gross']*100:+.2f}%")
    for co in COSTS_SIDE:
        print(f"  net @ {co*1e4:4.1f} bps/side: "
              f"{nets_agg[f'{co}']*100:+.2f}%")

    res = dict(folds=folds,
               aggregate=dict(n=agg["n"], acc=acc_agg, z=z,
                              gross=agg["gross"], nets=nets_agg))
    with open(f"{OUT}/walkforward.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"wrote {OUT}/walkforward.json")
    print("WALKFORWARD DONE", flush=True)


if __name__ == "__main__":
    main()
