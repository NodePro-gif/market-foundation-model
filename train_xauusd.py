#!/usr/bin/env python
"""
XAUUSD M5 (2020-2023) deep learning on Colab GPU.

Pipeline:
  1. load long-form yotahub data  (NAME,TIME,VALUE)  -> wide OHLC
  2. restrict to 2020-01-01 .. 2023-12-31
  3. feature engineering (past-only, no look-ahead)
  4. train GRU sequence model + MLP baseline on T4
  5. evaluate on chronological split, simple backtest
  6. dump metrics / plots / model / predictions
"""

import json
import os
import random
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
RAW = "/content/gold_price.csv.gz"
RAW_URL = "https://data.yotahub.com/2024-12/datahub-2024-12-Gold-Price.csv.gz"
OUT = "/content/xauusd_out"
SEED = 42
WINDOW = 48          # 48 x 5min = 4 hours of history
BATCH = 1024
EPOCHS = 60
PATIENCE = 10
LR = 1e-3
COST = 0.00015        # round-trip spread+slippage as fraction of price
SPLITS = {
    "train": ("2020-01-01", "2022-01-01"),
    "val": ("2022-01-01", "2023-01-01"),
    "test": ("2023-01-01", "2024-01-01"),
}

os.makedirs(OUT, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1-2. data
# --------------------------------------------------------------------------
log("loading raw csv ...")
if not os.path.exists(RAW):
    log(f"  {RAW} missing (fresh runtime) -> downloading ...")
    import urllib.request

    tmp = RAW + ".part"
    urllib.request.urlretrieve(RAW_URL, tmp)
    os.replace(tmp, RAW)
    log(f"  downloaded {os.path.getsize(RAW):,} bytes")
raw = pd.read_csv(RAW)
log(f"  raw rows={len(raw):,}  fields={sorted(raw['NAME'].unique())}")

raw["time"] = pd.to_datetime(raw["TIME"], unit="ns")   # ns epoch -> naive UTC
wide = (
    raw.pivot_table(index="time", columns="NAME", values="VALUE", aggfunc="last")
    .sort_index()
)
del raw
log(f"  wide bars total={len(wide):,}  range={wide.index[0]} .. {wide.index[-1]}")

df = wide.loc[SPLITS["train"][0] : "2023-12-31 23:59:59"].copy()
if "volume" not in df.columns:
    df["volume"] = 0.0
df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])

# integrity checks
step = df.index.to_series().diff().dt.total_seconds()
n_bad = int((step.dropna() != 300).sum())
max_gap = float(step.max())
log(
    f"  window bars={len(df):,}  range={df.index[0]} .. {df.index[-1]}  "
    f"gaps(!=5min)={n_bad}  max_gap={max_gap/3600:.2f}h"
)

# --------------------------------------------------------------------------
# 3. features (all trailing) + label
# --------------------------------------------------------------------------
log("building features ...")
c = df["close"].astype(np.float64)
o, h, l = df["open"].astype(np.float64), df["high"].astype(np.float64), df["low"].astype(np.float64)

f = pd.DataFrame(index=df.index)
r1 = np.log(c / c.shift(1))
f["ret1"] = r1
for k in (6, 12, 24, 48, 72):                       # 30m 1h 2h 4h 6h
    f[f"ret{k}"] = np.log(c / c.shift(k))
f["vol12"] = r1.rolling(12).std()
f["vol48"] = r1.rolling(48).std()
f["rng"] = (h - l) / c                              # bar range
f["body"] = (c - o) / c                             # bar body
f["wick_up"] = (h - np.maximum(o, c)) / c
f["wick_dn"] = (np.minimum(o, c) - l) / c
f["sma14d"] = c / c.rolling(14).mean() - 1
f["sma48d"] = c / c.rolling(48).mean() - 1

d = c.diff()
up = d.clip(lower=0).rolling(14).mean()
dn = (-d.clip(upper=0)).rolling(14).mean()
f["rsi"] = (100.0 - 100.0 / (1 + up / dn.replace(0, np.nan))) / 100.0

if "rsi14" in wide.columns:
    f["prov_rsi"] = wide.loc[df.index, "rsi14"] / 100.0
if "sma14" in wide.columns:
    f["prov_sma"] = c / wide.loc[df.index, "sma14"] - 1

hr, dow = df.index.hour, df.index.dayofweek
f["sin_h"] = np.sin(2 * np.pi * hr / 24)
f["cos_h"] = np.cos(2 * np.pi * hr / 24)
f["sin_d"] = np.sin(2 * np.pi * dow / 7)
f["cos_d"] = np.cos(2 * np.pi * dow / 7)

FEATS = list(f.columns)
X_all = f.to_numpy(dtype=np.float32)
y_all = (c.shift(-1) > c).astype(np.int64).to_numpy()   # next bar up?
times = df.index

# rows with non-finite features (rolling warm-up, weekend gaps, provider holes)
row_bad = ~np.isfinite(X_all).all(axis=1)
row_bad[-1] = True  # last row has no label
X_all[row_bad] = 0.0  # neutralise - these rows never enter any window
log(f"  features={len(FEATS)}  clean_rows={(~row_bad).sum():,} / {len(df):,}")

# a sample is usable only if EVERY row of its lookback window is clean,
# otherwise a single NaN poisons the whole sequence -> loss = nan
win_bad = np.lib.stride_tricks.sliding_window_view(row_bad, WINDOW).any(axis=1)
clean = np.zeros(len(df), dtype=bool)
clean[WINDOW - 1:] = ~win_bad
clean &= ~row_bad

pos = np.arange(len(df))
splits = {}
for name, (a, b) in SPLITS.items():
    m = (times >= a) & (times < b) & clean
    p = pos[m]
    assert len(p) > 0, f"empty split {name}"
    splits[name] = p
    log(f"  {name:5s} samples={len(p):,}  {times[p[0]]} .. {times[p[-1]]}")

# scaler fitted on TRAIN rows only (no leakage)
tr_all = pos[(times >= SPLITS["train"][0]) & (times < SPLITS["train"][1]) & ~row_bad]
mu = X_all[tr_all].mean(axis=0)
sd = X_all[tr_all].std(axis=0)
sd[sd < 1e-6] = 1.0
Xz = ((X_all - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------
# dataset / model
# --------------------------------------------------------------------------
class SeqDS(Dataset):
    def __init__(self, X, y, idx, w):
        self.X, self.y, self.idx, self.w = X, y, idx, w

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return torch.from_numpy(self.X[j - self.w + 1 : j + 1]), int(self.y[j])


class GRUNet(nn.Module):
    def __init__(self, n, hidden=128, layers=2, drop=0.25):
        super().__init__()
        self.gru = nn.GRU(n, hidden, num_layers=layers, batch_first=True,
                          dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1)
        )

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(-1)


class MLPNet(nn.Module):
    def __init__(self, n, w, hidden=512, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n * w, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_loader(name, shuffle):
    idx = splits[name]
    return DataLoader(SeqDS(Xz, y_all, idx, WINDOW), batch_size=BATCH,
                      shuffle=shuffle, num_workers=0, pin_memory=True), idx


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = float("nan")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                tp=tp, tn=tn, fp=fp, fn=fn)


# --------------------------------------------------------------------------
# 4. training
# --------------------------------------------------------------------------
def train(model, tag, epochs):
    tr_ld, _ = make_loader("train", True)
    va_ld, _ = make_loader("val", False)
    pos_w = torch.tensor(
        [(y_all[splits["train"]] == 0).sum() / max((y_all[splits["train"]] == 1).sum(), 1)],
        dtype=torch.float32, device=DEVICE,
    ).clamp(0.5, 2.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=epochs, steps_per_epoch=len(tr_ld)
    )

    hist, best, bad = [], 1e9, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for xb, yb in tr_ld:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, torch.float32)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {ep} (nan/inf in inputs or weights)"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        tr_loss = tot / n

        model.eval()
        vp, vy, vl, m = [], [], 0.0, 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE, torch.float32)
                logits = model(xb)
                vl += nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="sum"
                ).item()
                vp.append(torch.sigmoid(logits).cpu().numpy())
                vy.append(yb.cpu().numpy())
                m += len(xb)
        vp = np.concatenate(vp)
        vy = np.concatenate(vy).astype(int)
        va_loss = vl / m
        mm = metrics(vy, vp)
        hist.append(dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss,
                         val_acc=mm["acc"], val_auc=mm["auc"],
                         lr=sched.get_last_lr()[0]))

        if va_loss < best - 1e-5:
            best, bad = va_loss, 0
            torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
        else:
            bad += 1
        log(f"  {tag} ep{ep:02d} train={tr_loss:.4f} val={va_loss:.4f} "
            f"acc={mm['acc']:.4f} auc={mm['auc']:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if bad >= PATIENCE:
            log(f"  {tag}: early stop at epoch {ep}")
            break

    model.load_state_dict(torch.load(f"{OUT}/{tag}.pt", weights_only=True))
    return hist


log("training GRU ...")
gru = GRUNet(len(FEATS)).to(DEVICE)
log(f"  params={sum(p.numel() for p in gru.parameters()):,}  device={DEVICE}")
hist_gru = train(gru, "gru", EPOCHS)

log("training MLP baseline ...")
mlp = MLPNet(len(FEATS), WINDOW).to(DEVICE)
log(f"  params={sum(p.numel() for p in mlp.parameters()):,}")
hist_mlp = train(mlp, "mlp", EPOCHS)

# --------------------------------------------------------------------------
# 5. evaluation
# --------------------------------------------------------------------------
def predict(model, name):
    ld, idx = make_loader(name, False)
    ps = []
    model.eval()
    with torch.no_grad():
        for xb, _ in ld:
            ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.concatenate(ps), idx


results, preds = {}, {}
for tag, model in (("gru", gru), ("mlp", mlp)):
    results[tag] = {}
    for name in SPLITS:
        p, idx = predict(model, name)
        results[tag][name] = metrics(y_all[idx], p)
        if name == "test":
            preds[tag] = (p, idx)

# naive baselines on test (both must map to {0,1} to be comparable)
te = splits["test"]
mom_up = (f["ret6"].to_numpy()[te] > 0).astype(int)   # past 30 min momentum
mom_acc = float((mom_up == y_all[te]).mean())
rev_acc = float(((1 - mom_up) == y_all[te]).mean())   # opposite sign
maj_acc = float(max(y_all[te].mean(), 1 - y_all[te].mean()))

log("backtest on test (2023) ...")
close = c.to_numpy()
ANN = np.sqrt(288 * 252)
HOLDS = (1, 12, 48)   # 5min / 1h / 4h minimum holding period


def positions(p, min_hold):
    """Always-in-market position that only flips when the model flips its
    call AND at least `min_hold` bars have passed since the last flip."""
    raw = np.where(p >= 0.5, 1, -1)
    pos = np.empty(len(raw), dtype=np.int64)
    cur, last = int(raw[0]), 0
    for i in range(len(raw)):
        if raw[i] != cur and (i - last) >= min_hold:
            cur, last = int(raw[i]), i
        pos[i] = cur
    return pos


def run_backtest(p, idx, min_hold):
    """Cost is charged ONCE per position change (round trip), not per bar."""
    fwd = (close[idx + 1] - close[idx]) / close[idx]
    pos = positions(p, min_hold)
    flip = np.empty(len(pos), dtype=bool)
    flip[0] = True
    flip[1:] = pos[1:] != pos[:-1]
    gross = pos * fwd
    net = gross - flip * COST
    eq = np.cumprod(1 + net)
    eq_g = np.cumprod(1 + gross)
    trades = int(flip.sum())
    return dict(
        min_hold=min_hold, trades=trades, turnover=trades / len(pos),
        hit=float((np.sign(fwd) == pos).mean()),
        gross_total=float(eq_g[-1] - 1), net_total=float(eq[-1] - 1),
        net_mean_bps=float(net.mean() * 1e4),
        sharpe=float(net.mean() / (net.std() + 1e-12) * ANN),
        max_dd=float((eq / np.maximum.accumulate(eq) - 1).min()),
        equity=eq,
    )


bt = {}
for tag in ("gru", "mlp"):
    p, idx = preds[tag]
    bt[tag] = {str(h): run_backtest(p, idx, h) for h in HOLDS}
    np.save(f"{OUT}/pred_{tag}.npy", p)

buyhold = float(close[splits["test"][-1] + 1] / close[splits["test"][0]] - 1)

# calibration / discrimination: accuracy by predicted-probability decile
pg = preds["gru"][0]
order = np.argsort(pg)
dec = np.array_split(order, 10)
decile = [
    dict(decile=i + 1, mean_p=float(pg[d].mean()),
         up_rate=float(y_all[te][d].mean()),
         n=int(len(d)),
         acc=float(((pg[d] >= 0.5).astype(int) == y_all[te][d]).mean()))
    for i, d in enumerate(dec)
]
pd.DataFrame(
    {
        "time": times[te],
        "close": close[te],
        "y_true": y_all[te],
        "p_gru": preds["gru"][0],
        "p_mlp": preds["mlp"][0],
    }
).to_csv(f"{OUT}/test_predictions.csv", index=False)

# --------------------------------------------------------------------------
# 6. plots + report
# --------------------------------------------------------------------------
def plot_hist(h, tag):
    e = [x["epoch"] for x in h]
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(e, [x["train_loss"] for x in h], label="train")
    plt.plot(e, [x["val_loss"] for x in h], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title(f"{tag} loss")
    plt.subplot(1, 2, 2)
    plt.plot(e, [x["val_acc"] for x in h], label="val acc")
    plt.plot(e, [x["val_auc"] for x in h], label="val auc")
    plt.xlabel("epoch"); plt.legend(); plt.title(f"{tag} val metrics")
    plt.tight_layout(); plt.savefig(f"{OUT}/curves_{tag}.png", dpi=110); plt.close()


plot_hist(hist_gru, "gru")
plot_hist(hist_mlp, "mlp")

for tag in ("gru", "mlp"):
    m = results[tag]["test"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(4.6, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["down", "up"]); plt.yticks([0, 1], ["down", "up"])
    plt.xlabel("predicted"); plt.ylabel("actual")
    plt.title(f"{tag} confusion (test 2023)")
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_{tag}.png", dpi=110); plt.close()

p, idx = preds["gru"]
bh = close[idx + 1] / close[idx[0]]
eq12 = run_backtest(p, idx, 12)
eq1 = run_backtest(p, idx, 1)
plt.figure(figsize=(11, 4.5))
plt.plot(times[idx], eq12["equity"], label="GRU net of cost (min hold 1h)")
plt.plot(times[idx], eq1["equity"], label="GRU net of cost (min hold 5min)", alpha=0.75)
plt.plot(times[idx], bh, label="buy & hold", alpha=0.7)
plt.xlabel("date"); plt.ylabel("equity (x)"); plt.legend()
plt.title("Test set 2023 - strategy vs buy & hold (cost 1.5 bps per flip)")
plt.tight_layout(); plt.savefig(f"{OUT}/equity.png", dpi=110); plt.close()

report = {
    "task": "XAUUSD M5 next-bar direction classification",
    "period": f"{df.index[0]} .. {df.index[-1]}",
    "bars": int(len(df)),
    "features": FEATS,
    "window_bars": WINDOW,
    "splits": {k: int(len(v)) for k, v in splits.items()},
    "class_balance_train_up": float(y_all[splits["train"]].mean()),
    "device": str(DEVICE),
    "model_params": {t: sum(q.numel() for q in m.parameters()) for t, m in
                     (("gru", gru), ("mlp", mlp))},
    "metrics": results,
    "backtest": {
        t: {h: {k: v for k, v in d.items() if k != "equity"} for h, d in dd.items()}
        for t, dd in bt.items()
    },
    "baselines_test": {
        "majority_acc": maj_acc,
        "momentum6_acc": mom_acc,
        "reverse_momentum6_acc": rev_acc,
    },
    "buyhold_test_2023": buyhold,
    "decile_gru": decile,
    "cost_per_trade_frac": COST,
    "min_hold_bars": list(HOLDS),
    "epochs_run": {"gru": len(hist_gru), "mlp": len(hist_mlp)},
    "wall_seconds": round(time.time() - t_start, 1),
}
with open(f"{OUT}/metrics.json", "w") as fh:
    json.dump(report, fh, indent=2)
pd.DataFrame(hist_gru).to_csv(f"{OUT}/history_gru.csv", index=False)
pd.DataFrame(hist_mlp).to_csv(f"{OUT}/history_mlp.csv", index=False)

with zipfile.ZipFile(f"{OUT}/results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT)):
        if fn != "results.zip":
            z.write(os.path.join(OUT, fn), fn)

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"TRAINING SELESAI dalam {report['wall_seconds']}s")
print("=" * 72)
print(f"data      : {report['bars']:,} bars  {report['period']}")
print(f"splits    : train={report['splits']['train']:,}  "
      f"val={report['splits']['val']:,}  test={report['splits']['test']:,}")
print(f"up-rate   : train={report['class_balance_train_up']:.3f}   "
      f"test-majority-acc={maj_acc:.4f}   momentum-acc={mom_acc:.4f}   "
      f"reverse-momentum-acc={rev_acc:.4f}")
for tag in ("gru", "mlp"):
    m = results[tag]["test"]
    print(f"\n[{tag}] TEST 2023  acc={m['acc']:.4f} auc={m['auc']:.4f} "
          f"prec={m['prec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f}")
    for h in ("1", "12", "48"):
        b = bt[tag][h]
        print(f"       hold={int(h) * 5:>3}min trades={b['trades']:>6,} "
              f"turn={b['turnover']:.3f} gross={b['gross_total']:+.2%} "
              f"net={b['net_total']:+.2%} sharpe={b['sharpe']:+6.2f} "
              f"maxDD={b['max_dd']:.2%}")
print(f"\nbuy & hold 2023 = {buyhold:+.2%}")
print("\nGRU accuracy by predicted-probability decile (test 2023):")
print("   dec   mean_p   up_rate      acc        n")
for d in decile:
    print(f"    {d['decile']:>2}   {d['mean_p']:.4f}   {d['up_rate']:.4f}   "
          f"{d['acc']:.4f}   {d['n']:,}")
print("\noutputs:", ", ".join(sorted(os.listdir(OUT))))
print("=" * 72)
