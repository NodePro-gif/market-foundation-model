#!/usr/bin/env python
"""Diagnosis: kenapa hasilnya segitu? Apakah underfit, kehabisan epoch, atau
memang fiturnya yang tidak bawa sinyal?"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")

RAW = "/content/gold_price.csv.gz"
OUT = "/content/xauusd_out"
SEED = 42
WINDOW = 48
BATCH = 1024
SPLITS = {
    "train": ("2020-01-01", "2022-01-01"),
    "val": ("2022-01-01", "2023-01-01"),
    "test": ("2023-01-01", "2024-01-01"),
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- prep (same)
log("prep ...")
raw = pd.read_csv(RAW)
raw["time"] = pd.to_datetime(raw["TIME"], unit="ns")
wide = raw.pivot_table(index="time", columns="NAME", values="VALUE", aggfunc="last").sort_index()
del raw
df = wide.loc["2020-01-01" : "2023-12-31 23:59:59"].copy()
df = df[["open", "high", "low", "close"]].dropna(subset=["close"])

c = df["close"].astype(np.float64)
o, h, l = df["open"].astype(np.float64), df["high"].astype(np.float64), df["low"].astype(np.float64)
f = pd.DataFrame(index=df.index)
r1 = np.log(c / c.shift(1))
f["ret1"] = r1
for k in (6, 12, 24, 48, 72):
    f[f"ret{k}"] = np.log(c / c.shift(k))
f["vol12"] = r1.rolling(12).std()
f["vol48"] = r1.rolling(48).std()
f["rng"] = (h - l) / c
f["body"] = (c - o) / c
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
y_all = (c.shift(-1) > c).astype(np.int64).to_numpy()
times = df.index

row_bad = ~np.isfinite(X_all).all(axis=1)
row_bad[-1] = True
X_all[row_bad] = 0.0
win_bad = np.lib.stride_tricks.sliding_window_view(row_bad, WINDOW).any(axis=1)
clean = np.zeros(len(df), dtype=bool)
clean[WINDOW - 1 :] = ~win_bad
clean &= ~row_bad

pos = np.arange(len(df))
splits = {}
for name, (a, b) in SPLITS.items():
    m = (times >= a) & (times < b) & clean
    splits[name] = pos[m]

tr_all = pos[(times >= SPLITS["train"][0]) & (times < SPLITS["train"][1]) & ~row_bad]
mu = X_all[tr_all].mean(axis=0)
sd = X_all[tr_all].std(axis=0)
sd[sd < 1e-6] = 1.0
Xz = ((X_all - mu) / sd).astype(np.float32)


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
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(-1)


class MLPNet(nn.Module):
    def __init__(self, n, w, hidden=512, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n * w, hidden), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(hidden, hidden // 2),
                                 nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def acc_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    a = float(((p >= 0.5).astype(int) == y).mean())
    try:
        from sklearn.metrics import roc_auc_score

        u = float(roc_auc_score(y, p))
    except Exception:
        u = float("nan")
    return a, u


def eval_model(model, name):
    ld = DataLoader(SeqDS(Xz, y_all, splits[name], WINDOW), batch_size=2048)
    ps, ys = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in ld:
            ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            ys.append(yb.numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    bce = float(nn.functional.binary_cross_entropy(
        torch.from_numpy(p).clamp(1e-6, 1 - 1e-6), torch.from_numpy(y.astype(np.float32))))
    a, u = acc_auc(y, p)
    return dict(acc=a, auc=u, bce=bce, n=len(y))


# ================================================================ DIAGNOSIS
print("\n" + "=" * 74)
print("DIAGNOSIS 1 - apakah model yang sudah dilatih belajar apa-apa?")
print("=" * 74)
for tag, cls in (("gru", GRUNet), ("mlp", MLPNet)):
    m = cls(len(FEATS), WINDOW) if tag == "mlp" else cls(len(FEATS))
    m.load_state_dict(torch.load(f"{OUT}/{tag}.pt", weights_only=True, map_location=DEVICE))
    m.to(DEVICE)
    for s in ("train", "val", "test"):
        r = eval_model(m, s)
        print(f"  {tag:4s} {s:6s} acc={r['acc']:.4f} auc={r['auc']:.4f} "
              f"bce={r['bce']:.6f} n={r['n']:,}")

# ceiling of a constant predictor
for s in ("train", "val", "test"):
    y = y_all[splits[s]]
    base_acc = max(y.mean(), 1 - y.mean())
    base_bce = float(nn.functional.binary_cross_entropy(
        torch.full((len(y),), 0.5), torch.from_numpy(y.astype(np.float32))))
    print(f"  BASELINE const-0.5 {s:6s} acc={base_acc:.4f} bce={base_bce:.6f}")

print("\n" + "=" * 74)
print("DIAGNOSIS 2 - overfit 2048 sampel (apakah arsitektur & training bekerja?)")
print("=" * 74)
idx2k = np.random.default_rng(SEED).choice(splits["train"], 2048, replace=False)
ld2k = DataLoader(SeqDS(Xz, y_all, idx2k, WINDOW), batch_size=2048, shuffle=True)
probe = GRUNet(len(FEATS)).to(DEVICE)
opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
crit = nn.BCEWithLogitsLoss()
for ep in range(1, 151):
    probe.train()
    tot = n = 0
    for xb, yb in ld2k:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE, torch.float32)
        opt.zero_grad(set_to_none=True)
        loss = crit(probe(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
        n += len(xb)
    if ep % 30 == 0 or ep == 1:
        probe.eval()
        with torch.no_grad():
            ps = torch.sigmoid(probe(xb.to(DEVICE))).cpu().numpy()
        a, _ = acc_auc(yb.cpu().numpy(), ps)
        print(f"  ep{ep:>3}  train_bce={tot/n:.5f}  train_acc={a:.4f}")

print("\n" + "=" * 74)
print("DIAGNOSIS 3 - LogisticRegression pada fitur bar terakhir (tanpa sekuens)")
print("=" * 74)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=2000, C=1.0)
Xtr = Xz[splits["train"]]
ytr = y_all[splits["train"]]
lr.fit(Xtr, ytr)
for s in ("train", "val", "test"):
    p = lr.predict_proba(Xz[splits[s]])[:, 1]
    a, u = acc_auc(y_all[splits[s]], p)
    print(f"  LR {s:6s} acc={a:.4f} auc={u:.4f}")
top = np.argsort(-np.abs(lr.coef_[0]))[:6]
print("  top fitur by |coef|:",
      ", ".join(f"{FEATS[i]}={lr.coef_[0][i]:+.3f}" for i in top))

print("\n" + "=" * 74)
print("DIAGNOSIS 4 - seberapa acak labelnya? (autokorelasi & noise floor)")
print("=" * 74)
for k in (1, 6, 12, 48):
    same = float((np.sign(r1.to_numpy()[k:]) == np.sign(r1.to_numpy()[:-k])).mean())
    print(f"  arah bar t sama dengan arah bar t-{k:>2}: {same:.4f}")
# how much of the label is explained by the label's own past (persistence)
mom = (f["ret6"].to_numpy()[splits["test"]] > 0).astype(int)
print(f"  baseline momentum-30m test acc={float((mom == y_all[splits['test']]).mean()):.4f}")
print(f"  baseline mayoritas  test acc={float(max(y_all[splits['test']].mean(), 1-y_all[splits['test']].mean())):.4f}")

print("\nselesai.")
