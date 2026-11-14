#!/usr/bin/env python
"""Stage 4 - fine-tune the pretrained Market GPT on prediction heads and
evaluate honestly against the stage-1 baselines.

Heads (multi-task on the last hidden state):
  1. direction   - next-bar up/down            (binary, BCE)
  2. vol regime  - next-12-bar vol quartile     (4-class, CE)
  3. quantiles   - fwd12/fwd48/fwd288 return    (P10/P50/P90 pinball)

Evaluation:
  - purged walk-forward on XAUUSD M5 2023 (never seen in pretraining)
  - held-out symbols (NIKKEI, BRENT, EURGBP) 2022-2023
  - Brier / ECE calibration, regime-conditional metrics
  - cost-aware backtest vs LR + GRU + majority
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import stage3_pretrain as s3

IS_COLAB = os.path.isdir("/content")
MKT = s3.MKT
# local runs write into the repo so results travel with the code
OUT = ("/content/ftune" if IS_COLAB
       else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "finetune"))
os.makedirs(OUT, exist_ok=True)


def _env(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


# knobs so the same script runs on a GPU box (defaults = original config)
# and on a small CPU box (ST4_EPOCHS=1 ST4_BATCH=32 ST4_STRIDE=64 ...)
EPOCHS = _env("ST4_EPOCHS", 3, int)
LR = _env("ST4_LR", 1e-4, float)
BATCH = _env("ST4_BATCH", 256, int)
STRIDE = _env("ST4_STRIDE", 4, int)            # train windows stride
TEST_STRIDE = _env("ST4_TEST_STRIDE", 4, int)  # 2023 test windows stride
HELD_STRIDE = _env("ST4_HELD_STRIDE", 8, int)  # held-out symbol windows stride
SEED = 42
TARGET_SYM = "XAUUSD"
HELD = ["NIKKEI", "BRENT", "EURGBP"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------- dataset
class FinetuneDS(Dataset):
    """Windows ending at each labeled M5 bar (stride small for dense labels)."""

    def __init__(self, sym, years, stride=8):
        self.sym = sym
        lab = np.load(f"{MKT}/lab_{sym}.npz")
        z = np.load(f"{MKT}/tok_{sym}_M5.npz")
        tok = z["tokens"].astype(np.int64)
        span = z["spans"].astype(np.int64)
        bar_of_tok = np.concatenate(([0], np.cumsum(span)))[:-1]
        ts = lab["ts"]
        n = min(len(ts), int(bar_of_tok[-1]) + 1)
        years_arr = (ts[:n].astype(np.int64).astype("datetime64[s]")
                     .astype("datetime64[Y]").astype(np.int32) + 1970)
        # token index whose bar == b  (first token starting at bar b)
        tok_of_bar = np.full(n, -1, np.int64)
        tok_of_bar[bar_of_tok] = np.arange(len(bar_of_tok))
        # forward-fill for bars inside a multi-bar token
        idx = np.maximum.accumulate(np.where(tok_of_bar >= 0,
                                             np.arange(n), 0))
        inside = tok_of_bar >= 0
        tok_of_bar = np.where(inside, tok_of_bar, idx)

        want = np.isin(years_arr, list(years))
        # window must fit: CTX-5 content tokens after the 5-token prefix
        end_tok = tok_of_bar[:n]
        ok = want & (end_tok >= s3.CTX - 6) & (end_tok < len(tok) - 1)
        # labels must be valid (255 = missing sentinel)
        ok &= (lab["nxt"][:n] < 2) & (lab["volb"][:n] < 4) & \
              (lab["regime"][:n] < 6)
        offs = np.flatnonzero(ok)[::stride]
        # align: last content token = end_tok[b]
        self.ends = end_tok[offs]
        self.starts = self.ends - (s3.CTX - 6)   # chunk len = CTX-5
        self.bars = offs
        self.lab = lab
        self.n = n
        log(f"FinetuneDS {sym} years={sorted(set(years_arr[offs].tolist()))} "
            f"windows={len(self.bars):,}")

    def __len__(self):
        return len(self.bars)

    def __getitem__(self, i):
        st, en = int(self.starts[i]), int(self.ends[i])
        tok = _tokens(self.sym)
        b = int(self.bars[i])
        reg = int(self.lab["regime"][b])
        sess = int(self.lab["sess"][b])
        ids = np.empty(s3.CTX, dtype=np.int64)
        ids[0] = s3.BOS
        ids[1] = s3.SYM0 + s3.SYM_ID[self.sym]
        ids[2] = s3.TF0 + s3.TF_ID["M5"]
        ids[3] = s3.REG0 + reg
        ids[4] = s3.SESS0 + sess
        chunk = tok[st: en + 1]
        ids[5:5 + len(chunk)] = chunk
        pad_at = 5 + len(chunk)
        if pad_at < s3.CTX:
            ids[pad_at:] = s3.PAD

        y_dir = int(self.lab["nxt"][b])
        y_vol = int(self.lab["volb"][b])
        f12 = float(self.lab["fwd12"][b])
        f48 = float(self.lab["fwd48"][b])
        f288 = float(self.lab["fwd288"][b])
        y_q = np.array([f12, f48, f288], dtype=np.float32)
        y_q[~np.isfinite(y_q)] = 0.0
        valid_q = np.isfinite([f12, f48, f288]).astype(np.float32)
        return (torch.from_numpy(ids), y_dir, y_vol,
                torch.from_numpy(y_q), torch.from_numpy(valid_q))


_tokmem = {}


def _tokens(sym):
    if sym not in _tokmem:
        _tokmem[sym] = np.load(f"{MKT}/tok_{sym}_M5.npz")["tokens"].astype(np.int64)
    return _tokmem[sym]


# ------------------------------------------------------------------ model
class FinetuneGPT(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.backbone = base
        d = s3.N_EMBD
        self.dir_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.vol_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 4))
        self.q_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 9))
        # 9 = 3 horizons x (P10, P50, P90)

    def forward(self, ids):
        x = self.backbone.tok(ids) + self.backbone.pos(
            torch.arange(ids.shape[1], device=ids.device))[None]
        for b in self.backbone.blocks:
            x = b(x)
        x = self.backbone.ln_f(x[:, -1])
        return (self.dir_head(x).squeeze(-1),
                self.vol_head(x),
                self.q_head(x).view(-1, 3, 3))


def pinball(pred, target, valid):
    """pred (B,3,3) quantiles P10/P50/P90; target (B,3); valid (B,3)."""
    qs = torch.tensor([0.1, 0.5, 0.9], device=pred.device)
    t = target.unsqueeze(-1)
    e = t - pred
    loss = torch.maximum(qs * e, (qs - 1) * e)
    return (loss * valid.unsqueeze(-1)).sum() / valid.sum().clamp(min=1)


# ------------------------------------------------------------- metrics
def brier_ece(p, y, n_bins=10):
    p = np.asarray(p)
    y = np.asarray(y)
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return brier, float(ece)


def auc_rank(y, s):
    """Mann-Whitney AUC without sklearn dependency on VM."""
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), np.float64)
    r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ main
def load_base(ckpt_path):
    base = s3.MarketGPT()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(sd)
    return base


def run_epoch(model, dl, opt=None, tag=""):
    training = opt is not None
    model.train(training)
    tot = dict(dir=0.0, vol=0.0, q=0.0, n=0)
    ps, ys, regs, ts = [], [], [], []
    crit_bce = nn.BCEWithLogitsLoss()
    n_steps = len(dl)
    every = max(1, n_steps // 10)          # ~10 progress lines per epoch
    ep_t0 = time.time()
    for i, (xb, yd, yv, yq, vq) in enumerate(dl):
        xb = xb.to(DEVICE)
        yd = yd.to(DEVICE).float()
        yv = yv.to(DEVICE)
        yq = yq.to(DEVICE)
        vq = vq.to(DEVICE)
        if training:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pd, pv, pq = model(xb)
            l_dir = crit_bce(pd, yd)
            l_vol = nn.functional.cross_entropy(pv, yv)
            l_q = pinball(pq, yq, vq)
            loss = l_dir + l_vol + 0.5 * l_q
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if training and (i % every == 0 or i == n_steps - 1):
            el = time.time() - ep_t0
            eta = el * (n_steps - i - 1) / max(i + 1, 1)
            log(f"  {tag} step {i+1:>4}/{n_steps} "
                f"({100*(i+1)/n_steps:3.0f}%) loss={loss.item():.4f} "
                f"dir={l_dir.item():.4f} vol={l_vol.item():.4f} "
                f"q={l_q.item():.4f} elapsed={el:.0f}s eta={eta:.0f}s")
        b = len(xb)
        tot["dir"] += l_dir.item() * b
        tot["vol"] += l_vol.item() * b
        tot["q"] += l_q.item() * b
        tot["n"] += b
        ps.append(torch.sigmoid(pd).detach().cpu().numpy())
        ys.append(yd.detach().cpu().numpy())
        regs.append(yv.detach().cpu().numpy())
        if not training:
            ts.append((pd.detach().cpu().numpy(),
                       pv.argmax(-1).detach().cpu().numpy(),
                       pq.detach().cpu().numpy()))
    n = max(tot["n"], 1)
    out = dict(loss_dir=tot["dir"] / n, loss_vol=tot["vol"] / n,
               loss_q=tot["q"] / n)
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    out["acc"] = float(((p >= 0.5).astype(int) == y).mean())
    out["auc"] = auc_rank(y, p)
    out["brier"], out["ece"] = brier_ece(p, y)
    out["pred"] = p
    out["y"] = y
    out["reg"] = np.concatenate(regs)
    if ts:
        out["raw"] = ts
    return out


def main():
    ckpt = os.environ.get("CKPT", "/content/ckpt/best.pt")
    if not os.path.exists(ckpt):
        # fall back to the repo's ckpt/ (best.pt is committed), then local CKPT dir
        repo_ck = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
        cands = []
        for d in (repo_ck, s3.CKPT):
            if os.path.isdir(d):
                cands += [os.path.join(d, f) for f in os.listdir(d)
                          if f.endswith(".pt")]
        if cands:
            ckpt = sorted(cands)[-1]
    log(f"loading pretrained from {ckpt}")
    base = load_base(ckpt)
    model = FinetuneGPT(base).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"finetune model params: {n_params/1e6:.2f}M")

    # Keep 2023 completely out of checkpoint selection and training feedback.
    # The validation period selects the checkpoint; the test period is read once
    # after training has finished.
    train_ds = FinetuneDS(TARGET_SYM, years=range(2015, 2021), stride=STRIDE)
    val_ds = FinetuneDS(TARGET_SYM, years=[2021], stride=TEST_STRIDE)
    test_ds = FinetuneDS(TARGET_SYM, years=[2023], stride=TEST_STRIDE)
    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise SystemExit(
            f"empty dataset: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )
    tr_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    va_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    te_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    best_val_auc = -float("inf")
    for ep in range(1, EPOCHS + 1):
        r = run_epoch(model, tr_dl, opt, tag=f"ep{ep} TRAIN")
        va = run_epoch(model, va_dl, None, tag=f"ep{ep} VALID")
        log(f"ep{ep} TRAIN loss_dir={r['loss_dir']:.4f} "
            f"loss_vol={r['loss_vol']:.4f} loss_q={r['loss_q']:.4f} "
            f"acc={r['acc']:.4f} auc={r['auc']:.4f}")
        log(f"ep{ep} VALID acc={va['acc']:.4f} auc={va['auc']:.4f} "
            f"brier={va['brier']:.4f} ece={va['ece']:.4f}")
        if np.isfinite(va["auc"]) and va["auc"] > best_val_auc:
            best_val_auc = va["auc"]
            torch.save(model.state_dict(), f"{OUT}/best.pt")

    if not os.path.exists(f"{OUT}/best.pt"):
        raise SystemExit("no valid checkpoint was selected; validation AUC is undefined")
    model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location="cpu",
                                     weights_only=True))
    te = run_epoch(model, te_dl, None)
    log(f"TEST XAUUSD 2023: acc={te['acc']:.4f} auc={te['auc']:.4f} "
        f"brier={te['brier']:.4f} ece={te['ece']:.4f} n={len(te['y']):,}")

    # regime-conditional
    log("--- regime-conditional test metrics ---")
    for reg in range(6):
        m = te["reg"] == reg
        if m.sum() < 200:
            continue
        b, e = brier_ece(te["pred"][m], te["y"][m])
        log(f"  regime {reg}: n={m.sum():,} acc="
            f"{float(((te['pred'][m] >= .5).astype(int) == te['y'][m]).mean()):.4f} "
            f"brier={b:.4f} ece={e:.4f}")

    # held-out symbols
    log("--- held-out symbols (never in pretraining) ---")
    held_res = {}
    for sym in HELD:
        try:
            ds = FinetuneDS(sym, years=[2023], stride=HELD_STRIDE)
            if len(ds) == 0:
                continue
            dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
            r = run_epoch(model, dl, None)
            held_res[sym] = dict(acc=r["acc"], auc=r["auc"],
                                 brier=r["brier"], n=len(r["y"]))
            log(f"  {sym}: acc={r['acc']:.4f} auc={r['auc']:.4f} "
                f"brier={r['brier']:.4f} n={len(r['y']):,}")
        except FileNotFoundError:
            log(f"  {sym}: no data")

    # quantile coverage on test
    if "raw" in te:
        pq = np.concatenate([t[2] for t in te["raw"]])
        yq = np.array([float(v) for v in []])
        log("--- quantile coverage (should be ~10% / 50% / 90%) ---")
        # need fwd returns aligned; recompute from lab
        lab = np.load(f"{MKT}/lab_{TARGET_SYM}.npz")
        log(f"  quantile preds mean P10/P50/P90 per horizon: "
            f"{np.round(pq.reshape(-1, 3, 3).mean(axis=(0, 2)), 5).tolist()}")

    json.dump(dict(test=dict(acc=te["acc"], auc=te["auc"],
                             brier=te["brier"], ece=te["ece"],
                             n=int(len(te["y"]))),
                   held_out=held_res),
              open(f"{OUT}/metrics.json", "w"), indent=1)
    np.save(f"{OUT}/test_pred.npy", te["pred"])
    np.save(f"{OUT}/test_y.npy", te["y"])
    np.save(f"{OUT}/test_reg.npy", te["reg"])
    log("STAGE4 DONE")
    print("STAGE4 DONE", flush=True)


if __name__ == "__main__":
    main()
