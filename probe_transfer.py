#!/usr/bin/env python
"""Cross-asset transfer probe: train the heads on MANY symbols, test on the
held-out symbols that the XAUUSD-only fine-tune could not master.

stage4_finetune.py trains all heads on XAUUSD alone -> NIKKEI/BRENT/EURGBP
came out at chance (AUC 0.48/0.53/0.52). Hypothesis: the heads learned an
XAUUSD-specific mapping, not a symbol-agnostic one.

This probe:
  * FREEZES the pretrained backbone (cheap: forward only) - also tests
    whether the pretrained representations transfer at all,
  * trains direction/volatility/quantile heads on non-held-out symbols 2015-2020,
  * evaluates on 2023 as an exploratory diagnostic only (not an independent
    benchmark if the same final-test period has already been inspected).

Usage:
    $env:TR_STRIDE="1536"; $env:TR_EPOCHS="2"; $env:TR_BATCH="64"
    $env:ST4_TEST_STRIDE="64"
    python probe_transfer.py
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import stage3_pretrain as s3
import stage4_finetune as s4

OUT = s4.OUT
HELD = s4.HELD
TRAIN_SYM = [s for s in s3.SYMBOLS if s not in HELD and s != "XAUUSD"] + \
            ["XAUUSD"]                       # XAUUSD last so it is comparable
YEARS = tuple(range(2015, 2021))

STRIDE = int(os.environ.get("TR_STRIDE", 1536))
EPOCHS = int(os.environ.get("TR_EPOCHS", 2))
BATCH = int(os.environ.get("TR_BATCH", 64))
LR = float(os.environ.get("TR_LR", 2e-3))
EVAL_STRIDE = int(os.environ.get("ST4_TEST_STRIDE", 64))
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


class SymDS:
    """FinetuneDS wrapper that stamps its own symbol into the [SYM] token."""

    def __init__(self, sym, years, stride):
        self.sym = sym
        self.ds = s4.FinetuneDS(sym, years, stride)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        old = s4.TARGET_SYM
        s4.TARGET_SYM = self.sym
        try:
            return self.ds[i]
        finally:
            s4.TARGET_SYM = old


def evaluate(model, sym, year, stride=EVAL_STRIDE):
    ds = SymDS(sym, [year], stride)
    if len(ds) == 0:
        return None
    ps, ys, regs = [], [], []
    with torch.no_grad():
        for xb, yd, yv, *_ in DataLoader(ds, batch_size=BATCH, shuffle=False):
            xb = xb.to(s4.DEVICE)
            with torch.autocast(device_type=s4.DEVICE.type, dtype=torch.float16,
                                enabled=s4.DEVICE.type == "cuda" and s4.USE_AMP):
                pd, _, _ = model(xb)
            ps.append(torch.sigmoid(pd).float().cpu().numpy())
            ys.append(yd.numpy() if torch.is_tensor(yd) else np.asarray(yd))
    p, y = np.concatenate(ps), np.concatenate(ys)
    brier = float(((p - y) ** 2).mean())
    return dict(n=int(len(y)), acc=float(((p >= .5) == y).mean()),
                auc=s4.auc_rank(y, p), brier=brier)


def main():
    log(f"train symbols={len(TRAIN_SYM)} stride={STRIDE} epochs={EPOCHS} "
        f"batch={BATCH} lr={LR} (backbone FROZEN)")

    base = s4.load_base(os.environ.get("CKPT") or os.path.join(s3.CKPT, "best.pt"))
    model = s4.FinetuneGPT(base).to(s4.DEVICE)
    sd = torch.load(f"{OUT}/best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)              # start from the XAUUSD fine-tune
    for p in model.backbone.parameters():
        p.requires_grad = False
    heads = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params: {sum(p.numel() for p in heads)/1e6:.3f}M "
        f"(heads only, backbone frozen)")
    opt = torch.optim.AdamW(heads, lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

    parts = []
    for sym in TRAIN_SYM:
        try:
            d = SymDS(sym, YEARS, STRIDE)
            if len(d):
                parts.append(d)
                log(f"  {sym}: {len(d):,} windows")
        except FileNotFoundError:
            log(f"  {sym}: no data")
    ds = ConcatDataset(parts)
    log(f"total train windows: {len(ds):,}")

    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    model.train()
    model.backbone.eval()                  # frozen -> no dropout noise
    crit_bce = nn.BCEWithLogitsLoss()
    n_steps = len(dl)
    for ep in range(1, EPOCHS + 1):
        tot = dict(d=0.0, v=0.0, q=0.0, n=0)
        ps, ys = [], []
        for i, (xb, yd, yv, yq, vq) in enumerate(dl):
            xb = xb.to(s4.DEVICE)
            yd, yv, yq, vq = yd.to(s4.DEVICE), yv.to(s4.DEVICE), yq.to(s4.DEVICE), vq.to(s4.DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=s4.DEVICE.type, dtype=torch.float16,
                                enabled=s4.DEVICE.type == "cuda" and s4.USE_AMP):
                pd, pv, pq = model(xb)
                l_d = crit_bce(pd, yd.float())
                l_v = nn.functional.cross_entropy(pv, yv)
                l_q = s4.pinball(pq, yq, vq)
                loss = l_d + l_v + 0.5 * l_q
            s4.SCALER.scale(loss).backward()
            s4.SCALER.unscale_(opt)
            nn.utils.clip_grad_norm_(heads, 1.0)
            s4.SCALER.step(opt)
            s4.SCALER.update()
            b = len(xb)
            tot["d"] += l_d.item() * b
            tot["v"] += l_v.item() * b
            tot["q"] += l_q.item() * b
            tot["n"] += b
            ps.append(pd.detach().float().cpu().numpy())
            ys.append(yd.detach().cpu().numpy())
            if i % max(1, n_steps // 8) == 0:
                el = time.time() - t0
                eta = el * (n_steps - i - 1) / max(i + 1, 1)
                log(f"  ep{ep} step {i+1:>4}/{n_steps} "
                    f"({100*(i+1)/n_steps:3.0f}%) loss={loss.item():.4f} "
                    f"eta={eta:.0f}s")
        p = np.concatenate(ps)
        y = np.concatenate(ys)
        n = max(tot["n"], 1)
        log(f"ep{ep} TRAIN loss_dir={tot['d']/n:.4f} "
            f"loss_vol={tot['v']/n:.4f} loss_q={tot['q']/n:.4f} "
            f"acc={float(((p >= 0) == (y >= 1)).mean()):.4f}")

    log("--- EVAL held-out symbols 2023 (never in pretraining) ---")
    res = dict(train=dict(symbols=TRAIN_SYM, stride=STRIDE, epochs=EPOCHS,
                          batch=BATCH, lr=LR, windows=len(ds)),
               held_out={}, xauusd=None)
    for sym in HELD + ["XAUUSD"]:
        r = evaluate(model, sym, 2023)
        if r is None:
            log(f"  {sym}: no data")
            continue
        if sym in HELD:
            res["held_out"][sym] = r
        else:
            res["xauusd"] = r
        log(f"  {sym}: n={r['n']:,} acc={r['acc']:.4f} auc={r['auc']:.4f} "
            f"brier={r['brier']:.4f}")

    ref = json.load(open(f"{OUT}/metrics.json"))["held_out"]
    log("--- perbandingan vs kepala XAUUSD-only (stage4) ---")
    for sym in HELD:
        a, b = res["held_out"].get(sym), ref.get(sym)
        if a and b:
            log(f"  {sym}: AUC {b['auc']:.4f} -> {a['auc']:.4f} "
                f"({a['auc']-b['auc']:+.4f}), acc {b['acc']:.4f} -> {a['acc']:.4f}")

    with open(f"{OUT}/transfer.json", "w") as fh:
        json.dump(res, fh, indent=1)
    s4.s3.atomic_torch_save(model.state_dict(), f"{OUT}/transfer_best.pt")
    log(f"wrote {OUT}/transfer.json")
    print("TRANSFER DONE", flush=True)


if __name__ == "__main__":
    main()
