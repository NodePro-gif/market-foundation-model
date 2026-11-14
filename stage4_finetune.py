#!/usr/bin/env python
"""Stage 4 - fine-tune the pretrained Market GPT on prediction heads and
evaluate honestly against the stage-1 baselines.

Heads (multi-task on the last hidden state):
  1. direction   - next-bar up/down            (binary, BCE)
  2. vol regime  - next-12-bar vol quartile     (4-class, CE)
  3. quantiles   - fwd12/fwd48/fwd288 return    (P10/P50/P90 pinball)

Evaluation:
  - checkpoint selection on XAUUSD M5 2021, final test on 2023
  - held-out symbols (NIKKEI, BRENT, EURGBP) evaluated after selection
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import stage3_pretrain as s3

IS_COLAB = os.path.isdir("/content")
MKT = s3.MKT
# local runs write into the repo so results travel with the code
OUT = os.environ.get(
    "MFM_FT_DIR",
    "/content/ftune" if IS_COLAB
    else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "results", "finetune"),
)
os.makedirs(OUT, exist_ok=True)


def _env(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


# knobs so the same script runs on a GPU box (defaults = original config)
# and on a small CPU box (ST4_EPOCHS=1 ST4_BATCH=32 ST4_STRIDE=64 ...)
EPOCHS = _env("ST4_EPOCHS", 3, int)
LR = _env("ST4_LR", 1e-4, float)
BATCH = _env("ST4_BATCH", 64, int)
ACCUM_STEPS = max(1, _env("ST4_ACCUM", 4, int))
STRIDE = _env("ST4_STRIDE", 32, int)  # balanced multi-asset training windows
VAL_STRIDE = _env("ST4_VAL_STRIDE", 4, int)
TEST_STRIDE = _env("ST4_TEST_STRIDE", 4, int)
HELD_STRIDE = _env("ST4_HELD_STRIDE", 8, int)
NUM_WORKERS = _env("ST4_WORKERS", 2, int)
PREFETCH_FACTOR = max(2, _env("ST4_PREFETCH", 4, int))
USE_AMP = _env("ST4_AMP", 1, int) == 1
WARMUP_RATIO = _env("ST4_WARMUP_RATIO", 0.08, float)
MIN_LR_RATIO = _env("ST4_MIN_LR_RATIO", 0.10, float)
EARLY_STOP_PATIENCE = _env("ST4_PATIENCE", 3, int)
SEED = 42
TARGET_SYM = "XAUUSD"
HELD = ["NIKKEI", "BRENT", "EURGBP"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALER = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda" and USE_AMP)
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
torch.set_num_threads(max(1, _env("MFM_CPU_THREADS", 4, int)))
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------- dataset
class FinetuneDS(Dataset):
    """Windows ending at each labeled M5 bar (stride small for dense labels)."""

    def __init__(self, sym, years, stride=8):
        self.sym = sym
        lab = np.load(f"{MKT}/lab_{sym}.npz")
        tok = s3._tok_cache((sym, "M5"))
        with np.load(f"{MKT}/tok_{sym}_M5.npz") as z:
            span = z["spans"].astype(np.int64)
        bar_of_tok = np.concatenate(([0], np.cumsum(span)))[:-1]
        ts = lab["ts"]
        n = min(len(ts), int(bar_of_tok[-1] + span[-1]))
        years_arr = (ts[:n].astype(np.int64).astype("datetime64[s]")
                     .astype("datetime64[Y]").astype(np.int32) + 1970)
        # A BPE token is observable only after every bar it covers has closed.
        # Align each prediction to token END bars; never use a token that contains
        # bars occurring after the prediction timestamp.
        token_end_bars = bar_of_tok + span - 1
        valid_tokens = token_end_bars < n
        token_ids = np.flatnonzero(valid_tokens)
        token_bars = token_end_bars[valid_tokens]
        requested_years = list(years)
        token_years = years_arr[token_bars]
        want = np.isin(token_years, requested_years)

        # Purge samples whose longest target (288 bars) crosses the split.
        future_idx = np.minimum(token_bars + 288, n - 1)
        want &= np.isin(years_arr[future_idx], requested_years)
        # Need 507 content tokens after the five context tokens.
        want &= token_ids >= s3.CTX - 6
        want &= token_ids < len(tok) - 1
        # Labels must be valid at the token's final completed bar.
        want &= (lab["nxt"][token_bars] < 2) & (lab["volb"][token_bars] < 4) & \
                (lab["regime"][token_bars] < 6)

        offs = token_ids[want][::stride]
        self.ends = offs
        self.starts = offs - (s3.CTX - 6)   # exactly CTX-5 content tokens
        self.bars = token_end_bars[offs]
        self.lab = lab
        self.n = n
        log(f"FinetuneDS {sym} years={sorted(set(years_arr[self.bars].tolist()))} "
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


def _tokens(sym):
    # Reuse Stage-3's decompressed .npy memory map; worker processes share
    # the operating-system page cache instead of duplicating full token arrays.
    return s3._tok_cache((sym, "M5"))


# ------------------------------------------------------------------ model
class FinetuneGPT(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.backbone = base
        d = s3.N_EMBD
        hidden = 128
        # Independent normalized heads reduce scale coupling between the
        # heterogeneous direction, regime and return-quantile objectives.
        def prediction_head(out_dim):
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(s3.DROPOUT),
                nn.Linear(hidden, out_dim),
            )
        self.dir_head = prediction_head(1)
        self.vol_head = prediction_head(4)
        self.q_head = prediction_head(9)
        # 9 = 3 horizons x (P10, P50, P90)

    def forward(self, ids):
        x = self.backbone.drop(
            self.backbone.tok(ids) + self.backbone.pos(
                torch.arange(ids.shape[1], device=ids.device))[None]
        )
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
    # A raw state_dict cannot prove which BPE vocabulary its embedding rows mean.
    # Require the manifest written beside a best checkpoint by stage3.
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                 "best_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(
            f"missing {manifest_path}; retrain stage3 with the current tokenizer "
            "before fine-tuning (legacy checkpoints are not accepted)"
        )
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    fingerprint = s3.tokenizer_fingerprint()
    if manifest.get("tokenizer_fingerprint") != fingerprint:
        raise RuntimeError(
            "pretraining checkpoint/tokenizer fingerprint mismatch; "
            "run stage3_pretrain.py again"
        )
    if manifest.get("checkpoint_sha256") != s3.sha256_file(ckpt_path):
        raise RuntimeError("pretraining checkpoint bytes do not match best_manifest.json")
    base = s3.MarketGPT()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(sd)
    log(f"verified tokenizer fingerprint {fingerprint[:12]}...")
    return base


def run_epoch(model, dl, opt=None, tag="", scheduler=None):
    training = opt is not None
    model.train(training)
    tot = dict(dir=0.0, vol=0.0, q=0.0, n=0)
    ps, ys, regs, ts = [], [], [], []
    crit_bce = nn.BCEWithLogitsLoss()
    n_steps = len(dl)
    every = max(1, n_steps // 10)
    ep_t0 = time.time()
    if training:
        opt.zero_grad(set_to_none=True)
    for i, (xb, yd, yv, yq, vq) in enumerate(dl):
        xb = xb.to(DEVICE, non_blocking=True)
        yd = yd.to(DEVICE, non_blocking=True).float()
        yv = yv.to(DEVICE, non_blocking=True)
        yq = yq.to(DEVICE, non_blocking=True)
        vq = vq.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                enabled=DEVICE.type == "cuda" and USE_AMP):
                pd, pv, pq = model(xb)
                l_dir = crit_bce(pd, yd)
                l_vol = nn.functional.cross_entropy(pv, yv)
                l_q = pinball(pq, yq, vq)
                loss = l_dir + l_vol + 0.5 * l_q
        if training:
            group_start = (i // ACCUM_STEPS) * ACCUM_STEPS
            group_size = min(ACCUM_STEPS, n_steps - group_start)
            SCALER.scale(loss / group_size).backward()
            update_now = ((i + 1) % ACCUM_STEPS == 0) or (i + 1 == n_steps)
            if update_now:
                SCALER.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                SCALER.step(opt)
                SCALER.update()
                if scheduler is not None:
                    scheduler.step()
                opt.zero_grad(set_to_none=True)
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
                       pq.detach().float().cpu().numpy(),
                       yq.detach().cpu().numpy(),
                       vq.detach().cpu().numpy()))
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
    ckpt = os.environ.get("CKPT") or os.path.join(s3.CKPT, "best.pt")
    if not os.path.isfile(ckpt):
        raise SystemExit(
            f"pretraining checkpoint not found: {ckpt}; run stage3_pretrain.py first"
        )
    log(f"loading pretrained from {ckpt}")
    base = load_base(ckpt)
    model = FinetuneGPT(base).to(DEVICE)
    compile_enabled = DEVICE.type == "cuda" and os.environ.get("MFM_TORCH_COMPILE", "0") == "1"
    if compile_enabled and hasattr(model, "compile"):
        compile_mode = os.environ.get("MFM_COMPILE_MODE", "reduce-overhead")
        model.compile(mode=compile_mode)
        log(f"torch.compile enabled (mode={compile_mode}); first iterations include compilation warm-up")
    elif compile_enabled:
        log("torch.compile requested but unavailable in this PyTorch build; eager mode retained")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"finetune model params: {n_params/1e6:.2f}M on {DEVICE}; "
        f"batch={BATCH} x accum={ACCUM_STEPS}; AMP={USE_AMP}; workers={NUM_WORKERS}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        log(f"GPU={props.name} VRAM={props.total_memory / 1024**3:.1f}GiB")

    # Train the shared heads across all non-held-out assets, with equal sampling
    # probability per asset. Select the checkpoint on XAUUSD 2021 only; 2022 is
    # reserved for calibration and 2023 is not constructed until final testing.
    train_parts = []
    train_windows_by_symbol = {}
    train_symbols = [sym for sym in s3.SYMBOLS if sym not in s3.HELD_OUT]
    for sym in train_symbols:
        try:
            part = FinetuneDS(sym, years=range(2015, 2021), stride=STRIDE)
        except FileNotFoundError:
            log(f"skip training symbol {sym}: missing M5 token/label artifacts")
            continue
        if len(part):
            train_parts.append(part)
            train_windows_by_symbol[sym] = len(part)
    if not train_parts:
        raise SystemExit("no non-held-out multi-asset training data found")
    train_ds = ConcatDataset(train_parts)
    val_ds = FinetuneDS(TARGET_SYM, years=[2021], stride=VAL_STRIDE)
    if not len(val_ds):
        raise SystemExit("empty XAUUSD 2021 validation dataset")

    # Each asset contributes approximately the same number of sampled windows,
    # preventing the largest datasets from dominating the shared prediction heads.
    sample_weights = torch.cat([
        torch.full((len(part),), 1.0 / len(part), dtype=torch.double)
        for part in train_parts
    ])
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    loader_args = dict(num_workers=NUM_WORKERS,
                       pin_memory=DEVICE.type == "cuda")
    if NUM_WORKERS > 0:
        loader_args["persistent_workers"] = True
        loader_args["prefetch_factor"] = PREFETCH_FACTOR
    tr_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                       drop_last=False, **loader_args)
    va_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, **loader_args)
    log(f"balanced multi-asset fine-tuning: {len(train_parts)} symbols, "
        f"{len(train_ds):,} windows; train stride={STRIDE}")

    try:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=LR, betas=(0.9, 0.95), weight_decay=0.01,
                                fused=DEVICE.type == "cuda")
    except TypeError:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    updates_per_epoch = math.ceil(len(tr_dl) / ACCUM_STEPS)
    total_updates = max(1, EPOCHS * updates_per_epoch)
    warmup_updates = min(total_updates - 1, max(1, int(total_updates * WARMUP_RATIO)))
    def lr_lambda(update):
        if update < warmup_updates:
            return max(1e-3, (update + 1) / warmup_updates)
        progress = min(1.0, (update - warmup_updates) /
                      max(1, total_updates - warmup_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_val_auc = -float("inf")
    stale_epochs = 0
    # Remove cached predictions/calibration from earlier checkpoints so that
    # validation can never silently reuse stale model outputs.
    for stale in ("val_pred.npy", "val_y.npy", "val_pred_cal.npy",
                  "test_pred_cal.npy", "validation.json", "backtest.json"):
        stale_path = os.path.join(OUT, stale)
        if os.path.isfile(stale_path):
            os.remove(stale_path)
    best_path = os.path.join(OUT, "best.pt")
    if os.path.exists(best_path):
        os.remove(best_path)  # never accidentally reuse a checkpoint from an older run
    for ep in range(1, EPOCHS + 1):
        r = run_epoch(model, tr_dl, opt, tag=f"ep{ep} TRAIN",
                      scheduler=scheduler)
        va = run_epoch(model, va_dl, None, tag=f"ep{ep} VALID")
        log(f"ep{ep} TRAIN loss_dir={r['loss_dir']:.4f} "
            f"loss_vol={r['loss_vol']:.4f} loss_q={r['loss_q']:.4f} "
            f"acc={r['acc']:.4f} auc={r['auc']:.4f}")
        log(f"ep{ep} VALID acc={va['acc']:.4f} auc={va['auc']:.4f} "
            f"brier={va['brier']:.4f} ece={va['ece']:.4f}")
        current_lr = opt.param_groups[0]["lr"]
        log(f"  optimizer_updates={updates_per_epoch * ep} lr_next={current_lr:.3e}")
        if np.isfinite(va["auc"]) and va["auc"] > best_val_auc + 1e-5:
            best_val_auc = va["auc"]
            stale_epochs = 0
            s3.atomic_torch_save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
            log(f"  no validation AUC improvement ({stale_epochs}/{EARLY_STOP_PATIENCE})")
        if stale_epochs >= EARLY_STOP_PATIENCE:
            log("early stopping: validation AUC did not improve")
            break

    if not os.path.exists(best_path):
        raise SystemExit("no valid checkpoint was selected; validation AUC is undefined")
    model.load_state_dict(torch.load(best_path, map_location="cpu",
                                     weights_only=True))
    # Only now materialize the untouched final-test dataset.
    test_ds = FinetuneDS(TARGET_SYM, years=[2023], stride=TEST_STRIDE)
    if not len(test_ds):
        raise SystemExit("empty final-test dataset for 2023")
    te_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, **loader_args)
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

    # Quantile coverage: fraction of realized forward returns below each
    # predicted P10/P50/P90, calculated only after final-test inference.
    quantile_coverage = None
    if "raw" in te:
        pq = np.concatenate([t[2] for t in te["raw"]], axis=0)
        yq = np.concatenate([t[3] for t in te["raw"]], axis=0)
        vq = np.concatenate([t[4] for t in te["raw"]], axis=0)
        covered = (yq[:, :, None] <= pq).astype(np.float64)
        denom = vq.sum(axis=0)[:, None]
        numer = (covered * vq[:, :, None]).sum(axis=0)
        quantile_coverage = np.divide(
            numer, denom, out=np.full_like(numer, np.nan), where=denom > 0
        ).tolist()
        log("quantile coverage by horizon [P10, P50, P90]: "
            f"{np.round(np.asarray(quantile_coverage), 4).tolist()}")

    json.dump(dict(test=dict(acc=te["acc"], auc=te["auc"],
                             brier=te["brier"], ece=te["ece"],
                             n=int(len(te["y"]))),
                   held_out=held_res, quantile_coverage=quantile_coverage,
                   protocol=dict(train_years=[2015, 2020], checkpoint_selection_year=2021,
                                 calibration_year=2022, final_test_year=2023),
                   config=dict(epochs=EPOCHS, batch=BATCH, accumulation=ACCUM_STEPS,
                               learning_rate=LR, warmup_ratio=WARMUP_RATIO,
                               min_lr_ratio=MIN_LR_RATIO,
                               early_stop_patience=EARLY_STOP_PATIENCE,
                               optimizer="AdamW", scheduler="linear-warmup+cosine-decay",
                               total_optimizer_updates=total_updates,
                               train_stride=STRIDE,
                               validation_stride=VAL_STRIDE, test_stride=TEST_STRIDE,
                               train_symbols=list(train_windows_by_symbol),
                               train_windows_by_symbol=train_windows_by_symbol,
                               sampling="equal_probability_per_asset",
                               amp=USE_AMP, device=str(DEVICE))),
              open(f"{OUT}/metrics.json", "w"), indent=1)
    np.save(f"{OUT}/test_pred.npy", te["pred"])
    np.save(f"{OUT}/test_y.npy", te["y"])
    np.save(f"{OUT}/test_reg.npy", te["reg"])
    log("STAGE4 DONE")
    print("STAGE4 DONE", flush=True)


if __name__ == "__main__":
    main()
