#!/usr/bin/env python
"""Stage 3 - pretrain the Market Foundation Model (causal LM, next-token).

Reads tokenized streams from /content/mkt, builds packed sequences with
context tokens [SYM][TF][REG][SESS] prefix, trains a GPT-style decoder
with AMP fp16, periodic checkpointing to /content/ckpt.

Held-out rule: 2023 tokens and HELD_OUT symbols are excluded from pretraining.
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

IS_COLAB = os.path.isdir("/content")
MKT = "/content/mkt" if IS_COLAB else r"D:\xauusd-dl\mkt"
CKPT = "/content/ckpt" if IS_COLAB else r"D:\xauusd-dl\ckpt"
LOG = os.path.join(CKPT, "pretrain_log.json")

# ------------------------------------------------------------------ config
CTX = 512
BATCH = 64
LR = 3e-4
WARMUP = 300
MAX_STEPS = 6000          # ~45-60 min on T4 with 12M params
CKPT_EVERY = 1000
EVAL_EVERY = 250
SEED = 42
N_LAYER = 6
N_HEAD = 8
N_EMBD = 384              # 6 * 384^2 * 12 ~= 10.6M + heads/embeds ~= 12M
DROPOUT = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


vocab = json.load(open(f"{MKT}/vocab.json"))
VOCAB = vocab["vocab"]
PAD = vocab["PAD"]
BOS = vocab["BOS"]
EOS = vocab["EOS"]
SYM0 = vocab["SYM0"]
TF0 = vocab["TF0"]
REG0 = vocab["REG0"]
SESS0 = vocab["SESS0"]
SYMBOLS = vocab["symbols"]
HELD_OUT = set(vocab["held_out"])
TF_LIST = vocab["tf_list"]


def years_from_ts(ts):
    """Convert Unix-second timestamps to exact UTC calendar years."""
    return (np.asarray(ts, dtype=np.int64).astype("datetime64[s]")
            .astype("datetime64[Y]").astype(np.int32) + 1970)
SYM_ID = {s: i for i, s in enumerate(SYMBOLS)}
TF_ID = {t: i for i, t in enumerate(TF_LIST)}


# ------------------------------------------------------------------ data
_tok_mem = {}


def _tok_cache(key):
    if key not in _tok_mem:
        _tok_mem[key] = np.load(f"{MKT}/tok_{key[0]}_{key[1]}.npz")
    return _tok_mem[key]


LAB_CACHE = {}


def lab_for(sym):
    if sym not in LAB_CACHE and os.path.exists(f"{MKT}/lab_{sym}.npz"):
        LAB_CACHE[sym] = np.load(f"{MKT}/lab_{sym}.npz")
    return LAB_CACHE.get(sym)


class MarketLMSmart(Dataset):
    """Packed windows: [BOS SYM TF REG SESS content...] with real regime/session.

    Purged boundary: a window is kept only if BOTH its first and last bar
    fall inside the requested years, so no token from 2023 leaks into
    pretraining and vice versa.
    """

    def __init__(self, split_years, stride=None):
        self.stride = stride or CTX
        self.items = []
        years_set = set(int(y) for y in split_years)
        for sym in SYMBOLS:
            if sym in HELD_OUT:
                continue
            lab = lab_for(sym)
            for tf in TF_LIST:
                p = f"{MKT}/tok_{sym}_{tf}.npz"
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                tok = z["tokens"]
                span = z["spans"].astype(np.int64)
                bar_of_tok = np.concatenate(([0], np.cumsum(span)))[:-1]
                if lab is not None and tf == "M5":
                    ts = lab["ts"]
                    n = min(len(ts), int(bar_of_tok[-1]) + 1)
                    years = years_from_ts(ts[:n])
                    regime = lab["regime"][:n]
                    sess = lab["sess"][:n]
                    tok = tok[:int(np.searchsorted(bar_of_tok, n, "right"))]
                    bar_of_tok = bar_of_tok[:len(tok)]
                    last_bar = np.minimum(
                        bar_of_tok + CTX, len(years) - 1)  # approx window end
                    y0 = years[bar_of_tok]
                    y1 = years[last_bar]
                    ok = np.isin(y0, list(years_set)) & \
                         np.isin(y1, list(years_set))
                    offs = np.flatnonzero(ok)[:: self.stride]
                    for off in offs:
                        self.items.append((sym, tf, int(off),
                                           int(regime[bar_of_tok[off]]),
                                           int(sess[bar_of_tok[off]])))
                else:
                    # M1 (and any series without labels): filter by token-year
                    # map if available, else include everything
                    ymap_p = f"{MKT}/year_{sym}_{tf}.npy"
                    if not os.path.exists(ymap_p):
                        log(f"skip {sym}/{tf}: missing token-year map {ymap_p}")
                        continue
                    ymap = np.load(ymap_p)
                    m = min(len(ymap), len(tok))
                    if m <= CTX - 5:
                        continue
                    end_idx = np.minimum(np.arange(m) + CTX - 6, m - 1)
                    ok = (np.isin(ymap[:m], list(years_set)) &
                          np.isin(ymap[end_idx], list(years_set)))
                    offs = np.flatnonzero(ok)[:: self.stride]
                    for off in offs:
                        self.items.append((sym, tf, int(off), None, None))
        log(f"MarketLMSmart {min(years_set)}-{max(years_set)}: "
            f"{len(self.items):,} windows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sym, tf, off, reg, sess = self.items[i]
        t = _tok_cache((sym, tf))["tokens"][off: off + CTX - 5]
        ids = np.empty(len(t) + 5, dtype=np.int64)
        ids[0] = BOS
        ids[1] = SYM0 + SYM_ID[sym]
        ids[2] = TF0 + TF_ID[tf]
        ids[3] = REG0 + (reg if (reg is not None and reg < 6) else 0)
        ids[4] = SESS0 + (sess if sess is not None else 0)
        ids[5:] = t
        if len(ids) < CTX:
            ids = np.concatenate([ids, np.full(CTX - len(ids), PAD, np.int64)])
        ids = ids[:CTX]
        np.clip(ids, 0, VOCAB - 1, out=ids)   # guard: no embedding OOB on CUDA
        return torch.from_numpy(ids)


# ------------------------------------------------------------------ model
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask", torch.tril(torch.ones(CTX, CTX)).view(1, 1, CTX, CTX))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // N_HEAD)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        return self.drop(att @ v).transpose(1, 2).contiguous().view(B, T, C)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MarketGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(CTX, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.head.weight = self.tok.weight   # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.tok(ids) + self.pos(pos)[None])
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # shift: logits[t] (sees tokens[0..t]) must predict tokens[t+1]
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), targets[:, 1:].reshape(-1),
                ignore_index=PAD)
        return logits, loss


def main():
    os.makedirs(CKPT, exist_ok=True)
    model = MarketGPT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params: {n_params/1e6:.2f}M on {DEVICE}")

    # Fit on 2015-2020, select the language-model checkpoint on 2021-2022.
    # 2023 remains untouched by tokenizer fitting and pretraining.
    train_ds = MarketLMSmart(split_years=list(range(2015, 2021)))
    val_ds = MarketLMSmart(split_years=[2021, 2022])
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            f"empty pretraining dataset: train={len(train_ds)} val={len(val_ds)}; "
            "check MKT files and token-year maps"
        )

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = MAX_STEPS

    def lr_at(step):
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        p = (step - WARMUP) / max(total_steps - WARMUP, 1)
        return 0.1 * LR + 0.45 * LR * (1 + math.cos(math.pi * p))

    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    step = 0
    best_val = float("inf")
    hist = []

    def evaluate():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                if i >= 50:
                    break
                xb = batch.to(DEVICE, non_blocking=True)
                _, loss = model(xb, xb)
                tot += loss.item() * len(xb)
                n += len(xb)
        model.train()
        return tot / max(n, 1)

    model.train()
    while step < MAX_STEPS:
        for batch in train_dl:
            if step >= MAX_STEPS:
                break
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            xb = batch.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                _, loss = model(xb, xb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if step % 10 == 0:
                log(f"step {step:>5}  loss {loss.item():.4f}  "
                    f"ppl {math.exp(min(loss.item(), 20)):8.1f}  lr {lr:.2e}")
            if step % EVAL_EVERY == 0 and step > 0:
                vl = evaluate()
                hist.append(dict(step=step, val_loss=vl, train_loss=loss.item()))
                log(f"  EVAL step {step}: val_loss={vl:.4f} "
                    f"ppl={math.exp(min(vl, 20)):.1f}")
                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), f"{CKPT}/best.pt")
                    log(f"  new best val {vl:.4f} -> best.pt")
            if step % CKPT_EVERY == 0 and step > 0:
                torch.save(model.state_dict(), f"{CKPT}/step{step}.pt")
                json.dump(hist, open(LOG, "w"))
            step += 1

    torch.save(model.state_dict(), f"{CKPT}/final.pt")
    json.dump(hist, open(LOG, "w"), indent=1)
    log(f"DONE best_val={best_val:.4f}  params={n_params/1e6:.2f}M")
    print("STAGE3 DONE", flush=True)


if __name__ == "__main__":
    main()
