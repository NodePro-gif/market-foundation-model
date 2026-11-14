#!/usr/bin/env python
"""Stage 3 - pretrain the Market Foundation Model (causal LM, next-token).

Reads tokenized streams from /content/mkt, builds packed sequences with
context tokens [SYM][TF][REG][SESS] prefix, trains a GPT-style decoder
with AMP fp16, periodic checkpointing to /content/ckpt.

Held-out rule: 2023 tokens and HELD_OUT symbols are excluded from pretraining.
"""

import hashlib
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
BATCH = int(os.environ.get("PRETRAIN_BATCH", "32"))
ACCUM_STEPS = max(1, int(os.environ.get("PRETRAIN_ACCUM", "2")))
LR = float(os.environ.get("PRETRAIN_LR", "3e-4"))
WARMUP = int(os.environ.get("PRETRAIN_WARMUP", "300"))
MAX_STEPS = int(os.environ.get("PRETRAIN_STEPS", "6000"))
CKPT_EVERY = int(os.environ.get("PRETRAIN_CKPT_EVERY", "500"))
EVAL_EVERY = int(os.environ.get("PRETRAIN_EVAL_EVERY", "250"))
NUM_WORKERS = int(os.environ.get("PRETRAIN_WORKERS", "2"))
RESUME = os.environ.get("PRETRAIN_RESUME", "0") == "1"
SEED = int(os.environ.get("PRETRAIN_SEED", "42"))
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


def tokenizer_fingerprint():
    """Bind a checkpoint to the exact tokenizer artifacts used to train it."""
    digest = hashlib.sha256()
    for name in ("vocab.json", "bpe.npz"):
        path = os.path.join(MKT, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing tokenizer artifact: {path}")
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(payload, path):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
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
                    n = len(ts)
                    years = years_from_ts(ts)
                    regime = lab["regime"][:n]
                    sess = lab["sess"][:n]
                    if int(bar_of_tok[-1] + span[-1]) < n:
                        raise RuntimeError(f"{sym}/{tf}: token spans do not cover all bars")
                    last_tok_idx = np.minimum(
                        np.arange(len(tok)) + CTX - 6, len(tok) - 1)
                    last_bar = np.minimum(
                        bar_of_tok[last_tok_idx] + span[last_tok_idx] - 1,
                        n - 1)
                    y0 = years[bar_of_tok]
                    y1 = years[last_bar]
                    ok = np.isin(y0, list(years_set)) & \
                         np.isin(y1, list(years_set))
                    offs = np.flatnonzero(ok)[:: self.stride]
                    for off in offs:
                        context_end_bar = int(last_bar[off])
                        self.items.append((sym, tf, int(off),
                                           int(regime[context_end_bar]),
                                           int(sess[context_end_bar])))
                else:
                    # M1 (and any series without labels): filter by token-year
                    # map if available, else include everything
                    ymap_p = f"{MKT}/year_{sym}_{tf}.npy"
                    if not os.path.exists(ymap_p):
                        log(f"skip {sym}/{tf}: missing token-year map {ymap_p}")
                        continue
                    ymap = np.load(ymap_p)
                    yend_p = f"{MKT}/year_end_{sym}_{tf}.npy"
                    if not os.path.exists(yend_p):
                        log(f"skip {sym}/{tf}: missing token end-year map {yend_p}")
                        continue
                    yend = np.load(yend_p)
                    m = min(len(ymap), len(yend), len(tok))
                    if m <= CTX - 5:
                        continue
                    end_idx = np.minimum(np.arange(m) + CTX - 6, m - 1)
                    ok = (np.isin(ymap[:m], list(years_set)) &
                          np.isin(yend[end_idx], list(years_set)))
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

    # Fit on 2015-2020 and select the language-model checkpoint on 2021.
    # 2022 is reserved for calibration; 2023 remains the final untouched test.
    train_ds = MarketLMSmart(split_years=list(range(2015, 2021)))
    val_ds = MarketLMSmart(split_years=[2021])
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            f"empty pretraining dataset: train={len(train_ds)} val={len(val_ds)}; "
            "check MKT files and token-year maps"
        )

    loader_kwargs = dict(num_workers=NUM_WORKERS,
                         pin_memory=DEVICE.type == "cuda")
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
    generator = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          drop_last=False, generator=generator, **loader_kwargs)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        drop_last=False, **loader_kwargs)
    if len(train_dl) == 0 or len(val_dl) == 0:
        raise SystemExit("train/validation loader is empty; reduce batch size or check data")

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
    manifest_path = os.path.join(CKPT, "best_manifest.json")
    fingerprint = tokenizer_fingerprint()
    if RESUME and os.path.isfile(os.path.join(CKPT, "resume.pt")):
        state = torch.load(os.path.join(CKPT, "resume.pt"),
                           map_location=DEVICE, weights_only=False)
        if state.get("tokenizer_fingerprint") != fingerprint:
            raise RuntimeError("resume checkpoint tokenizer fingerprint mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        best_val = float(state["best_val"])
        hist = state.get("history", [])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if DEVICE.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        log(f"resumed from step {step}, best_val={best_val:.5f}")

    def evaluate():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                if i >= 50:
                    break
                xb = batch.to(DEVICE, non_blocking=True)
                _, loss = model(xb, xb)
                valid_tokens = int((xb[:, 1:] != PAD).sum().item())
                tot += loss.item() * valid_tokens
                n += valid_tokens
        model.train()
        if n == 0:
            raise RuntimeError("validation produced zero non-padding targets")
        return tot / n

    model.train()
    opt.zero_grad(set_to_none=True)
    while step < MAX_STEPS:
        for i, batch in enumerate(train_dl):
            if step >= MAX_STEPS:
                break
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            xb = batch.to(DEVICE, non_blocking=True)
            group_start = (i // ACCUM_STEPS) * ACCUM_STEPS
            group_size = min(ACCUM_STEPS, len(train_dl) - group_start)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                _, loss = model(xb, xb)
            scaler.scale(loss / group_size).backward()
            update_now = ((i + 1) % ACCUM_STEPS == 0) or (i + 1 == len(train_dl))
            if update_now:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

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
                    atomic_torch_save(model.state_dict(), f"{CKPT}/best.pt")
                    atomic_json(manifest_path, dict(
                        tokenizer_fingerprint=fingerprint,
                        vocab_size=VOCAB, context_length=CTX,
                        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                        train_years=[2015, 2020], checkpoint_selection_year=2021,
                        global_step=step, best_validation_loss=vl,
                    ))
                    log(f"  new best val {vl:.4f} -> best.pt (tokenizer bound)")
            step += 1
            if step % CKPT_EVERY == 0:
                atomic_torch_save(dict(
                    model=model.state_dict(), optimizer=opt.state_dict(),
                    scaler=scaler.state_dict(), step=step, best_val=best_val,
                    history=hist, tokenizer_fingerprint=fingerprint,
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None,
                ), os.path.join(CKPT, "resume.pt"))
                atomic_json(LOG, hist)
                log(f"saved resumable checkpoint at step {step}")

    atomic_torch_save(model.state_dict(), f"{CKPT}/final.pt")
    atomic_json(LOG, hist)
    log(f"DONE best_val={best_val:.4f}  params={n_params/1e6:.2f}M  batch={BATCH} accum={ACCUM_STEPS}")
    print("STAGE3 DONE", flush=True)


if __name__ == "__main__":
    main()
