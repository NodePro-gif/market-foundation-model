#!/usr/bin/env python
"""Verify a pretraining checkpoint against its tokenizer manifest and 2021 validation data.

The final 2023 test set is intentionally never used by this diagnostic.
Usage: python verify_ckpt.py [n_batches]
"""
import os
import sys
import time

import numpy as np
import torch

import stage3_pretrain as s3

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BATCH = 64                      # same as pretraining eval

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def main():
    ckpt = os.environ.get("CKPT") or os.path.join(s3.CKPT, "best.pt")
    log(f"loading {ckpt}")
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(ckpt)),
                                 "best_manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit("MISSING MANIFEST: retrain stage3 with the current tokenizer")
    import json
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    fingerprint = s3.tokenizer_fingerprint()
    if manifest.get("tokenizer_fingerprint") != fingerprint:
        raise SystemExit("TOKENIZER MISMATCH: retrain stage3 before fine-tuning")
    if manifest.get("checkpoint_sha256") != s3.sha256_file(ckpt):
        raise SystemExit("CHECKPOINT DIGEST MISMATCH: retrain stage3 or restore matching manifest")
    model = s3.MarketGPT().to(s3.DEVICE)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    # 2021 is the checkpoint-selection validation year. Never inspect 2023 here.
    ds = s3.MarketLMSmart(split_years=[2021])
    if len(ds) == 0:
        raise SystemExit("empty 2021 validation dataset")
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False,
                                     drop_last=False)
    tot = n = 0
    with torch.no_grad():
        for i, xb in enumerate(dl):
            if i >= N:
                break
            xb = xb.to(s3.DEVICE)
            with torch.autocast(device_type=s3.DEVICE.type, dtype=torch.float16,
                                enabled=s3.DEVICE.type == "cuda"):
                _, loss = model(xb, xb)
            valid_tokens = int((xb[:, 1:] != s3.PAD).sum().item())
            tot += loss.item() * valid_tokens
            n += valid_tokens
            if i % 5 == 0:
                log(f"  batch {i}: loss={loss.item():.4f}")
    if n == 0:
        raise SystemExit("validation contains no non-padding target tokens")
    vl = tot / n
    log(f"2021 validation loss={vl:.4f} over {n:,} target tokens")
    log(f"tokenizer fingerprint verified: {fingerprint}")
    print(f"VERIFY OK: tokenizer manifest matches; 2021 val_loss={vl:.4f}", flush=True)


if __name__ == "__main__":
    main()
