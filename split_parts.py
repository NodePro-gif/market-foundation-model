#!/usr/bin/env python
"""Split mkt_upload.zip into parts small enough for the Colab upload API."""
import os
import zipfile

SRC = r"D:\xauusd-dl\mkt_upload.zip"
PART = 40 * 1024 * 1024          # 40 MB per part

size = os.path.getsize(SRC)
n = (size + PART - 1) // PART
with open(SRC, "rb") as fh:
    for i in range(n):
        chunk = fh.read(PART)
        p = rf"D:\xauusd-dl\parts\mkt.z{i:02d}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as out:
            out.write(chunk)
        print(f"{p} {len(chunk)/1e6:.1f} MB")
print(f"total parts={n}")
