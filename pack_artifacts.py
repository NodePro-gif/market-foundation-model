#!/usr/bin/env python
"""Pack the lightweight tokenized artifacts for upload to the Colab VM."""
import os
import zipfile

SRC = r"D:\xauusd-dl\mkt"
DST = r"D:\xauusd-dl\mkt_upload.zip"

files = sorted(os.listdir(SRC))
total = 0
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        p = os.path.join(SRC, f)
        size = os.path.getsize(p)
        total += size
        z.write(p, f)

print(f"files={len(files)} raw={total/1e6:.1f} MB "
      f"zip={os.path.getsize(DST)/1e6:.1f} MB")
for f in files:
    print(f"  {f:28s} {os.path.getsize(os.path.join(SRC, f))/1e6:8.2f} MB")
