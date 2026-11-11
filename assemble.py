#!/usr/bin/env python
"""Reassemble the split zip and extract artifacts."""
import os
import zipfile

parts = sorted(f for f in os.listdir("/content") if f.startswith("mkt.z"))
print("parts:", parts)
with open("/content/mkt_upload.zip", "wb") as out:
    for p in parts:
        out.write(open(f"/content/{p}", "rb").read())
print("zip size:", os.path.getsize("/content/mkt_upload.zip"))

os.makedirs("/content/mkt", exist_ok=True)
with zipfile.ZipFile("/content/mkt_upload.zip") as z:
    z.extractall("/content/mkt")
files = os.listdir("/content/mkt")
print(f"extracted {len(files)} files")
assert len(files) >= 40, f"too few files: {len(files)}"
import json
v = json.load(open("/content/mkt/vocab.json"))
print("vocab ok:", v["vocab"], "symbols:", len(v["symbols"]))
print("ASSEMBLE DONE")
