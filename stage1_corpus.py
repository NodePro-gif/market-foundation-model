#!/usr/bin/env python
"""Stage 1 - download the multi-asset market corpus (M5 + M1) and verify it."""

import concurrent.futures as cf
import gzip
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://evtradelabs.com/api/simulator/data"
LOCAL_DATA_ROOT = os.environ.get(
    "MFM_DATA_ROOT",
    r"D:\\xauusd-dl" if os.name == "nt"
    else os.path.join(os.path.expanduser("~"), "market-foundation-model-data"),
)
ROOT = os.environ.get(
    "MFM_CORPUS_DIR",
    "/content/corpus" if os.path.isdir("/content")
    else os.path.join(LOCAL_DATA_ROOT, "corpus"),
)

SYMBOLS = [
    # FX
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURGBP", "GBPJPY",
    # metals
    "XAUUSD", "XAGUSD",
    # crypto
    "BTCUSD", "ETHUSD",
    # indices
    "SP500", "NQ", "DJ", "DAX", "FTSE", "NIKKEI",
    # energy
    "BRENT", "WTI",
]
YEARS = list(range(2015, 2024))
M1_SYMBOLS = ["XAUUSD", "EURUSD"]

JOBS = [(s, "M5", y) for s in SYMBOLS for y in YEARS]
JOBS += [(s, "M1", y) for s in M1_SYMBOLS for y in YEARS]

t0 = time.time()
os.makedirs(ROOT, exist_ok=True)


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def fetch(job):
    sym, tf, year = job
    d = os.path.join(ROOT, sym, tf)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{year}.json.gz")
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return (job, "cached", os.path.getsize(path), 0)
    url = f"{BASE}/{sym}/{tf}/{year}.json.gz"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 1024:
                return (job, "empty", len(data), 0)
            tmp = path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
            return (job, "ok", len(data), 0)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return (job, "missing", 0, 0)
            time.sleep(2 * (attempt + 1))
            last = f"HTTP {e.code}"
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            last = f"{type(e).__name__}"
    return (job, "failed", 0, last)


log(f"downloading {len(JOBS)} files ({len(SYMBOLS)} symbols x M5 {YEARS[0]}-{YEARS[-1]}"
    f" + M1 {M1_SYMBOLS}) ...")

ok = miss = fail = 0
bytes_total = 0
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    done = 0
    for res in ex.map(fetch, JOBS):
        job, status, size, err = res
        done += 1
        if status in ("ok", "cached"):
            ok += 1
            bytes_total += size
        elif status == "missing":
            miss += 1
        else:
            fail += 1
            log(f"  FAIL {job} {err}")
        if done % 20 == 0 or status not in ("ok", "cached"):
            log(f"  {done}/{len(JOBS)}  ok={ok} missing={miss} fail={fail} "
                f"downloaded={bytes_total/1e6:.1f} MB")

log(f"download done: ok={ok} missing={miss} fail={fail} "
    f"total={bytes_total/1e6:.1f} MB")

# ---- verify: parse every file, count bars and time coverage -------------
log("verifying ...")
manifest = []
total_bars = 0
for sym, tf, year in JOBS:
    path = os.path.join(ROOT, sym, tf, f"{year}.json.gz")
    if not os.path.exists(path):
        continue
    try:
        with gzip.open(path, "rb") as fh:
            arr = json.loads(fh.read())
    except Exception as e:
        log(f"  BAD {sym}/{tf}/{year}: {e}")
        continue
    if not arr:
        continue
    bars = len(arr)
    total_bars += bars
    manifest.append(dict(
        symbol=sym, tf=tf, year=year, bars=bars,
        t0=arr[0]["ts"], t1=arr[-1]["ts"],
        o0=arr[0]["o"], c1=arr[-1]["c"],
    ))

with open(os.path.join(ROOT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=1)

log(f"files={len(manifest)}  TOTAL BARS={total_bars:,}")
by_tf = {}
for m in manifest:
    by_tf[m["tf"]] = by_tf.get(m["tf"], 0) + m["bars"]
for k, v in sorted(by_tf.items()):
    log(f"  {k}: {v:,} bars")
log(f"disk usage: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(ROOT) for f in fs)/1e6:.1f} MB")
print("STAGE1 DONE", flush=True)
