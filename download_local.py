import concurrent.futures as cf
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://evtradelabs.com/api/simulator/data"
ROOT = r"D:\xauusd-dl\corpus"

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURGBP", "GBPJPY",
    "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "SP500", "NQ", "DJ", "DAX",
    "FTSE", "NIKKEI", "BRENT", "WTI",
]
YEARS = list(range(2015, 2024))
M1 = ["XAUUSD", "EURUSD"]

JOBS = [(s, "M5", y) for s in SYMBOLS for y in YEARS] + [(s, "M1", y) for s in M1 for y in YEARS]
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def fetch(job):
    sym, tf, year = job
    d = os.path.join(ROOT, sym, tf)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{year}.json.gz")
    if os.path.exists(path) and os.path.getsize(path) > 4096:
        return job, "cached", os.path.getsize(path), ""
    url = f"{BASE}/{sym}/{tf}/{year}.json.gz"
    err = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if len(data) < 4096:
                return job, "empty", len(data), ""
            tmp = path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
            return job, "ok", len(data), ""
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return job, "missing", 0, ""
            err = f"HTTP {e.code}"
        except Exception as e:
            err = type(e).__name__
        time.sleep(2 * (attempt + 1))
    return job, "failed", 0, err


def main():
    log(f"downloading {len(JOBS)} files -> {ROOT}")
    ok = miss = fail = 0
    total = 0
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for i, (job, status, size, err) in enumerate(ex.map(fetch, JOBS), 1):
            if status in ("ok", "cached"):
                ok += 1
                total += size
            elif status == "missing":
                miss += 1
            else:
                fail += 1
                log(f"  FAIL {job} {err}")
            if i % 20 == 0:
                log(f"  {i}/{len(JOBS)} ok={ok} miss={miss} fail={fail} {total/1e6:.0f} MB")
    log(f"DONE ok={ok} missing={miss} fail={fail} total={total/1e6:.1f} MB")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
