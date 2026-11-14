#!/usr/bin/env python
"""Stage 2 - market language: bar -> attributes -> BPE "words" -> streams.

Vocabulary layout
    0..255     bar characters (4 attributes x 4 bins)
    256..4095  BPE merges (learned recurring patterns = "kata")
    4096..     specials, symbol, timeframe, regime, session context tokens
"""

import gzip
import json
import os
import pickle
import time

import numpy as np

IS_COLAB = os.path.isdir("/content")
ROOT = "/content/corpus" if IS_COLAB else r"D:\xauusd-dl\corpus"
NPZ = "/content/npz" if IS_COLAB else r"D:\xauusd-dl\npz"
OUT = "/content/mkt" if IS_COLAB else r"D:\xauusd-dl\mkt"

BASE_V = 256
BPE_V = 4096          # total content vocab
N_MERGES = BPE_V - BASE_V
VOCAB = 4160          # padded final size

SPECIAL = BPE_V
PAD, BOS, EOS = SPECIAL, SPECIAL + 1, SPECIAL + 2
SYM0 = SPECIAL + 3                      # 19 symbols
TF0 = SYM0 + 19                         # 2 (M5, M1)
REG0 = TF0 + 2                          # 6 regimes
SESS0 = REG0 + 6                        # 4 sessions

YEARS = list(range(2015, 2024))
HELD_OUT = ["NIKKEI", "BRENT", "EURGBP"]

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURGBP", "GBPJPY",
    "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "SP500", "NQ", "DJ", "DAX",
    "FTSE", "NIKKEI", "BRENT", "WTI",
]
TF_LIST = ["M5", "M1"]

t0 = time.time()
os.makedirs(NPZ, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


# ----------------------------------------------------------------- helpers
def rolling_mean(x, w):
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (cs[w:] - cs[: len(x) - w + 1]) / w
    return out


def indicators(ts, o, h, l, c, v):
    n = len(c)
    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    atr = rolling_mean(tr, 14)
    ret = np.zeros(n)
    ret[1:] = np.log(c[1:] / c[:-1])
    rng = h - l
    pos = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1.0), 0.5)
    vm = rolling_mean(v, 20)
    a_val = ret / np.where(atr > 0, atr, np.nan)
    c_val = rng / np.where(atr > 0, atr, np.nan)
    d_val = v / np.where(vm > 0, vm, np.nan)
    return a_val, pos, c_val, d_val, ret, atr, rng


def regime_features(ret, atr, rng, c):
    f = np.zeros((len(c), 4), dtype=np.float32)
    s = rolling_mean(ret, 48)
    s2 = rolling_mean(ret ** 2, 48)
    sd = np.sqrt(np.maximum(s2 - s ** 2, 0.0))
    f[:, 0] = np.log(np.where(sd > 0, sd, np.nan))
    f[:, 1] = np.where(sd > 0, s / np.where(sd > 0, sd, np.nan), 0.0)
    f[:, 2] = np.where(c > 0, atr / c, np.nan)
    rr = rolling_mean(rng, 48)
    f[:, 3] = np.where(c > 0, rr / c, np.nan)
    f[:, 1] = np.clip(f[:, 1], -5, 5)
    return f


def session_of(ts):
    hr = ((ts // 3600) % 24).astype(np.int64)
    s = np.full(len(ts), 3, dtype=np.int64)   # 21-24 Off
    s[hr < 21] = 2                            # 13-21 NY
    s[hr < 13] = 1                            # 8-13 London
    s[hr < 8] = 0                             # 0-8 Asia
    return s


# ------------------------------------------------- STEP A: json -> npz
def step_a():
    log("STEP A - parsing json -> compact npz")
    done = 0
    for sym in SYMBOLS:
        for tf in TF_LIST:
            src = os.path.join(ROOT, sym, tf)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(NPZ, f"{sym}_{tf}.npz")
            if os.path.exists(dst):
                done += 1
                continue
            T, O, H, L, C, V = [], [], [], [], [], []
            for y in YEARS:
                p = os.path.join(src, f"{y}.json.gz")
                if not os.path.exists(p):
                    continue
                with gzip.open(p, "rb") as fh:
                    arr = json.loads(fh.read())
                if not arr:
                    continue
                T.append(np.fromiter((r["ts"] for r in arr), np.int64, len(arr)))
                O.append(np.fromiter((r["o"] for r in arr), np.float32, len(arr)))
                H.append(np.fromiter((r["h"] for r in arr), np.float32, len(arr)))
                L.append(np.fromiter((r["l"] for r in arr), np.float32, len(arr)))
                C.append(np.fromiter((r["c"] for r in arr), np.float32, len(arr)))
                # some feeds omit volume entirely - fall back to range as proxy
                V.append(np.fromiter((r.get("v", abs(r["h"] - r["l"]))
                                      for r in arr), np.float32, len(arr)))
            if not T:
                continue
            ts = np.concatenate(T)
            order = np.argsort(ts, kind="stable")
            np.savez_compressed(
                dst, ts=ts[order], o=np.concatenate(O)[order],
                h=np.concatenate(H)[order], l=np.concatenate(L)[order],
                c=np.concatenate(C)[order], v=np.concatenate(V)[order],
            )
            done += 1
            log(f"  {sym}/{tf}: {len(ts):,} bars  ({done} series)")
    log("STEP A done")


# ------------------------------------------------- STEP B: calibration
def iter_series():
    for sym in SYMBOLS:
        for tf in TF_LIST:
            p = os.path.join(NPZ, f"{sym}_{tf}.npz")
            if os.path.exists(p):
                yield sym, tf, np.load(p)


def step_b():
    log("STEP B - calibrating bins + regime clusters")
    A, P, R, D = [], [], [], []
    FR, AT, RG, PR = [], [], [], []
    vsample = []
    for sym, tf, z in iter_series():
        a, p, r, d, ret, atr, rng = indicators(z["ts"], z["o"], z["h"], z["l"], z["c"], z["v"])
        sl = slice(None, None, 7)
        for arr, acc in ((a, A), (p, P), (r, R), (d, D)):
            s = arr[sl]
            s = s[np.isfinite(s)]
            acc.append(s[:80000])
        if tf == "M5":
            rf = regime_features(ret, atr, rng, z["c"])
            FR.append(rf[sl][::3])
            # realised volatility of next 12 bars (label calibration)
            rr = np.concatenate(([0.0], np.log(z["c"][1:] / z["c"][:-1])))
            cs = np.concatenate(([0.0], np.cumsum(rr ** 2)))
            k = 12
            rv = np.full(len(rr), np.nan)
            # rv[i] = std of returns rr[i+1..i+k] = cs[i+k+1]-cs[i+1]
            rv[: len(rr) - k] = np.sqrt(
                np.maximum(cs[k + 1:len(rr) + 1] - cs[1:len(rr) - k + 1], 0) / k)
            v = rv[sl]
            vsample.append(v[np.isfinite(v)][:80000])

    pool_a = np.concatenate(A)
    pool_p = np.concatenate(P)
    pool_r = np.concatenate(R)
    pool_d = np.concatenate(D)
    edges = dict(
        a=np.quantile(pool_a, [0.25, 0.5, 0.75]),
        p=np.array([0.25, 0.5, 0.75]),
        r=np.quantile(pool_r, [0.25, 0.5, 0.75]),
        d=np.quantile(pool_d, [0.25, 0.5, 0.75]),
        v=np.quantile(np.concatenate(vsample), [0.25, 0.5, 0.75]),
    )
    log(f"  bin edges a={np.round(edges['a'],4)} r={np.round(edges['r'],4)} "
        f"d={np.round(edges['d'],4)} vol={np.round(edges['v'],4)}")

    from sklearn.cluster import KMeans

    X = np.concatenate(FR)
    X = X[np.isfinite(X).all(axis=1)]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit((X - mu) / sd)
    centers = km.cluster_centers_
    order = np.argsort(centers[:, 0])        # order regimes by volatility level
    inv = np.empty(6, dtype=np.int64)
    inv[order] = np.arange(6)
    km.cluster_centers_ = centers[order]
    np.savez(f"{OUT}/calib.npz", **{k: v for k, v in edges.items()},
             mu=mu, sd=sd, centers=km.cluster_centers_,
             held=np.array(HELD_OUT, dtype=object), allow_pickle=True)
    with open(f"{OUT}/kmeans.pkl", "wb") as fh:
        pickle.dump(km, fh)
    log(f"  regime centers (vol, trend, atr, range):\n{np.round(km.cluster_centers_, 3)}")
    log("STEP B done")


# ------------------------------------------------- STEP C: chars + BPE
def bin4(v, edges):
    return np.searchsorted(edges, v, side="right").astype(np.int8)


def _merge(data, idx, new_id):
    """Replace non-overlapping occurrences of a pair with a single new id.

    `cum` must carry a leading zero so that cum[idx+2] counts the elements
    kept strictly BEFORE position idx+2 (i.e. before the consumed pair).
    """
    n = len(data)
    if len(idx) == 0:
        return data
    keep = np.ones(n, dtype=bool)
    keep[idx] = False
    keep[idx + 1] = False
    kept = data[keep]
    cum = np.concatenate(([0], np.cumsum(keep)))          # length n+1
    new_pos = cum[np.minimum(idx + 2, n)] + np.arange(len(idx))
    out = np.empty(n - len(idx), dtype=data.dtype)
    free = np.ones(n - len(idx), dtype=bool)
    free[new_pos] = False
    out[new_pos] = new_id
    out[free] = kept
    return out


def train_bpe(sample, mul=BPE_V):
    data = sample.astype(np.int64)
    V = BASE_V
    merges, new_ids = [], []
    banned = set()          # pairs already merged - never re-merge them,
    log(f"  BPE training on {len(data):,} chars -> {N_MERGES} merges ...")  # or the
    tb = time.time()        # encoder's rank LUT becomes ambiguous
    for m in range(N_MERGES):
        pairs = data[:-1] * V + data[1:]
        cnt = np.bincount(pairs, minlength=V * V)
        while True:
            best = int(cnt.argmax())
            if cnt[best] == 0:
                break
            a, b = divmod(best, V)
            if (a, b) in banned:
                cnt[best] = 0
                continue
            break
        if cnt[best] == 0:
            log(f"  BPE: no pairs left at merge {m}")
            break
        banned.add((a, b))
        nid = V
        mask = (data[:-1] == a) & (data[1:] == b)
        idx = np.flatnonzero(mask)
        if len(idx) > 1:
            idx = idx[np.insert(np.diff(idx) > 1, 0, True)]
        data = _merge(data, idx, nid)
        merges.append((a, b))
        new_ids.append(nid)
        V += 1
        if m % 512 == 0:
            log(f"    merge {m:>4}/{N_MERGES}  size={len(data):,}  "
                f"{time.time()-tb:.0f}s")
    log(f"  BPE done: vocab={V} ({time.time()-tb:.0f}s)")
    return np.array(merges, dtype=np.int32), np.array(new_ids, dtype=np.int32)


def _rank_lut(merge_arr, mul=BPE_V):
    ranks = np.full(mul * mul, -1, dtype=np.int32)
    if len(merge_arr):
        ranks[merge_arr[:, 0].astype(np.int64) * mul + merge_arr[:, 1]] = np.arange(len(merge_arr))
    return ranks


def bpe_encode(chars, merge_arr, new_ids, mul=BPE_V):
    """Single pass returning (tokens, spans) - spans = how many bars each token covers."""
    ranks = _rank_lut(merge_arr, mul)
    data = chars.astype(np.int64)
    span = np.ones(len(data), dtype=np.int64)
    while len(data) > 1:
        pairs = data[:-1] * mul + data[1:]
        r = ranks[pairs]
        valid = r >= 0
        if not valid.any():
            break
        m = int(r[valid].min())
        idx = np.flatnonzero(r == m)
        if len(idx) > 1:
            idx = idx[np.insert(np.diff(idx) > 1, 0, True)]
        span = _merge(span, idx, span[idx] + span[idx + 1])
        data = _merge(data, idx, int(new_ids[m]))
    return data.astype(np.uint16), span.astype(np.uint32)


def step_c():
    log("STEP C - characterising bars, training BPE, encoding corpus")
    calib = np.load(f"{OUT}/calib.npz", allow_pickle=True)
    edges = {k: calib[k] for k in "aprdv"}
    km = pickle.load(open(f"{OUT}/kmeans.pkl", "rb"))
    mu, sd = calib["mu"], calib["sd"]

    # --- one pass: raw attribute arrays per series (cached) ---------------
    raws = {}
    for sym, tf, z in iter_series():
        a, p, r, d, ret, atr, rng = indicators(z["ts"], z["o"], z["h"], z["l"], z["c"], z["v"])
        raws[(sym, tf)] = dict(a=a, p=p, r=r, d=d, ret=ret, atr=atr, rng=rng,
                               ts=z["ts"], c=z["c"], h=z["h"], l=z["l"], v=z["v"])
        log(f"  indicators {sym}/{tf}: {len(a):,}")

    # --- BPE training sample ---------------------------------------------
    chunks = []
    for key in sorted(raws):
        rr = raws[key]
        a8 = bin4(rr["a"], edges["a"]).astype(np.int64)
        p8 = bin4(rr["p"], edges["p"]).astype(np.int64)
        r8 = bin4(rr["r"], edges["r"]).astype(np.int64)
        d8 = bin4(rr["d"], edges["d"]).astype(np.int64)
        ch = a8 * 64 + p8 * 16 + r8 * 4 + d8
        raws[key]["ch"] = ch
        chunks.append(ch[:: max(1, len(ch) // 120000)][:120000])
    sample = np.concatenate(chunks)
    merge_arr, new_ids = train_bpe(sample)

    # --- encode every series ---------------------------------------------
    summary = []
    for key in sorted(raws):
        sym, tf = key
        rr = raws[key]
        ch = rr["ch"].astype(np.int64)
        enc, spans = bpe_encode(ch, merge_arr, new_ids)
        np.savez_compressed(f"{OUT}/tok_{sym}_{tf}.npz", tokens=enc, spans=spans)
        summary.append((sym, tf, len(ch), len(enc), round(len(ch) / max(len(enc), 1), 3)))
        log(f"  encoded {sym}/{tf}: {len(ch):,} bars -> {len(enc):,} tokens "
            f"(compression {len(ch)/max(len(enc),1):.2f}x)")

    # --- labels + regime for M5 ------------------------------------------
    for key in sorted(raws):
        sym, tf = key
        if tf != "M5":
            continue
        rr = raws[key]
        c, ts = rr["c"], rr["ts"]
        n = len(c)
        nxt = np.full(n, 255, dtype=np.uint8)
        nxt[:-1] = (c[1:] > c[:-1]).astype(np.uint8)
        fwd = {}
        for h in (12, 48, 288):
            f = np.full(n, np.nan, dtype=np.float32)
            f[: n - h] = np.log(c[h:] / c[: n - h])
            fwd[h] = f
        ret = np.concatenate(([0.0], np.log(c[1:] / c[:-1])))
        cs = np.concatenate(([0.0], np.cumsum(ret ** 2)))
        rv = np.full(n, np.nan, dtype=np.float32)
        k = 12
        rv[: n - k] = np.sqrt(np.maximum(
            cs[k + 1:n + 1] - cs[1:n - k + 1], 0) / k)
        vb = np.full(n, 255, dtype=np.uint8)
        fin = np.isfinite(rv)
        vb[fin] = np.searchsorted(calib["v"], rv[fin], side="right").astype(np.uint8)
        rf = regime_features(rr["ret"], rr["atr"], rr["rng"], c)
        rfz = (rf - mu) / sd
        reg = np.full(n, 255, dtype=np.uint8)
        ok = np.isfinite(rfz).all(axis=1)
        reg[ok] = km.predict(np.nan_to_num(rfz[ok], nan=0.0)).astype(np.uint8)
        sess = session_of(ts).astype(np.uint8)
        np.savez_compressed(
            f"{OUT}/lab_{sym}.npz", ts=ts, nxt=nxt, vol=rv, volb=vb,
            regime=reg, sess=sess, fwd12=fwd[12], fwd48=fwd[48],
            fwd288=fwd[288], c=c,
        )
        log(f"  labels {sym}: {n:,} bars")

    np.savez(f"{OUT}/bpe.npz", merges=merge_arr, new_ids=new_ids)
    with open(f"{OUT}/vocab.json", "w") as fh:
        json.dump(dict(base=BASE_V, content=BPE_V, vocab=VOCAB, merges=int(len(merge_arr)),
                       PAD=PAD, BOS=BOS, EOS=EOS, SYM0=SYM0, TF0=TF0,
                       REG0=REG0, SESS0=SESS0, symbols=SYMBOLS,
                       held_out=HELD_OUT, tf_list=TF_LIST), fh, indent=1)
    log("STEP C done")
    log(f"compression summary (sym, tf, bars, tokens, ratio): {summary[:6]} ...")


if __name__ == "__main__":
    step_a()
    step_b()
    step_c()
    print("STAGE2 DONE", flush=True)
