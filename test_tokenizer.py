#!/usr/bin/env python
"""Self-test for _merge / bpe_encode correctness before trusting the corpus."""
import numpy as np
import stage2_tokenizer as st

fails = 0


def check(name, got, want):
    global fails
    ok = np.array_equal(np.asarray(got), np.asarray(want))
    if not ok:
        fails += 1
    print(f"  {'OK ' if ok else 'FAIL'} {name}: got={np.asarray(got).tolist()} want={np.asarray(want).tolist()}")


print("== _merge unit tests ==")
# simple: merge pair (1,2) -> 99 in [1,2,3,1,2,2,1]
check("basic", st._merge(np.array([1, 2, 3, 1, 2, 2, 1]), np.array([0, 3]), 99),
      [99, 3, 99, 2, 1])
# adjacent occurrences of same pair -> must not overlap
check("adjacent", st._merge(np.array([1, 2, 1, 2]), np.array([0]), 99), [99, 1, 2])
check("adjacent2", st._merge(np.array([1, 2, 1, 2]), np.array([0, 2]), 99), [99, 99])
# merge at the very end
check("tail", st._merge(np.array([5, 1, 2]), np.array([1]), 99), [5, 99])
# merge at the very start
check("head", st._merge(np.array([1, 2, 5]), np.array([0]), 99), [99, 5])
# no occurrences
check("none", st._merge(np.array([3, 4, 5]), np.array([], dtype=int), 99), [3, 4, 5])
# multi-merge preserves order of non-merged + positions of merged
rng = np.random.default_rng(0)
data = rng.integers(0, 20, 200)
idx = np.flatnonzero((data[:-1] == 7) & (data[1:] == 13))
if len(idx) > 1:
    idx = idx[np.insert(np.diff(idx) > 1, 0, True)]
ref = list(data)
for p in reversed(idx.tolist()):
    ref[p:p + 2] = [77]
check("random-7-13", st._merge(data, idx, 77), ref)

print("== bpe_encode vs naive reference ==")


def naive_encode(chars, merges):
    """Straightforward O(n*k) reference: repeatedly apply lowest-rank pair."""
    data = list(int(c) for c in chars)
    for rank, (a, b) in enumerate(merges):
        i = 0
        while i < len(data) - 1:
            if data[i] == a and data[i + 1] == b:
                data[i:i + 2] = [256 + rank]
                i = max(i - 1, 0)
            else:
                i += 1
    return data


# tiny hand-worked vocab: merges trained on a toy alphabet
toy = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1] * 5, dtype=np.int64)
merges = np.array([[0, 1], [2, 0], [256, 257], [1, 0]], dtype=np.int32)
new_ids = np.array([256, 257, 258, 259], dtype=np.int32)

enc, span = st.bpe_encode(toy, merges, new_ids)
want = naive_encode(toy, merges)
check("toy-encode", enc, want)
print(f"       span sum={int(span.sum())} n_bars={len(toy)} "
      f"{'OK ' if int(span.sum()) == len(toy) else 'FAIL'}")
if int(span.sum()) != len(toy):
    fails += 1

# random merges on random data
for trial in range(5):
    rng = np.random.default_rng(trial)
    n_merges = 30
    m = rng.integers(0, 256, (n_merges, 2))
    nid = 256 + np.arange(n_merges)
    seq = rng.integers(0, 256, 500)
    e, s = st.bpe_encode(seq, m, nid)
    w = naive_encode(seq, m)
    if not np.array_equal(e, np.asarray(w)) or int(s.sum()) != len(seq):
        fails += 1
        print(f"  FAIL random trial {trial}: len {len(e)} vs {len(w)}, span {int(s.sum())}")
    else:
        print(f"  OK  random trial {trial}: {len(seq)} chars -> {len(e)} tokens")

print("== BPE training sanity (toy corpus) ==")
# train_bpe uses the global N_MERGES; patch it for the toy run
orig = st.N_MERGES
st.N_MERGES = 40
try:
    m2, n2 = st.train_bpe(toy)
finally:
    st.N_MERGES = orig
e2, s2 = st.bpe_encode(toy, m2, n2)
w2 = naive_encode(toy, m2)
check("trained-bpe-encode", e2, w2)
print(f"       compression {len(toy)}/{len(e2)} = {len(toy)/max(len(e2),1):.2f}x, "
      f"span_ok={int(s2.sum()) == len(toy)}")

print()
print("SELF-TEST:", "FAILED" if fails else "PASSED", f"({fails} failures)")
raise SystemExit(1 if fails else 0)
