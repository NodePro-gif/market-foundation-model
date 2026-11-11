import csv

rows = list(csv.DictReader(open(r"D:\xauusd-dl\results\test_predictions.csv")))
close = [float(r["close"]) for r in rows]
y = [int(r["y_true"]) for r in rows]
pg = [float(r["p_gru"]) for r in rows]
pm = [float(r["p_mlp"]) for r in rows]
n = len(rows)


def acc(pred):
    return sum(int(p == t) for p, t in zip(pred, y)) / n


# direction of bar i = sign(close[i] - close[i-1])
dirb = [None] + [1 if close[i] > close[i - 1] else 0 for i in range(1, n)]

flip = [1 - dirb[i] if dirb[i] is not None else None for i in range(n)]
same = list(dirb)

ok_flip = [(f, t) for f, t in zip(flip, y) if f is not None]
ok_same = [(s, t) for s, t in zip(same, y) if s is not None]

print(f"n = {n:,}")
print(f"GRU                        acc = {acc([1 if p >= .5 else 0 for p in pg]):.4f}")
print(f"MLP                        acc = {acc([1 if p >= .5 else 0 for p in pm]):.4f}")
print(f"mayoritas                  acc = {max(sum(y), n - sum(y)) / n:.4f}")
print(f"prediksi = arah bar lalu   acc = {sum(s == t for s, t in ok_same) / len(ok_same):.4f}")
print(f"prediksi = BALIK arah bar  acc = {sum(f == t for f, t in ok_flip) / len(ok_flip):.4f}   <- flip rule")
print(f"up-rate test               = {sum(y) / n:.4f}")

# signal known at close[i] predicts direction of bar i+1 -> apply to
# return close[i] -> close[i+1]  (same alignment as the training backtest)
cost = 0.00015
gross, net = [], []
for i in range(1, n - 1):
    if flip[i] is None:
        continue
    fwd = (close[i + 1] - close[i]) / close[i]
    sig = 1 if flip[i] == 1 else -1
    gross.append(sig * fwd)
    net.append(sig * fwd - cost)  # position flips on (almost) every bar
eq_g = 1.0
for r in gross:
    eq_g *= 1 + r
eq_n = 1.0
for r in net:
    eq_n *= 1 + r
print(f"flip-rule gross only                    = {eq_g - 1:+.2%}  ({len(gross):,} bars)")
print(f"flip-rule net (cost per flip)           = {eq_n - 1:+.2%}")
