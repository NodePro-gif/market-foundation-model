# Market Foundation Model — XAUUSD & Multi-Asset

Sistem AI/ML trading yang dilatih untuk **memahami konteks pasar** (bukan sekadar
mencari pola OHLC), dengan arsitektur ala LLM: *tokenizer* pasar → *pretraining*
autoregresif di korpus multi-aset → *fine-tuning* ke kepala prediksi.

```
bar → 4 atribut × 4 bin (256 "huruf") → BPE merge → 4.096 "kata"
     → GPT decoder-only (12,44M param, ctx 512) → [arah | regime vol | kuantil return]
```

---

## 1. Cara kerja sistem (pipeline lengkap)

| Stage | Script | Apa yang dilakukan |
|---|---|---|
| **1** | `stage1_corpus.py` / `download_local.py` | Unduh korpus: 19 simbol × M5 2015–2023 + M1 (XAUUSD, EURUSD) dari EV Trading Labs. ±175 file, 359 MB. |
| **2** | `stage2_tokenizer.py` | Kalibrasi bin, cluster regime (KMeans 6 kelas), **BPE training** (3.840 merge → vocab 4.096), encoding seluruh korpus → token, plus label (arah, vol, regime, sesi, return 12/48/288). |
| **3** | `stage3_pretrain.py` | **Pretraining GPT** di seluruh korpus (2015–2022), model 12,44M param, ctx 512, AMP fp16, checkpoint tiap 250 step. 2023 di-holdout. |
| **4** | `stage4_finetune.py` | **Fine-tune** 3 kepala prediksi + evaluasi jujur: XAUUSD 2023, simbol held-out, kalibrasi Brier/ECE, metrik per-regime. |

### Detail arsitektur

- **Tokenizer**: tiap bar → 4 atribut (arah close, posisi close di range, besaran
  range, perubahan volume) masing-masing 4 bin → 256 "huruf" → BPE merge →
  **4.096 "kata"**. Layout vocab: `0–255` char, `256–4095` BPE, `4096+` specials
  (PAD/BOS/EOS), simbol, timeframe, regime, sesi.
- **Konteks**: tiap jendela diawali `[BOS][SIMBOL][TF][REGIME][SESI]`.
  Regime = 6 kelas keadaan pasar (tenang-trend, tenang-range, bergejolak-trend,
  bergejolak-range, krisis, recovery) dari KMeans.
- **Model**: GPT decoder-only, 6 layer, 8 head, d=384, weight tying, **12,44M param**.
- **Pretraining**: next-token prediction, 15.087 jendela (2015–2022), 3.000 step
  (±67 menit di T4), val loss 8,40 → **5,89**.
- **Fine-tune**: 3 kepala pada hidden state terakhir —
  (a) arah bar berikutnya (BCE), (b) regime volatilitas 4-kelas (CE),
  (c) kuantil return P10/P50/P90 untuk horizon 12/48/288 bar (pinball).

### Baseline (Fase 1, XAUUSD M5 2020–2023)

| Model | test acc | test AUC |
|---|---|---|
| GRU 2×128 | 0,5159 | 0,5206 |
| MLP | 0,5115 | 0,5148 |
| LogisticRegression | 0,5145 | 0,5190 |
| mayoritas | 0,5051 | — |

Diagnosis: arsitektur sehat (overfit 2.048 sampel → 99,76%), training konvergen,
tapi **fitur OHLC mentok** — inilah alasan sistem baru dibangun.

---

## 2. File yang TIDAK ada di GitHub (dan cara mendapatkannya)

Data mentah ±1,2 GB sengaja **dikecualikan** dari git (bisa diunduh ulang, bukan
sistem). Yang dikecualikan lewat `.gitignore`:

```
corpus/   342,7 MB  — korpus mentah (.json.gz)
npz/      178,2 MB  — hasil parse per series
mkt/      229,1 MB  — token + label (artefak tokenizer)
parts/    211,8 MB  — pecahan zip untuk upload ke Colab
*.zip     216 MB    — mkt_upload.zip, results.zip
gold_price.csv.gz 29,7 MB — data baseline XAUUSD
```

### Cara mendapatkan semuanya dari nol

```bash
# 1. Unduh korpus (19 simbol × M5 2015–2023 + M1 XAUUSD/EURUSD)
python download_local.py

# 2. Tokenisasi penuh (parse → kalibrasi → BPE → encoding → label)
python stage2_tokenizer.py

# 3. Pack artefak ringan untuk diunggah ke Colab
python pack_artifacts.py      # → mkt_upload.zip (±222 MB)
python split_parts.py         # → parts/mkt.z00 … mkt.z05 (±42 MB each)
```

Catatan:
- Sumber data: `https://evtradelabs.com/api/simulator/data/{SYMBOL}/{TF}/{YEAR}.json.gz`
  (tanpa login untuk tahun tertutup). 14 file tidak tersedia di penyedia.
- `stage2_tokenizer.py` bisa jalan di lokal (Windows) maupun di VM (`/content`).
- Untuk upload ke Colab: file >50 MB gagal dengan satu PUT — pecah jadi bagian
  ±42 MB (`split_parts.py`), lalu rakit di VM (`assemble.py`).

### File yang SUDAH ada di GitHub

- Semua script: `stage1_corpus.py`, `stage2_tokenizer.py`, `stage3_pretrain.py`,
  `stage4_finetune.py`, `train_xauusd.py`, `diag.py`, `test_tokenizer.py`, dll.
- `ckpt/best.pt` — **model pretraining terbaik** (val 5,8945 @ step 3000, 53 MB).
- `ckpt/pretrain_log.json` — kurva val loss.
- `results/` — hasil baseline (metrics, plot, prediksi, model GRU/MLP).

---

## 3. Menjalankan di Google Colab (GPU T4)

```bash
# 1. Buat session GPU
colab new -s gpu --gpu T4

# 2. Unggah artefak token (pecahan) + script
colab upload -s gpu parts/mkt.z00 /content/mkt.z00   # … z01–z05
colab upload -s gpu assemble.py /content/assemble.py
colab upload -s gpu stage3_pretrain.py /content/stage3_pretrain.py
colab upload -s gpu stage4_finetune.py /content/stage4_finetune.py
colab upload -s gpu ckpt/best.pt /content/ckpt_best.pt

# 3. Rakit + siapkan
colab exec -s gpu -f assemble.py
colab exec -s gpu -f prep_stage4.py

# 4. Pretraining (±67 menit) — GUNAKAN --timeout 0
colab exec -s gpu --timeout 0 -f stage3_pretrain.py

# 5. Fine-tune + evaluasi
colab exec -s gpu --timeout 0 -f stage4_finetune.py
```

### ⚠️ Jebakan yang sudah diketahui (penting!)

1. **Session Colab mati saat idle** → selalu `colab sessions` dulu; siapkan semua
   artefak lokal sebelum membuka session.
2. **Upload file >50 MB gagal** (SSL/500) → pecah jadi bagian ±42 MB.
3. **`colab exec` timeout default 30 detik tanpa output** → saat evaluasi,
   koneksi putus dan cell berhenti. Selalu pakai `--timeout 0`.
4. **CUDA "device-side assert"** = ada index ke luar vocab (mis. label regime
   sentinel 255). Setelah assert, context GPU keracunan → harus session baru.
5. **CUDA OOM** = sisa memori run crash sebelumnya menumpuk di kernel yang sama
   (kernel Colab persist antar-exec). Solusi: `gc.collect()` + `torch.cuda.empty_cache()`,
   atau batch lebih kecil.
6. **Autentikasi**: token OAuth Colab disimpan di Windows Credential Locker;
   bila hilang, `auth_helper.py` menjalankan ulang flow PKCE.

---

## 4. Struktur repo

```
.
├── stage1_corpus.py        # unduh korpus (versi VM)
├── download_local.py       # unduh korpus (versi lokal/Windows)
├── stage2_tokenizer.py     # tokenizer pasar + BPE + label
├── test_tokenizer.py       # self-test tokenizer (wajib lolos sebelum pakai)
├── stage3_pretrain.py      # pretraining GPT
├── verify_ckpt.py          # cek ckpt/best.pt masih selaras dgn tokenizer baru
├── stage4_finetune.py      # fine-tune + evaluasi
├── backtest_stage4.py      # backtest biaya-nyata dari prediksi stage4
├── train_xauusd.py         # baseline XAUUSD (GRU + MLP)
├── diag.py                 # diagnosis kenapa baseline mentok
├── assemble.py             # rakit artefak di VM
├── prep_stage4.py          # siapkan ckpt + year map di VM
├── pack_artifacts.py       # pack artefak ringan
├── split_parts.py          # pecah zip untuk upload
├── ckpt/                   # model pretraining + log
│   ├── best.pt
│   └── pretrain_log.json
├── results/                # hasil baseline
│   ├── metrics.json
│   ├── test_predictions.csv
│   ├── *.png
│   └── gru.pt / mlp.pt
└── results/finetune/       # hasil stage4 (metrics.json; bobot di-ignore)
```

## 5. Hasil pretraining (sistem baru)

| step | val_loss | ppl |
|---|---|---|
| 250 | 6,4024 | 603 |
| 500 | 6,1102 | 450 |
| 1000 | 5,9801 | 396 |
| 2000 | 5,9181 | 372 |
| **3000** | **5,8945** | **363** |

Val loss turun di setiap evaluasi tanpa overfit → model benar-benar belajar
"bahasa pasar". Hasil fine-tune + evaluasi lengkap lihat `results/` setelah
menjalankan `stage4_finetune.py`.

## 6. Kekuatan & batasan yang jujur

**Kekuatan:** konteks lintas-aset (19 simbol), kondisi pasar (regime/sesi),
output distribusi (bukan angka tunggal), evaluasi ter-holdout ketat
(2023 tak pernah dilihat; NIKKEI/BRENT/EURGBP tak pernah dipakai saat pretraining).

**Batasan:** arah harga M5 adalah target tersulit di keuangan — pemain dengan
data order-flow pun tidak banyak mendapatkan di sini. Model ini menghasilkan
**distribusi probabilitas terkalibrasi**, bukan ramalan pasti.

---

## 7. Status progres (11 Nov 2026) — pipeline dijalankan ulang dari nol

Seluruh pipeline di-reproduce end-to-end di Windows lokal (CPU saja, 4 thread,
tanpa Colab/GPU) untuk membuktikan repo bisa jalan dari nol:

| # | langkah | script | hasil |
|---|---|---|---|
| 1 | unduh korpus | `download_local.py` | 175/189 file, 359 MB, ±20 detik (14 missing = sudah diketahui) |
| — | self-test | `test_tokenizer.py` | **PASSED**, 0 failure |
| 2 | tokenisasi | `stage2_tokenizer.py` | 21 series, BPE **3.840 merge → vocab 4096**, kompresi ±1,45×, 21 menit |
| — | integritas ckpt | `verify_ckpt.py` | val_loss **5,9381** vs publikasi 5,8945 → tokenizer hasil regenerasi **SELARAS** dengan `ckpt/best.pt` |
| 4 | fine-tune + evaluasi | `stage4_finetune.py` | 1 epoch 23 menit → acc 0,5692/AUC 0,6199; **2 epoch 40 menit → acc 0,5872/AUC 0,6792** (tabel bawah) |
| — | backtest | `backtest_stage4.py` | 780 taruhan 1-bar, edge +0,78 bps/taruhan (t 5,28), breakeven cost 0,39 bps/side |

### Hasil fine-tune lokal (CPU)

Konfigurasi kecil karena keterbatasan CPU (lihat knob di bawah): stride latih
256 → 2.068 window (2015–2022), test 2023 stride 64 → 780 window, batch 16,
LR 3e-4, seed 42 (run 1 & 2 epoch identik sampai digit di epoch 1 → deterministik).

| model | test acc | test AUC | Brier | ECE |
|---|---|---|---|---|
| mayoritas (baseline) | 0,5051 | — | — | — |
| LogisticRegression | 0,5145 | 0,5190 | — | — |
| MLP (baseline) | 0,5115 | 0,5148 | — | — |
| GRU 2×128 (baseline) | 0,5159 | 0,5206 | — | — |
| Market GPT fine-tune, 1 epoch | 0,5692 | 0,6199 | 0,2443 | **0,0453** |
| **Market GPT fine-tune, 2 epoch** | **0,5872** | **0,6792** | **0,2279** | 0,1077 |

Catatan jujur soal kalibrasi: epoch ke-2 menaikkan AUC (0,6199 → 0,6792) dan
menurunkan Brier (0,2443 → 0,2279), tetapi **ECE memburuk** (0,0453 → 0,1077)
— fine-tune lanjut membuat probabilitas makin pekat (over-confidence). Bila
dipakai untuk sizing posisi, kepala sebaiknya di-platt-scale dulu.

Metrik lengkap: `results/finetune/metrics.json` (run 2 epoch; arsip run 1:
`metrics_cpu_1ep.json`, `backtest_1ep.json`). Per-regime pada test 2023 hanya
regime 1 yang cukup sampel (n=335, acc 0,5731, Brier 0,2237).

**Simbol held-out (tak pernah dilihat saat pretraining)** — dipakai untuk
menguji generalisasi lintas-aset (run 2 epoch):

| simbol | acc | AUC | Brier | n |
|---|---|---|---|---|
| NIKKEI | 0,4780 | 0,4838 | 0,3037 | 364 |
| BRENT | 0,5144 | 0,5293 | 0,2798 | 348 |
| EURGBP | 0,5196 | 0,5222 | 0,2800 | 408 |

Jujur: di simbol held-out model ≈ acak → **transfer lintas-aset belum terbukti**;
keunggulan hanya muncul di XAUUSD (simbol yang memang dilatih saat fine-tune).

### Backtest biaya-nyata (`backtest_stage4.py`, baru)

Skrip baru: ambil `test_pred.npy`, sejajarkan tiap window dengan bar yang
diakhirinya, lalu taruh taruhan **satu bar ke depan** (horizon yang memang
dilabeli `nxt`). 780 taruhan independen (stride 64, 2 Jan – 29 Des 2023),
biaya **1,5 bps/side** (sama seperti `check_flip.py`), round-trip 3 bps.

| strategi | total (net) | rata-rata/taruhan | menang-net | Sharpe/t-stat |
|---|---|---|---|---|
| **model, tanpa biaya** | **+6,10%** | **+0,78 bps** | 58,7% | **+5,28** |
| model, dgn biaya 3 bps RT | −17,30% | −2,22 bps | 22,2% | −14,98 |
| inverse model | −29,50% | −3,78 bps | 11,0% | −25,54 |
| selalu long | −23,92% | −3,07 bps | 15,6% | −20,35 |
| selalu short | −22,88% | −2,93 bps | 17,6% | −19,47 |
| persistence (ulangi arah bar lalu) | −1,11% | −0,14 bps | 33,2% | −1,29 |
| buy & hold (window sama) | −0,52% | — | — | — |

(run 1 epoch untuk perbandingan: model gross +4,27% / +0,55 bps / t 3,66.)

Kolom "menang-net" = fraksi taruhan yang **positif setelah biaya** (bukan
akurasi; akurasi kepala terhadap label = 0,5872 dan sudah sama persis dengan
yang dilaporkan `stage4_finetune.py`).

**Temuan kunci: breakeven cost = 0,39 bps/side (0,78 bps round-trip).**
Edge mentah kepala arah (+0,78 bps/taruhan, t-stat 5,28 → nyata, bukan
kebetulan; run 1 epoch: +0,55 bps, t 3,66) **masih di bawah** asumsi biaya
1,5 bps/side → pada M5 dengan biaya ini strategi tetap rugi. Jalan keluarnya:
spread/biaya lebih rendah (XAUUSD spread tipis justru masuk akal untuk
seleksi taruhan ber-`p` ekstrem), atau horizon lebih panjang dari satu bar.

### Knob untuk mesin kecil (defaults = konfigurasi asli Colab/GPU)

`stage4_finetune.py` bisa dijalankan di CPU tanpa mengubah kode lewat env var:

| env | default | arti |
|---|---|---|
| `ST4_EPOCHS` | 3 | jumlah epoch latih |
| `ST4_BATCH` | 256 | batch size |
| `ST4_LR` | 1e-4 | learning rate |
| `ST4_STRIDE` | 4 | stride window latih (per bar) |
| `ST4_TEST_STRIDE` | 4 | stride window test 2023 |
| `ST4_HELD_STRIDE` | 8 | stride window simbol held-out |
| `CKPT` | `ckpt/best.pt` (fallback lokal) | path checkpoint pretraining |

Output lokal kini masuk ke `results/finetune/` (bobot `*.pt`, `*.npy`, `*.csv`
di-ignore; `metrics.json` tetap di-commit).

### Cara menjalankan ulang di lokal (Windows, CPU)

```powershell
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install scikit-learn pandas matplotlib

python download_local.py     # 359 MB, ±20 detik
python test_tokenizer.py     # wajib PASSED
python stage2_tokenizer.py   # ±21 menit
python verify_ckpt.py        # wajib ALIGNED sebelum fine-tune

# konfigurasi CPU kecil (±40 menit untuk 2 epoch di 4 thread; 1 epoch ±23 menit)
$env:ST4_STRIDE="256"; $env:ST4_TEST_STRIDE="64"; $env:ST4_HELD_STRIDE="128"
$env:ST4_EPOCHS="2"; $env:ST4_BATCH="16"; $env:ST4_LR="0.0003"
python -u stage4_finetune.py

# backtest biaya-nyata (pakai env ST4_* yang sama)
python -u backtest_stage4.py
```

### Langkah berikutnya

1. **Fine-tune penuh di GPU (Colab T4)** dengan konfigurasi asli
   (`ST4_EPOCHS=3 ST4_STRIDE=4 ST4_BATCH=256`) — 15.087+ window, hasil
   resmi yang bisa dibandingkan dengan baseline GRU/MLP.
2. Uji transfer lintas-aset: fine-tune leave-one-symbol-out, atau tambah
   simbol held-out ke pretraining lalu ukur lagi.
3. ~~Backtest biaya-nyata (cost-aware) dari prediksi arah~~ — **sudah ada**:
   `backtest_stage4.py` (lihat tabel di atas); tinggal dijalankan ulang dengan
   konfigurasi GPU penuh.
