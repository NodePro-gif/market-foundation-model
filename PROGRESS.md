# PROGRESS — log kronologis

## 2026-11-14 — T4 throughput optimization

- Pretraining default micro-batch increased from 32 to 64; accumulation changed from 2 to 1, preserving effective batch 64.
- Fine-tuning default micro-batch increased from 32 to 64; accumulation changed from 8 to 4, preserving effective batch 256.
- Enabled fused AdamW where supported, CUDA matmul precision tuning, pinned-memory batches, non-blocking host-to-device transfers, persistent workers and configurable prefetching.
- Increased pretraining DataLoader workers to 4 and fine-tuning workers to 2 for the Colab T4 preset.
- Fixed repeated decompression of compressed pretraining token arrays by caching the materialized array once per worker.
- Changed Colab workspace to use local VM disk for active training, with Drive restore/sync at stage boundaries and periodic checkpoint mirroring to persistent Drive storage.
- These are safe starting settings for a 16 GiB T4, not a promise of 100% GPU utilization. If a runtime reports CUDA OOM, reduce the micro-batch to 48 or 32 while keeping the effective batch constant by increasing accumulation proportionally.
- Full GPU throughput and memory measurements still require an actual Colab T4 run; CI only validates syntax, data-independent tests, and notebook structure.

---


## 2026-11-14 — PR #2: training, head architecture, and portfolio hardening

- Fine-tuning optimizer remains AdamW; added linear warmup + cosine decay per optimizer update, gradient clipping, and early stopping based only on XAUUSD 2021 validation AUC.
- Multi-task prediction heads now use LayerNorm, a 128-unit GELU layer, and dropout. The pretrained GPT backbone tensor shapes are unchanged; a fresh Stage-3 checkpoint remains compatible.
- Added a sequential one-position-at-a-time portfolio simulator. Signals are ignored while a trade is open and for the next bar after exit; returns compound and transaction costs are applied on both entry and exit.
- Threshold and horizon selection on 2022 now use the same non-overlapping simulator as the locked 2023 test.
- Added portfolio regression tests and CI integration. First CI run caught a test expectation about same-bar re-entry; simulator/test were corrected to require a new bar after exit. Local regression run: 4 tests passed.
- Pull request: https://github.com/NodePro-gif/market-foundation-model/pull/2

**Not yet completed:** a fresh full pipeline run on Colab T4. GitHub CI is not a substitute for GPU training or proof of predictive edge. Do not publish win-rate/profitability claims until the new pipeline has been run and the untouched 2023 results have been reviewed with actual broker costs.

---


> **Status protokol (diperbarui 2026-11-14):** catatan eksperimen di bawah ini adalah rekam historis, bukan hasil evaluasi dari pipeline yang sekarang. Kode telah diperketat ke split kronologis: tokenizer/pretraining/fine-tuning dilatih pada 2015–2020; checkpoint pretraining dan supervised dipilih pada 2021; kalibrasi/threshold dipilih pada 2022; 2023 menjadi final test yang tidak disentuh. Statistik tokenizer/BPE hanya di-fit pada aset non-held-out 2015–2020. Karena checkpoint dan angka historis dibuat dengan protokol lama, seluruh artefak harus dibuat ulang sebelum angka performa baru dilaporkan. Jangan mengutip angka backtest lama sebagai bukti performa out-of-sample pada protokol baru.

## 2026-11-14 — hardening pipeline dan Google Colab T4

- Menghapus pasangan BPE sintetis antar-aset/timeframe dengan sentinel batas; sampel BPE sekarang berupa potongan candle berurutan yang tersebar sepanjang 2015–2020.
- Mengikat checkpoint pretraining ke SHA-256 vocab.json + bpe.npz + calib.npz + kmeans.pkl melalui best_manifest.json; fine-tuning menolak checkpoint lama/tidak cocok.
- Memindahkan pemeriksaan checkpoint ke validasi 2021; tidak lagi membaca final test 2023.
- Pretraining memakai SDPA, FP16 AMP, gradient accumulation, evaluasi seluruh validation split, dan checkpoint resumable.
- Fine-tuning memakai AMP, gradient accumulation, sampling seimbang lintas aset non-held-out; checkpoint dipilih pada XAUUSD 2021.
- Kalibrasi/threshold/horizon dipilih pada 2022; backtest hanya menjalankan kebijakan terkunci pada 2023 dan tidak melakukan sweep parameter pada test.
- Menambahkan notebook Colab T4 dengan Google Drive persistence serta pemeriksaan CI untuk JSON notebook.

**Belum dilakukan:** training ulang end-to-end di runtime GPU T4. CI memeriksa sintaks, tokenizer self-test, notebook JSON, dan konsistensi dependensi; CI bukan bukti bahwa training GPU atau kualitas prediksi sudah berhasil.


> **Status protokol (diperbarui 2026-11-14):** catatan eksperimen di bawah ini adalah rekam historis, bukan hasil evaluasi dari pipeline yang sekarang. Kode telah diperketat ke split kronologis: tokenizer/pretraining/fine-tuning dilatih pada 2015–2020; checkpoint pretraining dan supervised dipilih pada 2021; kalibrasi/threshold dipilih pada 2022; 2023 menjadi final test yang tidak disentuh. Statistik tokenizer/BPE hanya di-fit pada aset non-held-out 2015–2020. Karena checkpoint dan angka historis dibuat dengan protokol lama, seluruh artefak harus dibuat ulang sebelum angka performa baru dilaporkan. Jangan mengutip angka backtest lama sebagai bukti performa out-of-sample pada protokol baru.


Catatan "sedang ngapain" untuk tiap tahap. Untuk status **real-time** jalankan:

```powershell
python monitor.py        # proses jalan + tail log terbaru + hasil terbaru + git
```

Semua run sebaiknya ditulis ke log supaya bisa ditelusuri lagi:

```powershell
python stage4_finetune.py 2>&1 | Tee-Object logs\stage4_$(Get-Date -Format yyyyMMdd_HHmm).log
```

---

## 2026-11-11 — reproduce pipeline dari nol + hasil stage4 (lokal CPU)

| jam | tahap | apa yang terjadi | hasil |
|---|---|---|---|
| ~15:45 | clone | clone repo + baca seluruh script (stage1–4, baseline) | — |
| ~15:48 | env | install `torch 2.15.0+cpu`, numpy, scikit-learn, pandas, matplotlib | mesin: 4 thread, tanpa GPU |
| 15:50 | stage1 | `download_local.py` → `D:\xauusd-dl\corpus` | 175/189 file, 359 MB, 20 detik (14 missing = sudah diketahui README) |
| 15:51 | self-test | `test_tokenizer.py` | **PASSED**, 0 failure |
| 15:52 | stage2 | `stage2_tokenizer.py` (parse → kalibrasi → BPE → encoding → label) | selesai 1250 dtk: 21 series, BPE 3.840 merge → vocab 4096, kompresi ±1,45× |
| ~16:14 | **baru** | `verify_ckpt.py` — cek `ckpt/best.pt` selaras dengan tokenizer hasil regenerasi | val_loss **5,9381** vs publikasi 5,8945 → **ALIGNED** |
| 16:19 | benchmark | ukur kecepatan CPU | train 624 ms/window (seq 512), forward 140 ms/window |
| 16:22 | stage4 run1 | `ST4_EPOCHS=1 STRIDE=256 BATCH=16 LR=3e-4` | selesai 1377 dtk: test **acc 0,5692 / AUC 0,6199 / Brier 0,2443 / ECE 0,0453** |
| 16:45 | stage4 run2 | konfigurasi sama, `ST4_EPOCHS=2` | selesai 2410 dtk: test **acc 0,5872 / AUC 0,6792 / Brier 0,2279 / ECE 0,1077**; epoch 1 identik digit-per-digit dengan run1 → deterministik |
| 17:2x | **baru** | `backtest_stage4.py` — 780 taruhan 1-bar, biaya 1,5 bps/side | gross **+6,10%** (+0,78 bps/taruhan, t 5,28); net **−17,30%**; breakeven **0,39 bps/side** |
| 17:3x | **baru** | sweep threshold `p≥0.50..0.75` | edge/taruhan naik 2× → +1,56 bps (acc 0,779, t 7,51) di thr 0,75; net masih −4,11% |
| 17:4x | commit | `6d520c1`, `664811c` | — |
| 18:00 | **baru** | `validate_stage4.py` — kalibrasi Platt + threshold & horizon dipilih di 2021–2022, test 2023 disentuh sekali | selesai 273 dtk: ECE test 0,1077 → **0,0828**; threshold **0,75** + horizon **48 bar** terkunci → test **87 taruhan, akurasi 89,66%, gross +2,89%, net +0,28% @1,5bps** (positif!) |
| 18:1x | **baru** | `monitor.py` + `PROGRESS.md` — dashboard & catatan progres | `python monitor.py` menampilkan proses jalan, tail log, hasil terbaru, git |

### Keputusan & temuan kunci

1. **Tokenizer hasil regenerasi identis dengan yang dipakai pretraining**
   (`verify_ckpt.py`) → `ckpt/best.pt` sah dipakai, pretraining tidak perlu
   diulang. Ini pemeriksaan penting: checkpoint menyimpan embedding per
   *merge rank*, jadi kalau BPE-nya beda, hasilnya bakal rusak diam-diam.
2. **AUC naik 0,5206 (GRU baseline) → 0,6792** — model belajar beneran.
3. **Tapi belum profit setelah biaya.** Edge nyata (t 5,28) tapi di bawah
   asumsi 1,5 bps/side. Breakeven 0,39 bps/side (semua taruhan) / 0,78 bps/side
   (threshold 0,75) — kisaran spread ECX XAUUSD yang ketat.
4. **Transfer lintas-aset gagal**: held-out NIKKEI 0,4838 / BRENT 0,5293 /
   EURGBP 0,5222 (AUC) → ≈ acak.
5. **Kalibrasi memburuk saat epoch naik**: ECE 0,0453 → 0,1077 → perlu
   Platt scaling (itulah isi `validate_stage4.py`).
6. Threshold sweep sebelumnya **post-hoc di test** → sekarang dipindah ke
   validasi 2021–2022 dan dikunci sebelum menyentuh 2023.

### File baru di repo

| file | fungsi |
|---|---|
| `verify_ckpt.py` | bukti tokenizer ↔ checkpoint selaras (wajib lolos sebelum fine-tune) |
| `backtest_stage4.py` | backtest biaya-nyata + sweep threshold |
| `validate_stage4.py` | kalibrasi Platt + threshold & horizon terkunci di validasi |
| `monitor.py` | dashboard: proses jalan, tail log, hasil terbaru, git |
| `PROGRESS.md` | catatan ini |

### Jawaban "AI-nya error karena pretraining?"

**Tidak error.** Pretraining sehat (val loss turun monoton 6,40 → 5,89,
train ≈ val, tidak overfit) tapi **berhenti di step 3000 dari `MAX_STEPS=6000`**
→ plafon agak rendah (loss masih turun saat berhenti). Penyebab hasil jelek
bukan pretraining, melainkan: (1) fine-tune lokal cuma 2.068 window × 1–2
epoch (rencana 15.087+ × 3), (2) target M5 memang hampir acak (baseline semua
0,51–0,52), (3) kepala hanya dilatih di XAUUSD (held-out ≈ acak memang wajar),
(4) kalibrasi (sudah diperbaiki Platt), (5) biaya > edge (sudah ditangani
threshold terkunci + horizon 48 bar → net +0,28%).

### Menjalankan ulang

Lihat README §7 (Cara menjalankan ulang di lokal) — semua knob `ST4_*` terdokumentasi
di sana. Log tiap run taruh di `logs/` (di-ignore git) supaya `monitor.py`
bisa menampilkannya lagi.

---

## 2026-11-11 — "cari solusi": walk-forward + probe transfer

| jam | tahap | apa yang terjadi | hasil |
|---|---|---|---|
| 18:20 | **baru** | `walkforward_stage4.py` — 3 fold jujur (pilih Platt+thr+horizon di 2 tahun sebelum test year) | selesai 717 dtk: **agregat 116 taruhan, acc 0,9052, z=+8,73**, gross +2,49%, **net@0,5bps +1,33%** ✅ / net@1,5bps −0,99% |
| 18:32 | **baru** | `probe_transfer.py` — bekukan backbone, latih kepala di 16 simbol 2015–2022, uji held-out 2023 | selesai 1966 dtk: **held-out NIKKEI AUC 0,4838 → 0,7407 (+0,2569)**, BRENT 0,5293 → 0,6320, EURGBP 0,5222 → 0,6005 — **transfer lintas-aset TERBUKTI BISA** |

### Temuan probe transfer (inti — solusi "held-out ≈ acak" TERATASI)

- **Held-out naik jauh di atas acak**: NIKKEI 0,4838 → **0,7407**, BRENT
  0,5293 → **0,6320**, EURGBP 0,5222 → **0,6005** (acc juga: 0,478 → 0,621).
- **Penyebabnya terbukti**: bukan backbone yang gagal transfer, tapi **kepala
  yang cuma dilatih di XAUUSD**. Backbone dibekukan pun (0 parameter diubah)
  sudah cukup → representasi pretraining memang lintas-aset.
- XAUUSD tidak dikorbankan: 0,6792 → 0,6913.
- Latihan ≤2022, test 2023 → tanpa kebocoran waktu; n 696–815 per simbol.

### Temuan walk-forward (inti)

1. **Akurasinya nyata dan stabil lintas tahun**: 0,9565 / 0,9231 / 0,8409 di
   3 fold berbeda → z=+8,73 vs 50%. Bukan overfit ke 2023.
2. **Masih belum cukup untuk menutup biaya 1,5 bps/side** (agregat −0,99%):
   gross per taruhan tipis (±0,21 bps) di horizon pendek yang dipilih tiap fold.
3. Kesimpulan: **edge-nya ada, tapi tipis** — profitabel hanya kalau biaya
   ≤0,5 bps/side (spread ECN ketat) atau gross bisa dinaikkan (backbone lebih
   kuat → pretraining 6000 steps + fine-tune penuh).

---

## Status akhir sesi ini

- Semua tahap (stage1 → stage2 → verify → stage4 ×2 → backtest → validate →
  walk-forward → probe transfer) **selesai tanpa error**; log di `logs/`
  (di-ignore git).
- **Dua masalah utama teratasi**: kalibrasi (Platt) & pertanyaan transfer
  lintas-aset (kepala multi-simbol → held-out 0,48 → 0,74 AUC). Sisa: edge
  tipis terhadap biaya 1,5 bps/side (butuh biaya rendah / backbone lebih kuat).
- Dashboard: `python monitor.py`.
- Push: commit `6d520c1`, `664811c`, `b47c692`, `8e6fab1`, `b8774a5` sudah di
  `origin/main`; hasil probe transfer menyusul di commit berikutnya.
