# PROGRESS — log kronologis

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

## Status akhir sesi ini

- Semua tahap (stage1 → stage2 → verify → stage4 ×2 → backtest → validate)
  **selesai tanpa error**; log tersimpan di `logs/` (di-ignore git).
- Dashboard: `python monitor.py`.
- Push: commit `6d520c1`, `664811c`, `b47c692` + commit hasil validasi ini
  sudah/akan dikirim ke `origin/main`.
