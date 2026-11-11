#!/usr/bin/env python
"""monitor.py - see what the pipeline is doing right now.

Shows: running python processes (with command line), the tail of the newest
log in logs/, the latest numbers in results/finetune/*.json, and git state.

    python monitor.py            # 15 tail lines
    python monitor.py 40         # 40 tail lines
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "results", "finetune")
TAIL = int(sys.argv[1]) if len(sys.argv) > 1 else 15

PS_LIST = (
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "Sort-Object ProcessId | ForEach-Object { "
    "$c=($_.CommandLine -replace '\\s+',' '); "
    "if($c.Length -gt 110){$c=$c.Substring(0,110)+'...'}; "
    "'{0}|{1}|{2}' -f $_.ProcessId,$_.KernelModeTime,$c }"
)


def head(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def rel(p):
    return os.path.relpath(p, ROOT)


def newest(paths):
    paths = [p for p in paths if os.path.exists(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def main():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    head(f"PROGRESS MONITOR - {now}")

    # ---------------------------------------------------------- processes
    head("1. PROSES PYTHON BERJALAN")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", PS_LIST],
                             capture_output=True, text=True, timeout=30)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        if not lines:
            print("  (tidak ada proses python yang berjalan)")
        for l in lines:
            pid, ktime, cmd = (l.split("|", 2) + ["", ""])[:3]
            cpu = int(ktime) / 1e7 if ktime.isdigit() else 0.0
            print(f"  pid {pid:>6}  cpu {cpu:8.1f}s  {cmd}")
    except Exception as e:
        print("  gagal membaca proses:", e)

    # ----------------------------------------------------------------- logs
    head("2. LOG TERBARU (logs/)")
    log = newest([os.path.join(LOGS, f) for f in
                  (os.listdir(LOGS) if os.path.isdir(LOGS) else [])
                  if f.endswith((".log", ".txt"))])
    if log:
        with open(log, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        print(f"  {rel(log)}  ({len(lines)} baris, "
              f"diubah {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(log)))})")
        for l in lines[-TAIL:]:
            print("   |", l)
    else:
        print("  (belum ada log - jalankan run dengan: "
              "python stage4_finetune.py 2>&1 | Tee-Object logs\\stage4.log)")

    # ------------------------------------------------------- latest results
    head("3. HASIL TERBARU (results/finetune/)")
    m = os.path.join(OUT, "metrics.json")
    if os.path.exists(m):
        d = json.load(open(m))
        t = d.get("test", {})
        print(f"  stage4 metrics.json   acc={t.get('acc', 0):.4f} "
              f"auc={t.get('auc', 0):.4f} brier={t.get('brier', 0):.4f} "
              f"ece={t.get('ece', 0):.4f} n={t.get('n', 0)}")
        for sym, v in (d.get("held_out") or {}).items():
            print(f"    held-out {sym:8s} acc={v['acc']:.4f} "
                  f"auc={v['auc']:.4f} n={v['n']}")
    b = os.path.join(OUT, "backtest.json")
    if os.path.exists(b):
        d = json.load(open(b))
        sw = d.get("threshold_sweep") or []
        print(f"  backtest.json         bets={d.get('n_bets')} "
              f"breakeven={d.get('breakeven_cost_per_side_bps', 0):.2f} bps/side")
        if sw:
            best = max(sw, key=lambda s: s["mean_bps"])
            print(f"    sweep terbaik: thr={best['threshold']} "
                  f"n={best['n']} acc={best['acc']:.3f} "
                  f"gross={best['gross_total']*100:+.2f}% "
                  f"net@3bps={best['net_total']*100:+.2f}% "
                  f"t={best['t_stat']:.2f}")
    v = os.path.join(OUT, "validation.json")
    if os.path.exists(v):
        d = json.load(open(v))
        print(f"  validation.json       threshold terkunci={d.get('threshold')} "
              f"val_n={d.get('val_n')} horizon={d.get('chosen_horizon')}")
        print(f"    val  brier/ece      {d.get('val', {}).get('brier_ece')}"
              f" -> terkalibrasi {d.get('val_calibrated', {}).get('brier_ece')}")
        print(f"    test brier/ece      {d.get('test', {}).get('brier_ece')}"
              f" -> terkalibrasi {d.get('test', {}).get('brier_ece_calibrated')}")
        for h in d.get("horizons", []):
            nets = "  ".join(
                f"{float(k)*1e4:4.1f}bps={val*100:+.2f}%"
                for k, val in h["nets"].items())
            print(f"    horizon {h['horizon']:>2} bar  n={h['n']:>4} "
                  f"gross={h['gross_total']*100:+.2f}%  {nets}")
    if not any(os.path.exists(p) for p in (m, b, v)):
        print("  (belum ada hasil)")

    # ------------------------------------------------------------------ git
    head("4. GIT")
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%h %ad %s",
                                 "--date=format:%H:%M"], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        print("  HEAD:", commit.stdout.strip()[:110])
        st = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
        dirty = [l for l in st.stdout.splitlines() if l.strip()]
        print(f"  perubahan lokal: {len(dirty)} file")
        for l in dirty[:10]:
            print("   ", l)
    except Exception as e:
        print("  git:", e)
    print()


if __name__ == "__main__":
    main()
