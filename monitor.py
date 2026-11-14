#!/usr/bin/env python
"""Portable monitor for local and Colab pipeline runs.

Usage:
    python monitor.py       # show 15 recent log lines
    python monitor.py 40    # show 40 recent log lines
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_COLAB = os.path.isdir("/content")
LOCAL_DATA_ROOT = os.environ.get(
    "MFM_DATA_ROOT",
    r"D:\xauusd-dl" if os.name == "nt"
    else os.path.join(os.path.expanduser("~"), "market-foundation-model-data"),
)
LOGS = os.path.join(ROOT, "logs")
OUT = os.environ.get(
    "MFM_FT_DIR",
    "/content/ftune" if IS_COLAB else os.path.join(ROOT, "results", "finetune"),
)
CKPT_DIR = os.environ.get(
    "MFM_CKPT_DIR",
    "/content/ckpt" if IS_COLAB else os.path.join(LOCAL_DATA_ROOT, "ckpt"),
)
TAIL = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def head(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def rel(path):
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def newest(paths):
    paths = [p for p in paths if os.path.exists(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def show_processes():
    if os.name == "nt":
        command = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Sort-Object ProcessId | ForEach-Object { "
            "$c=($_.CommandLine -replace '\\s+',' '); "
            "if($c.Length -gt 110){$c=$c.Substring(0,110)+'...'}; "
            "'{0}|{1}|{2}' -f $_.ProcessId,$_.KernelModeTime,$c }",
        ]
        out = subprocess.run(command, capture_output=True, text=True, timeout=30)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        if not lines:
            print("  (tidak ada proses python yang berjalan)")
        for line in lines:
            pid, ktime, cmd = (line.split("|", 2) + ["", ""])[:3]
            cpu = int(ktime) / 1e7 if ktime.isdigit() else 0.0
            print(f"  pid {pid:>6}  cpu {cpu:8.1f}s  {cmd}")
    else:
        out = subprocess.run(["ps", "-eo", "pid,etime,args"],
                             capture_output=True, text=True, timeout=20)
        lines = [line for line in out.stdout.splitlines()
                 if "python" in line.lower() and "monitor.py" not in line.lower()]
        for line in lines[:30]:
            print(" ", line.strip())
        if not lines:
            print("  (tidak ada proses python yang berjalan)")


def main():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    head(f"PROGRESS MONITOR - {now}")

    head("1. PROSES PYTHON BERJALAN")
    try:
        show_processes()
    except Exception as exc:
        print("  gagal membaca proses:", exc)

    head("2. LOG TERBARU (logs/)")
    log_path = newest([
        os.path.join(LOGS, name)
        for name in (os.listdir(LOGS) if os.path.isdir(LOGS) else [])
        if name.endswith((".log", ".txt"))
    ])
    if log_path:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        print(f"  {rel(log_path)} ({len(lines)} baris)")
        for line in lines[-TAIL:]:
            print("   |", line)
    else:
        print("  (belum ada log; simpan output training ke folder logs/)")

    head("3. CHECKPOINT PRETRAINING")
    manifest_path = os.path.join(CKPT_DIR, "best_manifest.json")
    log_path = os.path.join(CKPT_DIR, "pretrain_log.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        print("  best step:", manifest.get("global_step"),
              "validation loss:", manifest.get("best_validation_loss"))
        print("  tokenizer fingerprint:", str(manifest.get("tokenizer_fingerprint", ""))[:16])
    else:
        print("  (best checkpoint manifest belum tersedia)")
    if os.path.isfile(log_path):
        with open(log_path, encoding="utf-8") as fh:
            history = json.load(fh)
        if history:
            last = history[-1]
            best = min(history, key=lambda item: item.get("val_loss", float("inf")))
            print(f"  last eval: step={last.get('step')} val_loss={last.get('val_loss')}")
            print(f"  best logged: step={best.get('step')} val_loss={best.get('val_loss')}")

    head("4. HASIL FINE-TUNING DAN EVALUASI")
    metrics_path = os.path.join(OUT, "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        test = metrics.get("test", {})
        config = metrics.get("config", {})
        print(f"  stage4: acc={test.get('acc', float('nan')):.4f} "
              f"auc={test.get('auc', float('nan')):.4f} "
              f"brier={test.get('brier', float('nan')):.4f} "
              f"ece={test.get('ece', float('nan')):.4f} n={test.get('n', 0)}")
        print(f"  train assets: {len(config.get('train_symbols', []))}; "
              f"stride={config.get('train_stride')} batch={config.get('batch')} "
              f"accum={config.get('accumulation')} device={config.get('device')}")
        for sym, result in (metrics.get("held_out") or {}).items():
            print(f"    held-out {sym:8s} acc={result['acc']:.4f} "
                  f"auc={result['auc']:.4f} n={result['n']}")
    else:
        print("  (metrics.json belum tersedia)")

    validation_path = os.path.join(OUT, "validation.json")
    if os.path.isfile(validation_path):
        with open(validation_path, encoding="utf-8") as fh:
            validation = json.load(fh)
        print(f"  calibration: threshold={validation.get('threshold')} "
              f"horizon={validation.get('chosen_horizon')} "
              f"validation n={validation.get('val_n')}")
        print("  validation brier/ece:",
              validation.get("val", {}).get("brier_ece"),
              "-> calibrated:",
              validation.get("val_calibrated", {}).get("brier_ece"))
        print("  final-test brier/ece:",
              validation.get("test", {}).get("brier_ece"),
              "-> calibrated:",
              validation.get("test", {}).get("brier_ece_calibrated"))

    backtest_path = os.path.join(OUT, "backtest.json")
    if os.path.isfile(backtest_path):
        with open(backtest_path, encoding="utf-8") as fh:
            backtest = json.load(fh)
        policy = backtest.get("policy", {})
        strategy = backtest.get("strategy", {})
        print(f"  locked backtest: trades={backtest.get('trades')} "
              f"threshold={policy.get('threshold')} "
              f"horizon={policy.get('horizon_bars')} bars")
        print(f"    gross sum={strategy.get('gross_sum', 0):+.4%} "
              f"net sum={strategy.get('net_sum', 0):+.4%} "
              f"win-after-cost={strategy.get('win_rate_after_cost', float('nan')):.3f}")

    transfer_path = os.path.join(OUT, "transfer.json")
    if os.path.isfile(transfer_path):
        with open(transfer_path, encoding="utf-8") as fh:
            transfer = json.load(fh)
        print(f"  exploratory transfer probe: "
              f"{len(transfer.get('train', {}).get('symbols', []))} training assets; "
              f"{transfer.get('train', {}).get('windows')} windows")
        for sym, result in (transfer.get("held_out") or {}).items():
            print(f"    {sym:8s} acc={result['acc']:.4f} "
                  f"auc={result['auc']:.4f} n={result['n']}")

    head("5. GIT")
    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad %s", "--date=format:%H:%M"],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        print("  HEAD:", commit.stdout.strip()[:110])
        status = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        dirty = [line for line in status.stdout.splitlines() if line.strip()]
        print(f"  perubahan lokal: {len(dirty)} file")
        for line in dirty[:10]:
            print("   ", line)
    except Exception as exc:
        print("  git:", exc)
    print()


if __name__ == "__main__":
    main()
