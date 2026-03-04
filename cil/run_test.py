#!/usr/bin/env python3
"""
Run 28 configs (1 baseline + 27 GoE grid), 3 times each; then generate summary.csv.
Windows-friendly: pathlib, forward slashes for Hydra, no reliance on os.path for joins.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# L1 configs only, in 03032025_uneven_cifar100/ (self-contained, new options enabled). 9 configs.
CONFIG_PATH = "configs/class"
CONFIG_NAMES = [
    "03032025_uneven_cifar100/GoE-L1-H512-HeadNone",
    "03032025_uneven_cifar100/GoE-L1-H512-Head512",
    "03032025_uneven_cifar100/GoE-L1-H512-Head512_256",
    "03032025_uneven_cifar100/GoE-L1-H768-HeadNone",
    "03032025_uneven_cifar100/GoE-L1-H768-Head512",
    "03032025_uneven_cifar100/GoE-L1-H768-Head512_256",
    "03032025_uneven_cifar100/GoE-L1-H1024-HeadNone",
    "03032025_uneven_cifar100/GoE-L1-H1024-Head512",
    "03032025_uneven_cifar100/GoE-L1-H1024-Head512_256",
]
NUM_RUNS = 3

# Resolve paths relative to script dir (works from any cwd; Windows-safe)
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH_ABS = SCRIPT_DIR / CONFIG_PATH.replace("\\", "/")


def path_for_hydra(p: Path) -> str:
    """Forward slashes for Hydra (avoids Windows backslash issues)."""
    return p.as_posix()


def is_oom(stderr: str) -> bool:
    if not stderr:
        return False
    s = stderr.lower()
    return "out of memory" in s or "outofmemoryerror" in s or "cuda out of memory" in s


def read_last_avg(metrics_path: Path):
    """Read last_acc and avg_acc from metrics.json (last line with 'last' and 'avg')."""
    try:
        text = metrics_path.read_text(encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None, None
        last_line = json.loads(lines[-1])
        if "last" in last_line and "avg" in last_line:
            return float(last_line["last"]), float(last_line["avg"])
        return None, None
    except Exception:
        return None, None


def write_summary_csv(run_dir: Path, runs: list, out_path: Path):
    """runs = [(config_name, run_id, exp_dir), ...]. Write CSV: per-run rows + per-config avg rows."""
    rows = []
    for config_name, run_id, exp_dir in runs:
        metrics_path = Path(exp_dir) / "metrics.json"
        last_acc, avg_acc = read_last_avg(metrics_path)
        rows.append({
            "config_name": config_name,
            "run_id": run_id,
            "last_acc": "" if last_acc is None else f"{last_acc:.2f}",
            "avg_acc": "" if avg_acc is None else f"{avg_acc:.2f}",
        })
    # Per-config averages (same config_name, 3 runs)
    by_config = defaultdict(list)
    for r in rows:
        if r["last_acc"] != "" and r["avg_acc"] != "":
            by_config[r["config_name"]].append((float(r["last_acc"]), float(r["avg_acc"])))
    for config_name in sorted(by_config.keys()):
        vals = by_config[config_name]
        mean_last = sum(x[0] for x in vals) / len(vals)
        mean_avg = sum(x[1] for x in vals) / len(vals)
        rows.append({
            "config_name": config_name,
            "run_id": "avg",
            "last_acc": f"{mean_last:.2f}",
            "avg_acc": f"{mean_avg:.2f}",
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config_name", "run_id", "last_acc", "avg_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")


def main():
    run_start = datetime.now().strftime("%m%d%Y-%H%M%S")  # mmddyyyy-HHMMSS
    results_dir = SCRIPT_DIR / "experiments" / run_start
    # Track (config_name, run_id, exp_dir) for summary
    completed_runs = []

    print("==========================================")
    print(f"{len(CONFIG_NAMES)} configs x {NUM_RUNS} runs = {len(CONFIG_NAMES) * NUM_RUNS} total")
    print("==========================================")
    print(f"Results: {path_for_hydra(results_dir)}")
    print("==========================================")
    print()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    for config_name in CONFIG_NAMES:
        print(f"--- Config: {config_name} ---")
        for i in range(1, NUM_RUNS + 1):
            exp_ts = datetime.now().strftime("%m%d%Y-%H%M%S")  # mmddyyyy-HHMMSS
            # Dir name = config_name with slash replaced (e.g. 02052026_uneven_cifar100_baseline)
            safe_name = config_name.replace("/", "_")
            exp_dir = SCRIPT_DIR / "experiments" / run_start / f"{safe_name}-run{i}-{exp_ts}"
            exp_dir_str = path_for_hydra(exp_dir)
            config_path_str = path_for_hydra(CONFIG_PATH_ABS)

            print(f"Run {i}/{NUM_RUNS}: {config_name} (batch_size=32)")

            main_py = SCRIPT_DIR / "main.py"
            cmd_base = [
                sys.executable,
                os.fspath(main_py),
                "--config-path", config_path_str,
                "--config-name", config_name,
                f"hydra.run.dir={exp_dir_str}",
            ]
            try:
                err_fd = tempfile.NamedTemporaryFile(mode="w", suffix=".err", delete=False)
                err_path = err_fd.name
                err_fd.close()
                stderr_content = ""
                try:
                    with open(err_path, "w", encoding="utf-8", errors="replace") as err_file:
                        ret = subprocess.run(
                            cmd_base,
                            cwd=os.fspath(SCRIPT_DIR),
                            env=env,
                            stdout=None,
                            stderr=err_file,
                        )
                finally:
                    if os.path.exists(err_path):
                        with open(err_path, "r", encoding="utf-8", errors="replace") as f:
                            stderr_content = f.read()
                        os.remove(err_path)

                if ret.returncode == 0:
                    completed_runs.append((config_name, i, exp_dir))
                    print(f"Run {i}/{NUM_RUNS} completed: {config_name}")
                    print()
                    continue
                if is_oom(stderr_content):
                    print("OOM detected; retrying with batch_size=12 ...")
                    cmd_retry = cmd_base + ["batch_size=12"]
                    err_fd2 = tempfile.NamedTemporaryFile(mode="w", suffix=".err", delete=False)
                    err_path2 = err_fd2.name
                    err_fd2.close()
                    try:
                        with open(err_path2, "w", encoding="utf-8", errors="replace") as err_file2:
                            ret2 = subprocess.run(
                                cmd_retry,
                                cwd=os.fspath(SCRIPT_DIR),
                                env=env,
                                stdout=None,
                                stderr=err_file2,
                            )
                    finally:
                        if os.path.exists(err_path2):
                            os.remove(err_path2)
                    if ret2.returncode == 0:
                        completed_runs.append((config_name, i, exp_dir))
                        print(f"Run {i}/{NUM_RUNS} completed: {config_name} (batch_size=12)")
                        print()
                        continue
                print(f"Failed: {config_name} run {i}")
                if stderr_content.strip():
                    for line in stderr_content.strip().splitlines()[-30:]:
                        print(line)
                continue
            except Exception as e:
                print(f"Failed: {config_name} run {i} ({e})")
                continue
        print()

    # Summary CSV: all runs + avg rows per config
    summary_path = results_dir / "summary.csv"
    write_summary_csv(results_dir, completed_runs, summary_path)

    print("==========================================")
    print("Summary")
    print("==========================================")
    print(f"Completed: {len(completed_runs)} / {len(CONFIG_NAMES) * NUM_RUNS}")
    print(f"Results: {path_for_hydra(results_dir)}")
    print(f"CSV: {path_for_hydra(summary_path)}")
    print("==========================================")

    sys.exit(0 if len(completed_runs) == len(CONFIG_NAMES) * NUM_RUNS else 1)


if __name__ == "__main__":
    main()
