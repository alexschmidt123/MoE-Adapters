#!/usr/bin/env python3
"""
Run uneven CIFAR-100 configs, each 3 times.
Same logic as run_test.sh, Windows-friendly (pathlib, subprocess).
Uses POSIX-style paths for Hydra to avoid Windows backslash issues.
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configs: same as run_test.sh
CONFIG_PATH = "configs/class"
CONFIG_NAMES = [
    "uneven_cifar100/cifar100_uneven10-MoE-Adapters-N4",
    "uneven_cifar100/cifar100_uneven10-MoE-Adapters-N4-GoE",
]
NUM_RUNS = 3

# Resolve paths relative to script directory (works from any cwd on Windows/Linux)
SCRIPT_DIR = Path(__file__).resolve().parent
# Use forward slashes so Path joins correctly on Windows (Path treats both / and \)
CONFIG_PATH_ABS = SCRIPT_DIR / CONFIG_PATH.replace("\\", "/")


def _path_for_hydra(p: Path) -> str:
    """Path string safe for Hydra on Windows: use forward slashes."""
    return p.as_posix()


def main():
    run_start = datetime.now().strftime("%m%d%Y-%H%M%S")
    results_dir = SCRIPT_DIR / "experiments" / run_start

    print("==========================================")
    print(f"Uneven CIFAR-100 test: {len(CONFIG_NAMES)} configs x {NUM_RUNS} runs")
    print("==========================================")
    print("Configs:", " ".join(CONFIG_NAMES))
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Results: {results_dir}")
    print("==========================================")
    print()

    successful = 0
    failed = 0
    failed_list = []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    for config_name in CONFIG_NAMES:
        print(f"--- Config: {config_name} ---")
        for i in range(1, NUM_RUNS + 1):
            exp_ts = datetime.now().strftime("%m%d%Y-%H%M%S")
            # Build path with forward slashes (Path handles them on Windows)
            exp_dir = SCRIPT_DIR / "experiments" / run_start / f"{config_name}-run{i}-{exp_ts}"
            # Use POSIX-style path for Hydra to avoid Windows backslash/escape issues
            exp_dir_str = _path_for_hydra(exp_dir)
            config_path_str = _path_for_hydra(CONFIG_PATH_ABS)

            print(f"Run {i}/{NUM_RUNS}: {config_name}")

            main_py = SCRIPT_DIR / "main.py"
            cmd = [
                sys.executable,
                os.fspath(main_py),
                "--config-path", config_path_str,
                "--config-name", f"{config_name}.yaml",
                f"hydra.run.dir={exp_dir_str}",
            ]
            try:
                # cwd: use native path for subprocess on this platform
                ret = subprocess.run(
                    cmd,
                    cwd=os.fspath(SCRIPT_DIR),
                    env=env,
                )
                if ret.returncode != 0:
                    failed += 1
                    failed_list.append(f"{config_name} run {i}")
                    print(f"Failed: {config_name} run {i}")
                    continue
            except Exception as e:
                failed += 1
                failed_list.append(f"{config_name} run {i}")
                print(f"Failed: {config_name} run {i} ({e})")
                continue
            successful += 1
            print(f"Run {i}/{NUM_RUNS} completed: {config_name}")
            print()
        print()

    print("==========================================")
    print("Summary")
    print("==========================================")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_list:
        print("Failed:")
        for item in failed_list:
            print(f"  - {item}")
    print(f"Results: {results_dir}")
    print("==========================================")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
