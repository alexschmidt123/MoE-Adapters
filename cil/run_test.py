#!/usr/bin/env python3
"""
Run uneven CIFAR-100 configs, each 3 times.
Same logic as run_test.sh, Windows-friendly (pathlib, subprocess).
Uses POSIX-style paths for Hydra to avoid Windows backslash issues.
"""
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Configs: N8 only (2 configs × 3 runs)
CONFIG_PATH = "configs/class"
CONFIG_NAMES = [
    "uneven_cifar100/cifar100_uneven10-MoE-Adapters-N8",
    "uneven_cifar100/cifar100_uneven10-MoE-Adapters-N8-GoE",
]
NUM_RUNS = 3

# Resolve paths relative to script directory (works from any cwd on Windows/Linux)
SCRIPT_DIR = Path(__file__).resolve().parent
# Use forward slashes so Path joins correctly on Windows (Path treats both / and \)
CONFIG_PATH_ABS = SCRIPT_DIR / CONFIG_PATH.replace("\\", "/")


def _path_for_hydra(p: Path) -> str:
    """Path string safe for Hydra on Windows: use forward slashes."""
    return p.as_posix()


def _is_oom(stderr: str) -> bool:
    """True if stderr indicates CUDA/system out-of-memory."""
    if not stderr:
        return False
    s = stderr.lower()
    return "out of memory" in s or "outofmemoryerror" in s or "cuda out of memory" in s


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

            print(f"Run {i}/{NUM_RUNS}: {config_name} (batch_size=32)")

            main_py = SCRIPT_DIR / "main.py"
            cmd_base = [
                sys.executable,
                os.fspath(main_py),
                "--config-path", config_path_str,
                "--config-name", f"{config_name}.yaml",
                f"hydra.run.dir={exp_dir_str}",
            ]
            try:
                # First try: use config default (32). Don't capture stdout so output streams and run doesn't look stuck.
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
                    successful += 1
                    print(f"Run {i}/{NUM_RUNS} completed: {config_name}")
                    print()
                    continue
                # Failed: retry with batch_size=12 if OOM
                if _is_oom(stderr_content):
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
                        successful += 1
                        print(f"Run {i}/{NUM_RUNS} completed: {config_name} (batch_size=12)")
                        print()
                        continue
                # Still failed or non-OOM failure
                failed += 1
                failed_list.append(f"{config_name} run {i}")
                print(f"Failed: {config_name} run {i}")
                if stderr_content.strip():
                    print("Last 30 lines of stderr:")
                    for line in stderr_content.strip().splitlines()[-30:]:
                        print(line)
                continue
            except Exception as e:
                failed += 1
                failed_list.append(f"{config_name} run {i}")
                print(f"Failed: {config_name} run {i} ({e})")
                continue
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
