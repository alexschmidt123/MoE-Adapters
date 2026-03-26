#!/usr/bin/env python3
"""
Windows-friendly runner with the same behavior as run.sh.

Usage:
  python run.py <config_file_path> [epochs]
  python run.py -directory <folder> [-times B]

Before each experiment, ImageNet-100/200/500 configs trigger
``scripts/prepare_imagenet_subsets.ensure_imagenet_subsets_from_full_data()`` when full ImageNet
is installed and subset assets are missing (see that module's docstring).
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH_BASE = Path("configs/class")

# Ensure cil/ and scripts/ are importable.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
_scripts = SCRIPT_DIR / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from prepare_imagenet_subsets import (  # noqa: E402
    config_needs_imagenet_subsets,
    ensure_imagenet_subsets_from_full_data,
)


def to_hydra_path(path: Path) -> str:
    """Hydra handles forward slashes reliably across OSes."""
    return path.as_posix()


def now_ts() -> str:
    return datetime.now().strftime("%m%d%Y-%H%M%S")


def detect_dataset(config_name: str) -> str:
    name = config_name.lower()
    if "cifar100" in name:
        return "cifar100"
    if "food101" in name:
        return "food101"
    if "tinyimagenet" in name:
        return "tinyimagenet"
    if "imagenet" in name:
        return "imagenet"
    print("Warning: Could not detect dataset from config name, defaulting to cifar100")
    return "cifar100"


def write_summary(run_start_timestamp: str) -> None:
    run_dir = SCRIPT_DIR / "experiments" / run_start_timestamp
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_run_summary.py"),
        to_hydra_path(run_dir),
    ]
    subprocess.run(cmd, cwd=str(SCRIPT_DIR))


def resolve_config_path_name(config_file: Path) -> tuple[str, str]:
    """
    Map file path to Hydra --config-path and --config-name.
    Example:
      configs/class/03122026_uneven_cifar100/MoE-N8.yaml
      -> (configs/class, 03122026_uneven_cifar100/MoE-N8)
    """
    config_file = config_file.resolve()
    try:
        rel = config_file.relative_to(SCRIPT_DIR)
    except ValueError:
        raise ValueError(f"Config must be inside project: {config_file}")

    if rel.suffix.lower() != ".yaml":
        raise ValueError(f"Config must be a .yaml file: {rel}")

    config_dir = rel.parent
    config_base = rel.stem
    class_dir = CONFIG_PATH_BASE

    if config_dir == class_dir:
        return to_hydra_path(config_dir), config_base

    # Nested under configs/class/<subdir>/...
    if len(config_dir.parts) >= 3 and config_dir.parts[:2] == class_dir.parts:
        sub = Path(*config_dir.parts[2:])
        return to_hydra_path(class_dir), f"{to_hydra_path(sub)}/{config_base}"

    return to_hydra_path(config_dir), config_base


def run_one(
    config_file: Path,
    epochs: Optional[str],
    parent_ts: Optional[str],
    run_index: int = 1,
    write_summary_csv: bool = False,
) -> int:
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return 1

    config_path, config_name = resolve_config_path_name(config_file)
    dataset = detect_dataset(config_name)

    safe_config_name = config_name.replace("/", "_")
    run_start_timestamp = parent_ts or now_ts()
    exp_timestamp = now_ts()
    out_dir = (
        f"experiments/{run_start_timestamp}/"
        f"{safe_config_name}-run{run_index}-{exp_timestamp}"
    )

    print("==========================================")
    print(f"Running experiment: {config_name}")
    print(f"Config file: {to_hydra_path(config_file)}")
    print(f"Dataset: {dataset}")
    if epochs is not None:
        print(f"Epochs override: {epochs}")
    print("==========================================")
    print(f"Results will be saved to: {out_dir}/")
    print("")

    if config_needs_imagenet_subsets(config_name):
        try:
            ensure_imagenet_subsets_from_full_data()
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "main.py"),
        "--config-path",
        config_path,
        "--config-name",
        config_name,
        f"hydra.run.dir={out_dir}",
    ]
    if epochs is not None:
        cmd.append(f"epochs={epochs}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    code = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env).returncode

    print("==========================================")
    print(f"Experiment completed: {config_name}")
    print("==========================================")
    if write_summary_csv:
        write_summary(run_start_timestamp)
    return code


def run_directory(folder: str, times: int) -> int:
    dir_full = SCRIPT_DIR / CONFIG_PATH_BASE / folder
    if not dir_full.is_dir():
        print(f"Error: Directory not found: {to_hydra_path(dir_full)}")
        return 1

    configs = sorted(dir_full.glob("*.yaml"))
    if not configs:
        print(f"Error: No .yaml configs in {to_hydra_path(dir_full)}")
        return 1

    run_start = now_ts()
    total = len(configs) * times

    print("==========================================")
    print(f"Running {len(configs)} configs x {times} runs = {total} total")
    print(f"Directory: {folder}")
    print(f"Results: experiments/{run_start}/")
    print("==========================================")

    final_code = 0
    for cfg in configs:
        for run_idx in range(1, times + 1):
            print("")
            code = run_one(
                cfg,
                None,
                run_start,
                run_index=run_idx,
                write_summary_csv=False,
            )
            if code != 0:
                final_code = code

    write_summary(run_start)

    print("==========================================")
    print(f"Done. Results: experiments/{run_start}/")
    print(f"CSV: experiments/{run_start}/summary.csv")
    print("==========================================")
    return final_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run experiments from one config or a whole directory."
    )
    parser.add_argument("config_file", nargs="?", help="Path to a config .yaml file")
    parser.add_argument("epochs", nargs="?", help="Optional epochs override")
    parser.add_argument("-directory", dest="directory", default=None, help="Folder under configs/class")
    parser.add_argument("-times", dest="times", type=int, default=3, help="Runs per config (default: 3)")
    args = parser.parse_args()

    if args.directory:
        return run_directory(args.directory, args.times)

    if not args.config_file:
        print("Error: No config file or -directory provided")
        print("Usage: python run.py <config_file_path> [epochs]")
        print("       python run.py -directory <folder> [-times B]")
        print("Example: python run.py configs/class/03122026_uneven_cifar100/MoE-N8.yaml")
        print("Example: python run.py -directory 03122026_uneven_cifar100 -times 3")
        return 1

    return run_one(
        SCRIPT_DIR / args.config_file,
        args.epochs,
        None,
        run_index=1,
        write_summary_csv=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
