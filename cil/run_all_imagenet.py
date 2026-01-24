#!/usr/bin/env python3
"""
Run all TinyImageNet experiments: 72 configs total (large base scenarios only)
Combinations: 2 MoE types × 3 GNN options × 4 N values × 3 scenarios
MoE types: Original MoE / HMoE-Hybrid
GNN options: No GNN / GNN ProtoDepth11 (no noise) / GNN ProtoDepth11 Noise001
N values: N4 / N8 / N16 / N32
Scenarios: 100-5 (20 step) / 100-10 (10 step) / 100-20 (5 step)
All scenarios use 100 base classes (aligns with original code)

Cross-platform Python version - works on Windows, Linux, and macOS
TinyImageNet downloads automatically (no manual download needed)
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Import summary generator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_results_summary import generate_summary, save_summary

# Get script directory for cross-platform path resolution
SCRIPT_DIR = Path(__file__).parent.absolute()

# Configuration (paths relative to script directory for Hydra, absolute for file checks)
CONFIG_PATH_REL = "configs/class/tinyimagenet_configs"  # Relative path for Hydra
CONFIG_PATH_ABS = str(SCRIPT_DIR / "configs" / "class" / "tinyimagenet_configs")  # Absolute for file checks
DATASET_ROOT = str(SCRIPT_DIR.parent / "datasets")
CLASS_ORDER = str(SCRIPT_DIR / "class_orders" / "tinyimagenet.yaml")
NUM_RUNS = 3  # Run each config 3 times

def check_tinyimagenet_dataset():
    """Check if TinyImageNet dataset is available (downloads automatically)"""
    # TinyImageNet downloads automatically via continuum library
    # Just verify the dataset directory exists or can be created
    dataset_path = SCRIPT_DIR.parent / "datasets" / "tinyimagenet"
    
    # The dataset will be downloaded automatically when first accessed
    # So we just return True - the download happens in the dataset loader
    return True

def config_file_exists(config_name: str) -> bool:
    config_file = Path(CONFIG_PATH_ABS) / config_name
    return config_file.exists()


def build_configs():
    scenarios = ["100-5", "100-10", "100-20"]
    n_values = [4, 8, 16, 32]
    variants = [
        {"suffix": "", "moe_type": "Original MoE", "gnn_type": "No GNN"},
        {"suffix": "-GoE-ProtoDepth11", "moe_type": "Original MoE", "gnn_type": "GNN ProtoDepth11"},
        {"suffix": "-GoE-ProtoDepth11-Noise001", "moe_type": "Original MoE", "gnn_type": "GNN ProtoDepth11 Noise001"},
        {"suffix": "-HMoE-Hybrid", "moe_type": "HMoE-Hybrid", "gnn_type": "No GNN"},
        {"suffix": "-HMoE-Hybrid-GoE-ProtoDepth11", "moe_type": "HMoE-Hybrid", "gnn_type": "GNN ProtoDepth11"},
        {"suffix": "-HMoE-Hybrid-GoE-ProtoDepth11-Noise001", "moe_type": "HMoE-Hybrid", "gnn_type": "GNN ProtoDepth11 Noise001"},
    ]

    configs = []
    for scenario in scenarios:
        for n_val in n_values:
            n_tag = f"N{n_val}"
            for variant in variants:
                config_name = f"tinyimagenet_{scenario}-MoE-Adapters-{n_tag}{variant['suffix']}.yaml"
                run_name = config_name.replace(".yaml", "")
                extra_args = []

                if not config_file_exists(config_name):
                    if n_val == 32:
                        fallback_name = config_name.replace("N32", "N16")
                        if not config_file_exists(fallback_name):
                            raise FileNotFoundError(f"Missing config: {fallback_name}")
                        config_name = fallback_name
                        method_name = run_name.replace(f"tinyimagenet_{scenario}-", "").replace(".yaml", "")
                        extra_args.extend([
                            "model.num_experts=32",
                            f"method={method_name}",
                        ])
                    else:
                        raise FileNotFoundError(f"Missing config: {config_name}")

                configs.append({
                    "config_name": config_name,
                    "run_name": run_name,
                    "scenario": scenario,
                    "n_val": n_tag,
                    "moe_type": variant["moe_type"],
                    "gnn_type": variant["gnn_type"],
                    "extra_args": extra_args,
                })
    return configs


CONFIGS = build_configs()

# ANSI color codes (works on Windows 10+ with ANSI support, or use colorama)
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def clear_gpu_memory():
    """Clear GPU memory cache"""
    print(f"{Colors.CYAN}Clearing GPU memory...{Colors.NC}")
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except:
        pass
    time.sleep(2)  # Give GPU time to free memory

def run_experiment(config_name, run_name, extra_args, run_num, total_runs, run_start_timestamp):
    """Run a single experiment"""
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    print(f"{Colors.BLUE}{'='*40}{Colors.NC}")
    print(f"{Colors.BLUE}Running: {config_name}{Colors.NC}")
    print(f"{Colors.BLUE}Run: {run_num} / {total_runs}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*40}{Colors.NC}")
    
    # Remove .yaml extension from config name for directory name
    config_dir_name = run_name
    
    # Generate timestamp for this specific experiment
    exp_timestamp = datetime.now().strftime("%m%d%Y-%H%M%S")
    
    # Build command with new save path: experiments/<run_start_timestamp>/<config-name>-<timestamp>/
    # Use relative paths since we'll run from SCRIPT_DIR
    experiments_dir = str(SCRIPT_DIR / "experiments" / run_start_timestamp)
    run_dir = str(Path(experiments_dir) / f"{config_dir_name}-{exp_timestamp}")
    
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "main.py",  # Relative path since we run from SCRIPT_DIR
        "--config-path", CONFIG_PATH_REL,
        "--config-name", config_name,
        f"dataset_root={DATASET_ROOT}",
        f"class_order={CLASS_ORDER}",
        f"hydra.run.dir={run_dir}"
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # TQDM_DISABLE is not set, so progress bars will be shown
    env["PYTHONUNBUFFERED"] = "1"  # Unbuffered output
    
    # Run experiment with proper output handling
    # Change to script directory to ensure relative paths work correctly
    exit_code = subprocess.call(
        cmd, 
        cwd=str(SCRIPT_DIR),  # Run from script directory
        env=env,
        stdout=sys.stdout,  # Keep stdout for important messages
        stderr=sys.stderr   # Keep stderr for errors
    )
    
    # Clear GPU memory after completion
    clear_gpu_memory()
    
    if exit_code == 0:
        print(f"{Colors.GREEN}✓ Run {run_num}/{total_runs} completed successfully: {config_name}{Colors.NC}")
        return True
    else:
        print(f"{Colors.RED}✗ Run {run_num}/{total_runs} failed: {config_name} (exit code: {exit_code}){Colors.NC}")
        return False

def main():
    # Generate run start timestamp (format: MMDDYYYY-HHMMSS)
    run_start_timestamp = datetime.now().strftime("%m%d%Y-%H%M%S")
    
    # Check dataset before starting (TinyImageNet downloads automatically)
    if not check_tinyimagenet_dataset():
        print("Exiting: TinyImageNet dataset check failed.")
        return 1
    
    print()
    print("=" * 50)
    print("TinyImageNet Comprehensive Test Suite (Large Base Only)")
    print("=" * 50)
    print(f"Total configs: {len(CONFIGS)}")
    print("  - 2 MoE types: Original MoE / HMoE-Hybrid")
    print("  - 3 GNN options: No GNN / GNN ProtoDepth11 (no noise) / GNN ProtoDepth11 Noise001")
    print("  - N values: N4 / N8 / N16 / N32")
    print("  - 3 scenarios: 100-5 (20 step) / 100-10 (10 step) / 100-20 (5 step)")
    print("  - All scenarios use 100 base classes (aligns with original code)")
    print()
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Total experiments: {len(CONFIGS) * NUM_RUNS}")
    results_dir = str(SCRIPT_DIR / "experiments" / run_start_timestamp)
    print(f"Results will be saved to: {results_dir}")
    print("=" * 50)
    print()
    
    # Track statistics
    total_experiments = len(CONFIGS) * NUM_RUNS
    successful = 0
    failed = 0
    failed_configs = []
    
    # Run all configs
    print(f"{Colors.CYAN}{'='*50}{Colors.NC}")
    print(f"{Colors.CYAN}Starting TinyImageNet Experiments{Colors.NC}")
    print(f"{Colors.CYAN}{'='*50}{Colors.NC}")
    print()
    
    for config in CONFIGS:
        # Extract variant info from config name
        scenario = ""
        if "100-5" in config:
            scenario = "100-5 (20 step)"
        elif "100-10" in config:
            scenario = "100-10 (10 step)"
        elif "100-20" in config:
            scenario = "100-20 (5 step)"
        
        scenario_label = config["scenario"]
        if scenario_label == "100-5":
            scenario_label = "100-5 (20 step)"
        elif scenario_label == "100-10":
            scenario_label = "100-10 (10 step)"
        elif scenario_label == "100-20":
            scenario_label = "100-20 (5 step)"

        variant = f"{scenario_label} | {config['n_val']} | {config['moe_type']} | {config['gnn_type']}"
        
        print(f"{Colors.GREEN}--- Variant: {variant} ---{Colors.NC}")
        print()
        
        for i in range(1, NUM_RUNS + 1):
            if run_experiment(
                config["config_name"],
                config["run_name"],
                config["extra_args"],
                i,
                NUM_RUNS,
                run_start_timestamp
            ):
                successful += 1
            else:
                failed += 1
                failed_configs.append(f"{config['run_name']} (run {i})")
            print()  # Add blank line between runs
        print()
    
    # Summary
    print()
    print("=" * 50)
    print("Test Suite Summary")
    print("=" * 50)
    print(f"Total experiments: {total_experiments}")
    print(f"{Colors.GREEN}Successful: {successful}{Colors.NC}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.NC}")
        print()
        print("Failed configs:")
        for failed_config in failed_configs:
            print(f"{Colors.RED}  - {failed_config}{Colors.NC}")
    print("=" * 50)
    run_folder = str(SCRIPT_DIR / "experiments" / run_start_timestamp)
    print(f"\n{Colors.CYAN}All results saved to: {run_folder}{Colors.NC}\n")
    
    # Generate results summary
    if Path(run_folder).exists():
        print(f"{Colors.CYAN}Generating results summary...{Colors.NC}")
        try:
            summary = generate_summary(run_folder)
            if summary:
                save_summary(run_folder, summary)
                print(f"{Colors.GREEN}✓ Results summary generated successfully{Colors.NC}")
            else:
                print(f"{Colors.YELLOW}⚠ No results found to summarize{Colors.NC}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Could not generate summary: {e}{Colors.NC}")
    
    if failed == 0:
        print(f"{Colors.GREEN}All tests passed!{Colors.NC}")
        return 0
    else:
        print(f"{Colors.YELLOW}Some tests failed.{Colors.NC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
