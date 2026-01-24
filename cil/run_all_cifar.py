#!/usr/bin/env python3
"""
Run all CIFAR-100 experiments: 72 configs total
Combinations: 2 MoE types × 3 GNN options × 4 N values × 3 scenarios
MoE types: Original MoE / HMoE-Hybrid
GNN options: No GNN / GNN ProtoDepth11 / GNN ProtoDepth11 Noise001
N values: N4 / N8 / N16 / N32
Scenarios: 2*2 / 5*5 / 10*10

Cross-platform Python version - works on Windows, Linux, and macOS
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

# Configuration
CONFIG_PATH = "configs/class/cifar_configs"
DATASET_ROOT = "../datasets/"
CLASS_ORDER = "class_orders/cifar100.yaml"
NUM_RUNS = 3  # Run each config 3 times

def config_file_exists(config_name: str) -> bool:
    return (Path(CONFIG_PATH) / config_name).exists()


def build_configs():
    scenarios = ["2-2", "5-5", "10-10"]
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
                config_name = f"cifar100_{scenario}-MoE-Adapters-{n_tag}{variant['suffix']}.yaml"
                run_name = config_name
                extra_args = []

                if not config_file_exists(config_name):
                    if n_val == 32:
                        fallback_name = config_name.replace("N32", "N16")
                        if not config_file_exists(fallback_name):
                            raise FileNotFoundError(f"Missing config: {fallback_name}")
                        config_name = fallback_name
                        method_name = run_name.replace(f"cifar100_{scenario}-", "").replace(".yaml", "")
                        extra_args.extend([
                            "model.num_experts=32",
                            f"method={method_name}",
                        ])
                    else:
                        raise FileNotFoundError(f"Missing config: {config_name}")

                configs.append({
                    "config_name": config_name,
                    "run_name": run_name.replace(".yaml", ""),
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
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "main.py",
        "--config-path", CONFIG_PATH,
        "--config-name", config_name,
        f"dataset_root={DATASET_ROOT}",
        f"class_order={CLASS_ORDER}",
        f"hydra.run.dir=experiments/{run_start_timestamp}/{config_dir_name}-{exp_timestamp}"
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # TQDM_DISABLE is not set, so progress bars will be shown
    env["PYTHONUNBUFFERED"] = "1"  # Unbuffered output
    
    # Run experiment with proper output handling
    exit_code = subprocess.call(
        cmd, 
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
    
    print("=" * 50)
    print("CIFAR-100 Comprehensive Test Suite")
    print("=" * 50)
    print(f"Total configs: {len(CONFIGS)}")
    print("  - 2 MoE types: Original MoE / HMoE-Hybrid")
    print("  - 3 GNN options: No GNN / GNN ProtoDepth11 / GNN ProtoDepth11 Noise001")
    print("  - 4 N values: N4 / N8 / N16 / N32")
    print("  - 3 scenarios: 2*2 / 5*5 / 10*10")
    print()
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Total experiments: {len(CONFIGS) * NUM_RUNS}")
    print(f"Results will be saved to: experiments/{run_start_timestamp}/")
    print("=" * 50)
    print()
    
    # Track statistics
    total_experiments = len(CONFIGS) * NUM_RUNS
    successful = 0
    failed = 0
    failed_configs = []
    
    # Run all configs
    print(f"{Colors.CYAN}{'='*50}{Colors.NC}")
    print(f"{Colors.CYAN}Starting CIFAR-100 Experiments{Colors.NC}")
    print(f"{Colors.CYAN}{'='*50}{Colors.NC}")
    print()
    
    for config in CONFIGS:
        scenario = config["scenario"].replace("-", "*")
        variant = f"{scenario} | {config['n_val']} | {config['moe_type']} | {config['gnn_type']}"
        
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
                failed_configs.append(f"{config} (run {i})")
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
    
    print(f"\n{Colors.CYAN}All results saved to: experiments/{run_start_timestamp}/{Colors.NC}\n")
    
    # Generate results summary
    run_folder = f"experiments/{run_start_timestamp}"
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
