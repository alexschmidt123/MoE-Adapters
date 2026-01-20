#!/usr/bin/env python3
"""
Run all new CIFAR-100 test experiments: 12 configs total
Combinations: 3 N values × 2 scenarios × 2 MoE types
N values: N4 / N8 / N16
Scenarios: 2*2 / 5*5
MoE types: MoE+GNN(ProtoDepth11) / HMoE-Hybrid+GNN(ProtoDepth11)
All configs use GNN with ProtoDepth=11 and NO noise

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
OUTPUT_DIR = "experiments/outputs"  # Output directory

# All 12 new configs
CONFIGS = [
    # 2*2 scenario - MoE+GNN (6 configs)
    "cifar100_2-2-MoE-Adapters-N4-GoE-ProtoDepth11.yaml",
    "cifar100_2-2-MoE-Adapters-N8-GoE-ProtoDepth11.yaml",
    "cifar100_2-2-MoE-Adapters-N16-GoE-ProtoDepth11.yaml",
    "cifar100_2-2-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "cifar100_2-2-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "cifar100_2-2-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    
    # 5*5 scenario - MoE+GNN (6 configs)
    "cifar100_5-5-MoE-Adapters-N4-GoE-ProtoDepth11.yaml",
    "cifar100_5-5-MoE-Adapters-N8-GoE-ProtoDepth11.yaml",
    "cifar100_5-5-MoE-Adapters-N16-GoE-ProtoDepth11.yaml",
    "cifar100_5-5-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "cifar100_5-5-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "cifar100_5-5-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
]

# ANSI color codes (for terminals that support them)
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def clear_gpu_memory():
    """Clear GPU memory cache"""
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"{Colors.CYAN}Clearing GPU memory...{Colors.NC}")
        time.sleep(2)  # Give GPU time to free memory
    except Exception:
        pass  # Ignore if torch is not available

def run_experiment(config_name, run_num, total_runs, run_start_timestamp):
    """Run a single experiment"""
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    print(f"{Colors.BLUE}{'='*40}{Colors.NC}")
    print(f"{Colors.BLUE}Running: {config_name}{Colors.NC}")
    print(f"{Colors.BLUE}Run: {run_num} / {total_runs}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*40}{Colors.NC}")
    
    # Remove .yaml extension from config name for directory name
    config_dir_name = config_name.replace('.yaml', '')
    
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
        f"hydra.run.dir=experiments/{run_start_timestamp}/{config_dir_name}-{exp_timestamp}",
        f"hydra.job.name={config_dir_name}_run{run_num}"
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
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

def extract_variant_info(config_name):
    """Extract variant information from config name"""
    scenario = ""
    if "2-2" in config_name:
        scenario = "2*2"
    elif "5-5" in config_name:
        scenario = "5*5"
    
    n_val = ""
    if "N4" in config_name:
        n_val = "N4"
    elif "N8" in config_name:
        n_val = "N8"
    elif "N16" in config_name:
        n_val = "N16"
    
    moe_type = "MoE+GNN"
    if "HMoE-Hybrid" in config_name:
        moe_type = "HMoE-Hybrid+GNN"
    
    return f"{scenario} | {n_val} | {moe_type} | ProtoDepth11 (No Noise)"

def main():
    print("=" * 50)
    print("CIFAR-100 Test Suite (New Configs)")
    print("=" * 50)
    print(f"Total configs: {len(CONFIGS)}")
    print("  - 3 N values: N4 / N8 / N16")
    print("  - 2 scenarios: 2*2 / 5*5")
    print("  - 2 MoE types: MoE+GNN / HMoE-Hybrid+GNN")
    print("  - All with ProtoDepth=11, NO noise")
    print("")
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Total experiments: {len(CONFIGS) * NUM_RUNS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)
    print("")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Track statistics
    total_experiments = len(CONFIGS) * NUM_RUNS
    successful = 0
    failed = 0
    failed_configs = []
    
    # Run all configs
    print(f"{Colors.CYAN}{'='*40}{Colors.NC}")
    print(f"{Colors.CYAN}Starting CIFAR-100 Test Experiments{Colors.NC}")
    print(f"{Colors.CYAN}{'='*40}{Colors.NC}")
    print("")
    
    for config in CONFIGS:
        variant = extract_variant_info(config)
        print(f"{Colors.GREEN}--- Variant: {variant} ---{Colors.NC}")
        print("")
        
        for i in range(1, NUM_RUNS + 1):
            if run_experiment(config, i, NUM_RUNS, run_start_timestamp):
                successful += 1
            else:
                failed += 1
                failed_configs.append(f"{config} (run {i})")
            print("")  # Add blank line between runs
        print("")
    
    # Summary
    print("")
    print("=" * 50)
    print("Test Suite Summary")
    print("=" * 50)
    print(f"Total experiments: {total_experiments}")
    print(f"{Colors.GREEN}Successful: {successful}{Colors.NC}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.NC}")
        print("")
        print("Failed configs:")
        for failed_config in failed_configs:
            print(f"{Colors.RED}  - {failed_config}{Colors.NC}")
    print("=" * 50)
    print("")
    print(f"All results saved to: experiments/{run_start_timestamp}/")
    print("")
    
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
