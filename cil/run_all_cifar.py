#!/usr/bin/env python3
"""
Run all CIFAR-100 experiments: 36 configs total
Combinations: 2 MoE types × 2 GNN options × 3 N values × 3 scenarios
MoE types: Original MoE / HMoE-Hybrid
GNN options: No GNN / GNN ProtoDepth11 Noise001
N values: N4 / N8 / N16
Scenarios: 2*2 / 5*5 / 10*10

Cross-platform Python version - works on Windows, Linux, and macOS
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Configuration
CONFIG_PATH = "configs/class/cifar_configs"
DATASET_ROOT = "../datasets/"
CLASS_ORDER = "class_orders/cifar100.yaml"
NUM_RUNS = 3  # Run each config 3 times

# All 36 configs organized by scenario
CONFIGS = [
    # 2*2 scenario (12 configs)
    "cifar100_2-2-MoE-Adapters-N4.yaml",
    "cifar100_2-2-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_2-2-MoE-Adapters-N4-HMoE-Hybrid.yaml",
    "cifar100_2-2-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_2-2-MoE-Adapters-N8.yaml",
    "cifar100_2-2-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_2-2-MoE-Adapters-N8-HMoE-Hybrid.yaml",
    "cifar100_2-2-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_2-2-MoE-Adapters-N16.yaml",
    "cifar100_2-2-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_2-2-MoE-Adapters-N16-HMoE-Hybrid.yaml",
    "cifar100_2-2-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    
    # 5*5 scenario (12 configs)
    "cifar100_5-5-MoE-Adapters-N4.yaml",
    "cifar100_5-5-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_5-5-MoE-Adapters-N4-HMoE-Hybrid.yaml",
    "cifar100_5-5-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_5-5-MoE-Adapters-N8.yaml",
    "cifar100_5-5-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_5-5-MoE-Adapters-N8-HMoE-Hybrid.yaml",
    "cifar100_5-5-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_5-5-MoE-Adapters-N16.yaml",
    "cifar100_5-5-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_5-5-MoE-Adapters-N16-HMoE-Hybrid.yaml",
    "cifar100_5-5-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    
    # 10*10 scenario (12 configs)
    "cifar100_10-10-MoE-Adapters-N4.yaml",
    "cifar100_10-10-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_10-10-MoE-Adapters-N4-HMoE-Hybrid.yaml",
    "cifar100_10-10-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_10-10-MoE-Adapters-N8.yaml",
    "cifar100_10-10-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_10-10-MoE-Adapters-N8-HMoE-Hybrid.yaml",
    "cifar100_10-10-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_10-10-MoE-Adapters-N16.yaml",
    "cifar100_10-10-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml",
    "cifar100_10-10-MoE-Adapters-N16-HMoE-Hybrid.yaml",
    "cifar100_10-10-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
]

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

def run_experiment(config_name, run_num, total_runs):
    """Run a single experiment"""
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    print(f"{Colors.BLUE}{'='*40}{Colors.NC}")
    print(f"{Colors.BLUE}Running: {config_name}{Colors.NC}")
    print(f"{Colors.BLUE}Run: {run_num} / {total_runs}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*40}{Colors.NC}")
    
    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "main.py",
        "--config-path", CONFIG_PATH,
        "--config-name", config_name,
        f"dataset_root={DATASET_ROOT}",
        f"class_order={CLASS_ORDER}"
    ]
    
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
    print("=" * 50)
    print("CIFAR-100 Comprehensive Test Suite")
    print("=" * 50)
    print(f"Total configs: {len(CONFIGS)}")
    print("  - 2 MoE types: Original MoE / HMoE-Hybrid")
    print("  - 2 GNN options: No GNN / GNN ProtoDepth11 Noise001")
    print("  - 3 N values: N4 / N8 / N16")
    print("  - 3 scenarios: 2*2 / 5*5 / 10*10")
    print()
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Total experiments: {len(CONFIGS) * NUM_RUNS}")
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
        # Extract variant info from config name
        scenario = ""
        if "2-2" in config:
            scenario = "2*2"
        elif "5-5" in config:
            scenario = "5*5"
        elif "10-10" in config:
            scenario = "10*10"
        
        n_val = ""
        if "N4" in config:
            n_val = "N4"
        elif "N8" in config:
            n_val = "N8"
        elif "N16" in config:
            n_val = "N16"
        
        moe_type = "Original MoE"
        if "HMoE-Hybrid" in config:
            moe_type = "HMoE-Hybrid"
        
        gnn_type = "No GNN"
        if "GoE-ProtoDepth11-Noise001" in config:
            gnn_type = "GNN ProtoDepth11 Noise001"
        
        variant = f"{scenario} | {n_val} | {moe_type} | {gnn_type}"
        
        print(f"{Colors.GREEN}--- Variant: {variant} ---{Colors.NC}")
        print()
        
        for i in range(1, NUM_RUNS + 1):
            if run_experiment(config, i, NUM_RUNS):
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
    
    if failed == 0:
        print(f"{Colors.GREEN}All tests passed!{Colors.NC}")
        return 0
    else:
        print(f"{Colors.YELLOW}Some tests failed.{Colors.NC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
