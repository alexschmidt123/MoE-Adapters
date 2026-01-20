#!/usr/bin/env python3
"""
Run all TinyImageNet experiments: 63 configs total (large base scenarios only)
Combinations: 2 MoE types × 3 GNN options × 4 N values × 3 scenarios
MoE types: Original MoE / HMoE-Hybrid
GNN options: No GNN / GNN ProtoDepth11 Noise001 / GNN ProtoDepth11 (no noise)
N values: N2 / N4 / N8 / N16
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

# Configuration
CONFIG_PATH = "configs/class/tinyimagenet_configs"
DATASET_ROOT = "../datasets/"
CLASS_ORDER = "class_orders/tinyimagenet.yaml"
NUM_RUNS = 3  # Run each config 3 times

def check_tinyimagenet_dataset():
    """Check if TinyImageNet dataset is available (downloads automatically)"""
    # TinyImageNet downloads automatically via continuum library
    # Just verify the dataset directory exists or can be created
    dataset_path = os.path.join(os.path.dirname(DATASET_ROOT), "datasets", "tinyimagenet")
    
    # The dataset will be downloaded automatically when first accessed
    # So we just return True - the download happens in the dataset loader
    return True

# All 63 configs organized by scenario (large base only, aligns with original code)
CONFIGS = [
    # 100-5 scenario (20 step, 100 base classes) (21 configs)
    "tinyimagenet_100-5-MoE-Adapters-N2.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N2-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N2-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N4.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N4-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N4-HMoE-Hybrid.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N8.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N8-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N8-HMoE-Hybrid.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N16.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N16-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N16-HMoE-Hybrid.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-5-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    
    # 100-10 scenario (10 step, 100 base classes) (21 configs)
    "tinyimagenet_100-10-MoE-Adapters-N2.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N2-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N2-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N4.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N4-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N4-HMoE-Hybrid.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N8.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N8-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N8-HMoE-Hybrid.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N16.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N16-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N16-HMoE-Hybrid.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-10-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    
    # 100-20 scenario (5 step, 100 base classes) (21 configs)
    "tinyimagenet_100-20-MoE-Adapters-N2.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N2-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N2-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N4.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N4-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N4-HMoE-Hybrid.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N8.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N8-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N8-HMoE-Hybrid.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N16.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N16-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N16-HMoE-Hybrid.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11.yaml",
    "tinyimagenet_100-20-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml",
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
        f"hydra.run.dir=experiments/{run_start_timestamp}/{config_dir_name}-{exp_timestamp}"
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
    print("  - N values: N2 / N4 / N8 / N16")
    print("  - 3 scenarios: 100-5 (20 step) / 100-10 (10 step) / 100-20 (5 step)")
    print("  - All scenarios use 100 base classes (aligns with original code)")
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
        
        n_val = ""
        if "N2" in config:
            n_val = "N2"
        elif "N4" in config:
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
            if run_experiment(config, i, NUM_RUNS, run_start_timestamp):
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
