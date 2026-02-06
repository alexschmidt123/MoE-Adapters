#!/usr/bin/env python3
"""
Run new GNN structure optimization: 64 configs (4×4×4), each runs 3 times (192 total experiments)
Windows-compatible version of run_test.sh

Configs: graph_num_layers × graph_hidden_dim × graph_head_layers
  - graph_num_layers: 1, 2, 3, 4
  - graph_hidden_dim: 256, 512, 768, 1024
  - graph_head_layers: None, [256], [512, 256], [256, 128]
"""

import os
import sys
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Try to import colorama for Windows color support (optional)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORS = True
except ImportError:
    # Fallback if colorama not available
    class Fore:
        GREEN = BLUE = YELLOW = CYAN = RED = ''
    class Style:
        RESET_ALL = ''
    HAS_COLORS = False

# Configuration
CONFIG_PATH = "configs/class/new_cifar_configs"
DATASET_ROOT = "../datasets/"
CLASS_ORDER = "class_orders/cifar100.yaml"
NUM_RUNS = 3

# Get script directory and resolve paths relative to it
SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_DIR = SCRIPT_DIR / CONFIG_PATH
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"


def print_colored(text: str, color: str = ''):
    """Print colored text if colors are available"""
    if HAS_COLORS:
        print(f"{color}{text}{Style.RESET_ALL}")
    else:
        print(text)


def clear_gpu_memory():
    """Clear GPU memory cache"""
    print_colored("Clearing GPU memory...", Fore.CYAN)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass  # PyTorch not available, skip
    except Exception:
        pass  # Ignore errors
    time.sleep(2)  # Give GPU time to free memory


def create_exp_dir_name(config_name: str, timestamp: str) -> str:
    """
    Create experiment directory name from config name.
    Format: cifar100_2-2-{numlayer}-{hiddendim}-{headlayer}-{timestamp}
    
    Args:
        config_name: Config filename (e.g., "cifar100_2-2-MoE-Adapters-N4-GoE-L2-H512-HeadNone.yaml")
        timestamp: Timestamp string (e.g., "01262026-123456")
    
    Returns:
        Experiment directory name
    """
    # Remove .yaml extension
    base_name = config_name.replace('.yaml', '')
    
    # Extract L{num}-H{hidden}-Head{head} pattern
    match = re.search(r'L(\d+)-H(\d+)-Head(.+)', base_name)
    if match:
        num_layer = match.group(1)
        hidden_dim = match.group(2)
        head_layer = match.group(3)
        return f"cifar100_2-2-{num_layer}-{hidden_dim}-{head_layer}-{timestamp}"
    else:
        return f"{base_name}-{timestamp}"


def extract_config_info(config_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract config parameters from filename.
    
    Returns:
        Tuple of (num_layers, hidden_dim, head_layers) or (None, None, None) if not found
    """
    match = re.search(r'L(\d+)-H(\d+)-Head(.+)\.yaml', config_name)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def load_configs() -> List[str]:
    """Load all YAML config files from the config directory"""
    if not CONFIG_DIR.exists():
        print_colored(f"Error: Config directory does not exist: {CONFIG_DIR}", Fore.RED)
        sys.exit(1)
    
    configs = sorted([f.name for f in CONFIG_DIR.glob("*.yaml")])
    
    if not configs:
        print_colored(f"Error: No config files found in {CONFIG_DIR}", Fore.RED)
        sys.exit(1)
    
    return configs


def run_experiment(
    config_name: str,
    run_num: int,
    total_runs: int,
    run_start_timestamp: str
) -> bool:
    """
    Run a single experiment.
    
    Returns:
        True if successful, False otherwise
    """
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    print_colored("=" * 40, Fore.BLUE)
    print_colored(f"Running: {config_name}", Fore.BLUE)
    print_colored(f"Run: {run_num} / {total_runs}", Fore.BLUE)
    print_colored("=" * 40, Fore.BLUE)
    
    # Generate timestamp for this specific experiment
    exp_timestamp = datetime.now().strftime("%m%d%Y-%H%M%S")
    
    # Create experiment directory name
    exp_dir_name = create_exp_dir_name(config_name, exp_timestamp)
    
    # Build experiment directory path
    exp_dir = EXPERIMENTS_DIR / run_start_timestamp / exp_dir_name
    
    # Build command
    # Use absolute paths to avoid issues
    config_path_abs = str(CONFIG_DIR)
    dataset_root_abs = str(SCRIPT_DIR / DATASET_ROOT)
    class_order_abs = str(SCRIPT_DIR / CLASS_ORDER)
    exp_dir_abs = str(exp_dir)
    
    # Build Python command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "main.py",
        "--config-path", config_path_abs,
        "--config-name", config_name,
        f"hydra.run.dir={exp_dir_abs}",
        f"dataset_root={dataset_root_abs}",
        f"class_order={class_order_abs}"
    ]
    
    # Run experiment
    try:
        # Change to script directory to run main.py
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            check=False,
            capture_output=False  # Show output in real-time
        )
        
        success = (result.returncode == 0)
        
    except Exception as e:
        print_colored(f"Error running experiment: {e}", Fore.RED)
        success = False
    
    # Clear GPU memory after completion
    clear_gpu_memory()
    
    if success:
        print_colored(f"✓ Run {run_num}/{total_runs} completed successfully: {config_name}", Fore.GREEN)
    else:
        print_colored(f"✗ Run {run_num}/{total_runs} failed: {config_name}", Fore.RED)
    
    return success


def generate_summary(run_start_timestamp: str):
    """Generate results summary using generate_results_summary.py"""
    run_folder = EXPERIMENTS_DIR / run_start_timestamp
    
    if not run_folder.exists():
        print_colored(f"Warning: Run folder does not exist: {run_folder}", Fore.YELLOW)
        return False
    
    print_colored("Generating results summary...", Fore.CYAN)
    
    try:
        summary_script = SCRIPT_DIR / "generate_results_summary.py"
        result = subprocess.run(
            [sys.executable, str(summary_script), str(run_folder)],
            cwd=str(SCRIPT_DIR),
            check=False,
            capture_output=True
        )
        
        if result.returncode == 0:
            print_colored("✓ Results summary generated successfully", Fore.GREEN)
            if result.stdout:
                print(result.stdout.decode('utf-8', errors='ignore'))
            return True
        else:
            print_colored("⚠ Could not generate summary", Fore.YELLOW)
            if result.stderr:
                print(result.stderr.decode('utf-8', errors='ignore'))
            return False
    except Exception as e:
        print_colored(f"Error generating summary: {e}", Fore.YELLOW)
        return False


def main():
    """Main execution function"""
    # Load configs
    configs = load_configs()
    
    # Print header
    print("=" * 40)
    print("New GNN Structure Optimization Test Suite")
    print("=" * 40)
    print(f"Configs: {len(configs)} (4×4×4 = 64 configs)")
    print("  graph_num_layers: 1, 2, 3, 4")
    print("  graph_hidden_dim: 256, 512, 768, 1024")
    print("  graph_head_layers: None, [256], [512, 256], [256, 128]")
    print()
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Total experiments: {len(configs) * NUM_RUNS}")
    print("=" * 40)
    print()
    
    # Generate run start timestamp
    run_start_timestamp = datetime.now().strftime("%m%d%Y-%H%M%S")
    print(f"Results will be saved to: experiments/{run_start_timestamp}/")
    print()
    
    # Create experiments directory if it doesn't exist
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    (EXPERIMENTS_DIR / run_start_timestamp).mkdir(exist_ok=True)
    
    # Track statistics
    total_experiments = len(configs) * NUM_RUNS
    successful = 0
    failed = 0
    failed_configs = []
    
    # Run all configs
    print_colored("=" * 40, Fore.CYAN)
    print_colored("Starting New GNN Structure Optimization", Fore.CYAN)
    print_colored("=" * 40, Fore.CYAN)
    print()
    
    for config in configs:
        # Extract config info for display
        num_layers, hidden_dim, head_layers = extract_config_info(config)
        
        if num_layers and hidden_dim and head_layers:
            variant = f"Layers={num_layers}, Hidden={hidden_dim}, Head={head_layers}"
        else:
            variant = "Unknown config"
        
        print_colored(f"--- Config: {variant} ---", Fore.GREEN)
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
    print("=" * 40)
    print("Test Suite Summary")
    print("=" * 40)
    print(f"Total experiments: {total_experiments}")
    print_colored(f"Successful: {successful}", Fore.GREEN)
    
    if failed > 0:
        print_colored(f"Failed: {failed}", Fore.RED)
        print()
        print("Failed configs:")
        for failed_config in failed_configs:
            print_colored(f"  - {failed_config}", Fore.RED)
    
    print("=" * 40)
    print()
    print(f"All results saved to: experiments/{run_start_timestamp}/")
    print()
    
    # Generate results summary
    generate_summary(run_start_timestamp)
    print()
    
    # Exit with appropriate code
    if failed == 0:
        print_colored("All tests passed!", Fore.GREEN)
        sys.exit(0)
    else:
        print_colored("Some tests failed.", Fore.YELLOW)
        sys.exit(1)


if __name__ == "__main__":
    main()
