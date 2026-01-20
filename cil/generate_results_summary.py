#!/usr/bin/env python3
"""
Generate results_summary.json for a batch run.
Collects all metrics.json files from experiments in a run folder and creates a summary.
"""

import os
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def extract_config_key(experiment_dir_name: str) -> str:
    """
    Extract a config key from experiment directory name.
    Examples:
    - cifar100_2-2-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001-01182026-100001
      -> N4-GoE-Noise001
    - cifar100_2-2-MoE-Adapters-N4-GoE-ProtoDepth11-01182026-100001
      -> N4-GoE-NoNoise
    - cifar100_2-2-MoE-Adapters-N8-HMoE-Hybrid-01182026-100045
      -> N8-HMoE-Hybrid
    - cifar100_2-2-MoE-Adapters-N4-01182026-100123
      -> N4-Baseline
    """
    # Remove timestamp suffix (format: MMDDYYYY-HHMMSS)
    # Timestamp is typically the last 2 parts separated by dash
    parts = experiment_dir_name.split('-')
    
    # Find N value (e.g., N2, N4, N8, N16)
    n_val = ""
    for part in parts:
        if part.startswith('N') and len(part) > 1 and part[1:].isdigit():
            n_val = part
            break
    
    if not n_val:
        # Fallback: use directory name without timestamp
        return experiment_dir_name.rsplit('-', 2)[0] if '-' in experiment_dir_name else experiment_dir_name
    
    # Check for HMoE-Hybrid
    has_hmoe = 'HMoE' in experiment_dir_name or 'Hybrid' in experiment_dir_name
    
    # Check for GoE and noise settings
    has_goe = 'GoE' in experiment_dir_name
    has_noise = 'Noise001' in experiment_dir_name
    has_protodepth = 'ProtoDepth11' in experiment_dir_name or 'ProtoDepth' in experiment_dir_name
    
    # Build key
    key_parts = [n_val]
    
    if has_hmoe:
        key_parts.append("HMoE-Hybrid")
        if has_goe:
            if has_noise:
                key_parts.append("GoE-Noise001")
            elif has_protodepth:
                key_parts.append("GoE-NoNoise")
            else:
                key_parts.append("GoE")
    else:
        # Not HMoE
        if has_goe:
            if has_noise:
                key_parts.append("GoE-Noise001")
            elif has_protodepth:
                key_parts.append("GoE-NoNoise")
            else:
                key_parts.append("GoE")
        else:
            # No GoE, no HMoE = Baseline
            key_parts.append("Baseline")
    
    return "-".join(key_parts)


def read_metrics_file(metrics_path: Path) -> Dict[str, Any]:
    """Read metrics.json file and extract final results"""
    try:
        with open(metrics_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            return None
        
        # Last line contains final summary
        final_line = json.loads(lines[-1])
        
        # Check if it's the final summary (has 'last' and 'avg')
        if 'last' in final_line and 'avg' in final_line:
            return {
                'final_acc': final_line['last'],
                'avg_acc': final_line['avg']
            }
        else:
            # If no final summary, use last task's metrics
            if len(lines) > 0:
                last_task = json.loads(lines[-1])
                if 'acc' in last_task and 'avg_acc' in last_task:
                    return {
                        'final_acc': last_task['acc'],
                        'avg_acc': last_task['avg_acc']
                    }
        
        return None
    except Exception as e:
        print(f"  Warning: Could not read {metrics_path}: {e}")
        return None


def calculate_statistics(values: List[float]) -> Dict[str, Any]:
    """Calculate mean, std, min, max for a list of values"""
    if not values:
        return {}
    
    return {
        'mean': round(statistics.mean(values), 2),
        'std': round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'values': [round(v, 2) for v in values]
    }


def generate_summary(run_folder: str) -> Dict[str, Any]:
    """
    Generate results summary for all experiments in a run folder.
    
    Args:
        run_folder: Path to the run folder (e.g., "experiments/01182026-100000")
    
    Returns:
        Dictionary with summary statistics grouped by config key
    """
    run_path = Path(run_folder)
    
    if not run_path.exists():
        print(f"Error: Run folder does not exist: {run_folder}")
        return {}
    
    # Group experiments by config key
    config_groups = defaultdict(lambda: {'final_acc': [], 'avg_acc': []})
    
    # Find all experiment directories
    for exp_dir in run_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        # Extract config key
        config_key = extract_config_key(exp_dir.name)
        
        # Read metrics
        metrics = read_metrics_file(metrics_file)
        if metrics:
            config_groups[config_key]['final_acc'].append(metrics['final_acc'])
            config_groups[config_key]['avg_acc'].append(metrics['avg_acc'])
    
    # Build summary
    summary = {}
    for config_key, values in config_groups.items():
        summary[config_key] = {
            'count': len(values['final_acc']),
            'final_acc': calculate_statistics(values['final_acc']),
            'avg_acc': calculate_statistics(values['avg_acc'])
        }
    
    return summary


def save_summary(run_folder: str, summary: Dict[str, Any], output_file: str = "results_summary.json"):
    """Save summary to JSON file in run folder"""
    run_path = Path(run_folder)
    output_path = run_path / output_file
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: {output_path}")
    return str(output_path)


def main():
    """Command-line interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_results_summary.py <run_folder>")
        print("Example: python generate_results_summary.py experiments/01182026-100000")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    summary = generate_summary(run_folder)
    
    if summary:
        save_summary(run_folder, summary)
        print(f"\nSummary generated for {len(summary)} config groups:")
        for key, data in summary.items():
            print(f"  {key}: {data['count']} experiments")
    else:
        print(f"No results found in {run_folder}")


if __name__ == "__main__":
    main()
