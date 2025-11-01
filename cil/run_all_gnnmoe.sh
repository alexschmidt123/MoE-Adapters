#!/bin/bash
# Run all Graph-over-Experts (GoE) MoE experiments
# This script runs N=2, N=4, and N=8 experiments sequentially

echo "=========================================="
echo "Running all GNN-MoE experiments"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to run an experiment and check for errors
run_experiment() {
    local script_name=$1
    local description=$2
    
    echo "----------------------------------------"
    echo "Running: $description"
    echo "Script: $script_name"
    echo "Started at: $(date)"
    echo "----------------------------------------"
    
    if bash "$script_name"; then
        echo ""
        echo "✓ Completed: $description"
        echo "Finished at: $(date)"
        echo ""
    else
        echo ""
        echo "✗ ERROR: Failed to run $description"
        echo "Aborting remaining experiments."
        echo ""
        exit 1
    fi
}

# Run N=2 experiment
run_experiment "run_cifar100-2-2-MoE-GoE-N2.sh" "GNN-MoE N=2"

# Run N=4 experiment
run_experiment "run_cifar100-2-2-MoE-GoE-N4.sh" "GNN-MoE N=4"

# Run N=8 experiment
run_experiment "run_cifar100-2-2-MoE-GoE-N8.sh" "GNN-MoE N=8"

echo "=========================================="
echo "All GNN-MoE experiments completed!"
echo "Finished at: $(date)"
echo "=========================================="

