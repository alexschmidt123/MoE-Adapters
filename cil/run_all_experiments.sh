#!/bin/bash
# Master script to run all three CIFAR-100 experiments for comparison
# 1. Baseline N=2, k=2 (original)
# 2. Baseline N=4, k=2 (fair comparison)
# 3. GoE N=4, k=2 (graph-enhanced)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the cil directory
cd "$SCRIPT_DIR"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "Running Complete Experimental Suite for Graph-over-Experts Evaluation"
echo "========================================================================"
echo ""

# ============================================================================
# Experiment 1: Original Baseline (N=2, k=2, no graph)
# ============================================================================
echo -e "${BLUE}[1/3] Starting Experiment 1: Original Baseline (N=2, k=2, no graph)${NC}"
echo "------------------------------------------------------------------------"
echo "This establishes the original baseline with 2 experts."
echo ""

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

echo ""
echo -e "${GREEN}✓ Experiment 1 completed!${NC}"
echo "Results saved to: experiments/class/cifar100_2-2-MoE-Adapters-0.0/"
echo ""
sleep 2

# ============================================================================
# Experiment 2: Fair Baseline (N=4, k=2, no graph)
# ============================================================================
echo -e "${BLUE}[2/3] Starting Experiment 2: Fair Baseline (N=4, k=2, no graph)${NC}"
echo "------------------------------------------------------------------------"
echo "This runs standard MoE with 4 experts (same as GoE but no graph)."
echo ""

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-N4.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

echo ""
echo -e "${GREEN}✓ Experiment 2 completed!${NC}"
echo "Results saved to: experiments/class/cifar100_2-2-MoE-Adapters-N4-0.0/"
echo ""
sleep 2

# ============================================================================
# Experiment 3: Graph-over-Experts (N=4, k=2, with graph)
# ============================================================================
echo -e "${BLUE}[3/3] Starting Experiment 3: Graph-over-Experts (N=4, k=2, with graph)${NC}"
echo "------------------------------------------------------------------------"
echo "This runs MoE with 4 experts enhanced by graph mixing."
echo ""

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-GoE.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

echo ""
echo -e "${GREEN}✓ Experiment 3 completed!${NC}"
echo "Results saved to: experiments/class/cifar100_2-2-MoE-Adapters-GoE-0.0/"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo "========================================================================"
echo ""
echo "Results Summary:"
echo "----------------"

if [ -f "experiments/class/cifar100_2-2-MoE-Adapters-0.0/metrics.json" ]; then
    echo -e "${YELLOW}1. Original Baseline (N=2):${NC}"
    tail -1 experiments/class/cifar100_2-2-MoE-Adapters-0.0/metrics.json
    echo ""
fi

if [ -f "experiments/class/cifar100_2-2-MoE-Adapters-N4-0.0/metrics.json" ]; then
    echo -e "${YELLOW}2. Fair Baseline (N=4, no graph):${NC}"
    tail -1 experiments/class/cifar100_2-2-MoE-Adapters-N4-0.0/metrics.json
    echo ""
fi

if [ -f "experiments/class/cifar100_2-2-MoE-Adapters-GoE-0.0/metrics.json" ]; then
    echo -e "${YELLOW}3. Graph-over-Experts (N=4, with graph):${NC}"
    tail -1 experiments/class/cifar100_2-2-MoE-Adapters-GoE-0.0/metrics.json
    echo ""
fi

echo "========================================================================"
echo "Compare results:"
echo "  cd experiments/class"
echo "  cat cifar100_2-2-MoE-Adapters-0.0/metrics.json"
echo "  cat cifar100_2-2-MoE-Adapters-N4-0.0/metrics.json"
echo "  cat cifar100_2-2-MoE-Adapters-GoE-0.0/metrics.json"
echo "========================================================================"

