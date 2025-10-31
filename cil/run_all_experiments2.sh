#!/bin/bash
# Master script to run three CIFAR-100 experiments for comparison
# 1. Baseline N=8, k=2 (no graph)
# 2. N=8 with GNN, k=2 (graph-enabled)
# 3. N=2 with GNN, k=2 (graph-enabled)

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
echo "Running Experimental Suite: N=8 Baseline vs N=8 GNN vs N=2 GNN"
echo "========================================================================"
echo ""

# ============================================================================
# Experiment 1: Baseline N=8, k=2 (no graph)
# ============================================================================
echo -e "${BLUE}[1/3] Starting Experiment 1: Baseline N=8, k=2 (no graph)${NC}"
echo "------------------------------------------------------------------------"
echo "This establishes the baseline with 8 experts and no graph mixer."
echo ""

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-N8.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

echo ""
echo -e "${GREEN}✓ Experiment 1 completed!${NC}"
echo "Results saved to: experiments/class/cifar100_2-2-MoE-Adapters-N8-0.0/"
echo ""
sleep 2

# ============================================================================
# Experiment 2: N=8 with GNN, k=2 (graph-enabled)
# ============================================================================
echo -e "${BLUE}[2/3] Starting Experiment 2: N=8 with GNN, k=2 (graph-enabled)${NC}"
echo "------------------------------------------------------------------------"
echo "This runs MoE with 8 experts enhanced by graph mixing."
echo ""

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-N8-GoE.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

echo ""
echo -e "${GREEN}✓ Experiment 2 completed!${NC}"
echo "Results saved to: experiments/class/cifar100_2-2-MoE-Adapters-N8-GoE-0.0/"
echo ""
sleep 2

# ============================================================================
# Experiment 3: N=2 with GNN, k=2 (graph-enabled)
# ============================================================================
echo -e "${BLUE}[3/3] Starting Experiment 3: N=2 with GNN, k=2 (graph-enabled)${NC}"
echo "------------------------------------------------------------------------"
echo "This runs MoE with 2 experts enhanced by graph mixing."
echo ""

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-N2-GoE.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

echo ""
echo -e "${GREEN}✓ Experiment 3 completed!${NC}"
echo "Results saved to: experiments/class/cifar100_2-2-MoE-Adapters-N2-GoE-0.0/"
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

if [ -f "experiments/class/cifar100_2-2-MoE-Adapters-N8-0.0/metrics.json" ]; then
    echo -e "${YELLOW}1. Baseline N=8 (no graph):${NC}"
    tail -1 experiments/class/cifar100_2-2-MoE-Adapters-N8-0.0/metrics.json
    echo ""
fi

if [ -f "experiments/class/cifar100_2-2-MoE-Adapters-N8-GoE-0.0/metrics.json" ]; then
    echo -e "${YELLOW}2. N=8 with GNN:${NC}"
    tail -1 experiments/class/cifar100_2-2-MoE-Adapters-N8-GoE-0.0/metrics.json
    echo ""
fi

if [ -f "experiments/class/cifar100_2-2-MoE-Adapters-N2-GoE-0.0/metrics.json" ]; then
    echo -e "${YELLOW}3. N=2 with GNN:${NC}"
    tail -1 experiments/class/cifar100_2-2-MoE-Adapters-N2-GoE-0.0/metrics.json
    echo ""
fi

echo "========================================================================"
echo "Compare results:"
echo "  cd experiments/class"
echo "  cat cifar100_2-2-MoE-Adapters-N8-0.0/metrics.json"
echo "  cat cifar100_2-2-MoE-Adapters-N8-GoE-0.0/metrics.json"
echo "  cat cifar100_2-2-MoE-Adapters-N2-GoE-0.0/metrics.json"
echo "========================================================================"

