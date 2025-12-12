#!/bin/bash
# Run HMoE experiments: each config runs 3 times
# Tests both HMoE-only and HMoE+GoE configurations

set -e  # Exit on error

# Configuration
CONFIG_PATH="configs/class"
DATASET_ROOT="../datasets/"
CLASS_ORDER="class_orders/cifar100.yaml"
NUM_RUNS=3

# Config files to test
CONFIG1="cifar100_2-2-MoE-Adapters-N4-HMoE.yaml"
CONFIG2="cifar100_2-2-MoE-Adapters-N4-HMoE-GoE.yaml"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "HMoE Test Suite"
echo "=========================================="
echo "Config 1: $CONFIG1 (HMoE only)"
echo "Config 2: $CONFIG2 (HMoE + GoE)"
echo "Runs per config: $NUM_RUNS"
echo "=========================================="
echo ""

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local run_num=$2
    local total_runs=$3
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $config_name${NC}"
    echo -e "${BLUE}Run: $run_num / $total_runs${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config-path "$CONFIG_PATH" \
        --config-name "$config_name" \
        dataset_root="$DATASET_ROOT" \
        class_order="$CLASS_ORDER"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Run $run_num/$total_runs completed successfully: $config_name${NC}"
    else
        echo -e "${YELLOW}✗ Run $run_num/$total_runs failed: $config_name${NC}"
        return 1
    fi
    echo ""
}

# Track statistics
TOTAL_EXPERIMENTS=$((NUM_RUNS * 2))
SUCCESSFUL=0
FAILED=0

# Run Config 1: HMoE only
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Testing Config 1: HMoE Only${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

for i in $(seq 1 $NUM_RUNS); do
    if run_experiment "$CONFIG1" "$i" "$NUM_RUNS"; then
        ((SUCCESSFUL++))
    else
        ((FAILED++))
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Testing Config 2: HMoE + GoE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Run Config 2: HMoE + GoE
for i in $(seq 1 $NUM_RUNS); do
    if run_experiment "$CONFIG2" "$i" "$NUM_RUNS"; then
        ((SUCCESSFUL++))
    else
        ((FAILED++))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Test Suite Summary"
echo "=========================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}Failed: $FAILED${NC}"
fi
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed.${NC}"
    exit 1
fi
