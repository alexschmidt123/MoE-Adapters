#!/bin/bash
# Run HMoE + GoE variant experiments: 6 configs (3 HMoE strategies × 2 GNN variants), each runs 3 times
# Tests: Geometric/Arithmetic/Hybrid × DeepProto/Noise001

# Configuration
CONFIG_PATH="configs/class"
DATASET_ROOT="../datasets/"
CLASS_ORDER="class_orders/cifar100.yaml"
NUM_RUNS=3

# HMoE + GoE variants: 3 HMoE strategies × 2 GNN variants
CONFIGS=(
    "cifar100_2-2-MoE-Adapters-N4-HMoE-GoE-Geometric-DeepProto.yaml"
    "cifar100_2-2-MoE-Adapters-N4-HMoE-GoE-Geometric-Noise001.yaml"
    "cifar100_2-2-MoE-Adapters-N4-HMoE-GoE-Arithmetic-DeepProto.yaml"
    "cifar100_2-2-MoE-Adapters-N4-HMoE-GoE-Arithmetic-Noise001.yaml"
    "cifar100_2-2-MoE-Adapters-N4-HMoE-GoE-Hybrid-DeepProto.yaml"
    "cifar100_2-2-MoE-Adapters-N4-HMoE-GoE-Hybrid-Noise001.yaml"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "HMoE + GoE Variants Test Suite"
echo "=========================================="
echo "Configs: ${#CONFIGS[@]}"
echo "  - Geometric + DeepProto"
echo "  - Geometric + Noise001"
echo "  - Arithmetic + DeepProto"
echo "  - Arithmetic + Noise001"
echo "  - Hybrid + DeepProto"
echo "  - Hybrid + Noise001"
echo ""
echo "Runs per config: $NUM_RUNS"
echo "Total experiments: $((${#CONFIGS[@]} * $NUM_RUNS))"
echo "=========================================="
echo ""

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local run_num=$2
    local total_runs=$3
    local exit_code=0
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $config_name${NC}"
    echo -e "${BLUE}Run: $run_num / $total_runs${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config-path "$CONFIG_PATH" \
        --config-name "$config_name" \
        dataset_root="$DATASET_ROOT" \
        class_order="$CLASS_ORDER" || exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Run $run_num/$total_runs completed successfully: $config_name${NC}"
        return 0
    else
        echo -e "${YELLOW}✗ Run $run_num/$total_runs failed: $config_name (exit code: $exit_code)${NC}"
        return 1
    fi
}

# Track statistics
TOTAL_EXPERIMENTS=$((${#CONFIGS[@]} * $NUM_RUNS))
SUCCESSFUL=0
FAILED=0

# Run all HMoE + GoE variant configs
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Testing HMoE + GoE Variants${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

for config in "${CONFIGS[@]}"; do
    # Extract strategy and variant from config name
    if [[ $config == *"Geometric"* ]]; then
        strategy="Geometric"
    elif [[ $config == *"Arithmetic"* ]]; then
        strategy="Arithmetic"
    elif [[ $config == *"Hybrid"* ]]; then
        strategy="Hybrid"
    fi
    
    if [[ $config == *"DeepProto"* ]]; then
        variant="DeepProto"
    elif [[ $config == *"Noise001"* ]]; then
        variant="Noise001"
    fi
    
    echo -e "${GREEN}--- Strategy: $strategy, Variant: $variant ---${NC}"
    echo ""
    
    for i in $(seq 1 $NUM_RUNS); do
        if run_experiment "$config" "$i" "$NUM_RUNS"; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        echo ""  # Add blank line between runs
    done
    echo ""
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
