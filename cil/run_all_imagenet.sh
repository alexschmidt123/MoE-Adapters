#!/bin/bash
# Run all ImageNet-100 experiments: 36 configs total
# Combinations: 2 MoE types × 2 GNN options × 3 N values × 3 scenarios
# MoE types: Original MoE / HMoE-Hybrid
# GNN options: No GNN / GNN ProtoDepth11 Noise001
# N values: N4 / N8 / N16
# Scenarios: 2*2 / 5*5 / 10*10

# Configuration
CONFIG_PATH="configs/class/imagenet_configs"
DATASET_ROOT="../datasets/"
CLASS_ORDER="class_orders/imagenet100.yaml"
NUM_RUNS=1  # Set to 1 for initial run, can be increased for multiple runs

# All 36 configs organized by scenario
CONFIGS=(
    # 2*2 scenario (12 configs)
    "imagenet100_2-2-MoE-Adapters-N4.yaml"
    "imagenet100_2-2-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_2-2-MoE-Adapters-N4-HMoE-Hybrid.yaml"
    "imagenet100_2-2-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_2-2-MoE-Adapters-N8.yaml"
    "imagenet100_2-2-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_2-2-MoE-Adapters-N8-HMoE-Hybrid.yaml"
    "imagenet100_2-2-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_2-2-MoE-Adapters-N16.yaml"
    "imagenet100_2-2-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_2-2-MoE-Adapters-N16-HMoE-Hybrid.yaml"
    "imagenet100_2-2-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    
    # 5*5 scenario (12 configs)
    "imagenet100_5-5-MoE-Adapters-N4.yaml"
    "imagenet100_5-5-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_5-5-MoE-Adapters-N4-HMoE-Hybrid.yaml"
    "imagenet100_5-5-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_5-5-MoE-Adapters-N8.yaml"
    "imagenet100_5-5-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_5-5-MoE-Adapters-N8-HMoE-Hybrid.yaml"
    "imagenet100_5-5-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_5-5-MoE-Adapters-N16.yaml"
    "imagenet100_5-5-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_5-5-MoE-Adapters-N16-HMoE-Hybrid.yaml"
    "imagenet100_5-5-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    
    # 10*10 scenario (12 configs)
    "imagenet100_10-10-MoE-Adapters-N4.yaml"
    "imagenet100_10-10-MoE-Adapters-N4-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_10-10-MoE-Adapters-N4-HMoE-Hybrid.yaml"
    "imagenet100_10-10-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_10-10-MoE-Adapters-N8.yaml"
    "imagenet100_10-10-MoE-Adapters-N8-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_10-10-MoE-Adapters-N8-HMoE-Hybrid.yaml"
    "imagenet100_10-10-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_10-10-MoE-Adapters-N16.yaml"
    "imagenet100_10-10-MoE-Adapters-N16-GoE-ProtoDepth11-Noise001.yaml"
    "imagenet100_10-10-MoE-Adapters-N16-HMoE-Hybrid.yaml"
    "imagenet100_10-10-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11-Noise001.yaml"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "ImageNet-100 Comprehensive Test Suite"
echo "=========================================="
echo "Total configs: ${#CONFIGS[@]}"
echo "  - 2 MoE types: Original MoE / HMoE-Hybrid"
echo "  - 2 GNN options: No GNN / GNN ProtoDepth11 Noise001"
echo "  - 3 N values: N4 / N8 / N16"
echo "  - 3 scenarios: 2*2 / 5*5 / 10*10"
echo ""
echo "Runs per config: $NUM_RUNS"
echo "Total experiments: $((${#CONFIGS[@]} * $NUM_RUNS))"
echo "=========================================="
echo ""

# Function to clear GPU memory
clear_gpu_memory() {
    echo -e "${CYAN}Clearing GPU memory...${NC}"
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()" 2>/dev/null || true
    sleep 2  # Give GPU time to free memory
}

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local run_num=$2
    local total_runs=$3
    local exit_code=0
    
    # Clear GPU memory before starting
    clear_gpu_memory
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $config_name${NC}"
    echo -e "${BLUE}Run: $run_num / $total_runs${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config-path "$CONFIG_PATH" \
        --config-name "$config_name" \
        +dataset_root="$DATASET_ROOT" \
        +class_order="$CLASS_ORDER" || exit_code=$?
    
    # Clear GPU memory after completion (success or failure)
    clear_gpu_memory
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Run $run_num/$total_runs completed successfully: $config_name${NC}"
        return 0
    else
        echo -e "${RED}✗ Run $run_num/$total_runs failed: $config_name (exit code: $exit_code)${NC}"
        return 1
    fi
}

# Track statistics
TOTAL_EXPERIMENTS=$((${#CONFIGS[@]} * $NUM_RUNS))
SUCCESSFUL=0
FAILED=0
FAILED_CONFIGS=()

# Run all configs
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Starting ImageNet-100 Experiments${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

for config in "${CONFIGS[@]}"; do
    # Extract variant info from config name
    scenario=""
    if [[ $config == *"2-2"* ]]; then
        scenario="2*2"
    elif [[ $config == *"5-5"* ]]; then
        scenario="5*5"
    elif [[ $config == *"10-10"* ]]; then
        scenario="10*10"
    fi
    
    n_val=""
    if [[ $config == *"N4"* ]]; then
        n_val="N4"
    elif [[ $config == *"N8"* ]]; then
        n_val="N8"
    elif [[ $config == *"N16"* ]]; then
        n_val="N16"
    fi
    
    moe_type="Original MoE"
    if [[ $config == *"HMoE-Hybrid"* ]]; then
        moe_type="HMoE-Hybrid"
    fi
    
    gnn_type="No GNN"
    if [[ $config == *"GoE-ProtoDepth11-Noise001"* ]]; then
        gnn_type="GNN ProtoDepth11 Noise001"
    fi
    
    variant="$scenario | $n_val | $moe_type | $gnn_type"
    
    echo -e "${GREEN}--- Variant: $variant ---${NC}"
    echo ""
    
    for i in $(seq 1 $NUM_RUNS); do
        if run_experiment "$config" "$i" "$NUM_RUNS"; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_CONFIGS+=("$config (run $i)")
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
    echo -e "${RED}Failed: $FAILED${NC}"
    echo ""
    echo "Failed configs:"
    for failed_config in "${FAILED_CONFIGS[@]}"; do
        echo -e "${RED}  - $failed_config${NC}"
    done
fi
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed.${NC}"
    exit 1
fi
