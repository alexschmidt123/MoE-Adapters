#!/bin/bash
# Run HMoE (Hybrid) experiments: 12 configs, each runs 3 times (36 total experiments)
# Configs: HMoE Hybrid × (no GNN / GNN ProtoDepth11 Noise001) × N4/N8/N16 × cifar5*5/cifar10*10
# 2 GNN options × 3 N values × 2 scenarios = 12 configs

# Configuration
CONFIG_PATH="configs/class"
DATASET_ROOT="../datasets/"
CLASS_ORDER="class_orders/cifar100.yaml"
NUM_RUNS=3

# HMoE (Hybrid) configs: 12 configs total
CONFIGS=(
    # No GNN configs - cifar5*5
    "cifar100_5-5-MoE-Adapters-N4-HMoE-Hybrid.yaml"
    "cifar100_5-5-MoE-Adapters-N8-HMoE-Hybrid.yaml"
    "cifar100_5-5-MoE-Adapters-N16-HMoE-Hybrid.yaml"
    # No GNN configs - cifar10*10
    "cifar100_10-10-MoE-Adapters-N4-HMoE-Hybrid.yaml"
    "cifar100_10-10-MoE-Adapters-N8-HMoE-Hybrid.yaml"
    "cifar100_10-10-MoE-Adapters-N16-HMoE-Hybrid.yaml"
    # GNN ProtoDepth11 Noise001 configs - cifar5*5
    "cifar100_5-5-MoE-Adapters-N4-HMoE-GoE-Hybrid-ProtoDepth11-Noise001.yaml"
    "cifar100_5-5-MoE-Adapters-N8-HMoE-GoE-Hybrid-ProtoDepth11-Noise001.yaml"
    "cifar100_5-5-MoE-Adapters-N16-HMoE-GoE-Hybrid-ProtoDepth11-Noise001.yaml"
    # GNN ProtoDepth11 Noise001 configs - cifar10*10
    "cifar100_10-10-MoE-Adapters-N4-HMoE-GoE-Hybrid-ProtoDepth11-Noise001.yaml"
    "cifar100_10-10-MoE-Adapters-N8-HMoE-GoE-Hybrid-ProtoDepth11-Noise001.yaml"
    "cifar100_10-10-MoE-Adapters-N16-HMoE-GoE-Hybrid-ProtoDepth11-Noise001.yaml"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "HMoE (Hybrid) Test Suite"
echo "=========================================="
echo "Configs: ${#CONFIGS[@]}"
echo "  HMoE Hybrid × (no GNN / GNN ProtoDepth11 Noise001) × N4/N8/N16 × cifar5*5/cifar10*10"
echo "  No GNN configs (6):"
echo "    - cifar5*5: N4, N8, N16"
echo "    - cifar10*10: N4, N8, N16"
echo "  GNN ProtoDepth11 Noise001 configs (6):"
echo "    - cifar5*5: N4, N8, N16"
echo "    - cifar10*10: N4, N8, N16"
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
        dataset_root="$DATASET_ROOT" \
        class_order="$CLASS_ORDER" || exit_code=$?
    
    # Clear GPU memory after completion (success or failure)
    clear_gpu_memory
    
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

# Run HMoE (Hybrid) configs
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Testing HMoE (Hybrid) Configs${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

for config in "${CONFIGS[@]}"; do
    # Extract variant from config name
    if [[ $config == *"5-5"* ]] && [[ $config == *"N4-HMoE-Hybrid.yaml"* ]] && [[ $config != *"GoE"* ]]; then
        variant="HMoE Hybrid N4 (no GNN, cifar5*5)"
    elif [[ $config == *"5-5"* ]] && [[ $config == *"N8-HMoE-Hybrid.yaml"* ]] && [[ $config != *"GoE"* ]]; then
        variant="HMoE Hybrid N8 (no GNN, cifar5*5)"
    elif [[ $config == *"5-5"* ]] && [[ $config == *"N16-HMoE-Hybrid.yaml"* ]] && [[ $config != *"GoE"* ]]; then
        variant="HMoE Hybrid N16 (no GNN, cifar5*5)"
    elif [[ $config == *"10-10"* ]] && [[ $config == *"N4-HMoE-Hybrid.yaml"* ]] && [[ $config != *"GoE"* ]]; then
        variant="HMoE Hybrid N4 (no GNN, cifar10*10)"
    elif [[ $config == *"10-10"* ]] && [[ $config == *"N8-HMoE-Hybrid.yaml"* ]] && [[ $config != *"GoE"* ]]; then
        variant="HMoE Hybrid N8 (no GNN, cifar10*10)"
    elif [[ $config == *"10-10"* ]] && [[ $config == *"N16-HMoE-Hybrid.yaml"* ]] && [[ $config != *"GoE"* ]]; then
        variant="HMoE Hybrid N16 (no GNN, cifar10*10)"
    elif [[ $config == *"5-5"* ]] && [[ $config == *"N4-HMoE-GoE-Hybrid-ProtoDepth11-Noise001"* ]]; then
        variant="HMoE Hybrid + ProtoDepth11 + GNN N4 (noise=0.001, cifar5*5)"
    elif [[ $config == *"5-5"* ]] && [[ $config == *"N8-HMoE-GoE-Hybrid-ProtoDepth11-Noise001"* ]]; then
        variant="HMoE Hybrid + ProtoDepth11 + GNN N8 (noise=0.001, cifar5*5)"
    elif [[ $config == *"5-5"* ]] && [[ $config == *"N16-HMoE-GoE-Hybrid-ProtoDepth11-Noise001"* ]]; then
        variant="HMoE Hybrid + ProtoDepth11 + GNN N16 (noise=0.001, cifar5*5)"
    elif [[ $config == *"10-10"* ]] && [[ $config == *"N4-HMoE-GoE-Hybrid-ProtoDepth11-Noise001"* ]]; then
        variant="HMoE Hybrid + ProtoDepth11 + GNN N4 (noise=0.001, cifar10*10)"
    elif [[ $config == *"10-10"* ]] && [[ $config == *"N8-HMoE-GoE-Hybrid-ProtoDepth11-Noise001"* ]]; then
        variant="HMoE Hybrid + ProtoDepth11 + GNN N8 (noise=0.001, cifar10*10)"
    elif [[ $config == *"10-10"* ]] && [[ $config == *"N16-HMoE-GoE-Hybrid-ProtoDepth11-Noise001"* ]]; then
        variant="HMoE Hybrid + ProtoDepth11 + GNN N16 (noise=0.001, cifar10*10)"
    else
        variant="Unknown"
    fi
    
    echo -e "${GREEN}--- Variant: $variant ---${NC}"
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
