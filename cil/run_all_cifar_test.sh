#!/bin/bash
# Run all new CIFAR-100 test experiments: 12 configs total
# Combinations: 3 N values × 2 scenarios × 2 MoE types
# N values: N4 / N8 / N16
# Scenarios: 2*2 / 5*5
# MoE types: MoE+GNN(ProtoDepth11) / HMoE-Hybrid+GNN(ProtoDepth11)
# All configs use GNN with ProtoDepth=11 and NO noise

# Configuration
CONFIG_PATH="configs/class/cifar_configs"
DATASET_ROOT="../datasets/"
CLASS_ORDER="class_orders/cifar100.yaml"
NUM_RUNS=3  # Run each config 3 times
OUTPUT_DIR="experiments/outputs"  # Output directory

# All 12 new configs
CONFIGS=(
    # 2*2 scenario - MoE+GNN (6 configs)
    "cifar100_2-2-MoE-Adapters-N4-GoE-ProtoDepth11.yaml"
    "cifar100_2-2-MoE-Adapters-N8-GoE-ProtoDepth11.yaml"
    "cifar100_2-2-MoE-Adapters-N16-GoE-ProtoDepth11.yaml"
    "cifar100_2-2-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11.yaml"
    "cifar100_2-2-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11.yaml"
    "cifar100_2-2-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11.yaml"
    
    # 5*5 scenario - MoE+GNN (6 configs)
    "cifar100_5-5-MoE-Adapters-N4-GoE-ProtoDepth11.yaml"
    "cifar100_5-5-MoE-Adapters-N8-GoE-ProtoDepth11.yaml"
    "cifar100_5-5-MoE-Adapters-N16-GoE-ProtoDepth11.yaml"
    "cifar100_5-5-MoE-Adapters-N4-HMoE-Hybrid-GoE-ProtoDepth11.yaml"
    "cifar100_5-5-MoE-Adapters-N8-HMoE-Hybrid-GoE-ProtoDepth11.yaml"
    "cifar100_5-5-MoE-Adapters-N16-HMoE-Hybrid-GoE-ProtoDepth11.yaml"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "CIFAR-100 Test Suite (New Configs)"
echo "=========================================="
echo "Total configs: ${#CONFIGS[@]}"
echo "  - 3 N values: N4 / N8 / N16"
echo "  - 2 scenarios: 2*2 / 5*5"
echo "  - 2 MoE types: MoE+GNN / HMoE-Hybrid+GNN"
echo "  - All with ProtoDepth=11, NO noise"
echo ""
echo "Runs per config: $NUM_RUNS"
echo "Total experiments: $((${#CONFIGS[@]} * $NUM_RUNS))"
echo "=========================================="
echo ""

# Generate run start timestamp (format: MMDDYYYY-HHMMSS)
RUN_START_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
echo "Results will be saved to: experiments/${RUN_START_TIMESTAMP}/"
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
    local run_start_timestamp=$4
    local exit_code=0
    
    # Clear GPU memory before starting
    clear_gpu_memory
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $config_name${NC}"
    echo -e "${BLUE}Run: $run_num / $total_runs${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Remove .yaml extension from config name for directory name
    config_dir_name="${config_name%.yaml}"
    
    # Generate timestamp for this specific experiment
    exp_timestamp=$(date +"%m%d%Y-%H%M%S")
    
    # Run with new save path: experiments/<run_start_timestamp>/<config-name>-<timestamp>/
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
        --config-path "$CONFIG_PATH" \
        --config-name "$config_name" \
        dataset_root="$DATASET_ROOT" \
        class_order="$CLASS_ORDER" \
        hydra.run.dir="experiments/${run_start_timestamp}/${config_dir_name}-${exp_timestamp}" \
        hydra.job.name="${config_dir_name}_run${run_num}" || exit_code=$?
    
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
echo -e "${CYAN}Starting CIFAR-100 Test Experiments${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

for config in "${CONFIGS[@]}"; do
    # Extract variant info from config name
    scenario=""
    if [[ $config == *"2-2"* ]]; then
        scenario="2*2"
    elif [[ $config == *"5-5"* ]]; then
        scenario="5*5"
    fi
    
    n_val=""
    if [[ $config == *"N4"* ]]; then
        n_val="N4"
    elif [[ $config == *"N8"* ]]; then
        n_val="N8"
    elif [[ $config == *"N16"* ]]; then
        n_val="N16"
    fi
    
    moe_type="MoE+GNN"
    if [[ $config == *"HMoE-Hybrid"* ]]; then
        moe_type="HMoE-Hybrid+GNN"
    fi
    
    variant="$scenario | $n_val | $moe_type | ProtoDepth11 (No Noise)"
    
    echo -e "${GREEN}--- Variant: $variant ---${NC}"
    echo ""
    
    for i in $(seq 1 $NUM_RUNS); do
        if run_experiment "$config" "$i" "$NUM_RUNS" "$RUN_START_TIMESTAMP"; then
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
echo ""
echo "All results saved to: experiments/${RUN_START_TIMESTAMP}/"
echo ""

# Generate results summary
if [ -d "experiments/${RUN_START_TIMESTAMP}" ]; then
    echo -e "${CYAN}Generating results summary...${NC}"
    python3 generate_results_summary.py "experiments/${RUN_START_TIMESTAMP}" 2>/dev/null && \
        echo -e "${GREEN}✓ Results summary generated successfully${NC}" || \
        echo -e "${YELLOW}⚠ Could not generate summary${NC}"
    echo ""
fi

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed.${NC}"
    exit 1
fi
