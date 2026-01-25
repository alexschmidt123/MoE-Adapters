#!/bin/bash
# Short test for GNN configs: 8 configs total (less than 10 for quick testing)
# Combinations: 2 N values × 1 scenario × 4 GNN variants
# N values: N4 / N8
# Scenarios: 2*2 (smallest, fastest)
# GNN variants: All 4 combinations (MoE/HMoE-Hybrid × No Noise/Noise001)
# 
# This is a quick test to verify GNN fixes work correctly.

# Configuration
CONFIG_PATH="configs/class/cifar_configs"
DATASET_ROOT="../datasets/"
CLASS_ORDER="class_orders/cifar100.yaml"
NUM_RUNS=1  # Run each config once for quick testing
OUTPUT_DIR="experiments/outputs"  # Output directory

# Build configs dynamically - Short test: 1 scenario × 2 N values × 4 variants = 8 configs
SCENARIOS=("2-2")  # Only smallest scenario for quick testing
N_VALUES=(4 8)  # Only N4 and N8 for quick testing
VARIANTS=(
    "-GoE-ProtoDepth11|MoE+GNN|ProtoDepth11 (No Noise)"
    "-GoE-ProtoDepth11-Noise001|MoE+GNN|ProtoDepth11 Noise001"
    "-HMoE-Hybrid-GoE-ProtoDepth11|HMoE-Hybrid+GNN|ProtoDepth11 (No Noise)"
    "-HMoE-Hybrid-GoE-ProtoDepth11-Noise001|HMoE-Hybrid+GNN|ProtoDepth11 Noise001"
)

CONFIGS=()
for scenario in "${SCENARIOS[@]}"; do
    for n_val in "${N_VALUES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            IFS='|' read -r suffix moe_type gnn_type <<< "$variant"
            config_name="cifar100_${scenario}-MoE-Adapters-N${n_val}${suffix}.yaml"
            run_name="${config_name%.yaml}"
            extra_args=""

            if [ ! -f "${CONFIG_PATH}/${config_name}" ]; then
                if [ "$n_val" -eq 32 ]; then
                    fallback_name="${config_name/N32/N16}"
                    if [ ! -f "${CONFIG_PATH}/${fallback_name}" ]; then
                        echo "Missing config: ${fallback_name}"
                        exit 1
                    fi
                    config_name="${fallback_name}"
                    extra_args="model.num_experts=32 method=${run_name}"
                else
                    echo "Missing config: ${config_name}"
                    exit 1
                fi
            fi

            CONFIGS+=("${config_name}|${scenario}|N${n_val}|${moe_type}|${gnn_type}|${extra_args}|${run_name}")
        done
    done
done

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "CIFAR-100 GNN Short Test Suite (Quick Test)"
echo "=========================================="
echo "Total configs: ${#CONFIGS[@]}"
echo "  - 1 scenario: 2*2 (smallest, fastest)"
echo "  - 2 N values: N4 / N8"
echo "  - 4 GNN variants: All combinations (MoE/HMoE × No Noise/Noise001)"
echo ""
echo -e "\033[0;36mThis is a quick test to verify GNN fixes work correctly.\033[0m"
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
    local run_name=$2
    local extra_args_str=$3
    local run_num=$4
    local total_runs=$5
    local run_start_timestamp=$6
    local exit_code=0
    
    # Clear GPU memory before starting
    clear_gpu_memory
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $config_name${NC}"
    echo -e "${BLUE}Run: $run_num / $total_runs${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Remove .yaml extension from config name for directory name
    config_dir_name="${run_name}"
    
    # Generate timestamp for this specific experiment
    exp_timestamp=$(date +"%m%d%Y-%H%M%S")
    
    # Run with new save path: experiments/<run_start_timestamp>/<config-name>-<timestamp>/
    cmd=(python -u main.py
        --config-path "$CONFIG_PATH"
        --config-name "$config_name"
        dataset_root="$DATASET_ROOT"
        class_order="$CLASS_ORDER"
        hydra.run.dir="experiments/${run_start_timestamp}/${config_dir_name}-${exp_timestamp}"
        hydra.job.name="${config_dir_name}_run${run_num}"
    )
    if [ -n "$extra_args_str" ]; then
        read -r -a extra_args <<< "$extra_args_str"
        cmd+=("${extra_args[@]}")
    fi
    CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" || exit_code=$?
    
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

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r config_name scenario n_val moe_type gnn_type extra_args run_name <<< "$entry"
    scenario="${scenario//-/*}"
    variant="$scenario | $n_val | $moe_type | $gnn_type"
    
    echo -e "${GREEN}--- Variant: $variant ---${NC}"
    echo ""
    
    for i in $(seq 1 $NUM_RUNS); do
        if run_experiment "$config_name" "$run_name" "$extra_args" "$i" "$NUM_RUNS" "$RUN_START_TIMESTAMP"; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_CONFIGS+=("$run_name (run $i)")
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
