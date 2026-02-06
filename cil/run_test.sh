#!/bin/bash
# Run uneven CIFAR-100 configs, each 3 times.
# Configs: cifar100_uneven10-MoE-Adapters-N4.yaml, cifar100_uneven10-MoE-Adapters-N4-GoE.yaml

CONFIG_PATH="configs/class"
CONFIG_NAMES=(
    "uneven_cifar100/cifar100_uneven10-MoE-Adapters-N4"
    "uneven_cifar100/cifar100_uneven10-MoE-Adapters-N4-GoE"
)
NUM_RUNS=3

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

RUN_START_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
echo "=========================================="
echo "Uneven CIFAR-100 test: 2 configs × $NUM_RUNS runs"
echo "=========================================="
echo "Configs: ${CONFIG_NAMES[*]}"
echo "Runs per config: $NUM_RUNS"
echo "Results: experiments/${RUN_START_TIMESTAMP}/"
echo "=========================================="
echo ""

SUCCESSFUL=0
FAILED=0
FAILED_LIST=()

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    echo -e "${CYAN}--- Config: $CONFIG_NAME ---${NC}"
    for i in $(seq 1 $NUM_RUNS); do
        EXP_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
        EXP_DIR="experiments/${RUN_START_TIMESTAMP}/${CONFIG_NAME}-run${i}-${EXP_TIMESTAMP}"
        echo -e "${BLUE}Run $i/$NUM_RUNS: $CONFIG_NAME${NC}"

        # All results go to experiments/ (override so no "output"/"outputs" folder is created)
        CUDA_VISIBLE_DEVICES=0 python main.py \
            --config-path "$CONFIG_PATH" \
            --config-name "${CONFIG_NAME}.yaml" \
            hydra.run.dir="$EXP_DIR" || {
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$CONFIG_NAME run $i")
            echo -e "${RED}✗ Failed: $CONFIG_NAME run $i${NC}"
            continue
        }
        SUCCESSFUL=$((SUCCESSFUL + 1))
        echo -e "${GREEN}✓ Run $i/$NUM_RUNS completed: $CONFIG_NAME${NC}"
        echo ""
    done
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "Failed:"
    printf '  - %s\n' "${FAILED_LIST[@]}"
fi
echo "Results: experiments/${RUN_START_TIMESTAMP}/"
echo "=========================================="

[ $FAILED -eq 0 ] && exit 0 || exit 1
