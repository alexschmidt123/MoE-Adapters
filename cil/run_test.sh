#!/bin/bash
# Run L1 configs only (9 in 03032025_uneven_cifar100/), 3 runs each; then generate summary.csv.
# Same configs as run_test.py. For Windows use run_test.py. Run from cil/.

CONFIG_PATH="configs/class"
CONFIG_NAMES=(
    "03032025_uneven_cifar100/GoE-L1-H512-HeadNone"
    "03032025_uneven_cifar100/GoE-L1-H512-Head512"
    "03032025_uneven_cifar100/GoE-L1-H512-Head512_256"
    "03032025_uneven_cifar100/GoE-L1-H768-HeadNone"
    "03032025_uneven_cifar100/GoE-L1-H768-Head512"
    "03032025_uneven_cifar100/GoE-L1-H768-Head512_256"
    "03032025_uneven_cifar100/GoE-L1-H1024-HeadNone"
    "03032025_uneven_cifar100/GoE-L1-H1024-Head512"
    "03032025_uneven_cifar100/GoE-L1-H1024-Head512_256"
)
NUM_RUNS=3

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# Timestamp format: mmddyyyy-HHMMSS
RUN_START_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
echo "=========================================="
echo "${#CONFIG_NAMES[@]} configs x $NUM_RUNS runs = $((${#CONFIG_NAMES[@]} * NUM_RUNS)) total"
echo "=========================================="
echo "Results: experiments/${RUN_START_TIMESTAMP}/"
echo "=========================================="
echo ""

SUCCESSFUL=0
FAILED=0
FAILED_LIST=()

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    # Dir name = config_name with slash replaced (e.g. 02052026_uneven_cifar100_baseline)
    SAFE_NAME="${CONFIG_NAME//\//_}"
    echo -e "${CYAN}--- Config: $CONFIG_NAME ---${NC}"
    for i in $(seq 1 $NUM_RUNS); do
        EXP_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
        EXP_DIR="experiments/${RUN_START_TIMESTAMP}/${SAFE_NAME}-run${i}-${EXP_TIMESTAMP}"
        echo -e "${BLUE}Run $i/$NUM_RUNS: $CONFIG_NAME (batch_size=32)${NC}"

        OOM_ERR=$(mktemp 2>/dev/null || echo /tmp/run_test_oom.$$)
        CUDA_VISIBLE_DEVICES=0 python main.py \
            --config-path "$CONFIG_PATH" \
            --config-name "$CONFIG_NAME" \
            hydra.run.dir="$EXP_DIR" 2> "$OOM_ERR"
        CODE=$?
        if [ $CODE -ne 0 ] && grep -qiE "out of memory|outofmemoryerror|cuda out of memory" "$OOM_ERR" 2>/dev/null; then
            echo -e "${BLUE}OOM detected; retrying with batch_size=12 ...${NC}"
            CUDA_VISIBLE_DEVICES=0 python main.py \
                --config-path "$CONFIG_PATH" \
                --config-name "$CONFIG_NAME" \
                hydra.run.dir="$EXP_DIR" \
                batch_size=12 2> "$OOM_ERR"
            CODE=$?
        fi
        rm -f "$OOM_ERR"

        if [ $CODE -ne 0 ]; then
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$CONFIG_NAME run $i")
            echo -e "${RED}✗ Failed: $CONFIG_NAME run $i${NC}"
            continue
        fi
        SUCCESSFUL=$((SUCCESSFUL + 1))
        echo -e "${GREEN}✓ Run $i/$NUM_RUNS completed: $CONFIG_NAME${NC}"
        echo ""
    done
    echo ""
done

# Generate summary.csv (last_acc, avg_acc per run + per-config avg rows)
python generate_run_summary.py "experiments/${RUN_START_TIMESTAMP}"

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
echo "CSV: experiments/${RUN_START_TIMESTAMP}/summary.csv"
echo "=========================================="

[ $FAILED -eq 0 ] && exit 0 || exit 1
