#!/bin/bash
# Run epoch study experiments for N2, N4, N8
# Usage: bash run_epoch_study.sh [epochs]
# Example: bash run_epoch_study.sh "3 5 10"
# Example: bash run_epoch_study.sh "5 10"  # Custom epoch values

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default epoch values (configurable)
if [ "$1" != "" ]; then
    EPOCH_VALUES="$1"
else
    EPOCH_VALUES="3 5 10"  # Default: skip epoch=1 (already have results)
fi

# N values to test
N_VALUES="2 4 8"

echo "=========================================="
echo "Epoch Study: N2, N4, N8"
echo "=========================================="
echo "N values: $N_VALUES"
echo "Epoch values: $EPOCH_VALUES"
echo "Configs per N: 11"
echo "=========================================="
echo ""

# Count total experiments
NUM_N=$(echo $N_VALUES | wc -w)
NUM_EPOCHS=$(echo $EPOCH_VALUES | wc -w)
CONFIGS_PER_N=11
TOTAL_EXPERIMENTS=$((CONFIGS_PER_N * NUM_N * NUM_EPOCHS))
TOTAL_RUNS=$((TOTAL_EXPERIMENTS * 3))  # Each experiment runs 3 times

echo "Calculation:"
echo "  Configs per N: $CONFIGS_PER_N"
echo "  N values: $NUM_N"
echo "  Epoch values: $NUM_EPOCHS"
echo "  Total experiments: $CONFIGS_PER_N × $NUM_N × $NUM_EPOCHS = $TOTAL_EXPERIMENTS"
echo "  Each experiment runs 3 times"
echo "  Total runs: $TOTAL_EXPERIMENTS × 3 = $TOTAL_RUNS"
echo "=========================================="
echo ""

CURRENT_RUN=0

# Loop through each N value
for n in $N_VALUES; do
    echo ""
    echo "=========================================="
    echo "Processing N=$n"
    echo "=========================================="
    
    # Find all configs for this N value
    CONFIG_FILES=$(find configs/class -name "*N${n}*.yaml" -not -path "*/reference/*" | sort)
    
    if [ -z "$CONFIG_FILES" ]; then
        echo "Warning: No configs found for N=$n"
        continue
    fi
    
    # Loop through each config
    while IFS= read -r config_file; do
        config_name=$(basename "$config_file" .yaml)
        
        echo ""
        echo "--- Config: $config_name ---"
        
        # Loop through each epoch value
        for epochs in $EPOCH_VALUES; do
            # Run 3 times for each (epoch, config) combination
            for run_num in 1 2 3; do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                echo ""
                echo "  [Run $run_num/3] Epochs=$epochs (Overall: $CURRENT_RUN/$TOTAL_RUNS)"
                
                # Run the experiment using run.sh with epoch override
                bash run.sh "$config_file" $epochs
                
                # Check if run was successful
                if [ $? -eq 0 ]; then
                    echo "  ✓ Completed successfully"
                else
                    echo "  ✗ Failed! Continuing..."
                fi
                
                # Small delay between runs
                sleep 2
            done
        done
        
        echo ""
        echo "  ✓ Completed all runs for $config_name"
        
    done <<< "$CONFIG_FILES"
    
    echo ""
    echo "✓ Completed all configs for N=$n"
    echo ""
done

echo "=========================================="
echo "All Epoch Study Experiments Completed!"
echo "=========================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Total runs: $TOTAL_RUNS"
echo "=========================================="
