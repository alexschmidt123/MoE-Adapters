#!/bin/bash
# Run all deeper layer experiments (DeepHead and DeepProto for N2 and N4)
# Each config will be run 3 times

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find all yaml files with deeper layers (DeepHead and DeepProto)
CONFIG_FILES=$(find configs/class -name "*.yaml" -not -path "*/reference/*" | grep -E "(DeepHead|DeepProto)" | sort)

# Check if any config files found
if [ -z "$CONFIG_FILES" ]; then
    echo "Error: No deeper layer config files found in configs/class (excluding reference folder)"
    exit 1
fi

echo "=========================================="
echo "Running Deeper Layer Experiments"
echo "=========================================="
echo "Found $(echo "$CONFIG_FILES" | wc -l) config files (DeepHead and DeepProto)"
echo "Each will be run 3 times"
echo "=========================================="
echo ""

# Counter for total experiments
TOTAL_CONFIGS=$(echo "$CONFIG_FILES" | wc -l)
TOTAL_RUNS=$((TOTAL_CONFIGS * 3))
CURRENT_RUN=0

# Loop through each config file
while IFS= read -r config_file; do
    config_name=$(basename "$config_file" .yaml)
    
    echo ""
    echo "=========================================="
    echo "Config: $config_name"
    echo "File: $config_file"
    echo "=========================================="
    
    # Run 3 times
    for run_num in 1 2 3; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        echo ""
        echo "--- Run $run_num/3 for $config_name (Overall: $CURRENT_RUN/$TOTAL_RUNS) ---"
        
        # Run the experiment
        bash run.sh "$config_file"
        
        # Check if run was successful
        if [ $? -eq 0 ]; then
            echo "✓ Run $run_num completed successfully"
        else
            echo "✗ Run $run_num failed!"
            echo "Continuing with next run..."
        fi
        
        # Small delay between runs
        sleep 2
    done
    
    echo ""
    echo "✓ Completed all 3 runs for $config_name"
    echo ""
    
done <<< "$CONFIG_FILES"

echo "=========================================="
echo "All Deeper Layer Experiments Completed!"
echo "Total configs: $TOTAL_CONFIGS"
echo "Total runs: $TOTAL_RUNS"
echo "=========================================="

