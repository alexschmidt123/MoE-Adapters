#!/bin/bash
# General script to run any experiment with its config file
# Usage: bash run.sh configs/class/xxxx.yaml [epochs]
# Example: bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml
# Example: bash run.sh configs/class/food101_11-10-MoE-Adapters-N4-GoE.yaml

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: No config file provided"
    echo "Usage: bash run.sh <config_file_path>"
    echo "Example: bash run.sh configs/class/food101_11-10-MoE-Adapters-N4-GoE.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract config path and name so that subfolder configs (e.g. configs/class/uneven_cifar100/xxx.yaml) work
CONFIG_DIR=$(dirname "$CONFIG_FILE")
CONFIG_BASE=$(basename "$CONFIG_FILE" .yaml)
# If config is inside configs/class/SUBFOLDER/, use configs/class as path and SUBFOLDER/basename as name
if [[ "$CONFIG_DIR" == *"/"*"/"* ]]; then
    CLASS_DIR="configs/class"
    if [[ "$CONFIG_DIR" == "$CLASS_DIR" ]]; then
        CONFIG_PATH=$(echo "$CONFIG_DIR" | sed 's|^\./||')
        CONFIG_NAME="$CONFIG_BASE"
    else
        CONFIG_PATH="$CLASS_DIR"
        CONFIG_NAME="${CONFIG_DIR#$CLASS_DIR/}/$CONFIG_BASE"
    fi
else
    CONFIG_PATH=$(echo "$CONFIG_DIR" | sed 's|^\./||')
    CONFIG_NAME="$CONFIG_BASE"
fi

# Auto-detect dataset from config name
if [[ "$CONFIG_NAME" == *"cifar100"* ]]; then
    DATASET="cifar100"
elif [[ "$CONFIG_NAME" == *"food101"* ]]; then
    DATASET="food101"
elif [[ "$CONFIG_NAME" == *"tinyimagenet"* ]]; then
    DATASET="tinyimagenet"
elif [[ "$CONFIG_NAME" == *"imagenet"* ]]; then
    DATASET="imagenet"
else
    # Default to cifar100 if cannot detect
    DATASET="cifar100"
    echo "Warning: Could not detect dataset from config name, defaulting to cifar100"
fi

echo "=========================================="
echo "Running experiment: $CONFIG_NAME"
echo "Config file: $CONFIG_FILE"
echo "Dataset: $DATASET"
if [ "$2" != "" ]; then
    echo "Epochs override: $2"
fi
echo "=========================================="

# Generate run start timestamp (format: MMDDYYYY-HHMMSS)
RUN_START_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
EXP_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
echo "Results will be saved to: experiments/${RUN_START_TIMESTAMP}/${CONFIG_NAME}-${EXP_TIMESTAMP}/"
echo ""

# Build command with optional epoch override and new save path
CMD="CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path \"$CONFIG_PATH\" \
    --config-name \"$CONFIG_NAME.yaml\" \
    dataset_root=\"../datasets/\" \
    class_order=\"class_orders/${DATASET}.yaml\" \
    hydra.run.dir=\"experiments/${RUN_START_TIMESTAMP}/${CONFIG_NAME}-${EXP_TIMESTAMP}\""

# Add epoch override if provided
if [ "$2" != "" ]; then
    CMD="$CMD epochs=$2"
fi

# Run the experiment
eval $CMD

echo "=========================================="
echo "Experiment completed: $CONFIG_NAME"
echo "=========================================="

