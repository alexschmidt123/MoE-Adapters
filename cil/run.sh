#!/bin/bash
# Run experiments: single config file, or all configs in a directory each B times.
# Usage: bash run.sh <config_file_path> [epochs]
#        bash run.sh -directory <folder> [-times B]   # run all .yaml in configs/class/<folder>, B times each (B default 3)
# Example: bash run.sh configs/class/03052025_uneven_cifar100/GoE-L1-H512-HeadNone-N8.yaml
# Example: bash run.sh -directory 03112026_uneven_cifar100
# Example: bash run.sh -directory 03112026_uneven_cifar100 -times 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH_BASE="configs/class"
NUM_RUNS=3
DIRECTORY=""

# Parse -directory A [-times B]
while [ $# -gt 0 ]; do
    case "$1" in
        -directory)
            DIRECTORY="$2"
            shift 2
            ;;
        -times)
            NUM_RUNS="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Mode: run all configs in a directory, each NUM_RUNS times
if [ -n "$DIRECTORY" ]; then
    DIR_FULL="$CONFIG_PATH_BASE/$DIRECTORY"
    if [ ! -d "$DIR_FULL" ]; then
        echo "Error: Directory not found: $DIR_FULL"
        exit 1
    fi
    RUN_START_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
    export HYDRARUN_PARENT_DIR="$RUN_START_TIMESTAMP"
    export HYDRARUN_SKIP_SUMMARY=1
    CONFIGS=()
    for f in "$DIR_FULL"/*.yaml; do
        [ -f "$f" ] && CONFIGS+=("$f")
    done
    if [ ${#CONFIGS[@]} -eq 0 ]; then
        echo "Error: No .yaml configs in $DIR_FULL"
        exit 1
    fi
    echo "=========================================="
    echo "Running ${#CONFIGS[@]} configs x $NUM_RUNS runs = $((${#CONFIGS[@]} * NUM_RUNS)) total"
    echo "Directory: $DIRECTORY"
    echo "Results: experiments/${RUN_START_TIMESTAMP}/"
    echo "=========================================="
    for CONFIG_FILE in "${CONFIGS[@]}"; do
        for run in $(seq 1 "$NUM_RUNS"); do
            export HYDRARUN_RUN_INDEX="$run"
            echo ""
            bash "$SCRIPT_DIR/run.sh" "$CONFIG_FILE"
        done
    done
    unset HYDRARUN_RUN_INDEX
    unset HYDRARUN_SKIP_SUMMARY
    python "$SCRIPT_DIR/generate_run_summary.py" "experiments/${RUN_START_TIMESTAMP}"
    echo "=========================================="
    echo "Done. Results: experiments/${RUN_START_TIMESTAMP}/"
    echo "CSV: experiments/${RUN_START_TIMESTAMP}/summary.csv"
    echo "=========================================="
    exit 0
fi

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: No config file or -directory provided"
    echo "Usage: bash run.sh <config_file_path> [epochs]"
    echo "       bash run.sh -directory <folder> [-times B]"
    echo "Example: bash run.sh configs/class/03052025_uneven_cifar100/GoE-L1-H512-HeadNone-N8.yaml"
    echo "Example: bash run.sh -directory 03112026_uneven_cifar100 -times 3"
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

# Timestamp format: mmddyyyy-HHMMSS. Use safe config name for dir (no slashes).
# If HYDRARUN_PARENT_DIR is set (e.g. when -directory is used), use it as parent so multiple runs share one dir.
SAFE_CONFIG_NAME="${CONFIG_NAME//\//_}"
RUN_START_TIMESTAMP="${HYDRARUN_PARENT_DIR:-$(date +"%m%d%Y-%H%M%S")}"
EXP_TIMESTAMP=$(date +"%m%d%Y-%H%M%S")
RUN_INDEX="${HYDRARUN_RUN_INDEX:-1}"
echo "Results will be saved to: experiments/${RUN_START_TIMESTAMP}/${SAFE_CONFIG_NAME}-run${RUN_INDEX}-${EXP_TIMESTAMP}/"
echo ""

# When full ImageNet is under ../datasets/ImageNet, build shared 100/200/500 splits once (same as run.py).
python -c "
import sys
sys.path.insert(0, 'scripts')
from prepare_imagenet_subsets import config_needs_imagenet_subsets, ensure_imagenet_subsets_from_full_data
name = sys.argv[1]
if not config_needs_imagenet_subsets(name):
    sys.exit(0)
ensure_imagenet_subsets_from_full_data()
" "$CONFIG_NAME" || exit 1

# Build command with optional epoch override and new save path (--config-name is name without .yaml)
# Do not override dataset_root/class_order here (config YAMLs define them; override triggers Hydra struct error).
CMD="CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path \"$CONFIG_PATH\" \
    --config-name \"$CONFIG_NAME\" \
    hydra.run.dir=\"experiments/${RUN_START_TIMESTAMP}/${SAFE_CONFIG_NAME}-run${RUN_INDEX}-${EXP_TIMESTAMP}\""

# Add epoch override if provided
if [ "$2" != "" ]; then
    CMD="$CMD epochs=$2"
fi

# Run the experiment
eval $CMD

# In single-config mode, write summary.csv for this run directory
if [ -z "$HYDRARUN_SKIP_SUMMARY" ]; then
    python "$SCRIPT_DIR/generate_run_summary.py" "experiments/${RUN_START_TIMESTAMP}"
fi

echo "=========================================="
echo "Experiment completed: $CONFIG_NAME"
echo "=========================================="
