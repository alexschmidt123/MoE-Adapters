#!/bin/bash
# Runner script for CIFAR-100 with 4 experts baseline (N=4, k=2, NO graph)
# This is the fair comparison baseline for Graph-over-Experts

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the cil directory
cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-N4.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

