#!/bin/bash
# Runner script for CIFAR-100 with Graph-over-Experts (GoE) Mixer
# This script runs the graph-enabled MoE configuration with 4 experts and top-k=2

set -e

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-GoE.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

