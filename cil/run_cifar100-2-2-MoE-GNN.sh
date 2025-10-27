#!/bin/bash

# Run CIFAR100 with 2 classes per task using MoE-Adapters with Graph-over-Experts (GoE) mixer
# This script demonstrates the Graph-based MoE variant where inactive experts contribute via learned adjacency

# Default number of experts (can be overridden with EXPERTS_N environment variable)
N="${EXPERTS_N:-4}"

# Fixed top-k selection (keep at 2)
K=2

echo "Running MoE-GNN with Configuration: k=${K}, n=${N} experts"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml" \
    experts.num_experts=${N} \
    experts.top_k=${K} \
    experts.graph_mixer_enabled=true \
    experts.graph_alpha_init=0.0 \
    experts.graph_entropy_weight=0.0

# Example usage:
# Run with default 4 experts:
#   bash run_cifar100-2-2-MoE-GNN.sh
#
# Run with 8 experts:
#   EXPERTS_N=8 bash run_cifar100-2-2-MoE-GNN.sh
#
# Or override directly:
#   bash run_cifar100-2-2-MoE-GNN.sh experts.num_experts=6

