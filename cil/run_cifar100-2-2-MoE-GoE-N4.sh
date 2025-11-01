#!/bin/bash
# Run CIFAR-100 with Graph-over-Experts (GoE) Mixer
# N=4 experts, k=2, with graph mixer enabled

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-GoE.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

