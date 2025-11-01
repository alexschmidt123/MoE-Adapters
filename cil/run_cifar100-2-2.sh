#!/bin/bash
# Run CIFAR-100 baseline (original paper configuration)
# N=2 experts, k=2, no graph mixer

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

