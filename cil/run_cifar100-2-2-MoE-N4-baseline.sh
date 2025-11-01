#!/bin/bash
# Run CIFAR-100 baseline with N=4 experts, k=2, no graph mixer

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-N4.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

