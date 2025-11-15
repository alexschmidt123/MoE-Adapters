#!/bin/bash
# Run CIFAR-100 with 5 classes per task (20 tasks total)

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_5-5-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

