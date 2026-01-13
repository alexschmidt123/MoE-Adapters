#!/bin/bash
# Run TinyImageNet with 100 base classes, 5 classes per incremental task (20 step)
# This corresponds to "20 step" in the paper table
# Paper: "100 base classes" with 20 incremental tasks (5 classes each)

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class/tinyimagenet_configs \
    --config-name tinyimagenet_100-5-MoE-Adapters-N2.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/tinyimagenet.yaml"

# for imagenet-1000 dataset; 100 classes/task
# python main.py \
#     --config-path configs/class \
#     --config-name imagenet1000_100-100.yaml \
#     dataset_root="../datasets/" \
#     class_order="class_orders/imagenet1000.yaml"