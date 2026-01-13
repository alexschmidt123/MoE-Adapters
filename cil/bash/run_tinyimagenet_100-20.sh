#!/bin/bash
# Run TinyImageNet with 100 base classes, 20 classes per incremental task (5 step)
# This corresponds to "5 step" in the paper table
# Paper: "100 base classes" with 5 incremental tasks (20 classes each)

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class/tinyimagenet_configs \
    --config-name tinyimagenet_100-20-MoE-Adapters-N2.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/tinyimagenet.yaml"
