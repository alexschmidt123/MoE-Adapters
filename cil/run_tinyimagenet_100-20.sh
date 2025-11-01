#!/bin/bash
# Run TinyImageNet with 20 classes per task (5 tasks total)

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name tinyimagenet_100-20.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/tinyimagenet.yaml"
