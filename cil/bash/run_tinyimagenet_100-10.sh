#!/bin/bash
# Run TinyImageNet with 10 classes per task (10 tasks total)

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name tinyimagenet_100-10.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/tinyimagenet.yaml"

# for imagenet-1000 dataset; 100 classes/task
# python main.py \
#     --config-path configs/class \
#     --config-name imagenet1000_100-100.yaml \
#     dataset_root="../datasets/" \
#     class_order="class_orders/imagenet1000.yaml"