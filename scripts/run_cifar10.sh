#!/bin/bash

set -e
set -x

# CUDA_VISIBLE_DEVICES=2 python train.py \
#     --dataset_name 'cifar10' \
#     --setting 'default'\
#     --batch_size 128 \
#     --grad_from_block 11 \
#     --epochs 200 \
#     --num_workers 8 \
#     --use_ssb_splits \
#     --sup_weight 0.35 \
#     --weight_decay 5e-5 \
#     --transform 'imagenet' \
#     --lr 0.1 \
#     --warmup_teacher_temp 0.07 \
#     --teacher_temp 0.04 \
#     --warmup_teacher_temp_epochs 30 \
#     --memax_weight 1 \
#     --exp_name cifar10_simgcd \

CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name 'cifar10' \
    --setting 'animal_2_transportation_0_0.5'\
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name 'cifar10_simgcd_animal_2_transportation_0_0.5' \
    # --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/cifar10_simgcd_(16.10.2023_|_59.345)/checkpoints/model.pt'
