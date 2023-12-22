#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python extract_features.py \
    --model_name 'twohead' \
    --dataset 'cifar100small' \
    --setting 'default' \
    --batch_size 128 \
    --num_workers 8 \
    --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/cifar100small_simgcd_default_twohead_ClusterContrastiveSupcontrastiveCoarse_weight_1-30_1.0-1.0_(21.12.2023_|_25.745)/checkpoints/model_200.pt'\
    --exp_name 'cifar100small_simgcd_default_twohead_ClusterContrastiveSupcontrastiveCoarse_weight_1-30_1.0-1.0_(21.12.2023_|_25.745)'

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --dataset_name 'cifar100' \
#     --setting 'default' \
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
#     --memax_weight 4 \
#     --eval_freq 10 \
#     --use_coarse_label \
#     --sup_coarse_con_weight 0.1 \
#     --exp_name cifar100_simgcd_default_coarselabel0.1
