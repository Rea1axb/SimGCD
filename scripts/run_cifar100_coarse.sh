#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train_with_coarse.py \
    --dataset_name 'cifar100' \
    --setting 'default' \
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
    --memax_weight 4 \
    --eval_freq 10 \
    --fine_weight 1.0 \
    --warmup_coarse_weight 0.0 \
    --coarse_weight 0.0 \
    --warmup_coarse_weight_epochs 30 \
    --use_coarse_label \
    --exp_name cifar100_simgcd_default_twohead_clustercoarse_weight0.0

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
