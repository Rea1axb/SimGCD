#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=3 python train_with_coarse.py \
    --do_test \
    --dataset_name 'cifar100small' \
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
    --warmup_coarse_weight_start_epoch 30 \
    --warmup_coarse_weight_end_epoch 60 \
    --coarse_weight 1.0 \
    --cooloff_coarse_weight_start_epoch 120 \
    --cooloff_coarse_weight_end_epoch 150 \
    --cooloff_coarse_weight 0.25 \
    --use_coarse_label \
    --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/cifar100small/cifar100small_simgcd_default_twohead_clustercoarse_weight_1-30_1.0-1.0_(20.12.2023_|_17.676)/checkpoints/model_200.pt' \
    --exp_name 'test_cifar100small_simgcd_default_twohead_clustercoarse_weight_1-30_1.0-1.0'

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
