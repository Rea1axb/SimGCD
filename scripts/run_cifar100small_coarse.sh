#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=2 python train_with_coarse.py \
    --dataset_name 'cifar100small' \
    --setting 'default' \
    --batch_size 256 \
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
    --coarse_weight 0.5 \
    --cooloff_coarse_weight_start_epoch 120 \
    --cooloff_coarse_weight_end_epoch 150 \
    --cooloff_coarse_weight 0.2 \
    --use_coarse_label 'True' \
    --use_memory_queue 'True' \
    --mq_start_add_epoch 0 \
    --mq_start_query_epoch 10 \
    --mq_query_mode 'soft' \
    --mq_maxsize 1024 \
    --use_prototypes_attention 'False' \
    --exp_name 'cifar100small_default_coarsefromfinehead(PrototypesSupclsmqsoftClusterContrastiveSupcontrastiveCoarse)_weight(warmup_30-60_0.0-0.5_cooloff_120-150_0.5-0.2)_fineweight(dynamic)_batchsz(256)'

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
