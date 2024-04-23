#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=3 python extract_features.py \
    --model_name 'doublecoarse' \
    --dataset 'cifar100' \
    --setting 'default' \
    --transform 'imagenet' \
    --batch_size 256 \
    --num_workers 8 \
    --extract_block \
    --extract_block_num 10 \
    --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/cifar100/cifar100_default_twohead(DoubleCoarsePrototypesSupclsmqsoftClusterContrastiveSupcontrastiveCoarse)_weight(warmup_30-60_0.0-0.5_cooloff_120-150_0.5-0.2)_fineweight(dynamic)_batchsz(256)_(06.04.2024_|_35.020)/checkpoints/model_30.pt'\
    --exp_name 'cifar100_default_doublecoarse_imagenet_(06.04.2024_|_35.020)'

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
