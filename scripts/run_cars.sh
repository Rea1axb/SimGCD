#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'scars' \
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
    --memax_weight 1 \
    --use_ema \
    --momentum_ema 0.99 \
    --exp_name scars_ema0.99_simgcd_default
