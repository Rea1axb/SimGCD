#!/bin/bash

set -e
set -x

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python train_LLM4GCD.py \
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
    --clip_train_epochs 1 \
    --query_freq 20 \
    --n_samples_1 30 \
    --n_samples_2 30 \
    --n_samples_3 30 \
    --n_samples_label 5 \
    --clip_train_epochs 2 \
    --prompt_dir './outputs/LLM4GCD/prompt/cifar100_default'\
    --exp_name cifar100_ProQuery_default
