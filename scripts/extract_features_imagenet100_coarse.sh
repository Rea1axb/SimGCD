#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python extract_features.py \
    --model_name 'doublecoarse' \
    --dataset 'imagenet' \
    --setting 'default' \
    --transform 'imagenet' \
    --batch_size 256 \
    --num_workers 8 \
    --extract_block \
    --extract_block_num 10 \
    --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/imagenet/imagenet_default_twohead(DoubleCoarseClusterContrastiveSupcontrastiveCoarse)_coarsenum(10)_weight(warmup_30-60_0.0-0.5_cooloff_120-150_0.5-0.5)_dcweight(0.0-0.5)_fineweight(dynamic)_batchsz(256)_(05.07.2024_|_13.639)/checkpoints/model_200.pt'\
    --exp_name 'imagenet_default_doublecoarse_imagenet_(05.07.2024_|_13.639)_200'
# CUDA_VISIBLE_DEVICES=3 python extract_features.py \
#     --model_name 'doublecoarse' \
#     --dataset 'imagenet10' \
#     --setting 'default' \
#     --transform 'imagenet' \
#     --batch_size 256 \
#     --num_workers 8 \
#     --extract_block \
#     --extract_block_num 10 \
#     --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/imagenet_10/imagenet10_default_twohead(DoubleCoarseClusterContrastiveSupcontrastiveCoarse)_coarsenum(3)_weight(warmup_30-60_0.0-0.5_cooloff_120-150_0.5-0.5)_dcweight(0.0-0.5)_fineweight(dynamic)_batchsz(256)_(30.08.2024_|_36.572)/checkpoints/model_60.pt'\
#     --exp_name 'imagenet10_default_doublecoarse_imagenet_(30.08.2024_|_36.572)_60'

# CUDA_VISIBLE_DEVICES=3 python extract_features.py \
#     --model_name 'doublecoarse' \
#     --dataset 'imagenet10' \
#     --setting 'default' \
#     --transform 'imagenet' \
#     --batch_size 256 \
#     --num_workers 8 \
#     --extract_block \
#     --extract_block_num 10 \
#     --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/imagenet_10/imagenet10_default_twohead(DoubleCoarseClusterContrastiveSupcontrastiveCoarse)_coarsenum(3)_weight(warmup_30-60_0.0-0.5_cooloff_120-150_0.5-0.5)_dcweight(0.0-0.5)_fineweight(dynamic)_batchsz(256)_(30.08.2024_|_36.572)/checkpoints/model_120.pt'\
#     --exp_name 'imagenet10_default_doublecoarse_imagenet_(30.08.2024_|_36.572)_120'

# CUDA_VISIBLE_DEVICES=3 python extract_features.py \
#     --model_name 'doublecoarse' \
#     --dataset 'imagenet10' \
#     --setting 'default' \
#     --transform 'imagenet' \
#     --batch_size 256 \
#     --num_workers 8 \
#     --extract_block \
#     --extract_block_num 10 \
#     --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/imagenet_10/imagenet10_default_twohead(DoubleCoarseClusterContrastiveSupcontrastiveCoarse)_coarsenum(3)_weight(warmup_30-60_0.0-0.5_cooloff_120-150_0.5-0.5)_dcweight(0.0-0.5)_fineweight(dynamic)_batchsz(256)_(30.08.2024_|_36.572)/checkpoints/model_150.pt'\
#     --exp_name 'imagenet10_default_doublecoarse_imagenet_(30.08.2024_|_36.572)_150'

# CUDA_VISIBLE_DEVICES=3 python extract_features.py \
#     --model_name 'doublecoarse' \
#     --dataset 'imagenet10' \
#     --setting 'default' \
#     --transform 'imagenet' \
#     --batch_size 256 \
#     --num_workers 8 \
#     --extract_block \
#     --extract_block_num 10 \
#     --warmup_model_dir '/home/czq/workspace/GCD/SimGCD/outputs/simgcd/log/imagenet_10/imagenet10_default_twohead(DoubleCoarseClusterContrastiveSupcontrastiveCoarse)_coarsenum(3)_weight(warmup_30-60_0.0-0.5_cooloff_120-150_0.5-0.5)_dcweight(0.0-0.5)_fineweight(dynamic)_batchsz(256)_(30.08.2024_|_36.572)/checkpoints/model_200.pt'\
#     --exp_name 'imagenet10_default_doublecoarse_imagenet_(30.08.2024_|_36.572)_200'

