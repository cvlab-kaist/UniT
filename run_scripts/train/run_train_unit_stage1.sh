#!/bin/bash

# Set local gpu rank
CUDA="0"
# Set number of gpus for training
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch  --num_processes ${NUM_GPU} train/train_dit4sr.py \
                                                --config run_configs/train/train_unit_stage1.yaml
