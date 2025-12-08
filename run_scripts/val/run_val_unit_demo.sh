#!/bin/bash

CUDA="7"
NUM_GPU=1

CUDA_VISIBLE_DEVICES=${CUDA} accelerate launch  --num_processes ${NUM_GPU} test/test.py \
                                                --config run_configs/val/val_unit_demo.yaml


