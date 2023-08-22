#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="2"

# choose epoch 19
python model_utkface.py --model-name UTKFace_lr_2e_3_b1 \
-b 128 --epochs 40 --lr 2e-3