#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 11
python model_fairface.py --model-name FairFace_lr_1e_3_b2 \
-b 256 --epochs 40 --lr 1e-3
