#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# choose epoch 20
python model_fairfaceage.py --model-name FairFaceAge_lr_1e_3_b1 \
--attr-list Age \
-b 128 --epochs 40 --lr 1e-3