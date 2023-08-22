#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="2"

# choose epoch 13
python model_utkfaceage.py --model-name UTKFaceAge_lr_5e_4_b1 \
--attr-list Age \
-b 128 --epochs 40 --lr 5e-4