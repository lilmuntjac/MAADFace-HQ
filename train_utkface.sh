#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python model_utkface.py --model-name UTKFace_test \
-b 256 --epochs 40 --lr 1e-3