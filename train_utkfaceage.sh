#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python model_utkfaceage.py --model-name UTKFaceAge_test \
--attr-list Age \
-b 256 --epochs 40 --lr 1e-3