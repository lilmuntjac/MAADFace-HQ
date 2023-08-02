#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python tweak_celeba.py --model-name CelebA_attr04 --model-ckpt-name 0031 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
--adv-type eyeglasses --advatk-ckpt-root /tmp2/npfe/eyeglasses --advatk-stat-root /tmp2/npfe/eyeglasses_stats --advatk-name celeba_poptim \
-b 128 --epochs 15 --lr 1e1 \
--fairness-matrix "equalized odds" --loss-type "perturb optim"\