#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python tweak_utkface.py --model-name UTKFace_lr_2e_3_b1 --model-ckpt-name 0019 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name utkface_p_t --loss-type "perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 5 --lr 1e-1 \
--coef-mode "static" \
--coef 0.0 \

python tweak_utkface.py --model-name UTKFace_lr_2e_3_b1 --model-ckpt-name 0019 \
--adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats \
--advatk-name utkface_pp_t --loss-type "perturb optim" --fairness-matrix "accuracy difference" \
-b 256 --epochs 5 --lr 1e-1 \
--coef-mode "dynamic" \
--coef 0.1 \