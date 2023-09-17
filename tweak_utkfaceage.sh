#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python tweak_utkfaceage.py --model-name UTKFaceAge_lr_5e_4_b1 --model-ckpt-name 0013 \
--attr-list Age \
--adv-type eyeglasses --advatk-ckpt-root /tmp2/npfe/eyeglasses --advatk-stat-root /tmp2/npfe/eyeglasses_stats \
--advatk-name utkfaceage_p_t --loss-type "perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 5 --lr 1e1 \
--coef-mode "static" \
--p-coef 0.0 \
--n-coef 0.0 \

python tweak_utkfaceage.py --model-name UTKFaceAge_lr_5e_4_b1 --model-ckpt-name 0013 \
--attr-list Age \
--adv-type eyeglasses --advatk-ckpt-root /tmp2/npfe/eyeglasses --advatk-stat-root /tmp2/npfe/eyeglasses_stats \
--advatk-name utkfaceage_pp_t --loss-type "perturb optim" --fairness-matrix "equalized odds" \
-b 256 --epochs 5 --lr 1e1 \
--coef-mode "dynamic" \
--p-coef 0.05 \
--n-coef 0.05 \