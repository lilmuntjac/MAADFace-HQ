#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# python model_celeba.py --model-name CelebA_attr39 \
# --attr-list 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs \
#             Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair \
#             Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair \
#             Heavy_Makeup High_Cheekbones  Mouth_Slightly_Open Mustache \
#             Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline \
#             Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings \
#             Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young \
# -b 256 --epochs 10 --lr 1e-3

# choose epoch 25
python model_celeba.py --model-name CelebA_lr_1e_3_b1 \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
-b 256 --epochs 40 --lr 1e-3