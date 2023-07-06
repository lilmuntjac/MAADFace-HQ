#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"


# train an attribute 46 model
python model_maadfacehq.py --model-name MAADFaceHQ_attr46 \
--attr-list Young Middle_Aged Senior Asian White Black \
            Rosy_Cheeks Shiny_Skin Bald Wavy_Hair Receding_Hairline Bangs Sideburns Black_Hair Blond_Hair Brown_Hair Gray_Hair \
            No_Beard Mustache 5_o_Clock_Shadow Goatee Oval_Face Square_Face Round_Face Double_Chin High_Cheekbones Chubby \
            Obstructed_Forehead Fully_Visible_Forehead Brown_Eyes Bags_Under_Eyes Bushy_Eyebrows Arched_Eyebrows \
            Mouth_Closed Smiling Big_Lips Big_Nose Pointy_Nose Heavy_Makeup \
            Wearing_Hat Wearing_Earrings Wearing_Necktie Wearing_Lipstick No_Eyewear Eyeglasses Attractive \
-b 256 --epochs 5 --lr 1e-3

# train an attribute 6 model
# python model_maadfacehq.py --model-name MAADFaceHQ_attr6 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# -b 256 --epochs 5 --lr 1e-3