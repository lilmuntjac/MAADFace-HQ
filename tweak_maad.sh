#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# --adv-type patch --advatk-ckpt-root /tmp2/npfe/patch --advatk-stat-root /tmp2/npfe/patch_stats --advatk-name test_direct \
# -b 256 --epochs 50 --lr 1e-3 \
# --fairness-matrix "equalized odds" --loss-type direct\

# python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# --adv-type patch --advatk-ckpt-root /tmp2/npfe/patch --advatk-stat-root /tmp2/npfe/patch_stats --advatk-name test_masking \
# -b 256 --epochs 50 --lr 1e-3 \
# --fairness-matrix "equalized odds" --loss-type masking\

# python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# --adv-type patch --advatk-ckpt-root /tmp2/npfe/patch --advatk-stat-root /tmp2/npfe/patch_stats --advatk-name test_poptim \
# -b 256 --epochs 50 --lr 1e-3 \
# --fairness-matrix "equalized odds" --loss-type "perturb optim"\

# python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# --adv-type noise --advatk-ckpt-root /tmp2/npfe/noise --advatk-stat-root /tmp2/npfe/noise_stats --advatk-name test \
# -b 256 --epochs 5 --lr 1e-1 \
# --fairness-matrix "equalized odds" --loss-type direct\

# python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# --adv-type patch --advatk-ckpt-root /tmp2/npfe/patch --advatk-stat-root /tmp2/npfe/patch_stats --advatk-name test \
# -b 256 --epochs 5 --lr 1e-1 \
# --fairness-matrix "equalized odds" --loss-type direct\

# python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# --adv-type frame --advatk-ckpt-root /tmp2/npfe/frame --advatk-stat-root /tmp2/npfe/frame_stats --advatk-name test \
# -b 256 --epochs 5 --lr 1e-1 \
# --fairness-matrix "equalized odds" --loss-type direct\

# python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# --adv-type eyeglasses --advatk-ckpt-root /tmp2/npfe/eyeglasses --advatk-stat-root /tmp2/npfe/eyeglasses_stats --advatk-name test \
# -b 256 --epochs 1 --lr 1e0 \
# --fairness-matrix "equalized odds" --loss-type direct\

python tweak_maadfacehq.py --model-name MAADFaceHQ_attr06 --model-ckpt-name 0005 \
--attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
--adv-type patch --advatk-ckpt-root /tmp2/npfe/patch --advatk-stat-root /tmp2/npfe/patch_stats --advatk-name lr_1_poptim \
-b 256 --epochs 15 --lr 1e-0 \
--fairness-matrix "equalized odds" --loss-type "perturb optim"\

# --attr-list Young Middle_Aged Senior Asian White Black \
#             Rosy_Cheeks Shiny_Skin Bald Wavy_Hair Receding_Hairline Bangs Sideburns Black_Hair Blond_Hair Brown_Hair Gray_Hair \
#             No_Beard Mustache 5_o_Clock_Shadow Goatee Oval_Face Square_Face Round_Face Double_Chin High_Cheekbones Chubby \
#             Obstructed_Forehead Fully_Visible_Forehead Brown_Eyes Bags_Under_Eyes Bushy_Eyebrows Arched_Eyebrows \
#             Mouth_Closed Smiling Big_Lips Big_Nose Pointy_Nose Heavy_Makeup \
#             Wearing_Hat Wearing_Earrings Wearing_Necktie Wearing_Lipstick No_Eyewear Eyeglasses Attractive \