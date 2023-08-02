#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# python get_model_status.py --stats /tmp2/npfe/patch_stats/test_poptim/MAADFaceHQ_attr06_val.npy \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# -o ./eval_old

# python get_model_status.py --stats /tmp2/npfe/patch_stats/lr_1_poptim/val.npy \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# -o ./eval

# python get_model_status.py --stats /tmp2/npfe/patch_stats/pmuilti_poptim/val.npy \
# --attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
# -o ./eval

python get_model_status.py --stats /tmp2/npfe/eyeglasses_stats/celeba_poptim/val.npy \
--attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
-o ./eval