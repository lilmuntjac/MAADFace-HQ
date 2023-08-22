#!/usr/bin/env bash

python get_attr_status.py --stats /tmp2/npfe/model_stats/MAADFaceHQ_attr46/MAADFaceHQ_attr46_val.npy \
--attr-list Young Middle_Aged Senior Asian White Black \
            Rosy_Cheeks Shiny_Skin Bald Wavy_Hair Receding_Hairline Bangs Sideburns Black_Hair Blond_Hair Brown_Hair Gray_Hair \
            No_Beard Mustache 5_o_Clock_Shadow Goatee Oval_Face Square_Face Round_Face Double_Chin High_Cheekbones Chubby \
            Obstructed_Forehead Fully_Visible_Forehead Brown_Eyes Bags_Under_Eyes Bushy_Eyebrows Arched_Eyebrows \
            Mouth_Closed Smiling Big_Lips Big_Nose Pointy_Nose Heavy_Makeup \
            Wearing_Hat Wearing_Earrings Wearing_Necktie Wearing_Lipstick No_Eyewear Eyeglasses Attractive \
-e 5 -o ./attribute46

python get_attr_status.py --stats /tmp2/npfe/model_stats/MAADFaceHQ_attr06/MAADFaceHQ_attr06_val.npy \
--attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
-e 5 -o ./attribute06

python get_attr_status.py --stats /tmp2/npfe/patch_stats/test_masking/MAADFaceHQ_attr06_val.npy \
--attr-list Young Shiny_Skin Oval_Face High_Cheekbones Smiling Big_Lips \
-e 5 -o ./attribute06_patch