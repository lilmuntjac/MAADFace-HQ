#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# python get_training_status.py \
# --train-stats /tmp2/npfe/model_stats/CelebA_lr_2e_3_b1/CelebA_lr_2e_3_b1_train.npy \
# --val-stats /tmp2/npfe/model_stats/CelebA_lr_2e_3_b1/CelebA_lr_2e_3_b1_val.npy \
# --attr-list Attractive High_Cheekbones Mouth_Slightly_Open Smiling \
# --pred-type binary -o ./eval_CelebA_lr_2e_3_b1

# python get_training_status.py \
# --train-stats /tmp2/npfe/model_stats/UTKFaceAge_lr_5e_4_b1/UTKFaceAge_lr_5e_4_b1_train.npy \
# --val-stats /tmp2/npfe/model_stats/UTKFaceAge_lr_5e_4_b1/UTKFaceAge_lr_5e_4_b1_val.npy \
# --attr-list Age \
# --pred-type binary -o ./eval_UTKFaceAge_lr_5e_4_b1

# python get_training_status.py \
# --train-stats /tmp2/npfe/model_stats/FairFaceAge_lr_1e_3_b1/FairFaceAge_lr_1e_3_b1_train.npy \
# --val-stats /tmp2/npfe/model_stats/FairFaceAge_lr_1e_3_b1/FairFaceAge_lr_1e_3_b1_val.npy \
# --attr-list Age \
# --pred-type binary -o ./eval_FairFaceAge_lr_1e_3_b1

# python get_training_status.py \
# --train-stats /tmp2/npfe/model_stats/UTKFace_lr_2e_3_b1/UTKFace_lr_2e_3_b1_train.npy \
# --val-stats /tmp2/npfe/model_stats/UTKFace_lr_2e_3_b1/UTKFace_lr_2e_3_b1_val.npy \
# --attr-list Gender Age \
# --pred-type categorical -o ./eval_UTKFace_lr_2e_3_b1

# python get_training_status.py \
# --train-stats /tmp2/npfe/model_stats/FairFace_lr_1e_3_b2/FairFace_lr_1e_3_b2_train.npy \
# --val-stats /tmp2/npfe/model_stats/FairFace_lr_1e_3_b2/FairFace_lr_1e_3_b2_val.npy \
# --attr-list Gender Age \
# --pred-type categorical -o ./eval_FairFace_lr_1e_3_b2