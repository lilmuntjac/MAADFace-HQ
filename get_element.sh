#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

python get_element.py --adv-type patch --stats /tmp2/npfe/patch/lr_1_poptim/0019.npy \
--show-type deploy -o ./_patch -n test