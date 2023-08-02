#!/usr/bin/env bash

# select GPUs on the server
export CUDA_VISIBLE_DEVICES="0"

# python get_element.py --adv-type patch --stats /tmp2/npfe/patch/psmallmom_poptim/0014.npy \
# --show-type element -o ./_patch -n test

python get_element.py --adv-type eyeglasses --stats /tmp2/npfe/eyeglasses/celeba_poptim/0014.npy \
--show-type element -o ./_eyeglasses -n test