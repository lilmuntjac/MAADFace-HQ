#!/usr/bin/env bash

python model_maadfacehq.py --model-name maadfacehq_1e_2 --epochs 5 --lr 1e-2
python model_maadfacehq.py --model-name maadfacehq_5e_3 --epochs 5 --lr 5e-3
python model_maadfacehq.py --model-name maadfacehq_1e_3 --epochs 5 --lr 1e-3
python model_maadfacehq.py --model-name maadfacehq_5e_4 --epochs 5 --lr 5e-4
python model_maadfacehq.py --model-name maadfacehq_1e_4 --epochs 5 --lr 1e-4