#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/base/segman_b_cityscapes.py \
4 \
--work-dir output/cityscapes/segman_b_cityscapes \
--drop-path 0.25