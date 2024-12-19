#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/small/segman_s_cityscapes.py \
4 \
--work-dir output/cityscapes/segman_s_cityscapes \
--drop-path 0.2