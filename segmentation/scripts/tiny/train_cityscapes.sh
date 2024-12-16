#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/tiny/segman_t_cityscapes.py \
4 \
--work-dir output/cityscapes/segman_t_cityscapes \
--drop-path 0.0