#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/tiny/segman_t_coco.py \
4 \
--work-dir output/coco/segman_t_coco \
--drop-path 0.0