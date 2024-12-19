#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/small/segman_s_coco.py \
4 \
--work-dir output/coco/segman_s_coco \
--drop-path 0.2