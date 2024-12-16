#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/base/segman_b_coco.py \
4 \
--work-dir output/coco/segman_b_coco \
--drop-path 0.25