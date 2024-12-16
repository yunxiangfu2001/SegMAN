#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/base/segman_b_ade.py \
4 \
--work-dir output/ade/segman_b_ade \
--drop-path 0.25