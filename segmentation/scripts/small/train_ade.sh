#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/small/segman_s_ade.py \
4 \
--work-dir output/ade/segman_s_ade \
--drop-path 0.15