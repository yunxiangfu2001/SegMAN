#!/bin/bash

bash tools/dist_train.sh \
local_configs/segman/tiny/segman_t_ade.py \
4 \
--work-dir output/ade/segman_t_ade \
--drop-path 0.0