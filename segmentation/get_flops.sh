#!/bin/bash
CONFIG_FILE='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/local_configs/segnext/large/cityscape.py'

python get_flops.py --config $CONFIG_FILE --shape 2048 1024 --bs 2