#!/bin/bash
python3 -m torch.distributed.launch \
--master_port=$((RANDOM+10000)) \
--nproc_per_node=8 \
train.py \
--data-dir /grp01/cs_yzyu/dataset/imagenet \
--batch-size 512 \
--model SegMANEncoder_t \
--lr 2e-3 \
--auto-lr \
--drop-path 0.0 \
--epochs 300 \
--warmup-epochs 5 \
--workers 16 \
--model-ema \
--model-ema-decay 0.99984 \
--output /path/to/save-checkpoint/ \
--clip-grad 5 