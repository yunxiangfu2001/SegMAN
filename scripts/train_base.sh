#!/bin/bash
python -m torch.distributed.launch \
--master_port=$((RANDOM+10000)) \
--nproc_per_node=8 \
train.py \
--data-dir dataset/imagenet \
--batch-size 256 \
--model SegMANEncoder_b \
--lr 1e-3 \
--auto-lr \
--drop-path 0.3 \
--epochs 300 \
--warmup-epochs 5 \
--workers 16 \
--model-ema \
--model-ema-decay 0.99984 \
--output /path/to/save-checkpoint/  \
--clip-grad 5 \
--amp \
--native-amp 
