python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=$((RANDOM+8888)) \
test.py \
CONFIG_FILE \
CKPT \
--eval mIoU \
--launcher pytorch
