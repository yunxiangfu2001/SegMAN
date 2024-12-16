export PATH=/grp01/cs_yzyu/yunxiang/anaconda/condabin:/grp01/cs_yzyu/yunxiang/.conda/envs/mmseg2/bin:$PATH
export HOME=/grp01/cs_yzyu/yunxiang
source /grp01/cs_yzyu/yunxiang/anaconda/bin/activate
conda activate mmseg2

##################################   without cpe ##################################
## base

# 52.34
# CONFIG_FILE='local_configs/segman/base/segman_b_ade.py'
# CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/ade/vmamba_former_b_v0_180_no_tricks_drop0.3_batc16_run1_52.47/best_mIoU_iter_160000.pth'

# 83.9
# CONFIG_FILE='local_configs/segman/base/segman_b_cityscapes.py'
# CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/cityscapes/segman_b_v0_180_320_o_tricks_drop0.25_batc8_run1_83.88/iter_144000.pth'

# 48.49
# CONFIG_FILE='local_configs/segman/base/segman_b_coco.py'
# CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/coco/segman_v0_180_drop0.25_batc16_run1_84.54/best_mIoU_iter_144000.pth'

## small
# 51.3
# CONFIG_FILE='local_configs/segman/small/segman_s_ade.py'
# CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/ade/vmamba_former_s_144_288_drop0.2_run2_best/iter_144000.pth'

# 83.17
# CONFIG_FILE='local_configs/segman/small/segman_s_cityscapes.py'
# CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/cityscapes/vmamba_former_s_152_c288_drop0.25_batc8_run2/best_mIoU_iter_160000.pth'

# 47.49
# CONFIG_FILE='local_configs/segman/small/segman_s_coco.py'
# CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/coco/vmamba_former_s_144_288_drop0.15_run2/iter_160000.pth'

##################################   without cpe ##################################

##################################   with cpe ##################################

## tiny 

# ade 43.0
CONFIG_FILE='local_configs/segman/tiny/segman_t_ade.py'
CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/ade/segman_t_new2_CPE_128_192_drop0.0_run5_42.95/best_mIoU_iter_160000.pth'

# cityscapes 80.3
CONFIG_FILE='local_configs/segman/tiny/segman_t_cityscapes.py'
CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/cityscapes/segman_t_new2_CPE_128_192_drop0.0_run5_80.31/best_mIoU_iter_120000.pth'

# coco 
CONFIG_FILE='local_configs/segman/tiny/segman_t_coco.py'
CKPT='/grp01/cs_yzyu/yunxiang/code/LaMamba_img/segmentation2/output/coco/segman_t_new2_CPE_128_192_drop0.0_run1_41.3/iter_152000.pth'


##################################   with cpe ##################################

python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=$((RANDOM+8888)) \
test.py \
$CONFIG_FILE \
$CKPT \
--eval mIoU \
--launcher pytorch
