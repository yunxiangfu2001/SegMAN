# SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation

Official Pytorch implementation of [SegMAN: Omni-scale Context Modeling with State Space Models
and Local Attention for Semantic Segmentation]()

![SegMAN](assets/model.png)

## Main Results

<img src="assets/main_results.png" width="50%" />

## Installation and data preparation

**Step 1:**  Create a new environment
```shell
conda create -n segman python=3.10
conda activate segman

pip install torch==2.1.2 torchvision==0.16.2
```
**Step 2:** Install [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0) by following the [installation guidelines](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/get_started.md) and prepare segmentation datasets by following [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/dataset_prepare.md).
The following installation commands works for me:
```
pip install -U openmim
mim install mmcv-full
cd segmentation
pip install -v -e .
```

To support torch>=2.1.0, you also need to replace ```Line 75``` of ```/miniconda3/envs/segman/lib/python3.10/site-packages/mmcv/parallel/_functions.py``` with the following:
```
if version.parse(torch.__version__) >= version.parse('2.1.0'):
    streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
else:
    streams = [_get_stream(device) for device in target_gpus]
```
**Step 3:** Install dependencies using the following commands.

To install [Natten](https://github.com/SHI-Labs/NATTEN), you should modify the following with your PyTorch and CUDA versions accordingly.
```shell
pip install natten==0.17.3+torch210cu121 -f https://shi-labs.com/natten/wheels/
```

The [Selective Scan 2D](https://github.com/MzeroMiko/VMamba) can be install with:
```shell
cd kernels/selective_scan && pip install .
```

Install other requirements:
```shell
pip install -r requirements.txt
```

## Training
Download the ImageNet-1k pretrained weights [here](https://drive.google.com/drive/folders/1QYU7nhpe0ddH7bPxI7VH4drc__07uEHs?usp=sharing) and put them in a folder ```pretrained/```. Navigate to the segmentation directory:
```shell
cd segmentation
```
Scripts to reproduce our paper results are provided in ```./scripts```
Example training script for ```SegMAN-B``` on ```ADE20K```:
```shell
# Single-gpu
python tools/train.py local_configs/segman/base/segman_b_ade.py --work-dir outputs/EXP_NAME

# Multi-gpu
bash tools/dist_train.sh local_configs/segman/base/segman_b_ade.py <GPU_NUM> --work-dir outputs/EXP_NAME
```

## Evaluation
Download `trained weights` for segmentation models at [google drive](https://drive.google.com/drive/folders/1C2bmb7KP7mECm9c04NCrUAJQGsEf_bQ4?usp=sharing). Navigate to the segmentation directory:
```shell
cd segmentation
```

Example for evaluating ```SegMAN-B``` on ```ADE20K```:
```
# Single-gpu
python tools/test.py local_configs/segman/base/segman_b_ade.py /path/to/checkpoint_file

# Multi-gpu
bash tools/dist_test.sh local_configs/segman/base/segman_b_ade.py /path/to/checkpoint_file <GPU_NUM>
```


## Encoder Pre-training
We provide scripts for pre-training the encoder from scratch.

**Step 1:** Download [ImageNet-1k](https://www.image-net.org/download.php) and using this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) to extract it.

**Step 2:** Start training with

```
bash scripts/train_segman-s.sh
``` 


## Acknowledgements

Our implementation is based on [MMSegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [Natten](https://github.com/SHI-Labs/NATTEN), [VMamba](https://github.com/MzeroMiko/VMamba), and [SegFormer](https://github.com/NVlabs/SegFormer). We gratefully thank the authors.

## Citation
```

```
