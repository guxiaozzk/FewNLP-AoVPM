## Requirements

- python 3.8
- pytorch 1.7.1
- timm 0.4.12
- scikit-learn 0.24.2 
- tqdm 4.59.0
- transformers 4.6.0
- pillow 8.2.0
- numpy 1.20.2

we mainly use `conda` commands to set up our environment and only use `pip` for installing `timm`.

## Prepare the data

You can download the datasets from the original links below and preprocess the data following the scrips in `datasets_pre_processing`, or you can download images from our [BaiduNetdisk](https://pan.baidu.com/s/1MBgH2SGg5D3WDepXhD_b4A) (access code `d2w2`) and use preprocessed data in `datasets` directly.

##  Experiments
You can run and update the run.sh directly, the related parameter and templates for prompt study maybe import from param.py and utils.py.
