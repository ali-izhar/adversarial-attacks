# Data Directory

This directory is for storing image datasets used for adversarial attack evaluation.

## Current Setup

The repository currently includes a sample ImageNet dataset with the following structure:

```
data/
└── imagenet/
    ├── sample_images/          # Contains 1000 sample images for testing
    └── imagenet_classes.txt    # List of 1000 ImageNet class labels
```

The `imagenet_classes.txt` file is sourced from [PyTorch's GitHub repository](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt) and contains the 1000 class labels used in ImageNet.

This sample setup is sufficient for quick testing and development of adversarial attacks.

## Obtaining Full ImageNet

1. **Official ImageNet**: Register at [image-net.org](https://image-net.org/) and download the validation set (~6.3 GB)
2. **Academic Access**: Use your institution's resources if available

## Alternatives

If full ImageNet is too large, the following can be used:

1. **ImageNette**: A 10-class subset of ImageNet available from [FastAI](https://github.com/fastai/imagenette)
2. **CIFAR-10**: A smaller dataset that can be automatically downloaded through `torchvision`

## Configuration

Modify `--data-dir` in `experiments/compare.py` to point to your dataset location:

```bash
python experiments/compare.py --data-dir=/path/to/data --dataset=imagenet
```
