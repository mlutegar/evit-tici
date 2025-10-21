# EViT - Efficient Vision Transformer for Building Segmentation

EViT (Efficient Vision Transformer) is a deep learning project for semantic segmentation of buildings in satellite and aerial imagery. The model uses a Vision Transformer-based architecture (GLBViT) with linear-window multi-head self-attention for efficient and accurate building extraction from remote sensing images.

## Overview

This project implements a state-of-the-art semantic segmentation model designed specifically for building detection in aerial and satellite imagery. The architecture combines the power of Vision Transformers with efficient local-window attention mechanisms and multi-scale feature processing.

## Key Features

- **GLBViT Architecture**: Global-Local Vision Transformer backbone with efficient linear-window multi-head self-attention (LWMSA)
- **Multi-Scale Processing**: Feature Pyramid Network (FPN) for capturing buildings at different scales
- **Detail Preservation**: Dedicated detail path to maintain fine-grained building boundaries
- **Attention Mechanisms**:
  - Linear complexity window-based attention
  - Squeeze-and-Excitation blocks for channel attention
  - Spatial Pyramid Pooling (SPP) for multi-scale context
- **Data Augmentation**: Mosaic augmentation and standard geometric transformations
- **Multiple Dataset Support**: WHU Building Dataset and SpaceNet
- **Test Time Augmentation**: D4 and Left-Right flip options for improved inference

## Architecture Components

### GLBViT Model (EViT/geoseg/models/EViT.py)
The main model consists of:
- **Stem**: Initial feature extraction with strided convolutions
- **Global-Local Backbone (Glb)**: 4-stage transformer encoder with window-based attention
- **Detail Path**: Preserves high-resolution features for boundary accuracy
- **FPN Decoder**: Multi-scale feature fusion
- **SPP Module**: Spatial pyramid pooling for context aggregation
- **SE Attention**: Channel-wise feature recalibration

### Training Pipeline (EViT/train.py)
- PyTorch Lightning-based training framework
- Supports multiple loss functions (EdgeLoss, DiceLoss, etc.)
- Metrics: mIoU, F1-Score, Overall Accuracy
- Automatic mixed precision training
- Model checkpointing and logging

### Inference Pipeline (EViT/test.py)
- Batch inference with configurable batch size
- Optional Test Time Augmentation (TTA)
- Multi-process result writing
- Evaluation metrics calculation

## Project Structure

```
EViT/
├── EViT/                          # Main source code
│   ├── config/                    # Configuration files
│   │   ├── whubuilding/          # WHU Building dataset configs
│   │   └── spacenet/             # SpaceNet dataset configs
│   ├── geoseg/                    # Core library
│   │   ├── datasets/             # Dataset implementations
│   │   ├── losses/               # Loss functions (Dice, Focal, EdgeLoss, etc.)
│   │   └── models/               # Model architectures
│   │       └── EViT.py           # GLBViT model implementation
│   ├── tools/                     # Utility tools
│   │   ├── cfg.py                # Configuration parser
│   │   └── metric.py             # Evaluation metrics
│   ├── train.py                   # Training script
│   ├── test.py                    # Inference script
│   └── mask_convert.py            # Mask format conversion
├── data/                          # Dataset directory (not included)
│   ├── whubuilding/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── spacenet/
├── model_weights/                 # Saved model checkpoints
├── results/                       # Inference results
├── lightning_logs/               # Training logs
└── requirements.txt              # Python dependencies
```

## Installation

### Requirements
- Python 3.9
- python -m pip install "pip<24.1"
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dh609/EViT.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- `timm==0.5.4` - PyTorch Image Models
- `pytorch-lightning==1.5.9` - Training framework
- `albumentations==1.1.0` - Data augmentation
- `catalyst==20.9` - Optimization utilities
- `ttach==0.0.3` - Test time augmentation
- `einops` - Tensor operations
- `scipy` - Scientific computing

## Dataset Preparation

### Supported Datasets

1. **WHU Building Dataset**: [Download here](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
2. **SpaceNet Buildings Dataset V2**: [Download here](https://spacenet.ai/spacenet-buildings-dataset-v2/)

### WHU Building Dataset Structure

After downloading the WHU Building dataset, you'll find the following original structure:

```
WHU Building Dataset/
├── train_val/
│   ├── A/              # Aerial images (original RGB images)
│   └── OUT/            # Ground truth masks
├── val/
│   ├── A/              # Aerial images
│   └── OUT/            # Ground truth masks
└── test/
    ├── A/              # Aerial images
    └── OUT/            # Ground truth masks
```

### Organizing the Dataset

You need to reorganize the dataset to match the expected structure:

**Step 1**: Create the proper directory structure:

```bash
mkdir -p data/whubuilding/train_val/images
mkdir -p data/whubuilding/train_val/masks_origin
mkdir -p data/whubuilding/train_val/masks
mkdir -p data/whubuilding/val/images
mkdir -p data/whubuilding/val/masks_origin
mkdir -p data/whubuilding/val/masks
mkdir -p data/whubuilding/test/images
mkdir -p data/whubuilding/test/masks_origin
mkdir -p data/whubuilding/test/masks
```

**Step 2**: Copy the files from the original dataset:

```bash
# For training set
cp path/to/WHU_Building_Dataset/train_val/A/* data/whubuilding/train_val/images/
cp path/to/WHU_Building_Dataset/train_val/OUT/* data/whubuilding/train_val/masks_origin/

# For validation set
cp path/to/WHU_Building_Dataset/val/A/* data/whubuilding/val/images/
cp path/to/WHU_Building_Dataset/val/OUT/* data/whubuilding/val/masks_origin/

# For test set
cp path/to/WHU_Building_Dataset/test/A/* data/whubuilding/test/images/
cp path/to/WHU_Building_Dataset/test/OUT/* data/whubuilding/test/masks_origin/
```

**Final structure should be**:

```
data/
├── whubuilding/
│   ├── train_val/
│   │   ├── images/           # Copied from A/ folder (.tif files)
│   │   ├── masks_origin/     # Copied from OUT/ folder (.tif files)
│   │   └── masks/            # Will be created by mask_convert.py
│   ├── val/                  # Same structure as train_val
│   ├── test/                 # Same structure as train_val
└── spacenet/
    └── [same structure as whubuilding]
```

### Mask Conversion

The original masks from the OUT folder need to be converted to binary format. The script processes one directory at a time, so you need to run it separately for train, val, and test sets:

```bash
# Convert training masks
python mask_convert.py --mask-dir data/whubuilding/train_val/masks_origin --output-mask-dir data/whubuilding/train_val/masks

# Convert validation masks
python mask_convert.py --mask-dir data/whubuilding/val/masks_origin --output-mask-dir data/whubuilding/val/masks

# Convert test masks
python mask_convert.py --mask-dir data/whubuilding/test/masks_origin --output-mask-dir data/whubuilding/test/masks
```

**What the script does**:
- Reads RGB masks from `masks_origin/` folders
- Converts colors to binary labels:
  - Black pixels (0, 0, 0) → 1 (building)
  - White pixels (255, 255, 255) → 0 (background)
- Saves converted binary masks (.tif) to `masks/` folders
- Uses multiprocessing for faster conversion

**Note**: You only need to convert the datasets you have. If you only have WHU Building, you don't need to download SpaceNet.

## Usage

### Training

Train a model using a configuration file:

```bash
python train_val.py --config_path config/whubuilding/evit.py
```

**Configuration options** (in `config/whubuilding/evit.py`):
- `max_epoch`: Number of training epochs
- `train_batch_size`: Batch size for training
- `lr`: Learning rate
- `weights_path`: Directory to save model checkpoints
- `gpus`: List of GPU IDs to use
- `num_classes`: Number of segmentation classes

The training process will:
- Save checkpoints to `model_weights/whubuilding/`
- Log metrics to `lightning_logs/`
- Display mIoU, F1, and OA metrics for train and validation

### Inference

Run inference on test data:

```bash
python EViT/test.py \
    --config_path config/whubuilding/evit.py \
    --output_path results/whubuilding \
    --tta d4 \
    --rgb
```

**Arguments**:
- `--config_path`: Path to configuration file
- `--output_path`: Directory to save predicted masks
- `--tta`: Test time augmentation (`None`, `lr`, or `d4`)
  - `lr`: Horizontal and vertical flips
  - `d4`: D4 group (flips + 90° rotations)
- `--rgb`: Output RGB visualization (optional)

### Evaluation Metrics

The model reports:
- **mIoU** (mean Intersection over Union): Pixel-level segmentation accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **OA** (Overall Accuracy): Percentage of correctly classified pixels
- **Precision**: Ratio of true positive predictions
- **Recall**: Ratio of detected ground truth pixels

## Model Architecture Details

### Linear-Window Multi-head Self-Attention (LWMSA)
- Partitions input into non-overlapping windows
- Applies linear attention within each window
- Uses ELU feature mapping for linear complexity: O(N) instead of O(N²)
- Progressive head interaction: each head builds on previous heads

### Feature Pyramid Network (FPN)
- Fuses features from all 4 encoder stages
- Top-down pathway with lateral connections
- Multi-scale upsampling for detail preservation

### Training Strategy
- **Optimizer**: AdamW with Lookahead wrapper
- **LR Scheduler**: Cosine Annealing with Warm Restarts
- **Losses**: EdgeLoss (boundary-aware), DiceLoss, FocalLoss, etc.
- **Data Augmentation**:
  - Mosaic augmentation (25% probability)
  - Horizontal/vertical flips
  - Normalization

## Configuration

Key hyperparameters in `config/whubuilding/evit.py`:

```python
max_epoch = 110
train_batch_size = 4
val_batch_size = 4
lr = 1e-4
weight_decay = 0.0025
num_classes = 2  # Building, Background
monitor = 'val_mIoU'  # Metric for checkpoint selection
```

## Results

The model outputs:
- **Predicted masks**: Binary or RGB segmentation maps
- **Quantitative metrics**: F1, IoU, OA, Precision, Recall per class
- **Checkpoints**: Top-k best models based on validation mIoU

## Checkpoints

Models are saved to `model_weights/{dataset_name}/`:
- Best models based on validation mIoU
- Last checkpoint (optional)
- Format: PyTorch Lightning `.ckpt` files

## Citation

If you use this code, please cite the original EViT repository:
```
https://github.com/dh609/EViT
```

## License

Please refer to the original repository for license information.

## Acknowledgments

This project uses:
- [timm](https://github.com/rwightman/pytorch-image-models) for model components
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning) for training
- [Albumentations](https://github.com/albumentations-team/albumentations) for augmentation
- WHU and SpaceNet for datasets
#   e v i t - t i c i 
 
 