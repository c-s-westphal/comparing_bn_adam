# ResNet Integration for Ablation Study

This document summarizes the integration of ResNet architectures into the existing VGG ablation study framework.

## Summary

ResNet models (20, 32, 44, 56, 110) have been added to the ablation study with full compatibility with the existing:
- Masking and MI calculation infrastructure
- Ablation parameters (weight_decay, batchnorm, random_crop, random_flip, batch_size)
- Training scripts and job management
- Visualization scripts

## New Files Created

### 1. Model Architecture
- **`models/resnet_cifar.py`**: ResNet implementations for CIFAR-10
  - ResNet-20 (n=3): 272K parameters
  - ResNet-32 (n=5): 467K parameters
  - ResNet-44 (n=7): 661K parameters
  - ResNet-56 (n=9): 856K parameters
  - ResNet-110 (n=18): 1.7M parameters

### 2. Job Files
- **`scripts/jobs_resnet32_ablation.txt`**: 64 job configurations for ResNet-32
- **`scripts/jobs_resnet44_ablation.txt`**: 64 job configurations for ResNet-44
- **`scripts/jobs_resnet56_ablation.txt`**: 64 job configurations for ResNet-56

### 3. Job Manager Scripts
- **`scripts/job_manager_resnet32_ablation.sh`**: SGE job array script for ResNet-32
- **`scripts/job_manager_resnet44_ablation.sh`**: SGE job array script for ResNet-44
- **`scripts/job_manager_resnet56_ablation.sh`**: SGE job array script for ResNet-56

## Modified Files

### 1. Model Package
- **`models/__init__.py`**: Added ResNet imports and exports

### 2. Training Script
- **`train_vgg_ablation.py`**:
  - Updated docstring to mention ResNet
  - Added ResNet imports
  - Added ResNet architectures to `--arch` argument choices
  - Added ResNet models to `model_map`

### 3. Visualization Script
- **`visualizations/plot_new_ablation.py`**:
  - Added ResNet architectures to `--arch` argument choices

## Architecture Details

### ResNet for CIFAR-10
The ResNet implementation follows the original paper's CIFAR-10 architecture:
- Initial 3×3 conv with 16 filters
- 3 stages with increasing filters: [16, 32, 64]
- Downsampling at start of stages 2 and 3 (stride=2)
- Global Average Pooling + single FC layer
- Total layers = 6n + 2 (where n is blocks per stage)

### Compatibility with MI Calculation
ResNet models have a `features` attribute containing the initial conv block (Conv->BN->ReLU), making them compatible with the existing `get_first_conv_block_output()` function used for MI evaluation.

The first layer activations (after Conv-BN-ReLU) have shape (batch, 16, 32, 32), which will be masked and used for MI calculations exactly like VGG.

## Ablation Configuration

Each architecture has 64 jobs (32 configurations × 2 seeds):
- **Weight decay**: wd / no_wd
- **Batch norm**: bn / no_bn
- **Random crop**: crop / no_crop
- **Random flip**: flip / no_flip
- **Batch size**: 8 / 128
- **Seeds**: 0, 1

Total: 2^5 = 32 configurations per seed

## Usage

### Training a single configuration
```bash
python train_vgg_ablation.py \\
    --arch resnet32 \\
    --seed 0 \\
    --weight_decay_ablation wd \\
    --batchnorm bn \\
    --random_crop crop \\
    --random_flip flip \\
    --batch_size 128 \\
    --epochs 500 \\
    --lr 0.001 \\
    --weight_decay 5e-4
```

### Running full ablation study on cluster
```bash
# ResNet-32
qsub scripts/job_manager_resnet32_ablation.sh

# ResNet-44
qsub scripts/job_manager_resnet44_ablation.sh

# ResNet-56
qsub scripts/job_manager_resnet56_ablation.sh
```

### Plotting results
```bash
# Plot ResNet-32 results
python visualizations/plot_new_ablation.py --arch resnet32

# Plot ResNet-44 results
python visualizations/plot_new_ablation.py --arch resnet44

# Plot ResNet-56 results
python visualizations/plot_new_ablation.py --arch resnet56
```

## Expected Results Format

Results will be saved as:
- Checkpoints: `checkpoints/resnet32_wd_bn_crop_flip_bs128_seed0_final.pt`
- Results: `results/resnet32_wd_bn_crop_flip_bs128_seed0_results.npz`
- Plots: `plots/resnet32_mi_diff_vs_gen_gap_new.png`

## Verification

All ResNet models have been tested for:
- ✓ Correct parameter counts
- ✓ Forward pass with CIFAR-10 input (3×32×32)
- ✓ Batch normalization / no batch normalization variants
- ✓ Dropout variants
- ✓ `features` attribute compatibility
- ✓ First layer hook capture for MI calculation
- ✓ Integration with training script

## Notes

1. **ResNet-20** and **ResNet-110** are available but job files were not created (ResNet-20 may be too small, ResNet-110 very large)
2. **No BatchNorm + Batch Size 8** configurations are expected to fail (as observed with VGG)
3. The same optimizer (AdamW) and learning rate schedule (cosine annealing) are used as with VGG
4. ResNet typically trains faster than VGG due to skip connections and better gradient flow

## Recommended Architectures for Comparison

For comparing with VGG-13 (9.4M params):
- **ResNet-32** (467K params): Much smaller, good baseline
- **ResNet-44** (661K params): Still smaller than VGG
- **ResNet-56** (856K params): Comparable depth, still smaller

For deeper networks:
- **ResNet-110** (1.7M params): Much deeper, interesting for MI analysis
