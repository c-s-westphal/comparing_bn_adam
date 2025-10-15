# VGG Ablation Study: Batch Normalization and Adam Optimizer

This repository contains code for studying how different training choices (optimizer, batch normalization, data augmentation) affect mutual information (MI) values calculated through masking in VGG models.

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   └── vgg_standard.py          # VGG11/13/16/19 with configurable BN
├── scripts/
│   ├── job_manager_vgg11_ablation.sh
│   ├── job_manager_vgg13_ablation.sh
│   ├── job_manager_vgg16_ablation.sh
│   ├── job_manager_vgg19_ablation.sh
│   ├── jobs_vgg11_ablation.txt
│   ├── jobs_vgg13_ablation.txt
│   ├── jobs_vgg16_ablation.txt
│   └── jobs_vgg19_ablation.txt
├── train_vgg_ablation.py        # Training script with MI evaluation
├── checkpoints/                 # Model checkpoints (created during training)
├── results/                     # Training results and MI evaluations
└── plots/                       # Visualization outputs (future)
```

## Ablation Conditions

We study 8 different training configurations (ablations):

1. `adam_no_bn_no_aug` - Adam optimizer, no batch normalization, no data augmentation
2. `adam_no_bn_aug` - Adam optimizer, no batch normalization, with data augmentation
3. `adam_bn_no_aug` - Adam optimizer, with batch normalization, no data augmentation
4. `adam_bn_aug` - Adam optimizer, with batch normalization, with data augmentation
5. `adamw_no_bn_no_aug` - AdamW optimizer, no batch normalization, no data augmentation
6. `adamw_no_bn_aug` - AdamW optimizer, no batch normalization, with data augmentation
7. `adamw_bn_no_aug` - AdamW optimizer, with batch normalization, no data augmentation
8. `adamw_bn_aug` - AdamW optimizer, with batch normalization, with data augmentation

Each configuration is trained with 3 different random seeds (0, 1, 2) for statistical robustness.

## Training Configuration

- **Architectures**: VGG11, VGG13, VGG16, VGG19
- **Dataset**: CIFAR-10
- **Epochs**: 200
- **Batch Size**: 128
- **Learning Rate**: 0.001 with cosine annealing
- **Weight Decay**: 5e-4 (for AdamW), 0 (for Adam)
- **Data Augmentation**: RandomCrop(32, padding=4) + RandomHorizontalFlip

## Mutual Information Evaluation

MI is evaluated using first-layer channel masking:

- **During Training**: Every 10 epochs using 20 masks and 20 batches
- **Final Evaluation**: At epoch 200 using 40 masks and 40 batches

The MI difference is calculated as:
```
MI_diff = MI(Y; full_predictions) - mean(MI(Y; masked_predictions))
```

## Usage

### Local Testing

To test a single configuration locally:

```bash
python train_vgg_ablation.py \
    --arch vgg11 \
    --seed 0 \
    --optimizer adamw \
    --batchnorm bn \
    --augmentation aug \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --device cuda
```

### HPC Submission (SGE)

To submit all jobs for a specific architecture:

```bash
# VGG11 (24 jobs: 8 ablations × 3 seeds)
qsub scripts/job_manager_vgg11_ablation.sh

# VGG13 (24 jobs)
qsub scripts/job_manager_vgg13_ablation.sh

# VGG16 (24 jobs)
qsub scripts/job_manager_vgg16_ablation.sh

# VGG19 (24 jobs)
qsub scripts/job_manager_vgg19_ablation.sh
```

To submit all architectures at once:
```bash
for arch in vgg11 vgg13 vgg16 vgg19; do
    qsub scripts/job_manager_${arch}_ablation.sh
done
```

This will submit a total of 96 jobs (4 architectures × 8 ablations × 3 seeds).

## Output Files

### Checkpoints
```
checkpoints/{arch}_{optimizer}_{batchnorm}_{augmentation}_seed{seed}_final.pt
```

Example: `checkpoints/vgg11_adamw_bn_aug_seed0_final.pt`

### Results (NumPy format)
```
results/{arch}_{optimizer}_{batchnorm}_{augmentation}_seed{seed}_results.npz
```

Contains:
- `epochs_evaluated`: Epochs at which MI was evaluated
- `mi_history`: MI difference at each evaluation
- `train_acc_history`: Training accuracy at each evaluation
- `test_acc_history`: Test accuracy at each evaluation
- `gen_gap_history`: Generalization gap at each evaluation
- `final_mi_full`: Final MI with full model
- `final_mean_mi_masked`: Final mean MI with masking
- `final_mi_diff`: Final MI difference
- `final_train_acc`, `final_test_acc`, `final_gen_gap`: Final accuracies

### Results (JSON format)
```
results/{arch}_{optimizer}_{batchnorm}_{augmentation}_seed{seed}_results.json
```

Human-readable JSON with the same information as the .npz file.

## Job Management

The job manager scripts automatically:
- Skip training if checkpoint and results already exist
- Create necessary directories
- Handle environment setup (CUDA, Python, virtual environment)
- Log outputs to job-specific files

## Requirements

- Python 3.9+
- PyTorch 1.8+
- torchvision
- numpy
- scikit-learn

## Next Steps

After training completes, you can:
1. Aggregate results across seeds
2. Create visualizations comparing MI vs generalization gap
3. Analyze how each ablation affects the MI-generalization relationship
