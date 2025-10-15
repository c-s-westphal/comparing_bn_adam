"""
Training script for VGG ablation study with integrated MI evaluation.

Trains VGG models with various ablations (optimizer, batch norm, data augmentation)
and evaluates mutual information at regular intervals.
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import mutual_info_score

from models.vgg_standard import VGG11, VGG13, VGG16, VGG19


def get_data_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = './data',
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders.

    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of data loading workers
        data_root: Root directory for CIFAR-10 data
        use_augmentation: If True, use data augmentation for training
    """
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Train transform
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = test_transform

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def get_eval_loader(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = './data'
) -> DataLoader:
    """Create evaluation data loader (no augmentation, no shuffling)."""
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    eval_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=eval_transform
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return eval_loader


def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0
) -> float:
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    batches_processed = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batches_processed += 1
            if max_batches and batches_processed >= max_batches:
                break

    return 100. * correct / total


def generate_random_channel_masks(
    n_channels: int,
    height: int,
    width: int,
    n_masks: int,
    seed: int = 42
) -> List[np.ndarray]:
    """Generate random masks for element selection in first conv layer activations.

    Each mask randomly selects between 1 and (n_channels*h*w - 2) individual elements to keep.
    Returns masks where True = keep element, False = zero out element.
    Mask shape: (n_channels, height, width)
    """
    np.random.seed(seed)
    masks = []

    # Total number of elements in activation tensor
    total_elements = n_channels * height * width

    # Maximum subset size: all elements except 2
    max_subset_size = max(1, total_elements - 2)

    for _ in range(n_masks):
        # Random subset size between 1 and max_subset_size
        subset_size = np.random.randint(1, max_subset_size + 1)

        # Create flat mask and randomly select elements to KEEP
        flat_mask = np.zeros(total_elements, dtype=bool)
        selected_indices = np.random.choice(total_elements, subset_size, replace=False)
        flat_mask[selected_indices] = True

        # Reshape to 3D: (n_channels, height, width)
        mask = flat_mask.reshape(n_channels, height, width)
        masks.append(mask)

    return masks


class ChannelMaskingHook:
    """Hook to mask individual elements in convolutional layer output."""
    def __init__(self, mask: np.ndarray):
        """
        Args:
            mask: Boolean array where True = keep element, False = zero out element
                  Shape: (channels, height, width) for 3D element masking
        """
        self.mask = torch.from_numpy(mask).bool()

    def __call__(self, module, input, output):
        """Apply element-wise masking during forward pass."""
        # output shape: (batch, channels, height, width)
        # mask shape: (channels, height, width)
        masked_output = output.clone()

        # Apply mask element-wise across all spatial positions and channels
        # Broadcast mask across batch dimension
        masked_output = masked_output * self.mask.unsqueeze(0).to(output.device)

        return masked_output


def get_first_conv_block_output(model: nn.Module) -> nn.Module:
    """Get the module after which to apply masking (after Conv->BN->ReLU or Conv->ReLU).

    Returns the ReLU module that follows the first conv (and possibly bn) block.
    """
    if hasattr(model, 'features'):
        # Find the first Conv2d, then find the ReLU that follows it
        found_first_conv = False

        for module in model.features:
            if isinstance(module, nn.Conv2d) and not found_first_conv:
                found_first_conv = True
            elif found_first_conv and isinstance(module, nn.ReLU):
                # This is the ReLU after Conv (and possibly BN)
                return module

        if found_first_conv:
            raise ValueError("Found first conv but no ReLU after it")

    raise ValueError("Could not find first conv block in model")


def get_predictions_and_labels(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mask: np.ndarray = None,
    hook_layer: nn.Module = None,
    max_batches: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions and true labels.

    Args:
        model: The model to evaluate
        data_loader: Data loader
        device: Device to use
        mask: Optional mask to apply at hook_layer
        hook_layer: Optional layer to hook for masking
        max_batches: Maximum number of batches to process (0 = all)

    Returns:
        (predictions, labels) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_labels = []

    hook_handle = None
    if mask is not None and hook_layer is not None:
        hook = ChannelMaskingHook(mask)
        hook_handle = hook_layer.register_forward_hook(hook)

    try:
        batches_processed = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

                batches_processed += 1
                if max_batches and batches_processed >= max_batches:
                    break
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    return predictions, labels


def calculate_mutual_information(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Calculate mutual information between predictions and true labels.

    Uses discrete mutual information: I(Y; predictions)
    Maximum MI is log2(10) ≈ 3.32 bits for CIFAR-10.
    """
    return mutual_info_score(labels, predictions)


def evaluate_first_layer_mi(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    n_subsets: int,
    seed: int = 42,
    max_batches: int = 0
) -> Tuple[float, float, float]:
    """Evaluate MI difference between full and masked first conv layer.

    Returns:
        (mi_full, mean_mi_masked, mi_difference)
    """
    model.eval()

    # Get the ReLU after first conv block
    first_block_output = get_first_conv_block_output(model)

    # Determine first layer dimensions
    sample_batch = next(iter(eval_loader))
    sample_input = sample_batch[0][:1].to(device)

    hook_output = None
    def capture_hook(module, input, output):
        nonlocal hook_output
        hook_output = output

    handle = first_block_output.register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(sample_input)
    handle.remove()

    n_channels = hook_output.shape[1]
    height = hook_output.shape[2]
    width = hook_output.shape[3]

    # Generate element masks
    masks = generate_random_channel_masks(n_channels, height, width, n_subsets, seed)

    # Get predictions for full model
    full_predictions, labels = get_predictions_and_labels(
        model, eval_loader, device, mask=None, hook_layer=None, max_batches=max_batches
    )

    # Calculate MI for full model
    mi_full = calculate_mutual_information(full_predictions, labels)

    # Calculate MI for each masked version
    masked_mis = []
    for mask in masks:
        masked_predictions, _ = get_predictions_and_labels(
            model, eval_loader, device, mask=mask, hook_layer=first_block_output, max_batches=max_batches
        )
        mi_masked = calculate_mutual_information(masked_predictions, labels)
        masked_mis.append(mi_masked)

    # Calculate mean MI across all masks
    mean_mi_masked = np.mean(masked_mis)
    mi_difference = mi_full - mean_mi_masked

    return mi_full, mean_mi_masked, mi_difference


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train VGG with ablations and MI evaluation')

    # Model arguments
    parser.add_argument('--arch', type=str, required=True,
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='Model architecture')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')

    # Ablation arguments
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['adam', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--batchnorm', type=str, required=True,
                        choices=['bn', 'no_bn'],
                        help='Whether to use batch normalization')
    parser.add_argument('--augmentation', type=str, required=True,
                        choices=['aug', 'no_aug'],
                        help='Whether to use data augmentation')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (for AdamW)')

    # Evaluation arguments
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluate MI every N epochs')
    parser.add_argument('--n_masks_train', type=int, default=20,
                        help='Number of masks for training MI evaluation')
    parser.add_argument('--n_masks_final', type=int, default=40,
                        help='Number of masks for final MI evaluation')
    parser.add_argument('--max_eval_batches_train', type=int, default=20,
                        help='Max batches for training MI evaluation')
    parser.add_argument('--max_eval_batches_final', type=int, default=40,
                        help='Max batches for final MI evaluation')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse ablation flags
    use_batchnorm = (args.batchnorm == 'bn')
    use_augmentation = (args.augmentation == 'aug')

    # Create model
    model_map = {
        'vgg11': VGG11,
        'vgg13': VGG13,
        'vgg16': VGG16,
        'vgg19': VGG19
    }
    model = model_map[args.arch](num_classes=10, use_batchnorm=use_batchnorm)
    model = model.to(device)

    print(f"\nModel: {args.arch.upper()}")
    print(f"Batch Normalization: {use_batchnorm}")
    print(f"Data Augmentation: {use_augmentation}")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_dir,
        use_augmentation=use_augmentation
    )

    # Create evaluation loader (no augmentation, no shuffle)
    eval_loader = get_eval_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_dir
    )

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Tracking
    mi_history = []
    train_acc_history = []
    test_acc_history = []
    gen_gap_history = []
    epochs_evaluated = []

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Step scheduler
        scheduler.step()

        # Evaluate accuracy
        test_acc = evaluate_accuracy(model, test_loader, device)
        gen_gap = train_acc - test_acc

        # Print progress
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%, "
                  f"Gen Gap: {gen_gap:.2f}%")

        # Evaluate MI at specified intervals
        if epoch % args.eval_interval == 0 or epoch == 1:
            print(f"  Evaluating MI (n_masks={args.n_masks_train}, max_batches={args.max_eval_batches_train})...")
            mi_full, mean_mi_masked, mi_diff = evaluate_first_layer_mi(
                model, eval_loader, device,
                n_subsets=args.n_masks_train,
                seed=args.seed + epoch,  # Different seed each time
                max_batches=args.max_eval_batches_train
            )
            print(f"  MI: {mi_full:.6f}, MI_masked: {mean_mi_masked:.6f}, MI_diff: {mi_diff:.6f}")

            mi_history.append(mi_diff)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            gen_gap_history.append(gen_gap)
            epochs_evaluated.append(epoch)

    print("\n" + "="*70)
    print("Training completed!")

    # Final MI evaluation with more masks and batches
    print(f"\nFinal MI evaluation (n_masks={args.n_masks_final}, max_batches={args.max_eval_batches_final})...")
    final_mi_full, final_mean_mi_masked, final_mi_diff = evaluate_first_layer_mi(
        model, eval_loader, device,
        n_subsets=args.n_masks_final,
        seed=args.seed,
        max_batches=args.max_eval_batches_final
    )

    print(f"Final MI: {final_mi_full:.6f}")
    print(f"Final MI_masked: {final_mean_mi_masked:.6f}")
    print(f"Final MI_diff: {final_mi_diff:.6f}")

    # Save final checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ablation_name = f"{args.optimizer}_{args.batchnorm}_{args.augmentation}"
    checkpoint_path = checkpoint_dir / f"{args.arch}_{ablation_name}_seed{args.seed}_final.pt"

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'arch': args.arch,
        'seed': args.seed,
        'use_batchnorm': use_batchnorm,
        'use_augmentation': use_augmentation,
        'optimizer_name': args.optimizer,
        'train_acc': train_acc_history[-1] if train_acc_history else train_acc,
        'test_acc': test_acc_history[-1] if test_acc_history else test_acc,
        'gen_gap': gen_gap_history[-1] if gen_gap_history else gen_gap,
    }, checkpoint_path)

    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{args.arch}_{ablation_name}_seed{args.seed}_results.npz"

    np.savez(
        results_path,
        epochs_evaluated=np.array(epochs_evaluated),
        mi_history=np.array(mi_history),
        train_acc_history=np.array(train_acc_history),
        test_acc_history=np.array(test_acc_history),
        gen_gap_history=np.array(gen_gap_history),
        final_mi_full=final_mi_full,
        final_mean_mi_masked=final_mean_mi_masked,
        final_mi_diff=final_mi_diff,
        final_train_acc=train_acc,
        final_test_acc=test_acc,
        final_gen_gap=gen_gap,
        arch=args.arch,
        seed=args.seed,
        use_batchnorm=use_batchnorm,
        use_augmentation=use_augmentation,
        optimizer=args.optimizer,
    )

    print(f"Results saved to: {results_path}")

    # Also save detailed JSON
    json_path = results_dir / f"{args.arch}_{ablation_name}_seed{args.seed}_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'arch': args.arch,
            'seed': args.seed,
            'optimizer': args.optimizer,
            'use_batchnorm': use_batchnorm,
            'use_augmentation': use_augmentation,
            'epochs': args.epochs,
            'final_train_acc': float(train_acc),
            'final_test_acc': float(test_acc),
            'final_gen_gap': float(gen_gap),
            'final_mi_full': float(final_mi_full),
            'final_mean_mi_masked': float(final_mean_mi_masked),
            'final_mi_diff': float(final_mi_diff),
            'epochs_evaluated': [int(e) for e in epochs_evaluated],
            'mi_history': [float(m) for m in mi_history],
            'gen_gap_history': [float(g) for g in gen_gap_history],
        }, f, indent=2)

    print(f"Detailed results saved to: {json_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
