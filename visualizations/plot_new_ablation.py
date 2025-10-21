"""
Visualization script for new ablation study: MI_diff vs Generalization Gap analysis.

Creates scatter plots showing the relationship between generalization gap
and MI difference (MI_full - mean(MI_masked)) for the new ablation structure:
- Weight decay (wd/no_wd)
- BatchNorm (bn/no_bn)
- Random crop (crop/no_crop)
- Random flip (flip/no_flip)
- Batch size (8 or 128)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(results_dir: Path, architecture: str, min_train_acc: float = 0.0):
    """Load all result files for a given architecture.

    Args:
        results_dir: Directory containing result files
        architecture: Architecture name to load
        min_train_acc: Minimum train accuracy threshold for filtering results

    Returns:
        dict: Dictionary with ablation names as keys and lists of results as values
    """
    results = defaultdict(list)
    filtered_count = 0

    # Find all result files for this architecture with new naming format
    pattern = f"{architecture}_*_results.npz"
    result_files = list(results_dir.glob(pattern))

    print(f"Found {len(result_files)} result files for {architecture}")

    for result_file in result_files:
        # Load the data
        data = np.load(result_file)

        # Extract ablation configuration from filename
        # Format: {arch}_{wd}_{bn}_{crop}_{flip}_bs{batch_size}_seed{seed}_results.npz
        filename = result_file.stem  # Remove .npz

        # Remove architecture prefix and _results suffix
        config_str = filename.replace(f"{architecture}_", "").replace("_results", "")

        # Extract seed (always at the end: seed0, seed1)
        import re
        seed_match = re.search(r'seed(\d+)$', config_str)
        if not seed_match:
            print(f"Warning: Could not parse seed from {filename}")
            continue
        seed = seed_match.group(1)

        # Remove seed from config string
        config_str = config_str[:seed_match.start()].rstrip('_')

        # Parse the 5 ablation dimensions
        # Format: {wd}_{bn}_{crop}_{flip}_bs{batch_size}
        parts = config_str.split('_')

        # Parse weight_decay
        if parts[0] == 'wd':
            weight_decay = 'wd'
            parts = parts[1:]
        elif parts[0] == 'no' and len(parts) > 1 and parts[1] == 'wd':
            weight_decay = 'no_wd'
            parts = parts[2:]
        else:
            print(f"Warning: Could not parse weight_decay from {filename}")
            continue

        # Parse batchnorm
        if parts[0] == 'bn':
            batchnorm = 'bn'
            parts = parts[1:]
        elif parts[0] == 'no' and len(parts) > 1 and parts[1] == 'bn':
            batchnorm = 'no_bn'
            parts = parts[2:]
        else:
            print(f"Warning: Could not parse batchnorm from {filename}")
            continue

        # Parse crop
        if parts[0] == 'crop':
            crop = 'crop'
            parts = parts[1:]
        elif parts[0] == 'no' and len(parts) > 1 and parts[1] == 'crop':
            crop = 'no_crop'
            parts = parts[2:]
        else:
            print(f"Warning: Could not parse crop from {filename}")
            continue

        # Parse flip
        if parts[0] == 'flip':
            flip = 'flip'
            parts = parts[1:]
        elif parts[0] == 'no' and len(parts) > 1 and parts[1] == 'flip':
            flip = 'no_flip'
            parts = parts[2:]
        else:
            print(f"Warning: Could not parse flip from {filename}")
            continue

        # Parse batch_size (format: bs8 or bs128)
        if parts[0].startswith('bs'):
            batch_size = parts[0]  # e.g., 'bs8' or 'bs128'
            batch_size_num = int(parts[0][2:])  # Extract the number
        else:
            print(f"Warning: Could not parse batch_size from {filename}")
            continue

        ablation_name = f"{weight_decay}_{batchnorm}_{crop}_{flip}_{batch_size}"

        # Extract values
        gen_gap = float(data['final_gen_gap'])
        mi_diff = float(data['final_mi_diff'])
        train_acc = float(data['final_train_acc'])

        # Filter by minimum train accuracy
        if train_acc < min_train_acc:
            filtered_count += 1
            continue

        results[ablation_name].append({
            'gen_gap': gen_gap,
            'mi_diff': mi_diff,
            'train_acc': train_acc,
            'seed': int(seed),
            'weight_decay': weight_decay,
            'batchnorm': batchnorm,
            'crop': crop,
            'flip': flip,
            'batch_size': batch_size,
            'batch_size_num': batch_size_num
        })

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} results with train_acc < {min_train_acc}%")

    return results


def plot_mi_diff_vs_gen_gap(results, architecture: str, output_dir: Path, group_by: str = 'weight_decay'):
    """Create scatter plot of MI_diff vs generalization gap.

    Args:
        results: Dictionary of results by ablation
        architecture: Architecture name for title
        output_dir: Directory to save the plot
        group_by: Which dimension to use for primary grouping ('weight_decay', 'batchnorm', 'augmentation')
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    # Define colors for weight decay
    wd_colors = {
        'wd': '#2ecc71',        # Green for weight decay
        'no_wd': '#e74c3c',     # Red for no weight decay
    }

    # Define markers for batchnorm + augmentation combinations
    markers = {
        'bn_both': 'o',         # Circle: BN + both augs
        'bn_crop': 's',         # Square: BN + crop only
        'bn_flip': '^',         # Triangle up: BN + flip only
        'bn_none': 'v',         # Triangle down: BN + no aug
        'no_bn_both': 'D',      # Diamond: No BN + both augs
        'no_bn_crop': 'p',      # Pentagon: No BN + crop only
        'no_bn_flip': '*',      # Star: No BN + flip only
        'no_bn_none': 'X',      # X: No BN + no aug
    }

    # Use different marker sizes for batch size
    sizes = {
        'bs8': 300,         # Larger for small batch
        'bs128': 150,       # Smaller for large batch
    }

    # Plot each ablation
    for ablation_name, data_points in sorted(results.items()):
        if not data_points:
            continue

        # Extract data
        gen_gaps = [d['gen_gap'] for d in data_points]
        mi_diffs = [d['mi_diff'] for d in data_points]

        # Get style parameters
        weight_decay = data_points[0]['weight_decay']
        batchnorm = data_points[0]['batchnorm']
        crop = data_points[0]['crop']
        flip = data_points[0]['flip']
        batch_size = data_points[0]['batch_size']
        batch_size_num = data_points[0]['batch_size_num']

        # Determine color (based on weight decay)
        color = wd_colors[weight_decay]

        # Determine marker (based on batchnorm + augmentation)
        aug_type = 'both' if crop == 'crop' and flip == 'flip' else \
                   'crop' if crop == 'crop' else \
                   'flip' if flip == 'flip' else \
                   'none'
        marker_key = f"{batchnorm}_{aug_type}"
        marker = markers[marker_key]

        # Determine size (based on batch_size)
        size = sizes[batch_size]

        # Create label
        wd_label = "WD" if weight_decay == "wd" else "No WD"
        bn_label = "BN" if batchnorm == "bn" else "No BN"
        aug_label = "Crop+Flip" if aug_type == "both" else \
                   "Crop" if aug_type == "crop" else \
                   "Flip" if aug_type == "flip" else \
                   "No Aug"
        bs_label = f"BS={batch_size_num}"

        label = f"{wd_label}, {bn_label}, {aug_label}, {bs_label}"

        # Plot
        ax.scatter(gen_gaps, mi_diffs,
                  c=color, marker=marker, s=size, alpha=0.6,
                  edgecolors='black', linewidths=1.5,
                  label=label)

    # Formatting
    ax.set_xlabel('Generalization Gap (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MI_diff (MI_full - mean(MI_masked))', fontsize=14, fontweight='bold')
    ax.set_title(f'MI Difference vs Generalization Gap\n{architecture.upper()} - New Ablation Study',
                fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{architecture}_mi_diff_vs_gen_gap_new.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF for publications
    output_path_pdf = output_dir / f"{architecture}_mi_diff_vs_gen_gap_new.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_path_pdf}")

    plt.close()


def plot_faceted_by_augmentation(results, architecture: str, output_dir: Path):
    """Create 4-panel faceted plot by augmentation type.

    Args:
        results: Dictionary of results by ablation
        architecture: Architecture name for title
        output_dir: Directory to save the plot
    """
    # Group results by augmentation type
    aug_groups = {
        'both': defaultdict(list),
        'crop': defaultdict(list),
        'flip': defaultdict(list),
        'none': defaultdict(list),
    }

    for ablation_name, data_points in results.items():
        if not data_points:
            continue

        crop = data_points[0]['crop']
        flip = data_points[0]['flip']

        aug_type = 'both' if crop == 'crop' and flip == 'flip' else \
                   'crop' if crop == 'crop' else \
                   'flip' if flip == 'flip' else \
                   'none'

        aug_groups[aug_type][ablation_name] = data_points

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    aug_titles = {
        'both': 'Crop + Flip',
        'crop': 'Crop Only',
        'flip': 'Flip Only',
        'none': 'No Augmentation'
    }

    # Colors for weight decay
    wd_colors = {
        'wd': '#2ecc71',
        'no_wd': '#e74c3c',
    }

    # Markers for batchnorm
    bn_markers = {
        'bn': 'o',
        'no_bn': 's',
    }

    # Sizes for batch_size
    sizes = {
        'bs8': 250,
        'bs128': 120,
    }

    for idx, (aug_type, aug_title) in enumerate(aug_titles.items()):
        ax = axes[idx]
        aug_results = aug_groups[aug_type]

        # Plot data for this augmentation type
        for ablation_name, data_points in sorted(aug_results.items()):
            if not data_points:
                continue

            gen_gaps = [d['gen_gap'] for d in data_points]
            mi_diffs = [d['mi_diff'] for d in data_points]

            weight_decay = data_points[0]['weight_decay']
            batchnorm = data_points[0]['batchnorm']
            batch_size = data_points[0]['batch_size']
            batch_size_num = data_points[0]['batch_size_num']

            color = wd_colors[weight_decay]
            marker = bn_markers[batchnorm]
            size = sizes[batch_size]

            wd_label = "WD" if weight_decay == "wd" else "No WD"
            bn_label = "BN" if batchnorm == "bn" else "No BN"
            bs_label = f"BS={batch_size_num}"

            label = f"{wd_label}, {bn_label}, {bs_label}"

            ax.scatter(gen_gaps, mi_diffs,
                      c=color, marker=marker, s=size, alpha=0.6,
                      edgecolors='black', linewidths=1.5,
                      label=label)

        # Format subplot
        ax.set_xlabel('Generalization Gap (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MI_diff', fontsize=12, fontweight='bold')
        ax.set_title(aug_title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'MI Difference vs Generalization Gap by Augmentation Type\n{architecture.upper()}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{architecture}_mi_diff_vs_gen_gap_faceted.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFaceted plot saved to: {output_path}")

    output_path_pdf = output_dir / f"{architecture}_mi_diff_vs_gen_gap_faceted.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Faceted plot saved to: {output_path_pdf}")

    plt.close()


def print_summary(results):
    """Print summary statistics of the loaded results."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for ablation_name, data_points in sorted(results.items()):
        if not data_points:
            continue

        gen_gaps = [d['gen_gap'] for d in data_points]
        mi_diffs = [d['mi_diff'] for d in data_points]
        train_accs = [d['train_acc'] for d in data_points]

        print(f"\n{ablation_name}:")
        print(f"  N runs: {len(data_points)}")
        print(f"  Train Acc: {np.mean(train_accs):.2f} ± {np.std(train_accs):.2f}%")
        print(f"  Gen Gap: {np.mean(gen_gaps):.2f} ± {np.std(gen_gaps):.2f}%")
        print(f"  MI_diff: {np.mean(mi_diffs):.4f} ± {np.std(mi_diffs):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot MI_diff vs Generalization Gap for new ablation study'
    )

    parser.add_argument('--arch', type=str, default='vgg13',
                       choices=['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19'],
                       help='Architecture to plot (default: vgg13)')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--min_train_acc', type=float, default=0.0,
                       help='Minimum train accuracy threshold for filtering results (default: 0.0)')
    parser.add_argument('--faceted', action='store_true',
                       help='Create faceted plot by augmentation type')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Load results
    print(f"Loading results for {args.arch}...")
    if args.min_train_acc > 0:
        print(f"Filtering results with train_acc >= {args.min_train_acc}%")
    results = load_results(results_dir, args.arch, args.min_train_acc)

    if not results:
        print(f"No results found for {args.arch} in {results_dir}")
        return

    # Print summary
    print_summary(results)

    # Create plots
    print(f"\nCreating main scatter plot...")
    plot_mi_diff_vs_gen_gap(results, args.arch, output_dir)

    if args.faceted:
        print(f"\nCreating faceted plot by augmentation type...")
        plot_faceted_by_augmentation(results, args.arch, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
